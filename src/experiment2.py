"""实验二：独立 LHS、固定容量运行和 15 个发生年份的生命周期评估。

正常状态保留消纳率约束且无上网限制；压力状态释放消纳率约束，
采用 LOCKED_EXPORT_MAX_KW。容量和设备退化的时间起点始终保持不变。
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import csv
import json

import numpy as np
from scipy.stats import norm

from src import config as cfg
from src.data import write_json, write_csv, source_fingerprint
from src.model import solve_locked, npv_at

DEFAULT_SCENARIOS = cfg.OUTPUT_ROOT / "lhs_domain_design_gamma1" / "lhs_scenarios_design.csv"
DEFAULT_BASELINE_SUMMARY = cfg.BASELINE_OUTPUT / "run_summary.json"
DEFAULT_OUTPUT = cfg.LHS_OUTPUT
CACHE_SCHEMA = 4

def compact_locked_result(result: dict[str, Any]) -> dict[str, Any]:
    """保留路径组装所需字段，并拒绝未认证或缺失逐年审计的结果。"""
    if result.get("status") != "optimal":
        raise RuntimeError(f"锁定容量求解未认证：{result.get('status')}")

    cashflows = tuple(float(v) for v in result.get("cashflows_actual_investor", ()))
    user_costs = tuple(float(v) for v in result.get("annual_user_costs", ()))
    outside_costs = tuple(float(v) for v in result.get("annual_outside_costs", ()))
    if len(cashflows) != cfg.LIFE + 1:
        raise RuntimeError("锁定容量结果缺少完整15年投资方现金流")
    if len(user_costs) != cfg.LIFE or len(outside_costs) != cfg.LIFE:
        raise RuntimeError("锁定容量结果缺少完整15年用户成本审计序列")

    pv_kw = float(result["pv_kw"])
    ess_kwh = float(result["ess_kwh"])
    if not np.isfinite([pv_kw, ess_kwh]).all() or min(pv_kw, ess_kwh) < 0.0:
        raise RuntimeError("锁定容量结果包含无效容量")

    return {
        "status": "optimal",
        "pv_kw": pv_kw,
        "ess_kwh": ess_kwh,
        "capacity": [pv_kw, ess_kwh],
        "npv_actual_investor": float(result["npv_actual_investor"]),
        "cashflows_actual_investor": list(cashflows),
        "annual_user_costs": list(user_costs),
        "annual_outside_costs": list(outside_costs),
        "audit": result.get("audit", {}),
    }


def assemble_locked_path(
    base: dict[str, Any],
    stress: dict[str, Any],
    onset_year: int,
    *,
    discount_rate: float = cfg.R_REQ,
    user_tolerance_yuan: float = 1.0,
) -> dict[str, Any]:
    """组装一条固定容量路径并重新审计NPV、容量和用户成本。"""
    if not isinstance(onset_year, int) or isinstance(onset_year, bool):
        raise TypeError("onset_year 必须为整数")
    if not 1 <= onset_year <= cfg.LIFE:
        raise ValueError(f"onset_year 必须位于 [1,{cfg.LIFE}]")

    base_capacity = np.asarray(base.get("capacity", ()), dtype=float)
    stress_capacity = np.asarray(stress.get("capacity", ()), dtype=float)
    if base_capacity.shape != (2,) or stress_capacity.shape != (2,):
        raise ValueError("基准与压力结果必须包含二元容量")
    capacity_locked = bool(np.allclose(base_capacity, stress_capacity, atol=1e-6, rtol=0.0))
    if not capacity_locked:
        raise ValueError(
            "路径两端容量不一致，不能用于锁定容量分析："
            f"base={base_capacity.tolist()}, stress={stress_capacity.tolist()}"
        )

    base_cf = np.asarray(base["cashflows_actual_investor"], dtype=float)
    stress_cf = np.asarray(stress["cashflows_actual_investor"], dtype=float)
    base_user = np.asarray(base["annual_user_costs"], dtype=float)
    stress_user = np.asarray(stress["annual_user_costs"], dtype=float)
    base_outside = np.asarray(base["annual_outside_costs"], dtype=float)
    stress_outside = np.asarray(stress["annual_outside_costs"], dtype=float)
    if base_cf.shape != (cfg.LIFE + 1,) or stress_cf.shape != (cfg.LIFE + 1,):
        raise ValueError("现金流序列长度必须为 LIFE+1")
    for values, label in (
        (base_user, "base annual_user_costs"),
        (stress_user, "stress annual_user_costs"),
        (base_outside, "base annual_outside_costs"),
        (stress_outside, "stress annual_outside_costs"),
    ):
        if values.shape != (cfg.LIFE,):
            raise ValueError(f"{label} 长度必须为 LIFE")

    path_cf = base_cf.copy()
    path_cf[onset_year:] = stress_cf[onset_year:]
    split = onset_year - 1
    path_user = np.concatenate((base_user[:split], stress_user[split:]))
    path_outside = np.concatenate((base_outside[:split], stress_outside[split:]))

    npv = float(npv_at(path_cf, discount_rate))
    base_npv = float(npv_at(base_cf, discount_rate))
    loss = float(base_npv - npv)
    user_project = float(path_user.sum())
    user_outside = float(path_outside.sum())
    user_margin = float(user_outside - user_project)
    user_constraint_ok = bool(user_margin >= -abs(float(user_tolerance_yuan)))

    return {
        "onset_year": onset_year,
        "capacity": base_capacity.tolist(),
        "capacity_locked": capacity_locked,
        "npv_actual_investor": npv,
        "base_npv_actual_investor": base_npv,
        "npv_loss": loss,
        "npv_loss_rate": loss / abs(base_npv) if abs(base_npv) > 1e-12 else None,
        "npv_negative": bool(npv < 0.0),
        "cashflows_actual_investor": path_cf.tolist(),
        "annual_user_costs": path_user.tolist(),
        "annual_outside_costs": path_outside.tolist(),
        "user_cost_project": user_project,
        "user_cost_outside": user_outside,
        "user_cost_margin": user_margin,
        "user_constraint_ok": user_constraint_ok,
        "npv_reconstruction_error": abs(npv - npv_at(path_cf, discount_rate)),
        "dispatch_policy": "statewise_persistent_dispatch",
        "dispatch_policy_note": (
            "基准与压力状态分别优化固定容量运行；路径按状态切换相应运行策略，"
            "容量不变，拼接后重新审计全生命周期用户成本。"
        ),
    }


def validate_onset_sequence(
    paths: list[dict[str, Any]], *, tolerance_yuan: float = 1.0
) -> dict[str, Any]:
    """检查同一下行情景下，压力推迟后损失是否非增。"""
    if not paths:
        raise ValueError("paths 不能为空")
    ordered = sorted(paths, key=lambda row: int(row["onset_year"]))
    losses = np.asarray([float(row["npv_loss"]) for row in ordered], dtype=float)
    increments = np.diff(losses)
    max_violation = float(max(0.0, float(increments.max()))) if increments.size else 0.0
    return {
        "onset_years": [int(row["onset_year"]) for row in ordered],
        "losses": losses.tolist(),
        "loss_nonincreasing_with_later_onset": bool(
            np.all(increments <= abs(float(tolerance_yuan)))
        ),
        "max_upward_violation_yuan": max_violation,
        "all_capacity_locked": all(bool(row["capacity_locked"]) for row in ordered),
        "all_user_constraints_ok": all(bool(row["user_constraint_ok"]) for row in ordered),
        "max_npv_reconstruction_error": float(
            max(float(row["npv_reconstruction_error"]) for row in ordered)
        ),
    }


def _lhs_uniform(n: int, bounds: tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    lo, hi = bounds
    u = (np.arange(n, dtype=float) + rng.random(n)) / n
    rng.shuffle(u)
    return lo + (hi - lo) * u


def _strata_complete(values: np.ndarray, bounds: tuple[float, float]) -> bool:
    lo, hi = bounds
    strata = np.floor((values - lo) / (hi - lo) * len(values)).astype(int)
    strata = np.clip(strata, 0, len(values) - 1)
    return len(np.unique(strata)) == len(values)


def _weights(raw: np.ndarray) -> np.ndarray:
    if np.any(~np.isfinite(raw)) or np.any(raw < 0) or raw.sum() <= 0:
        raise ValueError("无效权重")
    return raw / raw.sum()


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """以加权质量中心插值；等权时与 numpy 线性分位数一致。"""
    if not 0 <= q <= 1:
        raise ValueError("q必须位于[0,1]")
    order = np.argsort(values, kind="stable")
    v = np.asarray(values, dtype=float)[order]
    w = _weights(np.asarray(weights, dtype=float)[order])
    centers = np.cumsum(w) - 0.5 * w
    if len(v) == 1:
        return float(v[0])
    x = (centers - centers[0]) / (centers[-1] - centers[0])
    return float(np.interp(q, x, v))


def weighted_cvar_upper(values: np.ndarray, weights: np.ndarray, tail: float) -> float:
    """最高损失 tail 质量的均值，边界场景按剩余质量截取。"""
    if not 0 < tail <= 1:
        raise ValueError("tail必须位于(0,1]")
    order = np.argsort(values)[::-1]
    v = np.asarray(values, dtype=float)[order]
    w = _weights(np.asarray(weights, dtype=float)[order])
    used = np.minimum(w, np.maximum(0.0, tail - (np.cumsum(w) - w)))
    return float(np.dot(v, used) / tail)


def gaussian_copula_weights(u: np.ndarray, v: np.ndarray, rho_s: float) -> np.ndarray:
    """独立均匀提案样本到指定 Spearman 相关的高斯 copula 密度比。"""
    rho = 2.0 * np.sin(np.pi * rho_s / 6.0)
    z1 = norm.ppf(np.clip(u, 1e-8, 1.0 - 1e-8))
    z2 = norm.ppf(np.clip(v, 1e-8, 1.0 - 1e-8))
    log_density = (
        -0.5 * np.log(1.0 - rho * rho)
        + (2.0 * rho * z1 * z2 - rho * rho * (z1 * z1 + z2 * z2))
        / (2.0 * (1.0 - rho * rho))
    )
    return _weights(np.exp(log_density - log_density.max()))


def _weighted_corr(u: np.ndarray, v: np.ndarray, weights: np.ndarray) -> float:
    mean_u = float(np.dot(weights, u))
    mean_v = float(np.dot(weights, v))
    cu = u - mean_u
    cv = v - mean_v
    return float(
        np.dot(weights, cu * cv)
        / np.sqrt(np.dot(weights, cu * cu) * np.dot(weights, cv * cv))
    )


def _load_scenarios(
    path: Path,
    *,
    expected_n: int = 1000,
    alpha_bounds: tuple[float, float] = (0.25, 1.0),
    gamma_bounds: tuple[float, float] = cfg.LHS_GAMMA_BOUNDS,
) -> list[dict[str, float | int]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    scenarios = [
        {
            "scenario_id": int(row["scenario_id"]),
            "alpha": float(row["alpha"]),
            "gamma": float(row["gamma"]),
        }
        for row in raw
    ]
    if len(scenarios) != expected_n:
        raise RuntimeError(f"预期{expected_n}点，当前为{len(scenarios)}")
    ids = [int(row["scenario_id"]) for row in scenarios]
    if len(set(ids)) != len(ids) or sorted(ids) != list(range(1, expected_n + 1)):
        raise RuntimeError(f"场景编号必须唯一且连续为1..{expected_n}")
    alpha = np.asarray([float(row["alpha"]) for row in scenarios])
    gamma = np.asarray([float(row["gamma"]) for row in scenarios])
    if not (np.isfinite(alpha).all() and np.isfinite(gamma).all()):
        raise RuntimeError("LHS 场景包含非有限数值")
    if alpha.min() < alpha_bounds[0] or alpha.max() > alpha_bounds[1]:
        raise RuntimeError(f"alpha超出指定范围{list(alpha_bounds)}")
    if gamma.min() < gamma_bounds[0] or gamma.max() > gamma_bounds[1]:
        raise RuntimeError(f"gamma超出指定范围{list(gamma_bounds)}")
    return scenarios


def _load_capacity(path: Path) -> tuple[float, float]:
    value = json.loads(path.read_text(encoding="utf-8"))
    baseline = value.get("baseline", {})
    capacity = (float(baseline["pv_kw"]), float(baseline["ess_kwh"]))
    if not np.isfinite(capacity).all() or min(capacity) <= 0.0:
        raise RuntimeError(f"基准容量无效：{capacity}")
    return capacity


def _load_cache(
    path: Path,
    *,
    alpha: float,
    gamma: float,
    capacity: tuple[float, float],
    enforce_self_use: bool,
    export_max_kw: float | None,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    result = value.get("result", {})
    cached_export = value.get("export_max_kw")
    export_match = (
        cached_export is None
        if export_max_kw is None
        else cached_export is not None
        and np.isclose(float(cached_export), export_max_kw)
    )
    if (
        value.get("schema") != CACHE_SCHEMA
        or value.get("source_fingerprint") != source_fingerprint()
        or value.get("mutex_formulation") != cfg.MUTEX_FORMULATION
        or bool(value.get("enforce_self_use")) != enforce_self_use
        or not export_match
        or not np.isclose(float(value.get("alpha", np.nan)), alpha)
        or not np.isclose(float(value.get("gamma", np.nan)), gamma)
        or not np.allclose(value.get("capacity", ()), capacity)
        or len(result.get("annual_user_costs", ())) != cfg.LIFE
        or len(result.get("annual_outside_costs", ())) != cfg.LIFE
    ):
        return None
    return result


def generate_scenarios(
    output: Path = cfg.OUTPUT_ROOT / "lhs_domain_design_gamma1", *,
    n: int = cfg.LHS_N, seed: int = cfg.LHS_SEED,
    alpha_bounds: tuple[float, float] = cfg.LHS_ALPHA_BOUNDS,
    gamma_bounds: tuple[float, float] = cfg.LHS_GAMMA_BOUNDS,
) -> dict:
    """论文采样域的独立均匀 LHS；同一种子复现原有场景顺序。"""
    if n < 10:
        raise ValueError("n 至少为 10")
    if not (0 <= alpha_bounds[0] < alpha_bounds[1] and 0 <= gamma_bounds[0] < gamma_bounds[1]):
        raise ValueError("采样范围必须为非负递增区间")
    rng = np.random.default_rng(seed)
    alpha = _lhs_uniform(n, alpha_bounds, rng)
    gamma = _lhs_uniform(n, gamma_bounds, rng)
    rows = [{"scenario_id": i + 1, "alpha": float(alpha[i]), "gamma": float(gamma[i])}
            for i in range(n)]
    output = Path(output)
    scenario_path = output / "lhs_scenarios_design.csv"
    # 不覆盖已有不同设计，避免复跑时将场景与缓存错配。
    if scenario_path.exists():
        existing = _load_scenarios(scenario_path, expected_n=n,
                                   alpha_bounds=alpha_bounds, gamma_bounds=gamma_bounds)
        if existing != rows:
            raise RuntimeError(f"已有场景与本次种子或范围不同，请指定新输出目录：{scenario_path}")
    else:
        write_csv(scenario_path, rows)
    ranks_a, ranks_g = np.argsort(np.argsort(alpha)), np.argsort(np.argsort(gamma))
    correlation = float(np.corrcoef(ranks_a, ranks_g)[0, 1])
    checks = {
        "alpha_lhs_all_strata_once": _strata_complete(alpha, alpha_bounds),
        "gamma_lhs_all_strata_once": _strata_complete(gamma, gamma_bounds),
        "all_points_inside_domain": bool(np.all((alpha >= alpha_bounds[0]) & (alpha <= alpha_bounds[1]))
                                          and np.all((gamma >= gamma_bounds[0]) & (gamma <= gamma_bounds[1]))),
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "stage": "lhs_pressure_domain_design", "n": n, "seed": seed,
        "sampling": "independent_uniform_lhs", "alpha_range": list(alpha_bounds),
        "gamma_range": list(gamma_bounds), "spearman_alpha_gamma": correlation,
        "checks": checks, "artifacts": {"scenarios": str(scenario_path)},
        "interpretation": "独立均匀 LHS 的等权场景统计不代表现实发生概率。",
    }
    write_json(output / "domain_summary.json", report)
    return report


def run(
    *, scenario_path: Path | None = None,
    baseline_summary: Path = DEFAULT_BASELINE_SUMMARY,
    output: Path = DEFAULT_OUTPUT, threads: int = 20, workers: int = 1,
    mutex_formulation: str = "sos", time_limit: float = 600.0,
    limit: int | None = None, expected_n: int = cfg.LHS_N,
    seed: int = cfg.LHS_SEED,
    alpha_bounds: tuple[float, float] = cfg.LHS_ALPHA_BOUNDS,
    gamma_bounds: tuple[float, float] = cfg.LHS_GAMMA_BOUNDS,
    require_downside_monotonic: bool | None = None,
) -> dict:
    """一键准备采样并运行正式固定容量实验；存在基准时不重新规划。"""
    from src.experiment1 import run_baseline, _config_overrides

    output, baseline_summary = Path(output), Path(baseline_summary)
    if time_limit <= 0:
        raise ValueError("time_limit 必须为正")
    if scenario_path is None:
        design = generate_scenarios(output / "scenarios", n=expected_n, seed=seed,
                                    alpha_bounds=alpha_bounds, gamma_bounds=gamma_bounds)
        scenario_path = Path(design["artifacts"]["scenarios"])
    if not baseline_summary.exists():
        run_baseline(output=baseline_summary.parent, threads=threads,
                     time_limit=time_limit, mutex_formulation=mutex_formulation)
    value = json.loads(baseline_summary.read_text(encoding="utf-8"))
    baseline = value.get("baseline", {})
    if (value.get("status") != "complete" or not np.isclose(baseline.get("alpha", np.nan), 1.0)
            or not np.isclose(baseline.get("gamma", np.nan), cfg.BASELINE_GAMMA)):
        raise RuntimeError("实验二必须使用 alpha=1、gamma=BASELINE_GAMMA 的已认证基准容量")
    with _config_overrides(MUTEX=True, MUTEX_FORMULATION=mutex_formulation):
        if value.get("source_fingerprint") and value["source_fingerprint"] != source_fingerprint():
            raise RuntimeError("基准容量来自不同的代码、数据或参数，请先重新运行 baseline 或显式指定归档基准")
        return _run_lhs(
            scenario_path=Path(scenario_path), baseline_summary=baseline_summary,
            output=output, threads=threads, workers=workers,
            mutex_formulation=mutex_formulation, time_limit=time_limit, limit=limit,
            expected_n=expected_n, alpha_bounds=alpha_bounds, gamma_bounds=gamma_bounds,
            require_downside_monotonic=require_downside_monotonic,
        )


def _locked_state(
    *,
    alpha: float,
    gamma: float,
    capacity: tuple[float, float],
    cache_path: Path,
    threads: int,
    time_limit: float,
    enforce_self_use: bool,
    export_max_kw: float | None,
) -> tuple[dict[str, Any], bool]:
    cached = _load_cache(
        cache_path,
        alpha=alpha,
        gamma=gamma,
        capacity=capacity,
        enforce_self_use=enforce_self_use,
        export_max_kw=export_max_kw,
    )
    if cached is not None:
        return cached, True
    solved = solve_locked(
        alpha,
        gamma,
        capacity,
        include_user=True,
        enforce_self_use=enforce_self_use,
        export_max_kw=export_max_kw,
        threads=threads,
        time_limit=time_limit,
    )
    if export_max_kw is not None and solved["export_peak_kw"] > export_max_kw + 1e-5:
        raise RuntimeError(
            f"并网峰值超过锁定容量上限：{solved['export_peak_kw']} > {export_max_kw}"
        )
    compact = compact_locked_result(solved)
    write_json(
        cache_path,
        {
            "schema": CACHE_SCHEMA,
            "source_fingerprint": source_fingerprint(),
            "mutex_formulation": cfg.MUTEX_FORMULATION,
            "enforce_self_use": enforce_self_use,
            "export_max_kw": export_max_kw,
            "alpha": alpha,
            "gamma": gamma,
            "capacity": list(capacity),
            "result": compact,
        },
    )
    return compact, False


def _state_audit_ok(state: dict[str, Any]) -> bool:
    audit = state.get("audit", {})
    limits = (
        ("max_load_balance_error_kw", 1e-5),
        ("max_pv_balance_error_kw", 1e-5),
        ("max_pv_direct_priority_error_kw", 1e-5),
        ("max_soc_transition_error_kwh", 1e-5),
        ("soc_wrap_error_kwh", 1e-5),
        ("user_cost_reconstruction_error", 1e-4),
        ("cashflow_reconstruction_error", 1e-4),
    )
    return (
        state.get("status") == "optimal"
        and all(abs(float(audit.get(key, np.inf))) <= limit for key, limit in limits)
        and int(audit.get("simultaneous_export_grid_to_user_steps", -1)) == 0
        and int(audit.get("simultaneous_chg_dis_steps", -1)) == 0
        and bool(audit.get("monthly_demand_upper_ok", False))
    )


def _solve_one_process(
    scenario: dict[str, float | int],
    capacity: tuple[float, float],
    cache_dir: Path,
    threads: int,
    time_limit: float,
    mutex_formulation: str,
    enforce_self_use: bool,
    export_max_kw: float | None,
) -> tuple[dict[str, float | int], dict[str, Any], bool]:
    """独立进程求解一个状态，避免建模阶段受 Python GIL 限制。"""
    cfg.MUTEX_FORMULATION = mutex_formulation
    sid = int(scenario["scenario_id"])
    state, cached = _locked_state(
        alpha=float(scenario["alpha"]),
        gamma=float(scenario["gamma"]),
        capacity=capacity,
        cache_path=cache_dir / f"stress_{sid:04d}.json",
        threads=threads,
        time_limit=time_limit,
        enforce_self_use=enforce_self_use,
        export_max_kw=export_max_kw,
    )
    return scenario, state, cached


def _summaries(rows: list[dict[str, Any]], *, expected_n: int = 1000) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for onset in range(1, cfg.LIFE + 1):
        selected = [row for row in rows if int(row["onset_year"]) == onset]
        if len(selected) != expected_n:
            raise RuntimeError(f"第{onset}年结果不是{expected_n}个")
        losses = np.asarray([float(row["npv_loss"]) for row in selected])
        npvs = np.asarray([float(row["npv_actual_investor"]) for row in selected])
        order = np.argsort(losses)[::-1]
        worst_5 = losses[order[: max(1, int(np.ceil(0.05 * len(losses))))]]
        worst_10 = losses[order[: max(1, int(np.ceil(0.10 * len(losses))))]]
        summaries.append(
            {
                "onset_year": onset,
                "loss_q05": float(np.quantile(losses, 0.05)),
                "loss_q25": float(np.quantile(losses, 0.25)),
                "loss_q50": float(np.quantile(losses, 0.50)),
                "loss_q75": float(np.quantile(losses, 0.75)),
                "loss_q95": float(np.quantile(losses, 0.95)),
                "npv_q05": float(np.quantile(npvs, 0.05)),
                "npv_q25": float(np.quantile(npvs, 0.25)),
                "npv_q50": float(np.quantile(npvs, 0.50)),
                "npv_q75": float(np.quantile(npvs, 0.75)),
                "npv_q95": float(np.quantile(npvs, 0.95)),
                "npv_negative_count": int(np.sum(npvs < 0.0)),
                "npv_negative_share": float(np.mean(npvs < 0.0)),
                "conditional_cvar_worst_5pct": float(np.mean(worst_5)),
                "conditional_cvar_worst_10pct": float(np.mean(worst_10)),
                "loss_min": float(losses.min()),
                "loss_max": float(losses.max()),
            }
        )
    return summaries


def _run_lhs(
    *,
    scenario_path: Path = DEFAULT_SCENARIOS,
    baseline_summary: Path = DEFAULT_BASELINE_SUMMARY,
    output: Path = DEFAULT_OUTPUT,
    threads: int = 20,
    workers: int = 1,
    mutex_formulation: str = cfg.MUTEX_FORMULATION,
    time_limit: float = 600.0,
    limit: int | None = None,
    expected_n: int = 1000,
    alpha_bounds: tuple[float, float] = (0.25, 1.0),
    gamma_bounds: tuple[float, float] = cfg.LHS_GAMMA_BOUNDS,
    require_downside_monotonic: bool | None = None,
) -> dict[str, Any]:
    if workers < 1 or threads < 1:
        raise ValueError("workers和threads必须为正整数")
    if mutex_formulation not in {"binary", "sos", "indicator"}:
        raise ValueError("mutex_formulation无效")
    if expected_n < 10:
        raise ValueError("expected_n至少为10")
    cfg.MUTEX_FORMULATION = mutex_formulation
    if require_downside_monotonic is None:
        require_downside_monotonic = gamma_bounds[1] <= cfg.BASELINE_GAMMA
    scenarios = _load_scenarios(
        scenario_path,
        expected_n=expected_n,
        alpha_bounds=alpha_bounds,
        gamma_bounds=gamma_bounds,
    )
    if limit is not None:
        if not 1 <= limit <= len(scenarios):
            raise ValueError("limit超出范围")
        scenarios_to_run = scenarios[:limit]
    else:
        scenarios_to_run = scenarios
    capacity = _load_capacity(baseline_summary)
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = output / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output / "checkpoint.json"
    checkpoint = {
        "status": "running",
        "scenario_source": str(scenario_path),
        "capacity_source": str(baseline_summary),
        "capacity": list(capacity),
        "total": len(scenarios_to_run),
        "completed": 0,
        "last_scenario_id": None,
    }
    write_json(checkpoint_path, checkpoint)

    base, base_cached = _locked_state(
        alpha=1.0,
        gamma=cfg.BASELINE_GAMMA,
        capacity=capacity,
        cache_path=cache_dir / "baseline.json",
        threads=threads,
        time_limit=time_limit,
        enforce_self_use=True,
        export_max_kw=None,
    )
    if not _state_audit_ok(base):
        raise RuntimeError("基准固定容量状态未通过运行审计")

    states: dict[int, dict[str, Any]] = {}
    cache_hits = int(base_cached)

    def solve_one(scenario: dict[str, float | int]):
        sid = int(scenario["scenario_id"])
        state, cached = _locked_state(
            alpha=float(scenario["alpha"]),
            gamma=float(scenario["gamma"]),
            capacity=capacity,
            cache_path=cache_dir / f"stress_{sid:04d}.json",
            threads=threads,
            time_limit=time_limit,
            enforce_self_use=False,
            export_max_kw=cfg.LOCKED_EXPORT_MAX_KW,
        )
        return scenario, state, cached

    completed = 0
    if workers == 1:
        solved_items = (solve_one(scenario) for scenario in scenarios_to_run)
        for scenario, state, cached in solved_items:
            sid = int(scenario["scenario_id"])
            if not _state_audit_ok(state):
                raise RuntimeError(f"场景{sid}未通过运行审计")
            states[sid] = state
            cache_hits += int(cached)
            completed += 1
            checkpoint.update(
                {
                    "completed": completed,
                    "last_scenario_id": sid,
                    "cache_hits": cache_hits,
                }
            )
            write_json(checkpoint_path, checkpoint)
            print(
                f"[{completed}/{len(scenarios_to_run)}] scenario={sid} "
                f"alpha={scenario['alpha']:.6f} gamma={scenario['gamma']:.6f} "
                f"{'cache' if cached else 'solved'}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _solve_one_process, scenario, capacity, cache_dir,
                    threads, time_limit, mutex_formulation, False,
                    cfg.LOCKED_EXPORT_MAX_KW,
                ): scenario
                for scenario in scenarios_to_run
            }
            for future in as_completed(futures):
                scenario, state, cached = future.result()
                sid = int(scenario["scenario_id"])
                if not _state_audit_ok(state):
                    raise RuntimeError(f"场景{sid}未通过运行审计")
                states[sid] = state
                cache_hits += int(cached)
                completed += 1
                checkpoint.update(
                    {
                        "completed": completed,
                        "last_scenario_id": sid,
                        "cache_hits": cache_hits,
                    }
                )
                write_json(checkpoint_path, checkpoint)
                print(
                    f"[{completed}/{len(scenarios_to_run)}] scenario={sid} "
                    f"alpha={scenario['alpha']:.6f} gamma={scenario['gamma']:.6f} "
                    f"{'cache' if cached else 'solved'}",
                    flush=True,
                )

    rows: list[dict[str, Any]] = []
    invalid_user_paths: list[dict[str, int]] = []
    nonmonotonic_scenarios: list[int] = []
    for scenario in scenarios_to_run:
        sid = int(scenario["scenario_id"])
        scenario_rows: list[dict[str, Any]] = []
        for onset in range(1, cfg.LIFE + 1):
            path = assemble_locked_path(base, states[sid], onset)
            row = {
                "scenario_id": sid,
                "alpha": float(scenario["alpha"]),
                "gamma": float(scenario["gamma"]),
                "onset_year": onset,
                "pv_kw": capacity[0],
                "ess_kwh": capacity[1],
                "npv_actual_investor": float(path["npv_actual_investor"]),
                "npv_loss": float(path["npv_loss"]),
                "npv_loss_rate": float(path["npv_loss_rate"]),
                "npv_negative": int(bool(path["npv_negative"])),
                "user_cost_project": float(path["user_cost_project"]),
                "user_cost_outside": float(path["user_cost_outside"]),
                "user_cost_margin": float(path["user_cost_margin"]),
                "user_constraint_ok": int(bool(path["user_constraint_ok"])),
                "npv_reconstruction_error": float(path["npv_reconstruction_error"]),
            }
            scenario_rows.append(row)
            if not path["user_constraint_ok"]:
                invalid_user_paths.append({"scenario_id": sid, "onset_year": onset})
        losses = np.asarray([float(row["npv_loss"]) for row in scenario_rows])
        if np.any(np.diff(losses) > 1.0):
            nonmonotonic_scenarios.append(sid)
        rows.extend(scenario_rows)

    expected_rows = len(scenarios_to_run) * cfg.LIFE
    if len(rows) != expected_rows:
        raise RuntimeError("场景—发生年份结果行数不完整")
    if invalid_user_paths:
        raise RuntimeError(
            f"{len(invalid_user_paths)}条混合路径违反用户成本约束，正式统计已停止"
        )
    if nonmonotonic_scenarios and require_downside_monotonic:
        raise RuntimeError(
            f"{len(nonmonotonic_scenarios)}个下行情景的损失未随发生年份推迟而非增"
        )

    write_csv(output / "timing_results.csv", rows)
    summary = _summaries(rows, expected_n=expected_n) if limit is None else []
    if summary:
        write_csv(output / "timing_summary.csv", summary)

    checks = {
        "all_state_solves_certified_and_audited": True,
        "all_capacities_locked": True,
        "all_mixed_user_constraints_ok": True,
        "all_npv_reconstruction_errors_le_1e_6": max(
            float(row["npv_reconstruction_error"]) for row in rows
        )
        <= 1e-6,
    }
    if require_downside_monotonic:
        checks["all_downside_losses_nonincreasing_with_later_onset"] = not nonmonotonic_scenarios
    else:
        checks["extended_price_domain_audit_mode"] = True

    result = {
        "status": "complete",
        "stage": (
            "locked_capacity_lhs_main"
            if expected_n == 1000
            else "locked_capacity_lhs_sample_size"
        ),
        "formal_run": limit is None,
        "scenario_count": len(scenarios_to_run),
        "onset_years": list(range(1, cfg.LIFE + 1)),
        "result_rows": len(rows),
        "sampling": "independent_uniform_lhs",
        "mutex_formulation": mutex_formulation,
        "locked_export_max_kw": cfg.LOCKED_EXPORT_MAX_KW,
        "baseline_enforce_self_use": True,
        "stress_enforce_self_use": False,
        "alpha_range": list(alpha_bounds),
        "gamma_range": list(gamma_bounds),
        "scenario_source": str(scenario_path),
        "capacity_source": str(baseline_summary),
        "capacity": {"pv_kw": capacity[0], "ess_kwh": capacity[1]},
        "baseline_npv": float(base["npv_actual_investor"]),
        "cache_hits": cache_hits,
        "checks": checks,
        "summary": summary,
        "interpretation": (
            "结果是在独立均匀LHS状态测度下的条件场景统计，"
            "NPV负值占比不是真实亏损概率。"
        ),
        "onset_monotonicity": {
            "required": require_downside_monotonic,
            "violating_scenario_count": len(nonmonotonic_scenarios),
            "violating_scenario_ids": nonmonotonic_scenarios,
            "note": (
                "仅当价格域不高于基准价格、且压力状态被定义为纯下行情景时，"
                "才要求损失随压力推迟而非增；扩展到gamma>0.8后该假设不适用。"
            ),
        },
        "artifacts": {
            "timing_results": str(output / "timing_results.csv"),
            "timing_summary": (
                str(output / "timing_summary.csv") if summary else None
            ),
            "checkpoint": str(checkpoint_path),
        },
    }
    write_json(output / "run_summary.json", result)
    checkpoint.update(
        {
            "status": "complete",
            "completed": len(scenarios_to_run),
            "last_scenario_id": int(scenarios_to_run[-1]["scenario_id"]),
        }
    )
    write_json(checkpoint_path, checkpoint)
    return result
