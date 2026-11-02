"""实验一：基准容量、投资状态边界和逐情景重优化敏感性。

只执行计算并保存结果。图形由 plots.py 读取结果后独立生成。
边界搜索及断点续算沿用原有实现，保留 LP 边界近似的显式标记。
"""
from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

import numpy as np

from src import config as cfg
from src.data import load_data, write_json, write_csv, source_fingerprint
from src.model import solve_planning, npv_at

TOL = 1e-4
# 经济边界仍以 NPV=0 为零线；求解器的 MIP 间隙作为单独的结果不确定度
# 输出，不能直接改写经济可行性的定义。
V_TOL = 1e-5
SOLVER_VALUE_TOL = max(V_TOL, float(cfg.MIP_GAP_ABS))

# criterion -> 可行性随 γ 上升(True) 或下降(False)
_ASCENDING = {"investor": True, "joint": True, "user": False}
_PROBES = 5
_ALPHA_PROBES = 7
_ALPHA_SEARCH_MAX_ITER = 12


def _pv_only_physical_prefilter(alpha: float) -> dict:
    """快速判定 PV-only 是否可能满足最低消纳率。

    PV-only 没有储能时，给定装机容量 C 的最大可能自用电量为
    ``sum(min(C * pv, alpha * load))``。该比例随 C 增大不会提高，
    因此只需检查最小建设容量 200 kW（或当前配置值）。这与主模型
    的功率平衡、允许弃光/上网和 MIN_SELF_USE 约束同源；若此处失败，
    强制 PV-only 一定物理不可行，无需再让 MILP 花时间认证。
    """
    data = load_data()
    min_pv = float(cfg.PV_MIN_ENTRY_KW)
    generation = min_pv * np.asarray(data["pv"], dtype=float)
    load = float(alpha) * np.asarray(data["load"], dtype=float)
    max_self_use = float(np.minimum(generation, load).sum())
    required_self_use = float(cfg.MIN_SELF_USE * generation.sum())
    feasible = max_self_use + 1e-9 >= required_self_use
    return {
        "feasible": feasible,
        "minimum_pv_kw": min_pv,
        "max_self_use_kwh": max_self_use,
        "required_self_use_kwh": required_self_use,
        "reason": (
            "pv_only_min_build_fails_self_use"
            if not feasible
            else "pv_only_min_build_passes_self_use"
        ),
    }


def _planning_summary_cache(
    alpha: float,
    gamma: float,
    *,
    discount_rate: float | None,
    include_user: bool,
    investment_state: str,
    cache: dict | None,
    solver_threads: int | None,
    allow_uncertified: bool = False,
) -> dict:
    """Return a compact cached planning result for boundary searches.

    Boundary logic needs status, NPV, and both installed capacities. Caching
    the full 8760-step dispatch would consume substantial memory, so only
    these scalar fields are retained.

    ``include_user`` 只进入缓存键；用户总成本约束在所有边界探针中恒为
    开启（合同硬约束），该形参不再作为开关使用。
    """
    solve_kwargs = {
        "discount_rate": discount_rate,
        "include_user": include_user,
        "investment_state": investment_state,
        "allow_uncertified": allow_uncertified,
    }
    if solver_threads is not None:
        solve_kwargs["threads"] = solver_threads
    if cache is None:
        return solve_planning(alpha, gamma, **solve_kwargs)
    key = (
        float(alpha),
        float(gamma),
        None if discount_rate is None else float(discount_rate),
        bool(include_user),
        investment_state,
        solver_threads,
        bool(allow_uncertified),
    )
    if key not in cache:
        # 低负荷下 PV-only 的 70% 消纳约束可能使整个强制状态
        # 物理不可行。先做精确的快速筛选，避免把“无法认证”误写成
        # no_solution；此时若 PV+BESS 有解，边界逻辑会自然落到直接光储。
        if investment_state == "pv_only":
            prefilter = _pv_only_physical_prefilter(alpha)
            if not prefilter["feasible"]:
                cache[key] = {
                    "status": "infeasible",
                    "npv": None,
                    "pv_kw": None,
                    "ess_kwh": 0.0,
                    "solver_status": None,
                    "time_limited": False,
                    "sol_count": 0,
                    "physical_prefilter": True,
                    "prefilter_reason": prefilter["reason"],
                    "prefilter": prefilter,
                }
                return cache[key]
        result = solve_planning(alpha, gamma, **solve_kwargs)
        cache[key] = {
            "status": result.get("status"),
            "npv": result.get("npv"),
            "npv_planning": result.get("npv_planning", result.get("npv")),
            "npv_actual_investor": result.get("npv_actual_investor", result.get("npv")),
            "grid_charging_cost_pv": result.get("grid_charging_cost_pv"),
            "pv_kw": result.get("pv_kw"),
            "ess_kwh": result.get("ess_kwh"),
            "solver_status": result.get("solver_status"),
            "time_limited": result.get("time_limited"),
            "sol_count": result.get("sol_count"),
            "physical_prefilter": False,
        }
    return cache[key]


def _finite_number(value) -> bool:
    """Return whether a value can safely participate in a boundary test."""
    try:
        return value is not None and np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _boundary_npv(result: dict):
    """最终投资方边界 NPV；测试桩缺字段时回退到 npv。"""
    for key in ("npv_actual_investor", "boundary_npv", "npv"):
        value = result.get(key)
        if _finite_number(value):
            return float(value)
    return None


def _planning_npv(result: dict):
    """容量规划目标 NPV；仅用于记录，不能替代边界判据。"""
    for key in ("npv_planning", "npv"):
        value = result.get(key)
        if _finite_number(value):
            return float(value)
    return None


def _effective_ess_entry_kwh() -> float:
    """Return the effective energy entry threshold used by ``model.py``."""
    return max(
        float(cfg.ESS_MIN_ENTRY_KWH),
        float(cfg.ESS_ENTRY_POWER_KW) / float(cfg.C_RATE),
    )


def _state_capacity_available(result: dict, *, battery: bool) -> bool:
    """Check the half-continuous construction thresholds when capacities exist.

    The forced-state solver already imposes these constraints. Missing
    capacity fields are therefore treated as compatible for lightweight test
    stubs and for old in-memory callers; real solver results always include
    the fields and are checked explicitly.
    """
    pv_kw = result.get("pv_kw")
    if _finite_number(pv_kw) and float(pv_kw) < float(cfg.PV_MIN_ENTRY_KW) - 1e-6:
        return False
    ess_kwh = result.get("ess_kwh")
    if battery:
        if _finite_number(ess_kwh) and float(ess_kwh) < _effective_ess_entry_kwh() - 1e-6:
            return False
    elif _finite_number(ess_kwh) and float(ess_kwh) >= _effective_ess_entry_kwh() - 1e-6:
        return False
    return True


def _value_available(result: dict, *, battery: bool) -> bool:
    """Whether a forced state has a usable NPV value."""
    return (
        result.get("status") in {"optimal", "solver_not_certified"}
        and _finite_number(_boundary_npv(result))
        and _state_capacity_available(result, battery=battery)
    )


def classify_build_state(
    pv_only: dict,
    pv_battery: dict,
    *,
    value_tol: float = V_TOL,
    allow_uncertified: bool = False,
) -> dict:
    """Classify one ``(alpha, gamma)`` point using ``V_P`` and ``V_B``.

    ``pv_only`` and ``pv_battery`` are summaries of the corresponding forced
    construction states. Physical infeasibility is represented as ``-∞``;
    solver failure remains ``no_solution`` so it cannot silently create an
    economic region. A positive PV-only value wins ties and also wins when a
    profitable battery-equipped design has a lower NPV.
    """
    if value_tol < 0.0 or not np.isfinite(value_tol):
        raise ValueError("value_tol 必须为非负有限数")

    pv_status = pv_only.get("status")
    battery_status = pv_battery.get("status")
    allowed_statuses = {"optimal"}
    if allow_uncertified:
        allowed_statuses.add("solver_not_certified")
    pv_known = pv_status in allowed_statuses and _value_available(
        pv_only, battery=False
    )
    battery_known = battery_status in allowed_statuses and _value_available(
        pv_battery, battery=True
    )

    solver_unknown = {"no_solution"}
    if not allow_uncertified:
        solver_unknown.add("solver_not_certified")
    if (not pv_known and pv_status in solver_unknown) or (
        not battery_known and battery_status in solver_unknown
    ):
        return {
            "state": "no_solution",
            "reason": "forced_state_solver_unavailable",
            "pv_status": pv_status,
            "battery_status": battery_status,
            "pv_npv": _boundary_npv(pv_only) if pv_known else None,
            "battery_npv": _boundary_npv(pv_battery) if battery_known else None,
            "pv_npv_actual": _boundary_npv(pv_only) if pv_known else None,
            "battery_npv_actual": _boundary_npv(pv_battery) if battery_known else None,
            "pv_npv_planning": _planning_npv(pv_only) if pv_known else None,
            "battery_npv_planning": _planning_npv(pv_battery) if battery_known else None,
        }

    pv_value = _boundary_npv(pv_only) if pv_known else -np.inf
    battery_value = _boundary_npv(pv_battery) if battery_known else -np.inf

    if not pv_known and not battery_known:
        state = "no_invest_physical"
        reason = "both_forced_states_physically_unavailable"
    elif pv_value <= value_tol and battery_value <= value_tol:
        state = "no_invest_economic"
        reason = "both_forced_state_values_nonpositive"
    elif pv_value > value_tol and (
        not battery_known or pv_value >= battery_value - value_tol
    ):
        state = "pv_only"
        reason = "pv_only_positive_and_not_worse_than_battery"
    elif battery_value > value_tol and pv_value <= value_tol:
        state = "direct_to_pv_battery"
        reason = "battery_positive_while_pv_only_nonpositive"
    elif battery_value > pv_value + value_tol:
        state = "pv_battery"
        reason = "battery_value_strictly_exceeds_pv_only"
    else:
        # This is only reachable inside the numerical tolerance band. Keep
        # the tie-breaking rule explicit and conservative.
        state = "pv_only" if pv_value > value_tol else "no_invest_economic"
        reason = "tie_within_value_tolerance"

    return {
        "state": state,
        "reason": reason,
        "pv_status": pv_status,
        "battery_status": battery_status,
        "pv_npv": float(pv_value) if np.isfinite(pv_value) else None,
        "battery_npv": float(battery_value) if np.isfinite(battery_value) else None,
        "pv_npv_actual": float(pv_value) if np.isfinite(pv_value) else None,
        "battery_npv_actual": float(battery_value) if np.isfinite(battery_value) else None,
        "pv_npv_planning": _planning_npv(pv_only) if pv_known else None,
        "battery_npv_planning": _planning_npv(pv_battery) if battery_known else None,
        "pv_grid_charging_cost_pv": pv_only.get("grid_charging_cost_pv"),
        "battery_grid_charging_cost_pv": pv_battery.get("grid_charging_cost_pv"),
        "pv_kw": pv_only.get("pv_kw"),
        "ess_kwh": pv_battery.get("ess_kwh"),
        "investable": state in {"pv_only", "pv_battery", "direct_to_pv_battery"},
    }


def _evaluate(
    alpha: float,
    gamma: float,
    discount_rate: float | None,
    criterion: str,
    investment_state: str | None = None,
) -> dict:
    kwargs = {
        "discount_rate": discount_rate,
        # 用户总成本约束（用户不吃亏）是合同硬约束，边界探针必须与规划、
        # 锁定评估同口径；判据只决定用哪个指标判可行，不再决定该约束开关。
        "include_user": True,
    }
    if investment_state is None:
        # The boundary must evaluate the best forced-build value; the
        # no-investment solution has NPV=0 and is not an entry boundary.
        kwargs["require_build"] = True
    else:
        if investment_state not in {"pv_only", "pv_battery"}:
            raise ValueError("investment_state must be pv_only or pv_battery")
        kwargs["investment_state"] = investment_state
    return solve_planning(alpha, gamma, **kwargs)


def _feasible(result: dict, criterion: str) -> bool:
    if result["status"] != "optimal":
        return False
    npv = _boundary_npv(result)
    if not _finite_number(npv):
        return False
    if criterion == "user":
        return result["user_saving"] >= -V_TOL
    if criterion == "joint":
        return npv >= -V_TOL and result["user_saving"] >= -V_TOL
    return npv >= -V_TOL


def _scan_line(
    axis: str,
    fixed: float,
    lo: float,
    hi: float,
    discount_rate: float | None,
    criterion: str,
    investment_state: str | None = None,
) -> dict:
    """五点扫描定状态；全域结论只在对全部探针一致时给出。"""
    values = [float(v) for v in np.linspace(lo, hi, _PROBES)]
    flags, model_infeasible, no_solution = [], 0, 0
    for v in values:
        r = _evaluate(
            v if axis == "a" else fixed,
            fixed if axis == "a" else v,
            discount_rate,
            criterion,
            investment_state,
        )
        if r["status"] == "no_solution":
            no_solution += 1
            flags.append(False)
        elif r["status"] == "infeasible":
            model_infeasible += 1
            flags.append(False)
        elif r["status"] != "optimal":
            no_solution += 1
            flags.append(False)
        else:
            flags.append(_feasible(r, criterion))
    if no_solution or 0 < model_infeasible < len(values):
        return {
            "status": "no_solution",
            "threshold": None,
            "note": "探针含模型无解或求解异常，无法全域分类",
        }
    if model_infeasible == len(values):
        return {"status": "all_model_infeasible", "threshold": None}
    if all(flags):
        return {"status": "all_feasible", "threshold": None}
    if not any(flags):
        return {"status": "all_economically_infeasible", "threshold": None}
    monotone = all(b >= a for a, b in zip(flags, flags[1:])) or all(
        b <= a for a, b in zip(flags, flags[1:])
    )
    if not monotone:
        return {"status": "non_monotonic", "threshold": None}
    return {"status": "threshold", "threshold": None}


def _bisect(
    axis: str,
    fixed: float,
    lo: float,
    hi: float,
    discount_rate: float | None,
    criterion: str,
    tol: float,
    investment_state: str | None = None,
) -> tuple[str, float | None]:
    """二分临界点；方向按判据的可行性单调性；探针非 optimal 立即中止。"""
    ascending = _ASCENDING[criterion]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        r = _evaluate(
            mid if axis == "a" else fixed,
            fixed if axis == "a" else mid,
            discount_rate,
            criterion,
            investment_state,
        )
        if r["status"] != "optimal":
            return "no_solution", None  # 无解区间上不构造虚假阈值
        ok = _feasible(r, criterion)
        if (ok and ascending) or (not ok and not ascending):
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) <= tol:
            break
    return "threshold", 0.5 * (lo + hi)


def gamma_boundary(
    alpha: float,
    gamma_lo: float,
    gamma_hi: float,
    *,
    discount_rate: float | None = None,
    criterion: str = "investor",
    tol: float = TOL,
) -> dict:
    """固定 α，γ 方向二分 NPV=0 临界价格。"""
    scan = _scan_line("g", alpha, gamma_lo, gamma_hi, discount_rate, criterion)
    if scan["status"] != "threshold":
        return scan
    status, threshold = _bisect(
        "g", alpha, gamma_lo, gamma_hi, discount_rate, criterion, tol
    )
    return {"status": status, "threshold": threshold}


def alpha_boundary(
    gamma: float,
    alpha_lo: float,
    alpha_hi: float,
    *,
    discount_rate: float | None = None,
    criterion: str = "investor",
    tol: float = TOL,
) -> dict:
    """固定 γ，α 方向二分 NPV=0 临界负荷。"""
    scan = _scan_line("a", gamma, alpha_lo, alpha_hi, discount_rate, criterion)
    if scan["status"] != "threshold":
        return scan
    status, threshold = _bisect(
        "a", gamma, alpha_lo, alpha_hi, discount_rate, criterion, tol
    )
    return {"status": status, "threshold": threshold}


# ---------------------------------------------------------------------------
# 第一阶段：三状态投资区域的两条独立边界（唯一新实现，见下文）
# ---------------------------------------------------------------------------

_STATE_BOUNDARY_KINDS = {"pv_entry", "battery_upgrade"}
_ALPHA_EPS = 1e-6


def _state_metric(
    alpha: float,
    gamma: float,
    *,
    kind: str,
    discount_rate: float | None,
    include_user: bool,
    cache: dict | None,
    solver_threads: int | None = None,
    allow_uncertified: bool = False,
) -> dict:
    """计算第一阶段某条边界的标量指标。

    ``pv_entry`` 使用 ``max(V_P, V_B)``，其中物理不可行的强制状态按
    ``-∞`` 处理；``battery_upgrade`` 使用 ``V_B - V_P``，但只在两个
    强制状态都物理可用时计算。物理不可行不等于求解失败：前者可以
    形成“不投资”或“直接进入光储”分类，后者必须保留为未知状态。
    """
    if kind not in _STATE_BOUNDARY_KINDS:
        raise ValueError(f"invalid state boundary kind: {kind}")
    allowed_statuses = {"optimal"}
    if allow_uncertified:
        allowed_statuses.add("solver_not_certified")

    pv = _planning_summary_cache(
        alpha,
        gamma,
        discount_rate=discount_rate,
        include_user=include_user,
        investment_state="pv_only",
        cache=cache,
        solver_threads=solver_threads,
        allow_uncertified=allow_uncertified,
    )
    pv_status = pv.get("status")
    pv_npv_planning = pv.get("npv_planning", pv.get("npv"))
    pv_npv = _boundary_npv(pv)
    pv_ok = pv_status in allowed_statuses and _value_available(pv, battery=False)

    battery = _planning_summary_cache(
        alpha,
        gamma,
        discount_rate=discount_rate,
        include_user=include_user,
        investment_state="pv_battery",
        cache=cache,
        solver_threads=solver_threads,
        allow_uncertified=allow_uncertified,
    )
    battery_status = battery.get("status")
    battery_npv_planning = battery.get("npv_planning", battery.get("npv"))
    battery_npv = _boundary_npv(battery)
    battery_ok = battery_status in allowed_statuses and _value_available(
        battery, battery=True
    )

    classification = classify_build_state(
        pv,
        battery,
        value_tol=V_TOL,
        allow_uncertified=allow_uncertified,
    )

    def solver_unknown(status: str | None) -> bool:
        return status not in allowed_statuses and status != "infeasible"

    def result_status() -> str:
        if not solver_unknown(pv_status) and not solver_unknown(battery_status):
            if pv_status == "optimal" and battery_status == "optimal":
                return "optimal"
            # An infeasible alternative is a physical boundary condition, not
            # a solver failure; the remaining optimal state still defines
            # V_P or V_B for the entry boundary.
            return "optimal"
        return "solver_not_certified"

    common = {
        "pv_status": pv_status,
        "battery_status": battery_status,
        "pv_npv": float(pv_npv) if pv_ok else None,
        "battery_npv": float(battery_npv) if battery_ok else None,
        # pv_npv/battery_npv 保留为兼容字段，但明确其含义是实际投资方 NPV。
        "pv_npv_actual": float(pv_npv) if pv_ok else None,
        "battery_npv_actual": float(battery_npv) if battery_ok else None,
        "pv_npv_planning": float(pv_npv_planning) if pv_ok else None,
        "battery_npv_planning": float(battery_npv_planning) if battery_ok else None,
        "pv_grid_charging_cost_pv": pv.get("grid_charging_cost_pv"),
        "battery_grid_charging_cost_pv": battery.get("grid_charging_cost_pv"),
        "pv_kw": pv.get("pv_kw"),
        "ess_kwh": battery.get("ess_kwh"),
        "state_class": classification.get("state"),
        "state_reason": classification.get("reason"),
    }

    if kind == "pv_entry":
        if solver_unknown(pv_status) or solver_unknown(battery_status):
            return {"status": "no_solution", "metric": None, **common}
        if not pv_ok and not battery_ok:
            return {
                "status": "state_infeasible",
                "metric": None,
                **common,
            }
        candidates = [
            float(v)
            for v, ok in ((pv_npv, pv_ok), (battery_npv, battery_ok))
            if ok
        ]
        metric = max(candidates)
        return {
            "status": result_status(),
            "metric": float(metric),
            **common,
        }

    # battery_upgrade：差值需要两状态都有物理可用、可认证的定义。
    if solver_unknown(pv_status) or solver_unknown(battery_status):
        return {"status": "no_solution", "metric": None, **common}
    if not pv_ok or not battery_ok:
        return {
            "status": "state_unavailable",
            "metric": None,
            "reason": (
                "pv_only_state_unavailable"
                if not pv_ok
                else "pv_battery_state_unavailable"
            ),
            **common,
        }
    metric = float(battery_npv) - float(pv_npv)
    return {
        "status": result_status(),
        "metric": float(metric),
        **common,
    }


def _state_endpoint_crosses(
    alpha: float,
    *,
    kind: str,
    gamma_lo: float,
    gamma_hi: float,
    discount_rate: float | None,
    include_user: bool,
    cache: dict | None,
    solver_threads: int | None,
    probe_count: int = _PROBES,
    allow_uncertified: bool = False,
) -> bool:
    """只用 γ 两端点判断某个 α 是否可能存在价值边界。

    该函数用于寻找有效 α 区间，不输出边界点本身。两端点一正一负
    （允许一端在数值容差内等于0）才进入后续 γ 探针和二分。
    """
    if kind == "battery_upgrade":
        values = [float(v) for v in np.linspace(gamma_lo, gamma_hi, probe_count)]
    else:
        values = [float(gamma_lo), float(gamma_hi)]
    results = [
        _state_metric(
            alpha,
            gamma,
            kind=kind,
            discount_rate=discount_rate,
            include_user=include_user,
            cache=cache,
            solver_threads=solver_threads,
            allow_uncertified=allow_uncertified,
        )
        for gamma in values
    ]
    allowed_statuses = {"optimal"}
    if allow_uncertified:
        allowed_statuses.add("solver_not_certified")

    def solver_failed(result: dict) -> bool:
        return result.get("status") not in (
            allowed_statuses | {"state_infeasible", "state_unavailable"}
        )

    if any(solver_failed(result) for result in results):
        return False

    if kind == "battery_upgrade":
        usable = [
            (index, result)
            for index, result in enumerate(results)
            if result.get("status") in allowed_statuses
            and _finite_number(result.get("metric"))
        ]
        for (left_index, left), (right_index, right) in zip(usable, usable[1:]):
            if right_index != left_index + 1:
                continue
            left_ok = float(left["metric"]) >= -V_TOL
            right_ok = float(right["metric"]) >= -V_TOL
            if left_ok != right_ok:
                return True
        return False

    flags = [
        _finite_number(result.get("metric"))
        and float(result["metric"]) >= -V_TOL
        for result in results
    ]
    return any(left != right for left, right in zip(flags, flags[1:]))


def _auto_state_alpha_grid(
    *,
    n_points: int,
    alpha_lo: float,
    alpha_hi: float,
    gamma_lo: float,
    gamma_hi: float,
    kind: str,
    discount_rate: float | None,
    include_user: bool,
    cache: dict | None,
    solver_threads: int | None,
    workers: int,
    alpha_probe_count: int,
    alpha_search_max_iter: int,
    allow_uncertified: bool = False,
) -> dict:
    """自动寻找一条状态边界的有效 α 区间，再生成 n_points 个 α。

    α=0 被视为不投资区的理论点，不调用投资状态模型。先做少量端点
    价值探针定位“无穿越→有穿越”的起点，再在该起点附近二分；若
    穿越区间对 α 非单调，则拒绝拼出一条虚假的单段曲线。
    """
    alpha_start = max(float(alpha_lo), _ALPHA_EPS)
    if alpha_start >= alpha_hi:
        return {
            "status": "no_alpha_interval",
            "alpha_min": None,
            "alpha_values": [],
        }
    probe_values = np.linspace(alpha_start, alpha_hi, alpha_probe_count)
    endpoint_cache = cache if workers == 1 else None

    def endpoint_crosses(alpha_value):
        return _state_endpoint_crosses(
            alpha_value,
            kind=kind,
            gamma_lo=gamma_lo,
            gamma_hi=gamma_hi,
            discount_rate=discount_rate,
            include_user=include_user,
            cache=endpoint_cache,
            solver_threads=solver_threads,
            probe_count=_PROBES,
            allow_uncertified=allow_uncertified,
        )

    endpoint_values = [float(alpha) for alpha in probe_values]
    if workers == 1:
        crosses = []
        for alpha_value in endpoint_values:
            c = endpoint_crosses(alpha_value)
            crosses.append(c)
            print(
                f"[probe kind={kind}] α={alpha_value:.4f} endpoint_crosses={c}",
                flush=True,
            )
    else:
        load_data()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_alpha = {
                pool.submit(endpoint_crosses, a): a for a in endpoint_values
            }
            results = {}
            for fut, alpha_value in future_to_alpha.items():
                try:
                    c = fut.result()
                except Exception:  # noqa: BLE001
                    c = False
                results[alpha_value] = c
                print(
                    f"[probe kind={kind}] α={alpha_value:.4f} endpoint_crosses={c}",
                    flush=True,
                )
        crosses = [results[a] for a in endpoint_values]  # 保持原顺序
    valid_indices = [i for i, crossed in enumerate(crosses) if crossed]
    if not valid_indices:
        return {
            "status": "no_boundary_in_alpha_range",
            "alpha_min": None,
            "alpha_values": [],
        }
    first = valid_indices[0]
    if kind == "battery_upgrade":
        alpha_values = [
            float(probe_values[index]) for index in valid_indices
        ]
        return {
            "status": "ok",
            "alpha_min": min(alpha_values),
            "alpha_values": alpha_values,
        }
    if any(not crossed for crossed in crosses[first:]):
        return {
            "status": "non_monotonic_alpha",
            "alpha_min": None,
            "alpha_values": [],
        }
    if first == 0:
        alpha_min = float(alpha_start)
    else:
        lo = float(probe_values[first - 1])
        hi = float(probe_values[first])
        for _ in range(alpha_search_max_iter):
            if hi - lo <= 1e-5:
                break
            mid = 0.5 * (lo + hi)
            if _state_endpoint_crosses(
                mid,
                kind=kind,
                gamma_lo=gamma_lo,
                gamma_hi=gamma_hi,
                discount_rate=discount_rate,
                include_user=include_user,
                cache=cache,
                solver_threads=solver_threads,
                probe_count=_PROBES,
                allow_uncertified=allow_uncertified,
            ):
                hi = mid
            else:
                lo = mid
        alpha_min = hi
    return {
        "status": "ok",
        "alpha_min": alpha_min,
        "alpha_values": [
            float(value) for value in np.linspace(alpha_min, alpha_hi, n_points)
        ],
    }


def gamma_state_value_boundary(
    alpha: float,
    gamma_lo: float = 0.0,
    gamma_hi: float = 1.0,
    *,
    kind: str = "pv_entry",
    discount_rate: float | None = None,
    # 保留形参以兼容缓存键与既有调用；用户总成本约束实际恒为开启
    # （见 _evaluate 与 model.solve_planning/solve_locked 的默认值）。
    include_user: bool = True,
    tol: float = 1e-3,
    probe_count: int = _PROBES,
    probe_values: list[float] | None = None,
    max_iter: int = 40,
    cache: dict | None = None,
    solver_threads: int | None = None,
    allow_uncertified: bool = False,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    """固定 alpha 沿 gamma 搜索第一阶段状态边界。

    PV 进入线可以穿过“一个强制状态物理不可行、另一个状态可行”的
    区间；储能升级线则只在 PV-only 和 PV+BESS 都有数值定义的连续区间
    内寻找，不再因为低 γ 探针的物理不可行点而整体返回 no_solution。
    """
    if kind not in _STATE_BOUNDARY_KINDS:
        raise ValueError(f"invalid state boundary kind: {kind}")
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha 必须位于 [0,1]")
    if not np.isfinite(gamma_lo) or not np.isfinite(gamma_hi):
        raise ValueError("gamma 边界必须为有限数")
    if gamma_lo >= gamma_hi:
        raise ValueError("gamma_lo 必须小于 gamma_hi")
    if tol <= 0.0 or max_iter <= 0 or probe_count < 2:
        raise ValueError("tol、max_iter、probe_count 参数无效")

    value_cache: dict[float, dict] = {}
    phase = "gamma_probe"

    def evaluate(gamma: float) -> dict:
        key = float(gamma)
        if key not in value_cache:
            evaluation = len(value_cache) + 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "evaluation_started",
                        "axis": "gamma",
                        "phase": phase,
                        "alpha": float(alpha),
                        "gamma": key,
                        "kind": kind,
                        "evaluation": evaluation,
                    }
                )
            value_cache[key] = _state_metric(
                alpha,
                key,
                kind=kind,
                discount_rate=discount_rate,
                include_user=include_user,
                cache=cache,
                solver_threads=solver_threads,
                allow_uncertified=allow_uncertified,
            )
            if progress_callback is not None:
                result = value_cache[key]
                progress_callback(
                    {
                        "event": "evaluation_finished",
                        "axis": "gamma",
                        "phase": phase,
                        "alpha": float(alpha),
                        "gamma": key,
                        "kind": kind,
                        "evaluation": evaluation,
                        "result_status": result.get("status"),
                        "metric": result.get("metric"),
                        "state_class": result.get("state_class"),
                    }
                )
        return value_cache[key]

    if probe_values is None:
        probes = [float(v) for v in np.linspace(gamma_lo, gamma_hi, probe_count)]
    else:
        probes = [float(v) for v in probe_values]
        if len(probes) < 2 or any(
            not np.isfinite(value)
            or value < gamma_lo
            or value > gamma_hi
            for value in probes
        ) or any(right <= left for left, right in zip(probes, probes[1:])):
            raise ValueError("probe_values 必须是 gamma 区间内严格递增的有限点")
        probe_count = len(probes)
    probe_results = [evaluate(gamma) for gamma in probes]

    allowed_statuses = {"optimal"}
    if allow_uncertified:
        allowed_statuses.add("solver_not_certified")

    def solver_failed(result: dict) -> bool:
        return result.get("status") not in (
            allowed_statuses | {"state_infeasible", "state_unavailable"}
        )

    def metric_defined(result: dict) -> bool:
        return (
            result.get("status") in allowed_statuses
            and _finite_number(result.get("metric"))
        )

    def point_ok(result: dict) -> bool:
        return metric_defined(result) and float(result["metric"]) >= -V_TOL

    def summary(status: str, threshold: float | None = None, **extra) -> dict:
        def finite_value(result: dict, key: str):
            value = result.get(key)
            return float(value) if _finite_number(value) else None

        return {
            "status": status,
            "threshold": threshold,
            "kind": kind,
            "probe_count": probe_count,
            "solver_value_tolerance": SOLVER_VALUE_TOL,
            "metric_lo": finite_value(probe_results[0], "metric"),
            "metric_hi": finite_value(probe_results[-1], "metric"),
            "probes": [
                {
                    "gamma": gamma,
                    "status": result.get("status"),
                    "metric": finite_value(result, "metric"),
                    "state_class": result.get("state_class"),
                    "state_reason": result.get("state_reason")
                    or result.get("reason"),
                    "pv_status": result.get("pv_status"),
                    "battery_status": result.get("battery_status"),
                    "pv_npv": finite_value(result, "pv_npv"),
                    "battery_npv": finite_value(result, "battery_npv"),
                    "pv_npv_actual": finite_value(result, "pv_npv_actual"),
                    "battery_npv_actual": finite_value(result, "battery_npv_actual"),
                    "pv_npv_planning": finite_value(result, "pv_npv_planning"),
                    "battery_npv_planning": finite_value(result, "battery_npv_planning"),
                    "pv_grid_charging_cost_pv": finite_value(
                        result, "pv_grid_charging_cost_pv"
                    ),
                    "battery_grid_charging_cost_pv": finite_value(
                        result, "battery_grid_charging_cost_pv"
                    ),
                    "pv_kw": finite_value(result, "pv_kw"),
                    "ess_kwh": finite_value(result, "ess_kwh"),
                }
                for gamma, result in zip(probes, probe_results)
            ],
            "evaluations": len(value_cache),
            **extra,
        }

    if any(solver_failed(result) for result in probe_results):
        return summary("no_solution", reason="probe_solver_unavailable")

    if kind == "pv_entry":
        # 物理不可行点的 metric=None 只表示当前点没有可建状态，按不可投
        # 侧处理；只有求解器失败才阻断边界构造。
        flags = [point_ok(result) for result in probe_results]
        if not any(metric_defined(result) for result in probe_results):
            return summary("all_model_infeasible", reason="no_forced_state_value")
        if all(flags):
            return summary("all_feasible", orientation="ascending")
        if not any(flags):
            return summary("all_economically_infeasible", orientation="descending")

        ascending = all(not left or right for left, right in zip(flags, flags[1:]))
        descending = all(left or not right for left, right in zip(flags, flags[1:]))
        if not ascending and not descending:
            return summary("non_monotonic")
        transition = next(
            index
            for index, (left, right) in enumerate(zip(flags, flags[1:]))
            if left != right
        )
        lo_ok = flags[transition]
    else:
        # 储能差值只在两个状态均有定义的连续探针段内判断。无定义点
        # 可以是 PV-only 的物理不可行，不应被误报为求解失败。
        valid_indices = [
            index
            for index, result in enumerate(probe_results)
            if metric_defined(result)
        ]
        if not valid_indices:
            return summary(
                "not_applicable",
                reason="no_jointly_available_pv_and_battery_state",
            )

        runs: list[list[int]] = []
        for index in valid_indices:
            if not runs or index != runs[-1][-1] + 1:
                runs.append([index])
            else:
                runs[-1].append(index)
        transitions: list[tuple[int, bool]] = []
        for run in runs:
            run_flags = [point_ok(probe_results[index]) for index in run]
            ascending = all(
                not left or right for left, right in zip(run_flags, run_flags[1:])
            )
            descending = all(
                left or not right for left, right in zip(run_flags, run_flags[1:])
            )
            if not ascending and not descending:
                return summary("non_monotonic")
            transitions.extend(
                (run[offset], run_flags[offset])
                for offset, (left, right) in enumerate(
                    zip(run_flags, run_flags[1:])
                )
                if left != right
            )
        if len(transitions) != 1:
            if transitions:
                return summary("non_monotonic", reason="multiple_upgrade_crossings")
            profitable = any(
                metric_defined(result)
                and float(result.get("pv_npv", -np.inf)) > V_TOL
                and float(result.get("battery_npv", -np.inf)) > V_TOL
                for result in probe_results
            )
            return summary(
                "all_feasible" if profitable and all(
                    point_ok(probe_results[index]) for index in valid_indices
                ) else "not_applicable",
                orientation="ascending" if profitable else None,
                reason=(None if profitable else "no_upgrade_crossing_in_profitable_region"),
            )
        transition, lo_ok = transitions[0]

    lo, hi = float(probes[transition]), float(probes[transition + 1])
    phase = "gamma_bisection"
    for _ in range(max_iter):
        if hi - lo <= tol:
            break
        mid = 0.5 * (lo + hi)
        mid_result = evaluate(mid)
        if solver_failed(mid_result):
            return summary("no_solution", reason="bisection_solver_unavailable")
        if kind == "battery_upgrade" and not metric_defined(mid_result):
            return summary(
                "not_applicable",
                reason="upgrade_bisection_left_joint_state_region",
            )
        mid_ok = point_ok(mid_result)
        if mid_ok == lo_ok:
            lo = mid
        else:
            hi = mid

    threshold = 0.5 * (lo + hi)
    phase = "gamma_root"
    root = evaluate(threshold)
    if kind == "battery_upgrade":
        # 差值在亏损区的穿越不是投资状态边界；只有两种建设状态都
        # 在正 NPV 侧时，V_B-V_P=0 才能解释为 PV-only/PV+BESS 边界。
        if not (
            metric_defined(root)
            and _finite_number(root.get("pv_npv"))
            and _finite_number(root.get("battery_npv"))
            and float(root["pv_npv"]) > V_TOL
            and float(root["battery_npv"]) > V_TOL
        ):
            return summary(
                "not_applicable",
                reason="upgrade_crossing_outside_profitable_build_region",
            )
        root_state = root.get("state_class")
        if root_state not in {"pv_only", "pv_battery"}:
            return summary(
                "not_applicable",
                reason="upgrade_crossing_not_between_investable_states",
            )
    return summary(
        "threshold",
        threshold=threshold,
        orientation="ascending" if not lo_ok else "descending",
        state_at_threshold=root.get("state_class"),
        threshold_pv_npv=root.get("pv_npv"),
        threshold_battery_npv=root.get("battery_npv"),
    )


def alpha_state_value_boundary(
    gamma: float = 1.0,
    alpha_lo: float = 0.0,
    alpha_hi: float = 1.0,
    *,
    kind: str = "pv_entry",
    discount_rate: float | None = None,
    # 保留形参以兼容缓存键与既有调用；用户总成本约束实际恒为开启
    # （见 _evaluate 与 model.solve_planning/solve_locked 的默认值）。
    include_user: bool = True,
    tol: float = 1e-3,
    probe_count: int = _ALPHA_PROBES,
    max_iter: int = 40,
    cache: dict | None = None,
    solver_threads: int | None = None,
    allow_uncertified: bool = False,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    """在固定 ``gamma`` 下搜索最低可承受负荷。

    实验 1 的价格上界点 ``gamma=1`` 必须单独计算并写入正式结果；
    不能把它遗漏在仅从 ``alpha=0.05`` 开始的曲线网格之外。这里对
    ``pv_entry`` 的指标使用 ``max(V_P,V_B)``，与 γ 方向进入线完全同口径。
    """
    if kind not in _STATE_BOUNDARY_KINDS:
        raise ValueError(f"invalid state boundary kind: {kind}")
    if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma 必须位于 [0,1]")
    if not np.isfinite(alpha_lo) or not np.isfinite(alpha_hi):
        raise ValueError("alpha 边界必须为有限数")
    if alpha_lo >= alpha_hi:
        raise ValueError("alpha_lo 必须小于 alpha_hi")
    if tol <= 0.0 or max_iter <= 0 or probe_count < 2:
        raise ValueError("tol、max_iter、probe_count 参数无效")

    phase = "alpha_probe"
    value_cache: dict[float, dict] = {}

    def evaluate(alpha: float) -> dict:
        key = float(alpha)
        if key not in value_cache:
            evaluation = len(value_cache) + 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "evaluation_started",
                        "axis": "alpha",
                        "phase": phase,
                        "alpha": key,
                        "gamma": float(gamma),
                        "kind": kind,
                        "evaluation": evaluation,
                    }
                )
            value_cache[key] = _state_metric(
                key,
                gamma,
                kind=kind,
                discount_rate=discount_rate,
                include_user=include_user,
                cache=cache,
                solver_threads=solver_threads,
                allow_uncertified=allow_uncertified,
            )
            if progress_callback is not None:
                result = value_cache[key]
                progress_callback(
                    {
                        "event": "evaluation_finished",
                        "axis": "alpha",
                        "phase": phase,
                        "alpha": key,
                        "gamma": float(gamma),
                        "kind": kind,
                        "evaluation": evaluation,
                        "result_status": result.get("status"),
                        "metric": result.get("metric"),
                        "state_class": result.get("state_class"),
                    }
                )
        return value_cache[key]

    probes = [float(v) for v in np.linspace(alpha_lo, alpha_hi, probe_count)]
    probe_results = [evaluate(alpha) for alpha in probes]
    allowed_statuses = {"optimal"}
    if allow_uncertified:
        allowed_statuses.add("solver_not_certified")

    def solver_failed(result: dict) -> bool:
        # A low-alpha no_solution probe is treated as the non-investable side;
        # only an actual solver state outside the known statuses aborts search.
        return result.get("status") not in (
            allowed_statuses
            | {"state_infeasible", "state_unavailable", "no_solution"}
        )

    def metric_defined(result: dict) -> bool:
        return (
            result.get("status") in allowed_statuses
            and _finite_number(result.get("metric"))
        )

    def point_ok(result: dict) -> bool:
        return metric_defined(result) and float(result["metric"]) >= -V_TOL

    def finite_value(result: dict, key: str):
        value = result.get(key)
        return float(value) if _finite_number(value) else None

    def summary(status: str, threshold: float | None = None, **extra) -> dict:
        return {
            "status": status,
            "threshold": threshold,
            "axis": "alpha",
            "fixed_gamma": float(gamma),
            "kind": kind,
            "probe_count": probe_count,
            "solver_value_tolerance": SOLVER_VALUE_TOL,
            "probes": [
                {
                    "alpha": alpha,
                    "status": result.get("status"),
                    "metric": finite_value(result, "metric"),
                    "state_class": result.get("state_class"),
                    "state_reason": result.get("state_reason")
                    or result.get("reason"),
                    "pv_npv": finite_value(result, "pv_npv"),
                    "battery_npv": finite_value(result, "battery_npv"),
                    "pv_npv_actual": finite_value(result, "pv_npv_actual"),
                    "battery_npv_actual": finite_value(result, "battery_npv_actual"),
                    "pv_npv_planning": finite_value(result, "pv_npv_planning"),
                    "battery_npv_planning": finite_value(result, "battery_npv_planning"),
                    "pv_grid_charging_cost_pv": finite_value(
                        result, "pv_grid_charging_cost_pv"
                    ),
                    "battery_grid_charging_cost_pv": finite_value(
                        result, "battery_grid_charging_cost_pv"
                    ),
                    "pv_kw": finite_value(result, "pv_kw"),
                    "ess_kwh": finite_value(result, "ess_kwh"),
                }
                for alpha, result in zip(probes, probe_results)
            ],
            "evaluations": len(value_cache),
            **extra,
        }

    unknown_probe_count = sum(
        result.get("status") == "no_solution" for result in probe_results
    )
    if any(solver_failed(result) for result in probe_results):
        return summary("no_solution", reason="probe_solver_unavailable")
    if not any(metric_defined(result) for result in probe_results):
        return summary(
            "no_solution" if unknown_probe_count else "all_model_infeasible",
            reason=(
                "probe_solver_unavailable"
                if unknown_probe_count
                else "no_forced_state_value"
            ),
            unknown_probe_count=unknown_probe_count,
        )

    flags = [point_ok(result) for result in probe_results]
    if all(flags):
        return summary(
            "all_feasible",
            threshold=float(alpha_lo),
            orientation="ascending",
            reason="boundary_at_or_below_alpha_lo",
        )
    if not any(flags):
        return summary(
            "no_solution" if unknown_probe_count else "all_economically_infeasible",
            orientation="ascending",
            reason=(
                "probe_solver_unavailable"
                if unknown_probe_count
                else "all_probe_values_nonpositive"
            ),
            unknown_probe_count=unknown_probe_count,
        )

    ascending = all(not left or right for left, right in zip(flags, flags[1:]))
    descending = all(left or not right for left, right in zip(flags, flags[1:]))
    if not ascending and not descending:
        return summary("non_monotonic")
    transition = next(
        index
        for index, (left, right) in enumerate(zip(flags, flags[1:]))
        if left != right
    )
    lo, hi = float(probes[transition]), float(probes[transition + 1])
    lo_ok = flags[transition]
    phase = "alpha_bisection"
    for _ in range(max_iter):
        if hi - lo <= tol:
            break
        mid = 0.5 * (lo + hi)
        mid_result = evaluate(mid)
        if solver_failed(mid_result):
            return summary("no_solution", reason="bisection_solver_unavailable")
        mid_ok = point_ok(mid_result)
        if mid_ok == lo_ok:
            lo = mid
        else:
            hi = mid
    threshold = 0.5 * (lo + hi)
    phase = "alpha_root"
    root = evaluate(threshold)
    return summary(
        "threshold",
        threshold=threshold,
        orientation="ascending" if not lo_ok else "descending",
        state_at_threshold=root.get("state_class"),
        threshold_pv_npv=root.get("pv_npv"),
        threshold_battery_npv=root.get("battery_npv"),
    )


def state_boundary_curve(
    alpha_values=None,
    *,
    n_points: int = 7,
    alpha_lo: float = 0.0,
    alpha_hi: float = 1.0,
    gamma_lo: float = 0.0,
    gamma_hi: float = 1.0,
    kind: str = "pv_entry",
    discount_rate: float | None = None,
    # 保留形参以兼容缓存键与既有调用；用户总成本约束实际恒为开启
    # （见 _evaluate 与 model.solve_planning/solve_locked 的默认值）。
    include_user: bool = True,
    tol: float = 1e-3,
    probe_count: int = _PROBES,
    alpha_probe_count: int = _ALPHA_PROBES,
    alpha_search_max_iter: int = _ALPHA_SEARCH_MAX_ITER,
    max_iter: int = 40,
    cache: dict | None = None,
    workers: int = 1,
    solver_threads: int | None = None,
    allow_uncertified: bool = False,
) -> dict:
    """生成一条独立的状态边界曲线，每个负荷点最多一个 gamma 阈值。"""
    if kind not in _STATE_BOUNDARY_KINDS:
        raise ValueError(f"invalid state boundary kind: {kind}")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers 必须为正整数")
    if solver_threads is not None and (
        not isinstance(solver_threads, int)
        or isinstance(solver_threads, bool)
        or solver_threads < 1
    ):
        raise ValueError("solver_threads 必须为正整数或 None")
    if not isinstance(alpha_probe_count, int) or isinstance(alpha_probe_count, bool) or alpha_probe_count < 2:
        raise ValueError("alpha_probe_count 必须为不小于 2 的整数")
    if not isinstance(alpha_search_max_iter, int) or isinstance(alpha_search_max_iter, bool) or alpha_search_max_iter < 0:
        raise ValueError("alpha_search_max_iter 必须为非负整数")
    effective_solver_threads = (
        solver_threads
        if solver_threads is not None
        else max(1, cfg.THREADS // workers)
    ) if workers > 1 else solver_threads
    if workers > 1:
        load_data()
    shared_cache = {} if cache is None else cache
    auto = None
    if alpha_values is None:
        if n_points < 2:
            raise ValueError("n_points 至少为 2")
        auto = _auto_state_alpha_grid(
            n_points=n_points,
            alpha_lo=alpha_lo,
            alpha_hi=alpha_hi,
            gamma_lo=gamma_lo,
            gamma_hi=gamma_hi,
            kind=kind,
            discount_rate=discount_rate,
            include_user=include_user,
            cache=shared_cache,
            solver_threads=effective_solver_threads,
            workers=workers,
            alpha_probe_count=alpha_probe_count,
            alpha_search_max_iter=alpha_search_max_iter,
            allow_uncertified=allow_uncertified,
        )
        if auto["status"] != "ok":
            return {
                "kind": kind,
                "alpha_range": [float(alpha_lo), float(alpha_hi)],
                "gamma_range": [float(gamma_lo), float(gamma_hi)],
                "requested_points": n_points,
                "valid_points": 0,
                "alpha_probe_count": alpha_probe_count,
                "alpha_search_max_iter": alpha_search_max_iter,
                "gamma_probe_count": probe_count,
                "scan_mode": "exploratory",
                "scan_warning": "少量探针仅用于观察形状，不能保证发现所有非单调穿越",
                "alpha_search_status": auto["status"],
                "boundary_alpha_range": [None, None],
                "points": [],
            }
        alpha_values = auto["alpha_values"]
    alphas = [float(value) for value in alpha_values]
    if not alphas:
        raise ValueError("alpha_values 不能为空")
    if any(
        not np.isfinite(value)
        or value <= 0.0
        or not alpha_lo <= value <= alpha_hi
        for value in alphas
    ):
        raise ValueError("alpha_values 必须位于 (0,1] 且在给定 α 区间内")

    def solve_point(alpha):
        point_cache = shared_cache if workers == 1 else None
        result = gamma_state_value_boundary(
            alpha,
            gamma_lo,
            gamma_hi,
            kind=kind,
            discount_rate=discount_rate,
            include_user=include_user,
            tol=tol,
            probe_count=probe_count,
            max_iter=max_iter,
            cache=point_cache,
            solver_threads=effective_solver_threads,
            allow_uncertified=allow_uncertified,
        )
        # 进度日志：每个 α 完成即输出（便于实时观察"走到哪个点"）。
        print(
            f"[curves kind={kind}] α={alpha:.3f} → "
            f"status={result.get('status')} "
            f"threshold={result.get('threshold') if result.get('threshold') is not None else '-'} "
            f"(evaluations={result.get('evaluations', 0)})",
            flush=True,
        )
        return {"alpha": alpha, **result}

    if workers == 1:
        points = [solve_point(alpha) for alpha in alphas]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            points = list(pool.map(solve_point, alphas))
    valid = [point for point in points if point["status"] == "threshold"]
    return {
        "kind": kind,
        "alpha_range": [float(alpha_lo), float(alpha_hi)],
        "gamma_range": [float(gamma_lo), float(gamma_hi)],
        "requested_points": len(points),
        "valid_points": len(valid),
        "alpha_probe_count": alpha_probe_count,
        "alpha_search_max_iter": alpha_search_max_iter,
        "gamma_probe_count": probe_count,
        "scan_mode": "exploratory",
        "scan_warning": "少量探针仅用于观察形状，不能保证发现所有非单调穿越",
        "alpha_search_status": "explicit" if auto is None else auto["status"],
        "boundary_alpha_range": [
            float(alphas[0]) if alphas else None,
            float(alphas[-1]) if alphas else None,
        ],
        "points": points,
    }



def _write_boundary_progress(path: Path, payload: dict) -> None:
    payload = dict(payload)
    payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()

    write_json(path, payload)


def _resumable_state_boundaries(
    *,
    alpha_values: list[float],
    gamma_lo: float,
    gamma_hi: float,
    gamma_probe_count: int,
    boundary_tol: float,
    solver_threads: int,
    checkpoint_path: Path,
    progress_path: Path | None = None,
    reset_checkpoint: bool = False,
    pilot_mode: bool = False,
) -> tuple[dict | None, bool]:
    """Run the two boundary curves one alpha at a time with checkpointing.

    The checkpoint is deliberately per-alpha rather than per-solver call: a
    partially solved alpha is recomputed, while every completed alpha is
    reused on the next invocation. This keeps the final JSON compatible with
    the plotting code and avoids treating an interrupted MILP as certified.
    """
    if reset_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()
    if progress_path is None:
        progress_path = checkpoint_path.with_suffix(".progress.json")
    if reset_checkpoint and progress_path.exists():
        progress_path.unlink()

    alpha_values = [float(value) for value in alpha_values]
    # 与旧版 LP 边界图一致；pilot 只放宽到 0.01，不再使用 0.05 粗二分。
    effective_boundary_tol = 0.01 if pilot_mode else float(boundary_tol)
    metadata = {
        "schema_version": 5,
        "classification_version": "vp-vb-value-priority-v5-gamma-tou-actual-investor-npv",
        "boundary_formulation": (
            "lp_relaxed_mutex" if not cfg.MUTEX else "milp_mutex"
        ),
        "stage": "P1_state_boundaries",
        "preview_mode": False,
        "pilot_mode": bool(pilot_mode),
        "alpha_values": alpha_values,
        "gamma_range": [float(gamma_lo), float(gamma_hi)],
        "gamma_probe_count": int(gamma_probe_count),
        "boundary_tol": effective_boundary_tol,
        "solver_threads": int(solver_threads),
        "mip_gap": float(cfg.MIP_GAP),
        "mip_gap_abs": float(cfg.MIP_GAP_ABS),
        "allow_uncertified": bool(pilot_mode),
        # 用户总成本约束必须保留；边界扫描的 LP 近似只放松充放电互斥，
        # 不放松用户不吃亏约束。
        "include_user": True,
        "alpha_min_gamma": 1.0,
        "price_formula": "contract_price=gamma*TOU",
        "baseline_gamma": float(cfg.BASELINE_GAMMA),
        "investor_pays_grid_charging": True,
        "boundary_npv_field": "npv_actual_investor",
    }

    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        for key in (
            "schema_version",
            "classification_version",
            "stage",
            "preview_mode",
            "boundary_formulation",
            "solver_threads",
            "include_user",
            "alpha_min_gamma",
            "price_formula",
            "baseline_gamma",
            "investor_pays_grid_charging",
            "boundary_npv_field",
        ):
            if checkpoint.get(key) != metadata[key]:
                raise ValueError(
                    "checkpoint 参数与本次运行不一致；请使用 --reset-checkpoint "
                    "重新开始，避免混合不同模型口径。"
                )
        # pilot 网格是完整网格的子集；后续完整运行沿用已完成点，
        # 只更新 alpha_values，不清空 completed_points。
        checkpoint["alpha_values"] = alpha_values
        checkpoint["pilot_mode"] = bool(pilot_mode)
        checkpoint["gamma_range"] = [float(gamma_lo), float(gamma_hi)]
    else:
        checkpoint = {**metadata, "status": "paused", "completed_points": {}}

    progress = {
        "schema_version": metadata["schema_version"],
        "classification_version": metadata["classification_version"],
        "stage": metadata["stage"],
        "status": "paused",
        "phase": "initializing",
        "current_index": 0,
        "total": len(alpha_values),
        "current_alpha": None,
        "current_kind": None,
        "current_gamma": None,
        "axis": None,
        "evaluation": 0,
        "last_event": None,
        "last_result_status": None,
        "last_metric": None,
        "state_class": None,
    }
    _write_boundary_progress(progress_path, progress)

    # 立即落盘：即使第一个 alpha 仍在求解器中，也能安全暂停并续跑。
    checkpoint["status"] = "paused"
    write_json(checkpoint_path, checkpoint)
    completed = checkpoint.setdefault("completed_points", {})
    total = len(alpha_values)
    for index, alpha in enumerate(alpha_values, start=1):
        key = f"{alpha:.8f}"
        if key in completed:
            saved = completed[key]
            progress.update(
                {
                    "status": "running",
                    "phase": "resume_skip",
                    "current_index": index,
                    "total": total,
                    "current_alpha": alpha,
                    "current_kind": "completed",
                    "last_result_status": saved["pv_only_entry"].get("status"),
                }
            )
            _write_boundary_progress(progress_path, progress)
            print(
                f"[resume {index}/{total}] alpha={alpha:.2f} "
                f"pv_entry={saved['pv_only_entry'].get('status')} "
                f"battery_upgrade={saved['pv_battery_entry'].get('status')}",
                flush=True,
            )
            continue

        print(f"[start {index}/{total}] alpha={alpha:.2f}", flush=True)
        progress.update(
            {
                "status": "running",
                "phase": "boundary_start",
                "current_index": index,
                "total": total,
                "current_alpha": alpha,
                "current_kind": None,
                "current_gamma": None,
                "axis": "gamma",
                "evaluation": 0,
                "last_event": "alpha_started",
                "last_result_status": None,
                "last_metric": None,
                "state_class": None,
            }
        )
        _write_boundary_progress(progress_path, progress)

        def on_boundary_progress(event: dict) -> None:
            progress.update(
                {
                    "status": "running",
                    "phase": event.get("phase"),
                    "current_index": index,
                    "total": total,
                    "current_alpha": alpha,
                    "current_kind": event.get("kind"),
                    "current_gamma": event.get("gamma"),
                    "axis": event.get("axis"),
                    "evaluation": event.get("evaluation"),
                    "last_event": event.get("event"),
                    "last_result_status": event.get("result_status"),
                    "last_metric": event.get("metric"),
                    "state_class": event.get("state_class"),
                }
            )
            _write_boundary_progress(progress_path, progress)

        shared_cache = {}
        try:
            pv_entry = gamma_state_value_boundary(
                alpha,
                gamma_lo,
                gamma_hi,
                kind="pv_entry",
                include_user=True,
                tol=effective_boundary_tol,
                probe_count=gamma_probe_count,
                probe_values=(
                    [
                        float(value)
                        for value in np.linspace(
                            gamma_lo, gamma_hi, gamma_probe_count
                        )
                    ]
                    if pilot_mode
                    else None
                ),
                max_iter=4 if pilot_mode else 40,
                cache=shared_cache,
                solver_threads=solver_threads,
                allow_uncertified=pilot_mode,
                progress_callback=on_boundary_progress,
            )
            battery_upgrade = gamma_state_value_boundary(
                alpha,
                gamma_lo,
                gamma_hi,
                kind="battery_upgrade",
                include_user=True,
                tol=effective_boundary_tol,
                probe_count=gamma_probe_count,
                probe_values=(
                    [
                        float(value)
                        for value in np.linspace(
                            gamma_lo, gamma_hi, gamma_probe_count
                        )
                    ]
                    if pilot_mode
                    else None
                ),
                max_iter=4 if pilot_mode else 40,
                cache=shared_cache,
                solver_threads=solver_threads,
                allow_uncertified=pilot_mode,
                progress_callback=on_boundary_progress,
            )
        except KeyboardInterrupt:
            progress.update(
                {
                    "status": "paused",
                    "phase": "alpha_paused",
                    "current_index": index,
                    "total": total,
                    "current_alpha": alpha,
                    "last_event": "keyboard_interrupt",
                }
            )
            _write_boundary_progress(progress_path, progress)
            checkpoint["status"] = "paused"
            checkpoint["last_completed_alpha"] = (
                max((item["alpha"] for item in completed.values()), default=None)
            )
            write_json(checkpoint_path, checkpoint)
            print(
                f"[paused] alpha={alpha:.2f} 未完成；已保存到 "
                f"alpha={checkpoint['last_completed_alpha']}",
                flush=True,
            )
            return None, True
        except Exception as exc:  # noqa: BLE001
            progress.update(
                {
                    "status": "failed",
                    "phase": "exception",
                    "current_index": index,
                    "current_alpha": alpha,
                    "error": repr(exc),
                }
            )
            _write_boundary_progress(progress_path, progress)
            raise

        completed[key] = {
            "alpha": alpha,
            "pv_only_entry": {"alpha": alpha, **pv_entry},
            "pv_battery_entry": {"alpha": alpha, **battery_upgrade},
        }
        checkpoint["status"] = "paused"
        checkpoint["last_completed_alpha"] = alpha
        write_json(checkpoint_path, checkpoint)
        progress.update(
            {
                "status": "running",
                "phase": "alpha_complete",
                "current_index": index,
                "total": total,
                "current_alpha": alpha,
                "current_kind": "completed",
                "current_gamma": None,
                "axis": None,
                "last_event": "alpha_complete",
                "last_result_status": pv_entry.get("status"),
                "last_metric": None,
                "state_class": None,
            }
        )
        _write_boundary_progress(progress_path, progress)
        print(
            f"[point {index}/{total}] alpha={alpha:.2f} "
            f"pv_entry={pv_entry.get('status')} "
            f"gamma*={pv_entry.get('threshold')} "
            f"battery_upgrade={battery_upgrade.get('status')} "
            f"gamma*={battery_upgrade.get('threshold')}",
            flush=True,
        )

    points_by_kind = {"pv_only_entry": [], "pv_battery_entry": []}
    for saved in completed.values():
        for key in points_by_kind:
            points_by_kind[key].append(saved[key])
    for points in points_by_kind.values():
        points.sort(key=lambda point: point["alpha"])

    def curve(kind: str, points: list[dict]) -> dict:
        valid = [point for point in points if point.get("status") == "threshold"]
        return {
            "kind": kind,
            "alpha_range": [0.0, 1.0],
            "gamma_range": [float(gamma_lo), float(gamma_hi)],
            "formulation": (
                "lp_relaxed_mutex" if not cfg.MUTEX else "milp_mutex"
            ),
            "requested_points": len(alpha_values),
            "valid_points": len(valid),
            "alpha_probe_count": None,
            "alpha_search_max_iter": None,
            "gamma_probe_count": gamma_probe_count,
            "scan_mode": "explicit_grid_resumable",
            "scan_warning": "α 使用显式网格；边界点沿 γ 用探针+二分。",
            "alpha_search_status": "explicit",
            "boundary_alpha_range": [
                float(min(alpha_values)) if alpha_values else None,
                float(max(alpha_values)) if alpha_values else None,
            ],
            "points": points,
        }

    # γ=1 的最低可承受负荷不是 α 网格中的一个普通点，必须单独写入
    # 正式结果。它使用与 PV 进入线完全相同的 max(V_P, V_B) 指标。
    alpha_min_gamma_1 = checkpoint.get("alpha_min_gamma_1")
    # A cached no_solution at alpha=0 must not suppress a later retry.
    if (
        alpha_min_gamma_1 is None
        or alpha_min_gamma_1.get("status") == "no_solution"
    ):
        progress.update(
            {
                "status": "running",
                "phase": "alpha_min_gamma_1",
                "current_index": total + 1,
                "total": total + 1,
                "current_alpha": None,
                "current_kind": "pv_entry",
                "current_gamma": 1.0,
                "axis": "alpha",
                "evaluation": 0,
                "last_event": "alpha_min_started",
                "last_result_status": None,
                "last_metric": None,
                "state_class": None,
            }
        )
        _write_boundary_progress(progress_path, progress)

        def on_alpha_min_progress(event: dict) -> None:
            progress.update(
                {
                    "status": "running",
                    "phase": event.get("phase"),
                    "current_index": total + 1,
                    "total": total + 1,
                    "current_alpha": event.get("alpha"),
                    "current_kind": event.get("kind"),
                    "current_gamma": 1.0,
                    "axis": event.get("axis"),
                    "evaluation": event.get("evaluation"),
                    "last_event": event.get("event"),
                    "last_result_status": event.get("result_status"),
                    "last_metric": event.get("metric"),
                    "state_class": event.get("state_class"),
                }
            )
            _write_boundary_progress(progress_path, progress)

        try:
            alpha_min_gamma_1 = alpha_state_value_boundary(
                gamma=1.0,
                alpha_lo=0.0,
                alpha_hi=1.0,
                kind="pv_entry",
                include_user=True,
                tol=effective_boundary_tol,
                probe_count=max(7, gamma_probe_count),
                max_iter=4 if pilot_mode else 40,
                cache={},
                solver_threads=solver_threads,
                allow_uncertified=pilot_mode,
                progress_callback=on_alpha_min_progress,
            )
        except KeyboardInterrupt:
            progress.update(
                {
                    "status": "paused",
                    "phase": "alpha_min_paused",
                    "current_index": total + 1,
                    "total": total + 1,
                    "current_kind": "pv_entry",
                    "current_gamma": 1.0,
                    "last_event": "keyboard_interrupt",
                }
            )
            _write_boundary_progress(progress_path, progress)
            checkpoint["status"] = "paused"
            checkpoint["pause_stage"] = "alpha_min_gamma_1"
            write_json(checkpoint_path, checkpoint)
            print(
                "[paused] γ=1 最低负荷边界尚未完成；已保存已完成 α 点，"
                "下次运行将继续并重新计算该边界。",
                flush=True,
            )
            return None, True
        except Exception as exc:  # noqa: BLE001
            progress.update(
                {
                    "status": "failed",
                    "phase": "alpha_min_exception",
                    "current_index": total + 1,
                    "total": total + 1,
                    "current_kind": "pv_entry",
                    "current_gamma": 1.0,
                    "error": repr(exc),
                }
            )
            _write_boundary_progress(progress_path, progress)
            raise
        checkpoint["alpha_min_gamma_1"] = alpha_min_gamma_1
        checkpoint["status"] = "paused"
        write_json(checkpoint_path, checkpoint)
        progress.update(
            {
                "status": "running",
                "phase": "alpha_min_complete",
                "current_index": total + 1,
                "total": total + 1,
                "current_kind": "completed",
                "current_gamma": 1.0,
                "last_event": "alpha_min_complete",
                "last_result_status": alpha_min_gamma_1.get("status"),
            }
        )
        _write_boundary_progress(progress_path, progress)

    result = {
        "stage": "P1_state_boundaries",
        "preview_mode": False,
        "preview_assumptions": None,
        "alpha_range_enforced": [0.0, 1.0],
        "gamma_range": [float(gamma_lo), float(gamma_hi)],
        "n_points": len(alpha_values),
        "alpha_probe_count": None,
        "gamma_probe_count": gamma_probe_count,
        "classification_version": "vp-vb-value-priority-v5-gamma-tou-actual-investor-npv",
        "boundary_formulation": (
            "lp_relaxed_mutex" if not cfg.MUTEX else "milp_mutex"
        ),
        "pilot_mode": bool(pilot_mode),
        "checkpoint_path": str(checkpoint_path),
        "progress_path": str(progress_path),
        "price_formula": "contract_price=gamma*TOU",
        "baseline_gamma": float(cfg.BASELINE_GAMMA),
        "investor_pays_grid_charging": True,
        "boundary_npv_field": "npv_actual_investor",
        "alpha_min_gamma_1": alpha_min_gamma_1,
        "pv_only_entry": curve("pv_entry", points_by_kind["pv_only_entry"]),
        "pv_battery_entry": curve(
            "battery_upgrade", points_by_kind["pv_battery_entry"]
        ),
    }
    checkpoint["status"] = "complete"
    write_json(checkpoint_path, checkpoint)
    progress.update(
        {
            "status": "complete",
            "phase": "complete",
            "current_index": total + 1,
            "total": total + 1,
            "current_kind": "complete",
            "current_gamma": None,
            "last_event": "boundary_scan_complete",
        }
    )
    _write_boundary_progress(progress_path, progress)
    return result, False


def _run_boundaries(argv: list[str]) -> int:
    preview = "--preview" in argv
    pilot = "--pilot" in argv
    reset_checkpoint = "--reset-checkpoint" in argv
    args = [
        value
        for value in argv[1:]
        if value not in {"--preview", "--pilot", "--reset-checkpoint"}
    ]
    n_points = int(args[0]) if len(args) > 0 else 20
    workers = int(args[1]) if len(args) > 1 else 1
    alpha_probe_count = int(args[2]) if len(args) > 2 else 7
    gamma_probe_count = int(args[3]) if len(args) > 3 else 5
    boundary_tol = 0.05 if preview else 1e-3
    alpha_search_max_iter = 0 if preview else 12
    if preview:
        # 边界绘图沿用旧版快速口径：当前模型的 LP 互斥松弛。
        cfg.MUTEX = False
        cfg.TIME_LIMIT = min(float(cfg.TIME_LIMIT), 60.0)
    else:
        # 仅边界扫描使用当前模型的 LP 快速分支；baseline 仍保持正式 MILP。
        cfg.MUTEX = False
        if workers != 1:
            raise ValueError("正式断点续算要求 workers=1，solver_threads 单独设置。")
        # 与本次已暂停的正式运行保持同一数值口径：允许适度求解误差，
        # 但不把未认证结果写入边界。
        cfg.MIP_GAP = 0.05 if pilot else 0.005
        cfg.MIP_GAP_ABS = 20000.0 if pilot else 3000.0
        # 当前模型的 LP 边界扫描使用已验证的对偶单纯形提速设置。
        cfg.TOU_METHOD = 1
        if pilot:
            # pilot 只用于确认区域形状；沿用旧图的 240 秒单次时限，
            # 只放宽边界扫描范围和输出认证，不改变当前模型。
            cfg.TIME_LIMIT = min(float(cfg.TIME_LIMIT), 240.0)
        alpha_values = (
            [0.10, 0.15, 0.25, 0.50, 1.00]
            if pilot
            else [
                round(float(value), 2)
                for value in np.linspace(0.05, 1.0, n_points)
            ]
        )
        checkpoint_path = (
            cfg.OUTPUT_ROOT
            / "f2_e1"
            / "e1_state_boundaries.gamma_tou.v5.checkpoint.json"
        )
        result, paused = _resumable_state_boundaries(
            alpha_values=alpha_values,
            gamma_lo=0.5 if pilot else 0.0,
            gamma_hi=1.0,
            gamma_probe_count=gamma_probe_count,
            boundary_tol=0.0025,
            solver_threads=20,
            checkpoint_path=checkpoint_path,
            reset_checkpoint=reset_checkpoint,
            pilot_mode=pilot,
        )
        if paused:
            return 130
        out_path = (
            cfg.OUTPUT_ROOT
            / "f2_e1"
            / (
                "e1_state_boundaries.gamma_tou.v5.pilot.json"
                if pilot
                else "e1_state_boundaries.json"
            )
        )
        write_json(out_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=1))
        return 0
    stage1_cache = {}
    result = {
        "stage": "P1_state_boundaries",
        "preview_mode": preview,
        "preview_assumptions": (
            {
                "mutex": cfg.MUTEX,
                "time_limit_seconds": cfg.TIME_LIMIT,
                "boundary_tol": boundary_tol,
                "alpha_search_max_iter": alpha_search_max_iter,
                "allow_uncertified": True,
            }
            if preview
            else None
        ),
        "alpha_range_enforced": [0.0, 1.0],  # 自动 α 区间会跳过不可行区
        "gamma_range": [0.0, 1.0],
        "n_points": n_points,
        "alpha_probe_count": alpha_probe_count,
        "gamma_probe_count": gamma_probe_count,
        "pv_only_entry": state_boundary_curve(
            n_points=n_points,
            kind="pv_entry",
            workers=workers,
            alpha_probe_count=alpha_probe_count,
            probe_count=gamma_probe_count,
            tol=boundary_tol,
            alpha_search_max_iter=alpha_search_max_iter,
            allow_uncertified=preview,
            cache=stage1_cache,
        ),
        "pv_battery_entry": state_boundary_curve(
            n_points=n_points,
            kind="battery_upgrade",
            workers=workers,
            alpha_probe_count=alpha_probe_count,
            probe_count=gamma_probe_count,
            tol=boundary_tol,
            alpha_search_max_iter=alpha_search_max_iter,
            allow_uncertified=preview,
            cache=stage1_cache,
        ),
    }
    out_path = (
        cfg.OUTPUT_ROOT
        / "f2_e1"
        / "e1_state_boundaries.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


@contextmanager
def _config_overrides(**values):
    """同一进程运行多个实验时，求解设置和敏感性参数不得相互污染。"""
    previous = {name: getattr(cfg, name) for name in values}
    try:
        for name, value in values.items():
            setattr(cfg, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(cfg, name, value)


def run_boundaries(argv: list[str] | None = None, *, output_root: Path | None = None) -> int:
    names = ("MUTEX", "TIME_LIMIT", "MIP_GAP", "MIP_GAP_ABS", "TOU_METHOD", "OUTPUT_ROOT")
    values = {name: getattr(cfg, name) for name in names}
    if output_root is not None:
        values["OUTPUT_ROOT"] = Path(output_root)
    with _config_overrides(**values):
        return _run_boundaries(["state_boundaries", *(argv or [])])


def run_baseline(
    alpha: float = 1.0, gamma: float = cfg.BASELINE_GAMMA, *,
    output: Path = cfg.BASELINE_OUTPUT, threads: int = 12,
    time_limit: float = 1800.0, mutex_formulation: str = "sos",
) -> dict:
    """求基准容量并保存完整逐时调度，供实验二及独立绘图复用。"""
    if threads < 1 or time_limit <= 0:
        raise ValueError("threads 和 time_limit 必须为正")
    with _config_overrides(MUTEX=True, MUTEX_FORMULATION=mutex_formulation):
        result = solve_planning(alpha, gamma, threads=threads, time_limit=time_limit)
        if result.get("status") != "optimal":
            raise RuntimeError(f"基准容量未获得认证解：{result.get('status')}")
        audit = result.get("audit", {})
        if (float(audit.get("max_pv_direct_priority_error_kw", np.inf)) > 1e-5
                or int(audit.get("simultaneous_export_grid_to_user_steps", -1)) != 0):
            raise RuntimeError("基准光伏优先直供审计失败")
        fingerprint = source_fingerprint()
    baseline = {
        "alpha": alpha, "gamma": gamma,
        "pv_kw": float(result["pv_kw"]), "ess_kwh": float(result["ess_kwh"]),
        "npv_actual_investor": float(result["npv_actual_investor"]), "audit": audit,
    }
    report = {
        "status": "complete", "stage": "pv_direct_priority_baseline",
        "mutex_formulation": mutex_formulation,
        "export_max_kw": cfg.EXPORT_MAX_KW,
        "min_self_use_planning": cfg.MIN_SELF_USE,
        "source_fingerprint": fingerprint, "baseline": baseline,
    }
    output = Path(output)
    write_json(output / "run_summary.json", report)
    inputs = load_data()
    write_json(output / "baseline_result.json", {
        "alpha": alpha, "gamma": gamma, "result": result,
        "load": alpha * inputs["load"], "pv_profile": inputs["pv"],
        "tou": inputs["tou"], "source_fingerprint": fingerprint,
    })
    return report


def sensitivity_cases() -> list[dict[str, Any]]:
    """论文 Fig.8 的 19 个单因素情景，包含两档上网电价下调。"""
    baseline = {
        "gamma": cfg.BASELINE_GAMMA,
        "pv_capex_yuan_per_kw": cfg.PV_CAPEX,
        "ess_capex_yuan_per_kwh": cfg.ESS_CAPEX,
        "demand_share": cfg.DEMAND_SHARE,
        "tou_multiplier": 1.0, "discount_rate": cfg.R_REQ,
        "export_price_yuan_per_kwh": cfg.EXPORT_PRICE,
    }
    definitions = [
        ("baseline", "基准", None, None),
        ("pv_capex_minus_10pct", "光伏投资成本", "pv_capex_yuan_per_kw", 3420.0),
        ("pv_capex_minus_20pct", "光伏投资成本", "pv_capex_yuan_per_kw", 3040.0),
        ("ess_capex_minus_10pct", "储能投资成本", "ess_capex_yuan_per_kwh", 900.0),
        ("ess_capex_minus_20pct", "储能投资成本", "ess_capex_yuan_per_kwh", 800.0),
        ("gamma_080", "合同价格系数", "gamma", 0.80),
        ("gamma_085", "合同价格系数", "gamma", 0.85),
        ("gamma_090", "合同价格系数", "gamma", 0.90),
        ("gamma_095", "合同价格系数", "gamma", 0.95),
        ("demand_share_10pct", "需量分成", "demand_share", 0.10),
        ("demand_share_15pct", "需量分成", "demand_share", 0.15),
        ("demand_share_20pct", "需量分成", "demand_share", 0.20),
        ("tou_minus_2pct", "分时电价", "tou_multiplier", 0.98),
        ("tou_plus_2pct", "分时电价", "tou_multiplier", 1.02),
        ("discount_rate_4pct", "折现率", "discount_rate", 0.04),
        ("discount_rate_5pct", "折现率", "discount_rate", 0.05),
        ("discount_rate_6pct", "折现率", "discount_rate", 0.06),
        ("fit_minus_2_5pct", "上网电价", "export_price_yuan_per_kwh", 0.34125),
        ("fit_minus_5pct", "上网电价", "export_price_yuan_per_kwh", 0.3325),
    ]
    cases = []
    for case_id, group, parameter, value in definitions:
        inputs = dict(baseline)
        if parameter:
            inputs[parameter] = float(value)
        cases.append({"case_id": case_id, "group": group,
                      "changed_parameter": parameter, "changed_value": value,
                      "inputs": inputs})
    return cases


def run_sensitivity(
    output: Path = cfg.SENSITIVITY_OUTPUT, *, threads: int = 12,
    time_limit: float = 1200.0,
) -> dict:
    """逐情景重优化；从零运行不依赖历史 LHS 缓存或补算脚本。"""
    if threads < 1 or time_limit <= 0:
        raise ValueError("threads 和 time_limit 必须为正")
    output = Path(output)
    fingerprint = source_fingerprint()
    cases = sensitivity_cases()
    results = []
    baseline_item = None
    for index, case in enumerate(cases, 1):
        path = output / "cases" / f"{case['case_id']}.json"
        if path.exists():
            item = json.loads(path.read_text(encoding="utf-8"))
            if (item.get("case") != case or item.get("source_fingerprint") != fingerprint
                    or not item.get("checks") or not all(item["checks"].values())):
                raise RuntimeError(f"敏感性缓存与当前模型不一致，请指定新的输出目录：{path}")
        elif baseline_item is not None and case["inputs"] == cases[0]["inputs"]:
            item = {**baseline_item, "case": case}
        else:
            item = _solve_sensitivity(case, threads, time_limit)
            item["source_fingerprint"] = fingerprint
        write_json(path, item)
        if case["case_id"] == "baseline":
            baseline_item = item
        results.append(item)
        write_json(output / "checkpoint.json", {
            "status": "running", "completed": index, "total": len(cases),
            "last_case_id": case["case_id"],
        })
        print(f"[{index}/{len(cases)}] {case['case_id']} complete", flush=True)
    base = baseline_item["result"]
    rows = []
    for item in results:
        case, res = item["case"], item["result"]
        rows.append({
            "case_id": case["case_id"], "group": case["group"],
            "changed_parameter": case["changed_parameter"],
            "changed_value": case["changed_value"], **case["inputs"],
            "pv_kw": res["pv_kw"], "ess_kwh": res["ess_kwh"], "npv_yuan": res["npv_yuan"],
            "delta_npv_yuan": res["npv_yuan"] - base["npv_yuan"],
            "delta_pv_kw": res["pv_kw"] - base["pv_kw"],
            "delta_ess_kwh": res["ess_kwh"] - base["ess_kwh"],
            "user_saving_yuan": res["user_saving_yuan"],
            "investment_state": res["investment_state"],
        })
    fit = [row["npv_yuan"] for row in rows if row["group"] == "上网电价"]
    monotone = fit[1] <= fit[0] + 1000 and fit[0] <= base["npv_yuan"] + 1000
    if not monotone:
        raise RuntimeError("上网电价下调的 NPV 单调性审计失败")
    write_csv(output / "sensitivity_results.csv", rows)
    report = {
        "status": "complete", "stage": "final_one_factor_reoptimized_capacity_sensitivity_with_fit",
        "source_fingerprint": fingerprint, "case_count": len(cases),
        "alpha": 1.0, "mutex_formulation": "sos; binary for configured difficult cases", "baseline": base,
        "capacity_rule": "每个非基准情景重新优化光伏与储能容量",
        "cases": cases, "results": rows,
        "checks": {"all_cases_optimal_and_audited": True, "fit_npv_monotone_on_decline": bool(monotone)},
        "artifacts": {"table": str(output / "sensitivity_results.csv")},
    }
    write_json(output / "design.json", {"cases": cases, "source_fingerprint": fingerprint})
    write_json(output / "run_summary.json", report)
    write_json(output / "checkpoint.json", {"status": "complete", "completed": len(cases), "total": len(cases)})
    return report




def _audit_sensitivity(result: dict[str, Any], rate: float) -> dict[str, bool]:
    audit = result.get("audit", {})
    flows = np.asarray(result.get("cashflows_actual_investor", ()), dtype=float)
    built = float(result.get("pv_kw", 0)) > 0 or float(result.get("ess_kwh", 0)) > 0
    npv = float(result.get("npv_actual_investor", np.nan))
    npv_ok = (
        flows.shape == (cfg.LIFE + 1,) and np.isfinite(flows).all()
        and abs(npv - npv_at(flows, rate)) <= 1e-5
    ) if built else abs(npv) <= 1e-5
    return {
        "optimal": result.get("status") == "optimal",
        "npv_reconstructed": bool(npv_ok),
        "user_not_worse_off": float(result.get("user_saving", -np.inf)) >= -1.0,
        "load_balance_ok": abs(float(audit.get("max_load_balance_error_kw", np.inf))) <= 1e-5,
        "pv_balance_ok": abs(float(audit.get("max_pv_balance_error_kw", np.inf))) <= 1e-5,
        "soc_ok": abs(float(audit.get("max_soc_transition_error_kwh", np.inf))) <= 1e-5,
        "no_simultaneous_charge_discharge": int(audit.get("simultaneous_chg_dis_steps", -1)) == 0,
        "monthly_demand_ok": bool(audit.get("monthly_demand_upper_ok", False)),
        "economic_reconstruction_ok": (
            abs(float(audit.get("user_cost_reconstruction_error", np.inf))) <= 1e-4
            and abs(float(audit.get("cashflow_reconstruction_error", np.inf))) <= 1e-4
        ),
        "self_use_ok": bool(audit.get("self_use_ok", False)) if built else True,
    }


def _solve_sensitivity(case: dict[str, Any], threads: int, time_limit: float) -> dict[str, Any]:
    inputs = case["inputs"]
    formulation = "binary" if case["case_id"] in cfg.SENSITIVITY_BINARY_CASES else "sos"
    with _config_overrides(
        PV_CAPEX=float(inputs["pv_capex_yuan_per_kw"]),
        ESS_CAPEX=float(inputs["ess_capex_yuan_per_kwh"]),
        DEMAND_SHARE=float(inputs["demand_share"]),
        EXPORT_PRICE=float(inputs["export_price_yuan_per_kwh"]),
        MUTEX=True, MUTEX_FORMULATION=formulation,
    ):
        result = solve_planning(
            1.0, float(inputs["gamma"]),
            tou_multiplier=float(inputs["tou_multiplier"]),
            discount_rate=float(inputs["discount_rate"]),
            include_user=True, threads=threads, time_limit=time_limit,
        )
    checks = _audit_sensitivity(result, float(inputs["discount_rate"]))
    if not all(checks.values()):
        flows = np.asarray(result.get("cashflows_actual_investor", ()), dtype=float)
        reconstructed = npv_at(flows, float(inputs["discount_rate"])) if flows.size else None
        raise RuntimeError(
            f"情景 {case['case_id']} 审计失败：{checks}; "
            f"pv={result.get('pv_kw')}, ess={result.get('ess_kwh')}, "
            f"npv={result.get('npv_actual_investor')}, reconstructed={reconstructed}, "
            f"flow_shape={flows.shape}, model_error={result.get('npv_reconciliation_error')}"
        )
    return {
        "case": case,
        "checks": checks,
        "solver_formulation": formulation,
        "result": {
            "status": result["status"],
            "investment_state": result["investment_state"],
            "pv_kw": float(result["pv_kw"]),
            "ess_kwh": float(result["ess_kwh"]),
            "npv_yuan": float(result["npv_actual_investor"]),
            "user_saving_yuan": float(result["user_saving"]),
            "cashflows_actual_investor": [float(v) for v in result["cashflows_actual_investor"]],
            "audit": result["audit"],
        },
    }
