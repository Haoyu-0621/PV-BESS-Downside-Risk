"""压缩重构后的冒烟测试：纯逻辑，不调用 Gurobi。

覆盖：现金流指标、差异化结算组件、参数工具、第一阶段边界状态机。
v4.3 裁剪：删除随事后分析移除的测试（锁定生命周期、路径因子、
风险前沿、损失分解、价格让步、k_BE、E2 风险度量）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src import config as cfg
from src.data import _tou
from src.data import source_fingerprint
from src.model import _capacity_spec, _pricing_components, err, irr, npv_at
from src import model as model_mod

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def boundary_input_profile(monkeypatch):
    """边界逻辑测试使用已知负荷与光伏，不依赖未公开的真实数据。"""
    import src.experiment1 as bd
    monkeypatch.setattr(bd, "load_data", lambda: {
        "load": np.full(24, 1000.0), "pv": np.ones(24),
    })


def test_pv_only_prefilter_respects_minimum_capacity_and_self_use():
    import src.experiment1 as bd
    assert bd._pv_only_physical_prefilter(0.139)["feasible"] is False
    assert bd._pv_only_physical_prefilter(0.141)["feasible"] is True


# ---- 现金流指标 ----


def test_npv_known_value():
    cf = np.array([-100.0, 60.0, 60.0])
    assert npv_at(cf, 0.0) == pytest.approx(20.0)
    assert npv_at(cf, 0.1) == pytest.approx(-100 + 60 / 1.1 + 60 / 1.21, rel=1e-9)


def test_irr_known_value():
    cf = np.array([-100.0, 60.0, 60.0])
    r = irr(cf)
    assert r is not None and abs(r - 0.130662) < 1e-5


def test_irr_undefined_on_no_positive():
    assert irr(np.array([-100.0, -5.0])) is None


def test_err_known_value():
    cf = np.array([-100.0, 50.0, 60.0])
    r = err(cf)
    assert r is not None and 0 < r < 0.2


# ---- 参数工具 ----


def test_gamma_floor_positive():
    assert cfg.gamma_floor(0.6) == pytest.approx(cfg.EXPORT_PRICE / 0.6)
    with pytest.raises(ValueError):
        cfg.gamma_floor(0.0)


def test_sell_price_floor():
    # 托底已去掉：结算价 = γ×基准售价，可低于上网电价
    assert cfg.sell_price(0.1, 0.6) == pytest.approx(0.06)
    assert cfg.sell_price(1.0, 0.6) == pytest.approx(0.6)
    assert cfg.sell_price(0.5, 0.6) == pytest.approx(0.3)  # 低于上网价 0.35 也不再托底


# ---- 差异化结算组件 ----


def test_pricing_components_unified_discount_settles_pv_and_all_ess():
    """统一绿电折扣口径（2026-09 定稿）：储能放电不区分来源，
    光伏直供与全部储能放电都按使用/放电时刻 gamma × tou 结算，
    无套利分成（arbitrage_saving/fee 恒 0）。"""
    tou = np.array([0.7, 0.2, 0.6])
    p = _pricing_components(
        pricing_rule=cfg.PRICING_RULE_TOU_SERVICE,
        arbitrage_share=0.0,
        gamma=1.0,
        base_sell=0.4,
        tou=tou,
        pv_to_user=np.array([2.0, 0.0, 0.0]),
        ess_to_user=np.array([0.0, 0.0, 4.0]),
    )
    assert p["pv_settlement"] == pytest.approx(1.0 * 0.7 * 2)
    # 全部储能放电（不区分光伏源/电网源）统一按放电时刻 gamma×tou 结算
    assert p["ess_settlement"] == pytest.approx(1.0 * 0.6 * 4)
    assert p["arbitrage_saving"] == pytest.approx(0.0)
    assert p["arbitrage_fee"] == pytest.approx(0.0)
    assert p["total_settlement"] == pytest.approx(1.0 * 0.7 * 2 + 1.0 * 0.6 * 4)


def test_pricing_components_unified_discount_gamma_scales_settlement():
    """γ 为统一绿电折扣的合同价格系数：γ 缩放全部结算（光伏直供+储能放电）。"""
    tou = np.array([0.7, 0.2, 0.6])
    p = _pricing_components(
        pricing_rule=cfg.PRICING_RULE_TOU_SERVICE,
        arbitrage_share=0.0,
        gamma=0.9,
        base_sell=0.4,
        tou=tou,
        pv_to_user=np.array([2.0, 0.0, 0.0]),
        ess_to_user=np.array([0.0, 0.0, 4.0]),
    )
    assert p["pv_settlement"] == pytest.approx(0.9 * 0.7 * 2)
    assert p["ess_settlement"] == pytest.approx(0.9 * 0.6 * 4)
    assert p["total_settlement"] == pytest.approx(p["pv_settlement"] + p["ess_settlement"])


def test_baseline_gamma_is_eight_tenths_of_tou():
    assert cfg.BASELINE_GAMMA == pytest.approx(0.8)
    p = _pricing_components(
        pricing_rule=cfg.PRICING_RULE_TOU_SERVICE,
        arbitrage_share=0.0,
        gamma=cfg.BASELINE_GAMMA,
        base_sell=0.4,
        tou=np.array([0.7]),
        pv_to_user=np.array([2.0]),
        ess_to_user=np.zeros(1),
    )
    assert p["pv_settlement"] == pytest.approx(0.8 * 0.7 * 2)


def test_pricing_components_reject_invalid_parameters():
    common = dict(
        gamma=1.0,
        base_sell=0.4,
        tou=np.ones(1),
        pv_to_user=np.ones(1),
        ess_to_user=np.zeros(1),
    )
    with pytest.raises(ValueError, match="pricing_rule"):
        _pricing_components(pricing_rule="other", arbitrage_share=0.5, **common)
    with pytest.raises(ValueError, match="arbitrage_share"):
        _pricing_components(
            pricing_rule=cfg.PRICING_RULE_TOU_SERVICE,
            arbitrage_share=1.1,
            **common,
        )


def test_tou_expands_monthly_prices_to_8760_hours(monkeypatch):
    import pandas as pd
    table = pd.DataFrame([["month"] + list(range(24))] + [
        [month] + [0.1 * month + 0.01 * hour for hour in range(24)]
        for month in range(1, 13)
    ])
    monkeypatch.setattr(pd, "read_excel", lambda *args, **kwargs: table)
    tou, month = _tou()
    assert tou.shape == (cfg.N,)
    assert month.shape == (cfg.N,)
    assert np.array_equal(
        np.bincount(month)[1:], np.asarray(cfg.MONTH_DAYS) * 24
    )
    offset = 0
    for days in cfg.MONTH_DAYS:
        block = tou[offset : offset + days * 24].reshape(days, 24)
        assert np.allclose(block, block[0])
        offset += days * 24


def test_capacity_entry_semantics(monkeypatch, tmp_path):
    # 指纹性质检查不依赖未公开的真实 Excel 输入。
    for name in ("RAW_LOAD_FILE", "PV_PROFILE_FILE", "PV_REF_FILE", "TOU_FILE"):
        path = tmp_path / f"{name}.xlsx"
        path.write_bytes(name.encode())
        monkeypatch.setattr(cfg, name, path)
    assert _capacity_spec() == (
        cfg.PV_MIN,
        cfg.PV_MAX,
        cfg.ESS_MIN,
        cfg.ESS_MAX,
        False,
        0.0,
    )
    assert cfg.PV_MIN == 0.0
    assert cfg.ESS_ENTRY_POWER_KW == 0.0
    fingerprint = source_fingerprint()

    monkeypatch.setattr(cfg, "ESS_ENTRY_POWER_KW", 50.0)
    assert source_fingerprint() != fingerprint
    spec = _capacity_spec()
    assert spec[-2] is True
    assert spec[-1] == pytest.approx(100.0)

    monkeypatch.setattr(cfg, "ESS_ENTRY_POWER_KW", cfg.C_RATE * cfg.ESS_MAX + 1.0)
    with pytest.raises(ValueError, match="储能进入功率"):
        _capacity_spec()


def _stub_solver(feasible_at):
    """feasible_at: callable (alpha, gamma) -> bool 的可行性函数。"""

    def fake(alpha, gamma, **kw):
        return {
            "status": "optimal",
            "npv": 1.0 if feasible_at(alpha, gamma) else -1.0,
            "user_saving": 1.0,
        }

    return fake


def test_gamma_boundary_all_feasible(monkeypatch):
    import src.experiment1 as bd

    monkeypatch.setattr(bd, "solve_planning", _stub_solver(lambda a, g: True))
    r = bd.gamma_boundary(1.0, 0.5, 1.0)
    assert r["status"] == "all_feasible"


def test_gamma_boundary_threshold(monkeypatch):
    import src.experiment1 as bd

    monkeypatch.setattr(bd, "solve_planning", _stub_solver(lambda a, g: g >= 0.7))
    r = bd.gamma_boundary(1.0, 0.5, 1.0)
    assert r["status"] == "threshold" and 0.69 < r["threshold"] < 0.71


def test_alpha_boundary_threshold(monkeypatch):
    import src.experiment1 as bd

    monkeypatch.setattr(bd, "solve_planning", _stub_solver(lambda a, g: a >= 0.6))
    r = bd.alpha_boundary(1.0, 0.3, 1.0)
    assert r["status"] == "threshold" and 0.59 < r["threshold"] < 0.61


def test_user_boundary_descending_direction(monkeypatch):
    """用户可行性随 γ 下降：阈值应出现在高 γ 侧。"""
    import src.experiment1 as bd

    def fake(alpha, gamma, **kw):
        return {
            "status": "optimal",
            "npv": 1.0,
            "user_saving": 1.0 if gamma <= 0.7 else -1.0,
        }

    monkeypatch.setattr(bd, "solve_planning", fake)
    r = bd.gamma_boundary(1.0, 0.5, 1.0, criterion="user")
    assert r["status"] == "threshold" and 0.69 < r["threshold"] < 0.71


# ---- 用户总成本约束（合同硬约束）必须处处默认卡死 ----


class _FakeSolver:
    """捕获 _solve 的关键字参数，不触发 Gurobi。"""

    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return {"status": "optimal", "npv_actual_investor": 0.0}


def test_solve_planning_enforces_user_constraint_by_default(monkeypatch):
    fake = _FakeSolver()
    monkeypatch.setattr(model_mod, "_solve", fake)
    model_mod.solve_planning(1.0, 0.8)
    assert fake.kwargs["include_user"] is True


def test_solve_locked_enforces_user_constraint_by_default(monkeypatch):
    fake = _FakeSolver()
    monkeypatch.setattr(model_mod, "_solve", fake)
    model_mod.solve_locked(1.0, 0.8, (1800.0, 500.0))
    assert fake.kwargs["include_user"] is True


def test_boundary_probe_enforces_user_constraint_for_all_criteria(monkeypatch):
    import src.experiment1 as bd

    captured = []

    def fake_planning(alpha, gamma, **kw):
        captured.append(kw)
        return {"status": "optimal", "npv": 100.0}

    monkeypatch.setattr(bd, "solve_planning", fake_planning)
    for criterion in ("investor", "joint", "user"):
        bd._evaluate(0.5, 0.7, None, criterion, investment_state="pv_only")
    assert captured and all(kw["include_user"] is True for kw in captured)


# ---- 六类边界状态（桩求解器）----


def _status_stub(probe_fn):
    def fake(alpha, gamma, **kw):
        return dict(probe_fn(gamma))

    return fake


def test_boundary_six_states(monkeypatch):
    import src.experiment1 as bd

    ok = {"status": "optimal", "npv": 1.0, "user_saving": 1.0}
    bad = {"status": "optimal", "npv": -1.0, "user_saving": 1.0}
    inf = {"status": "infeasible", "npv": None, "user_saving": None}

    monkeypatch.setattr(bd, "solve_planning", _status_stub(lambda g: ok))
    assert bd.gamma_boundary(1.0, 0.5, 1.0)["status"] == "all_feasible"

    monkeypatch.setattr(bd, "solve_planning", _status_stub(lambda g: bad))
    assert bd.gamma_boundary(1.0, 0.5, 1.0)["status"] == "all_economically_infeasible"

    monkeypatch.setattr(bd, "solve_planning", _status_stub(lambda g: inf))
    assert bd.gamma_boundary(1.0, 0.5, 1.0)["status"] == "all_model_infeasible"

    # 混合：1 个模型无解 + 2 个经济不可行 → no_solution，不得归入经济不可行
    monkeypatch.setattr(
        bd, "solve_planning", _status_stub(lambda g: inf if g < 0.6 else bad)
    )
    assert bd.gamma_boundary(1.0, 0.5, 1.0)["status"] == "no_solution"

    monkeypatch.setattr(
        bd, "solve_planning", _status_stub(lambda g: ok if 0.6 < g < 0.9 else bad)
    )
    assert bd.gamma_boundary(1.0, 0.5, 1.0)["status"] == "non_monotonic"


def test_build_state_classification_uses_value_priority(monkeypatch):
    import src.experiment1 as bd

    def state(npv, *, status="optimal", pv_kw=250.0, ess_kwh=None):
        return {
            "status": status,
            "npv": npv,
            "pv_kw": pv_kw,
            "ess_kwh": ess_kwh,
        }

    # 两种方案均不盈利：不投资（经济不可行）。
    assert bd.classify_build_state(state(-1), state(-2, ess_kwh=100))["state"] == (
        "no_invest_economic"
    )
    # 储能方案虽然盈利，但 NPV 更低：仍选仅光伏。
    assert bd.classify_build_state(state(10), state(5, ess_kwh=100))["state"] == (
        "pv_only"
    )
    # PV-only 不盈利、光储盈利：直接进入光伏+储能。
    assert bd.classify_build_state(state(-1), state(5, ess_kwh=100))["state"] == (
        "direct_to_pv_battery"
    )
    # PV-only 物理不可行、光储可行且盈利：同样是直接进入光储。
    assert bd.classify_build_state(
        state(-1, status="infeasible"), state(5, ess_kwh=100)
    )["state"] == "direct_to_pv_battery"
    # 光储的储能容量低于门槛时，不把它当作有效光储状态。
    assert bd.classify_build_state(state(10), state(20, ess_kwh=50))["state"] == (
        "pv_only"
    )
    # 两种强制建设状态都物理不可行：不投资（物理不可行）。
    assert bd.classify_build_state(
        state(None, status="infeasible"), state(None, status="infeasible")
    )["state"] == "no_invest_physical"


def test_build_state_classification_prefers_actual_investor_npv():
    import src.experiment1 as bd

    def state(planning, actual, *, ess_kwh=None):
        return {
            "status": "optimal",
            "npv": planning,
            "npv_planning": planning,
            "npv_actual_investor": actual,
            "pv_kw": 500.0,
            "ess_kwh": ess_kwh,
            "grid_charging_cost_pv": 100.0 if ess_kwh else 0.0,
        }

    # 规划目标口径下光储更高，但加回投资方代垫的电池电网充电成本后，
    # 仅光伏更高；边界分类必须使用实际投资方 NPV。
    result = bd.classify_build_state(
        state(200.0, 900.0), state(300.0, 800.0, ess_kwh=100.0)
    )
    assert result["state"] == "pv_only"
    assert result["pv_npv_actual"] == pytest.approx(900.0)
    assert result["battery_npv_actual"] == pytest.approx(800.0)
    assert result["pv_npv_planning"] == pytest.approx(200.0)
    assert result["battery_npv_planning"] == pytest.approx(300.0)

    # 反向检查：实际 NPV 变为光储更高时，才进入光储状态。
    assert bd.classify_build_state(
        state(300.0, 800.0), state(200.0, 900.0, ess_kwh=100.0)
    )["state"] == "pv_battery"


def test_state_boundaries_use_actual_investor_npv(monkeypatch):
    import src.experiment1 as bd

    def fake(alpha, gamma, **kw):
        state = kw["investment_state"]
        pv_npv_actual = 100.0 * (gamma - 0.60)
        value = (
            pv_npv_actual
            if state == "pv_only"
            else pv_npv_actual + 100.0 * (gamma - 0.70)
        )
        # 故意让规划 NPV 给出不同的边界，防止测试只读到了 npv。
        planning = 100.0 * (gamma - (0.90 if state == "pv_only" else 0.80))
        return {
            "status": "optimal",
            "npv": planning,
            "npv_planning": planning,
            "npv_actual_investor": value,
            "pv_kw": 500.0,
            "ess_kwh": 100.0 if state == "pv_battery" else 0.0,
            "grid_charging_cost_pv": 50.0 if state == "pv_battery" else 0.0,
        }

    monkeypatch.setattr(bd, "solve_planning", fake)
    pv = bd.gamma_state_value_boundary(
        0.5, 0.0, 1.0, kind="pv_entry", tol=1e-3
    )
    bat = bd.gamma_state_value_boundary(
        0.5, 0.0, 1.0, kind="battery_upgrade", tol=1e-3
    )
    assert pv["threshold"] == pytest.approx(0.60, abs=1e-3)
    assert bat["threshold"] == pytest.approx(0.70, abs=1e-3)


def test_state_boundaries_are_separate_and_use_expected_metrics(monkeypatch):
    import src.experiment1 as bd

    calls = []

    def fake(alpha, gamma, **kw):
        state = kw["investment_state"]
        calls.append((alpha, gamma, state, kw["include_user"]))
        pv_value = 100.0 * (gamma - 0.60)
        if state == "pv_only":
            return {"status": "optimal", "npv": pv_value}
        return {"status": "optimal", "npv": pv_value + 100.0 * (gamma - 0.70)}

    monkeypatch.setattr(bd, "solve_planning", fake)
    pv = bd.gamma_state_value_boundary(
        0.5, 0.0, 1.0, kind="pv_entry", tol=1e-3
    )
    bat = bd.gamma_state_value_boundary(
        0.5, 0.0, 1.0, kind="battery_upgrade", tol=1e-3
    )
    assert pv["status"] == "threshold"
    assert pv["threshold"] == pytest.approx(0.60, abs=1e-3)
    assert bat["status"] == "threshold"
    assert bat["threshold"] == pytest.approx(0.70, abs=1e-3)
    # 用户总成本约束（合同硬约束）必须在边界探针中恒为开启；
    # 判据 criterion 只决定用哪个指标判可行，不再决定该约束开关。
    assert all(include_user for _, _, _, include_user in calls)

    curve = bd.state_boundary_curve(n_points=20, kind="pv_entry", tol=1e-3)
    assert curve["requested_points"] == 20
    assert len(curve["points"]) == 20
    assert curve["valid_points"] == 20
    assert curve["alpha_search_status"] == "ok"
    assert all(point["alpha"] > 0.0 for point in curve["points"])
    assert all(point["status"] == "threshold" for point in curve["points"])


def test_state_boundary_curve_auto_finds_lower_alpha(monkeypatch):
    import src.experiment1 as bd

    seen = []

    def fake(alpha, gamma, **kw):
        seen.append(alpha)
        threshold = 1.2 - alpha
        return {"status": "optimal", "npv": 100.0 * (gamma - threshold)}

    monkeypatch.setattr(bd, "solve_planning", fake)
    curve = bd.state_boundary_curve(n_points=20, kind="pv_entry", tol=1e-3)
    assert curve["alpha_search_status"] == "ok"
    assert curve["valid_points"] == 20
    assert curve["boundary_alpha_range"][0] > 0.19
    assert min(seen) > 0.0  # alpha=0 仅作不投资区解释，不调用投资模型


def test_alpha_state_value_boundary_at_gamma_one(monkeypatch):
    import src.experiment1 as bd

    def fake(alpha, gamma, **kw):
        state = kw["investment_state"]
        return {
            "status": "optimal",
            "npv": 100.0 * (alpha - 0.4) + (10.0 if state == "pv_battery" else 0.0),
            "pv_kw": 500.0,
            "ess_kwh": 100.0 if state == "pv_battery" else 0.0,
        }

    monkeypatch.setattr(bd, "solve_planning", fake)
    r = bd.alpha_state_value_boundary(
        gamma=1.0, alpha_lo=0.0, alpha_hi=1.0, tol=1e-3, probe_count=5
    )
    assert r["status"] == "threshold"
    assert r["threshold"] == pytest.approx(0.30, abs=2e-3)


def test_state_metric_reuses_compact_planning_cache(monkeypatch):
    import src.experiment1 as bd

    calls = []

    def fake(alpha, gamma, **kw):
        calls.append((alpha, gamma, kw["investment_state"]))
        state = kw["investment_state"]
        npv = 10.0 if state == "pv_only" else 20.0
        # free 状态返回大容量(≥门槛)避免触发 entry_too_small；pv_only/pv_battery 也大容量
        return {"status": "optimal", "npv": npv, "pv_kw": 500.0}

    monkeypatch.setattr(bd, "solve_planning", fake)
    cache = {}
    bd._state_metric(
        0.5, 0.75, kind="pv_entry", discount_rate=None, include_user=True, cache=cache
    )
    bd._state_metric(
        0.5,
        0.75,
        kind="battery_upgrade",
        discount_rate=None,
        include_user=True,
        cache=cache,
    )
    bd._state_metric(
        0.5,
        0.75,
        kind="battery_upgrade",
        discount_rate=None,
        include_user=True,
        cache=cache,
    )
    # 两个强制建设状态各缓存一次，重复调用命中缓存；自由规划不再参与
    # 边界指标计算，避免把"不投资"经济状态误作模型无解。
    assert len(calls) == 2
    assert {state for _, _, state in calls} == {"pv_only", "pv_battery"}
    assert len(cache) == 2
    assert all(
        {"status", "npv", "pv_kw", "ess_kwh"} <= set(value)
        for value in cache.values()
    )
