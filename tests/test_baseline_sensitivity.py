"""编号5：单因素设计与逐时电价调整的关键验证。"""

import numpy as np
import pytest

from src import config as cfg, data
from src.model import npv_at
from src import experiment1 as sensitivity


def test_case_matrix_changes_one_input_at_a_time():
    cases = sensitivity.sensitivity_cases()
    assert len(cases) == 19
    assert cases[0]["case_id"] == "baseline"
    assert {case["group"] for case in cases} == {
        "基准", "光伏投资成本", "储能投资成本", "合同价格系数", "需量分成", "分时电价", "折现率", "上网电价"
    }
    for case in cases:
        changes = [
            key for key, value in case["inputs"].items()
            if value != cases[0]["inputs"][key]
        ]
        parameter = case["changed_parameter"]
        expected = [] if parameter is None or case["inputs"][parameter] == cases[0]["inputs"][parameter] else [parameter]
        assert changes == expected


def test_tou_multiplier_updates_all_energy_price_inputs_without_mutating_cache(monkeypatch):
    raw_tou = np.array([0.5, 1.0, 1.5])
    raw_load = np.array([2.0, 2.0, 2.0])
    baseline = {
        "load": raw_load,
        "pv": np.ones(3),
        "tou": raw_tou,
        "month": np.ones(3, dtype=int),
        "base_sell": 1.0,
        "no_project": float(np.dot(raw_load, raw_tou)) + 40.0,
    }
    monkeypatch.setattr(data, "_CACHE", baseline)
    monkeypatch.setattr(data, "_base_sell", lambda pv, tou: float(np.mean(tou)))
    scaled = data.load_data(tou_multiplier=1.02)
    assert scaled["tou"] == pytest.approx(raw_tou * 1.02)
    assert scaled["base_sell"] == pytest.approx(1.02)
    assert scaled["no_project"] == pytest.approx(float(np.dot(raw_load, raw_tou * 1.02)) + 40.0)
    assert data.load_data()["tou"] == pytest.approx(raw_tou)
    with pytest.raises(ValueError):
        data.load_data(tou_multiplier=0.0)


def test_case_solver_receives_correct_override_and_restores_config(monkeypatch):
    original_capex = cfg.PV_CAPEX
    calls = []

    def fake_solve(alpha, gamma, *, tou_multiplier, discount_rate, include_user, threads, time_limit):
        calls.append((alpha, gamma, tou_multiplier, cfg.PV_CAPEX, cfg.ESS_CAPEX, cfg.DEMAND_SHARE, cfg.EXPORT_PRICE))
        cashflows = np.array([-100.0] + [20.0] * cfg.LIFE)
        return {
            "status": "optimal",
            "investment_state": "pv_battery",
            "pv_kw": 200.0,
            "ess_kwh": 100.0,
            "invest": 100.0,
            "npv_actual_investor": npv_at(cashflows, cfg.R_REQ),
            "cashflows_actual_investor": cashflows.tolist(),
            "irr_actual_investor": 0.1,
            "user_saving": 1.0,
            "audit": {
                "max_load_balance_error_kw": 0.0,
                "max_pv_balance_error_kw": 0.0,
                "max_soc_transition_error_kwh": 0.0,
                "simultaneous_chg_dis_steps": 0,
                "monthly_demand_upper_ok": True,
                "user_cost_reconstruction_error": 0.0,
                "cashflow_reconstruction_error": 0.0,
                "self_use_ok": True,
                "self_use_ratio": 0.7,
            },
        }

    monkeypatch.setattr(sensitivity, "solve_planning", fake_solve)
    case = next(item for item in sensitivity.sensitivity_cases() if item["case_id"] == "pv_capex_minus_10pct")
    solved = sensitivity._solve_sensitivity(case, threads=1, time_limit=5.0)
    assert all(solved["checks"].values())
    assert calls == [(1.0, 0.8, 1.0, 3420.0, 1000.0, 0.2, 0.35)]
    assert cfg.PV_CAPEX == original_capex
