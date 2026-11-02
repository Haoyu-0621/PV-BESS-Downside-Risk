"""合并后的真实工作流检查：缓存、调用口径、结果表和异常恢复。"""
import ast
from pathlib import Path

import numpy as np
import pytest

from src import config as cfg, experiment1 as e1, experiment2 as e2
from src.data import write_json
from src.model import npv_at


@pytest.fixture(autouse=True)
def isolated_input_fingerprints(tmp_path, monkeypatch):
    """工作流使用模拟求解结果；指纹仅需测试输入，不依赖私有数据。"""
    for name in ("RAW_LOAD_FILE", "PV_PROFILE_FILE", "PV_REF_FILE", "TOU_FILE"):
        path = tmp_path / f"{name}.xlsx"
        path.write_bytes(name.encode())
        monkeypatch.setattr(cfg, name, path)


def _result(*, alpha=1.0, gamma=0.8, pv=1800.0, ess=480.0):
    cashflows = [-1000.0] + [(100 + 100 * alpha * gamma + 100 * cfg.EXPORT_PRICE) * (1 - cfg.PV_DEGRADE) ** y for y in range(cfg.LIFE)]
    audit = {key: 0.0 for key in (
        "max_load_balance_error_kw", "max_pv_balance_error_kw",
        "max_pv_direct_priority_error_kw", "max_soc_transition_error_kwh",
        "soc_wrap_error_kwh", "user_cost_reconstruction_error", "cashflow_reconstruction_error")}
    audit.update(simultaneous_chg_dis_steps=0, simultaneous_export_grid_to_user_steps=0,
                 monthly_demand_upper_ok=True, self_use_ok=True)
    return {
        "status": "optimal", "investment_state": "pv_battery", "pv_kw": pv, "ess_kwh": ess,
        "npv_actual_investor": npv_at(np.array(cashflows), cfg.R_REQ),
        "cashflows_actual_investor": cashflows,
        "annual_user_costs": [80.0 * alpha] * cfg.LIFE,
        "annual_outside_costs": [100.0 * alpha] * cfg.LIFE,
        "user_saving": 20.0 * alpha * cfg.LIFE, "audit": audit, "export_peak_kw": 500.0,
    }


def test_lhs_full_pipeline_and_resume_preserve_operating_regimes(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.json"
    write_json(baseline, {"status": "complete", "baseline": {
        "alpha": 1.0, "gamma": 0.8, "pv_kw": 1800.0, "ess_kwh": 480.0}})
    calls = []

    def solve(alpha, gamma, capacity, **kw):
        calls.append((alpha, gamma, capacity, kw))
        return _result(alpha=alpha, gamma=gamma, pv=capacity[0], ess=capacity[1])

    monkeypatch.setattr(e2, "solve_locked", solve)
    original_formulation = cfg.MUTEX_FORMULATION
    output = tmp_path / "lhs"
    first = e2.run(baseline_summary=baseline, output=output, expected_n=10, threads=1)
    assert first["scenario_count"] == 10 and first["result_rows"] == 150
    assert len(first["summary"]) == 15
    assert first["gamma_range"] == [0.4, 1.0]
    assert all(first["checks"].values())
    assert len(calls) == 11
    assert calls[0][3]["enforce_self_use"] is True
    assert calls[0][3]["export_max_kw"] is None
    assert all(call[3]["enforce_self_use"] is False and call[3]["export_max_kw"] == 1000.0 for call in calls[1:])
    assert all(call[2] == (1800.0, 480.0) for call in calls)
    second = e2.run(baseline_summary=baseline, output=output, expected_n=10, threads=1)
    assert len(calls) == 11
    assert second["cache_hits"] == 11
    assert second["summary"] == first["summary"]
    assert cfg.MUTEX_FORMULATION == original_formulation
    # 输入参数变化后，旧缓存不得继续使用。
    monkeypatch.setattr(cfg, "LOCKED_EXPORT_MAX_KW", 900.0)
    e2.run(baseline_summary=baseline, output=output, expected_n=10, threads=1)
    assert len(calls) == 22


def test_sensitivity_full_pipeline_contains_fit_and_resumes(tmp_path, monkeypatch):
    calls = []

    def solve(alpha, gamma, **kw):
        calls.append((cfg.PV_CAPEX, cfg.ESS_CAPEX, cfg.EXPORT_PRICE, gamma, kw["discount_rate"], cfg.MUTEX_FORMULATION))
        result = _result(alpha=alpha, gamma=gamma)
        result["npv_actual_investor"] = npv_at(np.array(result["cashflows_actual_investor"]), kw["discount_rate"])
        return result

    monkeypatch.setattr(e1, "solve_planning", solve)
    original = (cfg.PV_CAPEX, cfg.ESS_CAPEX, cfg.EXPORT_PRICE, cfg.MUTEX_FORMULATION)
    result = e1.run_sensitivity(tmp_path, threads=1)
    assert result["case_count"] == 19
    assert all(result["checks"].values())
    assert len(calls) == 16  # 3 个与基准相同的参数等级复用已认证结果。
    assert sum(call[5] == "binary" for call in calls) == 2
    assert any(call[2] == pytest.approx(0.34125) for call in calls)
    assert any(call[2] == pytest.approx(0.3325) for call in calls)
    assert (cfg.PV_CAPEX, cfg.ESS_CAPEX, cfg.EXPORT_PRICE, cfg.MUTEX_FORMULATION) == original
    assert e1.run_sensitivity(tmp_path, threads=1)["results"] == result["results"]
    assert len(calls) == 16


def test_baseline_dispatch_is_saved_for_independent_plotting(tmp_path, monkeypatch):
    result = _result()
    result["dispatch"] = {
        "pv_to_user": np.full(cfg.N, 0.4), "ess_to_user": np.full(cfg.N, 0.1),
        "grid_to_user": np.full(cfg.N, 0.5), "export": np.zeros(cfg.N),
        "soc": np.full(cfg.N + 1, 50.0),
    }
    monkeypatch.setattr(e1, "solve_planning", lambda *a, **kw: result)
    monkeypatch.setattr(e1, "load_data", lambda: {
        "load": np.ones(cfg.N), "pv": np.ones(cfg.N) / 1800,
        "tou": np.full(cfg.N, 0.5),
    })
    e1.run_baseline(output=tmp_path)
    from src.plots import weekly_dispatch
    weekly_dispatch(tmp_path / "baseline_result.json", output=tmp_path / "figures")
    assert (tmp_path / "figures" / "fig_weekly_dispatch.png").is_file()
    assert (tmp_path / "figures" / "fig_weekly_dispatch.svg").is_file()


def test_sensitivity_restores_parameters_when_solver_fails(monkeypatch):
    original = (cfg.PV_CAPEX, cfg.EXPORT_PRICE, cfg.MUTEX_FORMULATION)

    def fail(*args, **kwargs):
        raise RuntimeError("solver failed")

    monkeypatch.setattr(e1, "solve_planning", fail)
    case = next(c for c in e1.sensitivity_cases() if c["case_id"] == "pv_capex_minus_20pct")
    with pytest.raises(RuntimeError, match="solver failed"):
        e1._solve_sensitivity(case, 1, 5.0)
    assert (cfg.PV_CAPEX, cfg.EXPORT_PRICE, cfg.MUTEX_FORMULATION) == original


def test_boundary_wrapper_restores_solver_settings(monkeypatch):
    original = (cfg.MUTEX, cfg.TIME_LIMIT, cfg.TOU_METHOD)

    def fail(argv):
        cfg.MUTEX = False
        cfg.TIME_LIMIT = 1.0
        cfg.TOU_METHOD = 1
        raise RuntimeError("paused")

    monkeypatch.setattr(e1, "_run_boundaries", fail)
    with pytest.raises(RuntimeError, match="paused"):
        e1.run_boundaries(["--preview"])
    assert (cfg.MUTEX, cfg.TIME_LIMIT, cfg.TOU_METHOD) == original


def test_boundary_entry_saves_completed_result(tmp_path, monkeypatch):
    expected = {"stage": "P1_state_boundaries", "pv_only_entry": {}, "pv_battery_entry": {}}
    monkeypatch.setattr(e1, "_resumable_state_boundaries", lambda **kw: (expected, False))
    assert e1.run_boundaries(["2"], output_root=tmp_path) == 0
    import json
    assert json.loads((tmp_path / "f2_e1" / "e1_state_boundaries.json").read_text(encoding="utf-8")) == expected


def test_sampling_refuses_to_overwrite_a_different_design(tmp_path):
    e2.generate_scenarios(tmp_path, n=10, seed=1)
    with pytest.raises(RuntimeError, match="已有场景"):
        e2.generate_scenarios(tmp_path, n=10, seed=2)


def test_plotting_and_calculation_dependencies_are_separate():
    src = Path(__file__).resolve().parents[1] / "src"
    for name in ("config.py", "data.py", "model.py", "experiment1.py", "experiment2.py"):
        tree = ast.parse((src / name).read_text(encoding="utf-8-sig"))
        imports = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        assert not any("matplotlib" in text or "src.plots" in text for text in imports)
    tree = ast.parse((src / "plots.py").read_text(encoding="utf-8-sig"))
    imports = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any("src.model" in text or "src.experiment" in text or "gurobi" in text for text in imports)
