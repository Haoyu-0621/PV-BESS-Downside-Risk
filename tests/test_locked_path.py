from __future__ import annotations

import numpy as np
import pytest

from src import config as cfg
from src.experiment2 import assemble_locked_path, validate_onset_sequence
from src.model import npv_at


def _state(*, annual_cf: float, annual_user: float, annual_outside: float, capacity=(100.0, 50.0)):
    cashflows = [-1000.0] + [annual_cf] * cfg.LIFE
    return {
        "capacity": list(capacity),
        "cashflows_actual_investor": cashflows,
        "annual_user_costs": [annual_user] * cfg.LIFE,
        "annual_outside_costs": [annual_outside] * cfg.LIFE,
        "npv_actual_investor": npv_at(np.asarray(cashflows), cfg.R_REQ),
    }


def test_locked_path_switches_at_one_based_onset_and_keeps_capacity():
    base = _state(annual_cf=120.0, annual_user=80.0, annual_outside=100.0)
    stress = _state(annual_cf=60.0, annual_user=45.0, annual_outside=50.0)

    path = assemble_locked_path(base, stress, onset_year=6)

    assert path["capacity_locked"] is True
    assert path["cashflows_actual_investor"][1:6] == [120.0] * 5
    assert path["cashflows_actual_investor"][6:] == [60.0] * 10
    assert path["annual_user_costs"][:5] == [80.0] * 5
    assert path["annual_user_costs"][5:] == [45.0] * 10
    assert path["user_constraint_ok"] is True
    assert path["npv_loss"] > 0.0


def test_onset_one_matches_persistent_stress_and_later_onset_reduces_loss():
    base = _state(annual_cf=120.0, annual_user=80.0, annual_outside=100.0)
    stress = _state(annual_cf=60.0, annual_user=45.0, annual_outside=50.0)
    paths = [assemble_locked_path(base, stress, onset) for onset in (1, 6, 11, 15)]

    assert paths[0]["npv_actual_investor"] == pytest.approx(
        stress["npv_actual_investor"]
    )
    validation = validate_onset_sequence(paths)
    assert validation["loss_nonincreasing_with_later_onset"] is True
    assert validation["all_capacity_locked"] is True
    assert validation["all_user_constraints_ok"] is True


def test_locked_path_rejects_capacity_change():
    base = _state(annual_cf=120.0, annual_user=80.0, annual_outside=100.0)
    stress = _state(
        annual_cf=60.0,
        annual_user=45.0,
        annual_outside=50.0,
        capacity=(101.0, 50.0),
    )

    with pytest.raises(ValueError, match="容量不一致"):
        assemble_locked_path(base, stress, onset_year=6)
