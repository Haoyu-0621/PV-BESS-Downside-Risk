"""样本量设计和加权尾部统计的关键性质。"""

import numpy as np
import pytest

from src.experiment2 import (
    gaussian_copula_weights,
    weighted_cvar_upper,
    weighted_quantile,
)
from src.experiment2 import _strata_complete
from pathlib import Path
from src.experiment2 import generate_scenarios
from src.experiment2 import _load_scenarios


def test_equal_weights_match_main_quantile_definition():
    values = np.array([7.0, 1.0, 10.0, 3.0, 5.0])
    weights = np.ones(len(values))
    for q in (0.05, 0.5, 0.95):
        assert weighted_quantile(values, weights, q) == pytest.approx(
            np.quantile(values, q)
        )


def test_cvar_uses_fraction_of_boundary_scenario():
    values = np.array([1.0, 2.0, 10.0])
    weights = np.array([0.2, 0.3, 0.5])
    assert weighted_cvar_upper(values, weights, 0.6) == pytest.approx(
        (0.5 * 10.0 + 0.1 * 2.0) / 0.6
    )


def test_gaussian_copula_weights_are_normalized_and_distinct():
    rng = np.random.default_rng(9)
    u, v = rng.random(1000), rng.random(1000)
    negative = gaussian_copula_weights(u, v, -0.5)
    positive = gaussian_copula_weights(u, v, 0.5)
    assert negative.sum() == pytest.approx(1.0)
    assert positive.sum() == pytest.approx(1.0)
    assert np.all(negative > 0)
    assert np.all(positive > 0)
    assert not np.allclose(negative, positive)


def test_independent_lhs_case_has_exact_strata_and_expected_size(tmp_path):
    path = Path(generate_scenarios(tmp_path, n=250, seed=20260921, gamma_bounds=(0.4,0.8))["artifacts"]["scenarios"])
    scenarios = _load_scenarios(path, expected_n=250, gamma_bounds=(0.4,0.8))
    alpha = np.array([row["alpha"] for row in scenarios])
    gamma = np.array([row["gamma"] for row in scenarios])
    assert _strata_complete(alpha, (0.25, 1.0))
    assert _strata_complete(gamma, (0.4, 0.8))
    assert Path(generate_scenarios(tmp_path, n=250, seed=20260921, gamma_bounds=(0.4,0.8))["artifacts"]["scenarios"]) == path
