"""Independent full-history Gaussian SCAR-TM-OU oracle tests."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from pyscarcopula import BivariateGaussianCopula
from pyscarcopula._types import LatentResult, ou_params
from pyscarcopula.api import predict as api_predict
from pyscarcopula.strategy._base import get_strategy_for_result

from ._analytical_oracles import (
    gaussian_conditional_cdf,
    gaussian_copula_parameter_from_state,
)
from ._scar_tm_ou_oracle import (
    GaussianScarOuReference,
    QuadratureDistribution,
    simulate_gaussian_scar_history,
)
from ._statistical_assertions import assert_uniform_pit


KAPPA = 2.0
MU = 0.4
NU = 3.0
N_OBS = 21


def _history(seed=20260920):
    return simulate_gaussian_scar_history(
        N_OBS, KAPPA, MU, NU, seed=seed
    )[0]


def _reference(history, *, n_nodes=501, range_sigma=7.0):
    oracle = GaussianScarOuReference(
        KAPPA,
        MU,
        NU,
        len(history),
        n_nodes=n_nodes,
        range_sigma=range_sigma,
    )
    return oracle, oracle.filter(history)


def _result(*, K=700, grid_range=7.0):
    return LatentResult(
        log_likelihood=0.0,
        method="SCAR-TM-OU",
        copula_name="Gaussian copula",
        success=True,
        params=ou_params(KAPPA, MU, NU),
        K=K,
        grid_range=grid_range,
        grid_method="dense",
        adaptive=False,
        transition_method="matrix",
        max_K=None,
    )


def _production_distribution(history, horizon, *, K=700):
    copula = BivariateGaussianCopula()
    result = _result(K=K)
    state = get_strategy_for_result(result).predictive_state(
        copula, history, result, horizon=horizon
    )
    return QuadratureDistribution(state.z_grid, state.prob)


def _assert_distribution_close(production, reference, stationary_sigma):
    assert abs(production.mean - reference.mean) < 0.025 * stationary_sigma
    assert abs(production.variance - reference.variance) < (
        0.045 * stationary_sigma ** 2
    )
    q = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    error = np.max(np.abs(production.quantile(q) - reference.quantile(q)))
    assert error < 0.045 * stationary_sigma
    production_r = production.expectation(
        gaussian_copula_parameter_from_state(production.nodes)
    )
    reference_r = reference.expectation(
        gaussian_copula_parameter_from_state(reference.nodes)
    )
    assert abs(production_r - reference_r) < 0.006


def test_reference_stationary_distribution_and_transition_preserve_ou_law():
    oracle = GaussianScarOuReference(
        KAPPA, MU, NU, N_OBS, n_nodes=601, range_sigma=8.0
    )
    stationary = oracle.stationary
    propagated = oracle.propagate(stationary)
    sigma = NU / np.sqrt(2.0 * KAPPA)
    assert stationary.mean == pytest.approx(MU, abs=2e-13)
    assert stationary.variance == pytest.approx(sigma ** 2, rel=2e-10)
    assert propagated.mean == pytest.approx(MU, abs=2e-10)
    assert propagated.variance == pytest.approx(sigma ** 2, rel=3e-8)
    assert oracle.stationary_tail_mass_bound < 2e-15


def test_reference_filter_converges_in_nodes_range_and_mixture_cdf():
    history = _history()
    coarse, coarse_result = _reference(
        history, n_nodes=301, range_sigma=7.0
    )
    fine, fine_result = _reference(
        history, n_nodes=601, range_sigma=8.0
    )
    sigma = NU / np.sqrt(2.0 * KAPPA)
    assert abs(coarse_result.current.mean - fine_result.current.mean) < 2e-4 * sigma
    assert abs(
        coarse_result.predictive.variance
        - fine_result.predictive.variance
    ) < 4e-4 * sigma ** 2
    values = np.linspace(0.01, 0.99, 199)
    coarse_cdf = coarse.mixture_cdf(
        values, 0.87, coarse_result.predictive
    )
    fine_cdf = fine.mixture_cdf(
        values, 0.87, fine_result.predictive
    )
    assert np.max(np.abs(coarse_cdf - fine_cdf)) < 3e-4
    assert fine_result.stationary_tail_mass_bound < 2e-15


def test_reference_mixture_sample_has_uniform_reference_pit():
    history = _history()
    oracle, result = _reference(history, n_nodes=501, range_sigma=7.0)
    sample = oracle.sample_mixture(
        20_000,
        0.87,
        result.predictive,
        np.random.default_rng(20260921),
    )
    pit = oracle.mixture_pit(sample, 0.87, result.predictive)
    assert_uniform_pit(pit)


@pytest.mark.parametrize("horizon", ["current", "next"])
def test_production_state_distribution_matches_continuous_reference(horizon):
    history = _history()
    oracle, result = _reference(history, n_nodes=601, range_sigma=8.0)
    expected = result.current if horizon == "current" else result.predictive
    observed = _production_distribution(history, horizon, K=700)
    _assert_distribution_close(observed, expected, oracle.stationary_sigma)


@pytest.mark.parametrize("horizon", ["current", "next"])
def test_production_conditional_prediction_passes_full_history_mixture_pit(
        horizon):
    history = _history()
    oracle, reference_result = _reference(
        history, n_nodes=601, range_sigma=8.0
    )
    distribution = (
        reference_result.current
        if horizon == "current"
        else reference_result.predictive
    )
    given = 0.87
    samples = api_predict(
        BivariateGaussianCopula(),
        history,
        _result(K=700),
        18_000,
        given={0: given},
        horizon=horizon,
        predictive_r_mode="grid",
        rng=np.random.default_rng(20260922),
    )
    np.testing.assert_array_equal(samples[:, 0], given)
    pit = oracle.mixture_pit(samples[:, 1], given, distribution)
    assert_uniform_pit(pit, numerical_floor=0.006)


def test_histories_with_same_last_row_recompute_different_full_history_state():
    prefix = np.linspace(0.12, 0.88, N_OBS - 1)
    last = np.array([[0.54, 0.46]])
    concordant = np.vstack((np.column_stack((prefix, prefix)), last))
    discordant = np.vstack((np.column_stack((prefix, 1.0 - prefix)), last))

    oracle_c, filtered_c = _reference(
        concordant, n_nodes=501, range_sigma=8.0
    )
    oracle_d, filtered_d = _reference(
        discordant, n_nodes=501, range_sigma=8.0
    )
    reference_gap = (
        filtered_c.predictive.expectation(
            gaussian_copula_parameter_from_state(filtered_c.predictive.nodes)
        )
        - filtered_d.predictive.expectation(
            gaussian_copula_parameter_from_state(filtered_d.predictive.nodes)
        )
    )
    assert reference_gap > 0.25

    production_c = _production_distribution(concordant, "next", K=700)
    production_d = _production_distribution(discordant, "next", K=700)
    production_gap = (
        production_c.expectation(
            gaussian_copula_parameter_from_state(production_c.nodes)
        )
        - production_d.expectation(
            gaussian_copula_parameter_from_state(production_d.nodes)
        )
    )
    assert production_gap > 0.25
    assert production_gap == pytest.approx(reference_gap, abs=0.015)
    assert np.array_equal(concordant[-1], discordant[-1])
    assert oracle_c.n_observations == oracle_d.n_observations


def test_current_and_next_reference_targets_are_not_interchanged():
    history = _history()
    oracle, result = _reference(history, n_nodes=601, range_sigma=8.0)
    q = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    reference_quantile_gap = np.max(np.abs(
        result.current.quantile(q) - result.predictive.quantile(q)
    ))
    assert reference_quantile_gap > 0.02 * oracle.stationary_sigma
    current_production = _production_distribution(history, "current", K=700)
    next_production = _production_distribution(history, "next", K=700)
    production_quantile_gap = np.max(np.abs(
        current_production.quantile(q) - next_production.quantile(q)
    ))
    assert production_quantile_gap > 0.02 * oracle.stationary_sigma
    assert current_production.mean == pytest.approx(
        result.current.mean, abs=0.025 * oracle.stationary_sigma
    )
    assert next_production.mean == pytest.approx(
        result.predictive.mean, abs=0.025 * oracle.stationary_sigma
    )


def test_nearly_degenerate_ou_mixture_approaches_fixed_parameter_copula():
    kappa, mu, nu = 2.0, 1.7, 0.002
    history = np.full((N_OBS, 2), 0.5)
    oracle = GaussianScarOuReference(
        kappa, mu, nu, N_OBS, n_nodes=301, range_sigma=8.0
    )
    distribution = oracle.filter(history).predictive
    values = np.linspace(0.01, 0.99, 301)
    mixture = oracle.mixture_cdf(values, 0.83, distribution)
    fixed = gaussian_conditional_cdf(
        values,
        0.83,
        gaussian_copula_parameter_from_state(mu),
    )
    assert np.max(np.abs(mixture - fixed)) < 2e-6


def test_reference_mixture_cdf_is_monotone_and_has_exact_endpoints():
    history = _history()
    oracle, result = _reference(history)
    grid, cdf = oracle.mixture_cdf_grid(
        0.87, result.predictive, grid_size=2049
    )
    assert grid[0] == 0.0 and grid[-1] == 1.0
    assert cdf[0] == 0.0 and cdf[-1] == 1.0
    assert np.all(np.diff(cdf) >= 0.0)
