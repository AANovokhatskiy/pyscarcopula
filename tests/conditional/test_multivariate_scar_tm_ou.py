"""Full-history multivariate SCAR-TM-OU conditional-mixture tests."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

from pyscarcopula import EquicorrGaussianCopula, StochasticStudentCopula
from pyscarcopula._types import LatentResult, ou_params
from pyscarcopula.api import predict as api_predict
from pyscarcopula.strategy._base import get_strategy_for_result

from ._multivariate_scar_oracle import (
    ScalarScarOuReference,
    equicorr_conditional_cdf,
    equicorr_gaussian_log_density,
    equicorr_parameter_from_state,
    simulate_equicorr_scar_history,
    simulate_student_scar_history,
    student_conditional_cdf,
    student_copula_log_density,
    student_df_parameter_from_state,
)
from ._scar_tm_ou_oracle import QuadratureDistribution
from ._statistical_assertions import assert_uniform_pit


DIMENSION = 4
KAPPA = 2.0
MU = 0.4
NU = 2.3
N_OBSERVATIONS = 21
CORRELATION = np.array([
    [1.00, 0.45, 0.20, 0.10],
    [0.45, 1.00, 0.30, 0.15],
    [0.20, 0.30, 1.00, 0.40],
    [0.10, 0.15, 0.40, 1.00],
])
GIVEN = {0: 0.14, 1: 0.37, 3: 0.86}
FREE_INDEX = 2


def _model(family):
    if family == "equicorr":
        return EquicorrGaussianCopula(DIMENSION)
    return StochasticStudentCopula(DIMENSION, R=CORRELATION)


@lru_cache(maxsize=None)
def _history(family):
    if family == "equicorr":
        return simulate_equicorr_scar_history(
            N_OBSERVATIONS,
            DIMENSION,
            KAPPA,
            MU,
            NU,
            seed=20261020,
        )[0]
    return simulate_student_scar_history(
        N_OBSERVATIONS,
        CORRELATION,
        KAPPA,
        MU,
        NU,
        seed=20261021,
    )[0]


def _parameter_link(family):
    if family == "equicorr":
        return lambda state: equicorr_parameter_from_state(state, DIMENSION)
    return student_df_parameter_from_state


def _emission(family):
    if family == "equicorr":
        return equicorr_gaussian_log_density
    return lambda row, df: student_copula_log_density(row, df, CORRELATION)


def _conditional_cdf(family):
    if family == "equicorr":
        return lambda values, parameter: equicorr_conditional_cdf(
            values, parameter, DIMENSION, GIVEN
        )
    return lambda values, parameter: student_conditional_cdf(
        values, parameter, CORRELATION, GIVEN
    )


def _reference(family, *, n_nodes=501, range_sigma=8.0):
    oracle = ScalarScarOuReference(
        KAPPA,
        MU,
        NU,
        N_OBSERVATIONS,
        _parameter_link(family),
        _emission(family),
        n_nodes=n_nodes,
        range_sigma=range_sigma,
    )
    return oracle, oracle.filter(_history(family))


def _result(model, *, K=700, grid_range=8.0):
    return LatentResult(
        log_likelihood=0.0,
        method="SCAR-TM-OU",
        copula_name=model.name,
        success=True,
        params=ou_params(KAPPA, MU, NU),
        K=K,
        grid_range=grid_range,
        grid_method="dense",
        adaptive=False,
        transition_method="matrix",
        max_K=None,
    )


def _production_distribution(family, history, horizon):
    model = _model(family)
    result = _result(model)
    state = get_strategy_for_result(result).predictive_state(
        model, history, result, horizon=horizon
    )
    return QuadratureDistribution(state.z_grid, state.prob)


def _parameter_expectation(family, distribution):
    return distribution.expectation(
        _parameter_link(family)(distribution.nodes)
    )


@pytest.mark.parametrize("family", ["equicorr", "student"])
def test_multivariate_reference_filter_converges_in_nodes_and_range(family):
    coarse, coarse_result = _reference(
        family, n_nodes=301, range_sigma=7.0
    )
    fine, fine_result = _reference(
        family, n_nodes=501, range_sigma=8.0
    )
    sigma = fine.stationary_sigma
    assert abs(coarse_result.current.mean - fine_result.current.mean) < 1e-4 * sigma
    assert abs(
        coarse_result.predictive.variance - fine_result.predictive.variance
    ) < 2e-4 * sigma ** 2
    values = np.linspace(0.01, 0.99, 101)
    coarse_cdf = coarse.mixture_cdf(
        values, coarse_result.predictive, _conditional_cdf(family)
    )
    fine_cdf = fine.mixture_cdf(
        values, fine_result.predictive, _conditional_cdf(family)
    )
    assert np.max(np.abs(coarse_cdf - fine_cdf)) < 3e-4
    assert fine.stationary_tail_mass_bound < 2e-15


@pytest.mark.parametrize("horizon", ["current", "next"])
@pytest.mark.parametrize("family", ["equicorr", "student"])
def test_multivariate_production_state_matches_continuous_reference(
        family, horizon):
    oracle, filtered = _reference(family)
    expected = filtered.current if horizon == "current" else filtered.predictive
    observed = _production_distribution(family, _history(family), horizon)
    sigma = oracle.stationary_sigma
    assert abs(observed.mean - expected.mean) < 0.025 * sigma
    assert abs(observed.variance - expected.variance) < 0.05 * sigma ** 2
    quantiles = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    assert np.max(np.abs(
        observed.quantile(quantiles) - expected.quantile(quantiles)
    )) < 0.04 * sigma
    parameter_tolerance = 0.007 if family == "equicorr" else 0.025
    assert abs(
        _parameter_expectation(family, observed)
        - _parameter_expectation(family, expected)
    ) < parameter_tolerance


@pytest.mark.validation
@pytest.mark.parametrize("horizon", ["current", "next"])
@pytest.mark.parametrize("family", ["equicorr", "student"])
def test_multivariate_prediction_passes_full_history_mixture_pit(
        family, horizon):
    model = _model(family)
    history = _history(family)
    oracle, filtered = _reference(family)
    distribution = (
        filtered.current if horizon == "current" else filtered.predictive
    )
    samples = api_predict(
        model,
        history,
        _result(model),
        14_000,
        given=GIVEN,
        horizon=horizon,
        predictive_r_mode="grid",
        rng=np.random.default_rng(20261022),
    )
    for index, value in GIVEN.items():
        np.testing.assert_array_equal(samples[:, index], value)
    pit = oracle.mixture_cdf(
        samples[:, FREE_INDEX], distribution, _conditional_cdf(family)
    )
    assert_uniform_pit(pit, numerical_floor=0.007)


def _same_last_histories(family):
    last = np.array([[0.54, 0.46, 0.51, 0.49]])
    if family == "equicorr":
        pattern = np.linspace(0.12, 0.88, N_OBSERVATIONS - 1)
        first = np.tile(pattern[:, None], (1, DIMENSION))
        second = np.column_stack(
            (pattern, 1.0 - pattern, pattern, 1.0 - pattern)
        )
    else:
        extreme = np.where(
            np.arange(N_OBSERVATIONS - 1) % 2 == 0, 0.03, 0.97
        )
        first = np.tile(extreme[:, None], (1, DIMENSION))
        second = np.full((N_OBSERVATIONS - 1, DIMENSION), 0.5)
    return np.vstack((first, last)), np.vstack((second, last))


@pytest.mark.parametrize("family", ["equicorr", "student"])
def test_same_last_row_but_different_prefix_changes_multivariate_state(family):
    first, second = _same_last_histories(family)
    oracle_first = ScalarScarOuReference(
        KAPPA,
        MU,
        NU,
        N_OBSERVATIONS,
        _parameter_link(family),
        _emission(family),
        n_nodes=501,
        range_sigma=8.0,
    )
    oracle_second = ScalarScarOuReference(
        KAPPA,
        MU,
        NU,
        N_OBSERVATIONS,
        _parameter_link(family),
        _emission(family),
        n_nodes=501,
        range_sigma=8.0,
    )
    expected_first = oracle_first.filter(first).predictive
    expected_second = oracle_second.filter(second).predictive
    observed_first = _production_distribution(family, first, "next")
    observed_second = _production_distribution(family, second, "next")
    expected_gap = (
        _parameter_expectation(family, expected_first)
        - _parameter_expectation(family, expected_second)
    )
    observed_gap = (
        _parameter_expectation(family, observed_first)
        - _parameter_expectation(family, observed_second)
    )
    minimum_gap = 0.5 if family == "equicorr" else 0.10
    assert abs(expected_gap) > minimum_gap
    assert observed_gap == pytest.approx(expected_gap, abs=0.01)
    np.testing.assert_array_equal(first[-1], second[-1])


@pytest.mark.parametrize("family", ["equicorr", "student"])
def test_current_and_propagated_next_are_distinct_multivariate_targets(family):
    oracle, filtered = _reference(family)
    quantiles = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    reference_gap = np.max(np.abs(
        filtered.current.quantile(quantiles)
        - filtered.predictive.quantile(quantiles)
    ))
    current = _production_distribution(family, _history(family), "current")
    following = _production_distribution(family, _history(family), "next")
    production_gap = np.max(np.abs(
        current.quantile(quantiles) - following.quantile(quantiles)
    ))
    assert reference_gap > 0.005 * oracle.stationary_sigma
    assert production_gap > 0.005 * oracle.stationary_sigma


@pytest.mark.parametrize("family", ["equicorr", "student"])
def test_predict_batches_are_reproducible_from_one_frozen_scar_state(family):
    model = _model(family)
    model.fit_result = _result(model)
    model._last_u = _history(family)
    kwargs = {"n_threads": 2} if family == "student" else {}

    def draw():
        return np.vstack(list(model.predict_batches(
            17,
            batch_rows=6,
            given=GIVEN,
            horizon="next",
            predictive_r_mode="grid",
            rng=np.random.default_rng(20261023),
            **kwargs,
        )))

    first = draw()
    second = draw()
    np.testing.assert_array_equal(second, first)
    for index, value in GIVEN.items():
        np.testing.assert_array_equal(first[:, index], value)
