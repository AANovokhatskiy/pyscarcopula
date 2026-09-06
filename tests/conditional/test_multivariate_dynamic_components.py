"""Component and execution-contract tests for dynamic models."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from pyscarcopula import EquicorrGaussianCopula, StochasticStudentCopula
from pyscarcopula._types import GASResult, LatentResult, gas_params, ou_params
from pyscarcopula.api import predict as api_predict
from pyscarcopula.strategy._base import get_strategy_for_result

from ._analytical_oracles import (
    gaussian_conditional_parameters,
    student_conditional_parameters,
)
from ._multivariate_scar_oracle import (
    ScalarScarOuReference,
    equicorr_conditional_cdf,
    equicorr_gaussian_log_density,
    equicorr_parameter_from_state,
    equicorrelation_matrix,
    student_conditional_cdf,
    student_copula_log_density,
    student_df_parameter_from_state,
)
from ._statistical_assertions import (
    assert_covariance_with_whitening,
    assert_mean_with_mc_error,
    assert_uniform_pit,
)


DIMENSION = 6
LOADINGS = np.array([
    [0.31, 0.04],
    [0.23, -0.16],
    [-0.08, 0.29],
    [0.18, 0.21],
    [-0.24, 0.11],
    [0.07, -0.27],
])
CORRELATION = LOADINGS @ LOADINGS.T
np.fill_diagonal(CORRELATION, 1.0)
GIVEN = {0: 0.16, 3: 0.79}
POINT_GIVEN = {0: 0.12, 1: 0.29, 3: 0.67, 4: 0.82, 5: 0.93}


def _assert_given(samples):
    for index, value in GIVEN.items():
        np.testing.assert_array_equal(samples[:, index], value)


@pytest.mark.validation
def test_equicorr_rowwise_rho_path_matches_each_gaussian_component():
    model = EquicorrGaussianCopula(DIMENSION)
    parameters = np.repeat(np.array([-0.14, 0.08, 0.58]), 8_000)
    samples = model.sample_conditional(
        len(parameters),
        r=parameters,
        given=GIVEN,
        rng=np.random.default_rng(20261010),
        n_threads=4,
    )
    _assert_given(samples)
    for parameter in np.unique(parameters):
        selected = parameters == parameter
        oracle = gaussian_conditional_parameters(
            equicorrelation_matrix(DIMENSION, parameter), GIVEN
        )
        latent = norm.ppf(samples[selected][:, oracle.free_indices])
        assert_mean_with_mc_error(latent, oracle.mean, oracle.covariance)
        assert_covariance_with_whitening(
            latent, oracle.mean, oracle.covariance, numerical_floor=0.025
        )


@pytest.mark.validation
@pytest.mark.parametrize("corr_mode", ["fixed", "factor"])
def test_stochastic_student_rowwise_df_path_matches_each_t_component(corr_mode):
    if corr_mode == "fixed":
        model = StochasticStudentCopula(DIMENSION, R=CORRELATION)
    else:
        model = StochasticStudentCopula(
            DIMENSION,
            corr_mode="factor",
            factor_rank=2,
            factor_loadings=LOADINGS,
        )
    parameters = np.repeat(np.array([3.5, 7.0, 30.0]), 10_000)
    samples = model.sample_conditional(
        len(parameters),
        r=parameters,
        given=GIVEN,
        rng=np.random.default_rng(20261011),
        n_threads=4,
    )
    _assert_given(samples)
    for parameter in np.unique(parameters):
        selected = parameters == parameter
        oracle = student_conditional_parameters(
            (
                model.to_correlation_matrix()
                if corr_mode == "factor"
                else CORRELATION
            ),
            parameter,
            GIVEN,
        )
        latent = oracle.latent_from_copula(samples[selected])
        assert_mean_with_mc_error(latent, oracle.location, oracle.covariance)
        assert_covariance_with_whitening(
            latent,
            oracle.location,
            oracle.covariance,
            sigma=9.0,
            numerical_floor=0.035,
        )


@pytest.mark.parametrize("corr_mode", ["fixed", "factor"])
def test_stochastic_student_rowwise_path_is_seed_exact_across_threads(corr_mode):
    model = (
        StochasticStudentCopula(DIMENSION, R=CORRELATION)
        if corr_mode == "fixed"
        else StochasticStudentCopula(
            DIMENSION,
            corr_mode="factor",
            factor_rank=2,
            factor_loadings=LOADINGS,
        )
    )
    parameters = np.resize(np.array([3.5, 7.0, 30.0]), 301)
    one = model.sample_conditional(
        len(parameters),
        r=parameters,
        given=GIVEN,
        rng=np.random.default_rng(20261012),
        n_threads=1,
    )
    four = model.sample_conditional(
        len(parameters),
        r=parameters,
        given=GIVEN,
        rng=np.random.default_rng(20261012),
        n_threads=4,
    )
    np.testing.assert_array_equal(four, one)


def test_equicorr_conditional_memory_budget_has_exact_output_boundary():
    model = EquicorrGaussianCopula(DIMENSION)
    required = 7 * DIMENSION * 8
    sample = model.sample_conditional(
        7,
        r=0.3,
        given=GIVEN,
        rng=np.random.default_rng(20261013),
        memory_budget_bytes=required,
    )
    assert sample.shape == (7, DIMENSION)
    with pytest.raises(MemoryError, match="requires"):
        model.sample_conditional(
            7, r=0.3, given=GIVEN, memory_budget_bytes=required - 1
        )


def test_dense_stochastic_student_memory_budget_has_output_boundary():
    model = StochasticStudentCopula(DIMENSION, R=CORRELATION)
    required = 7 * DIMENSION * 8
    sample = model.sample_conditional(
        7,
        r=6.0,
        given=GIVEN,
        rng=np.random.default_rng(20261014),
        memory_budget_bytes=required,
    )
    assert sample.shape == (7, DIMENSION)
    with pytest.raises(MemoryError, match="sampling requires"):
        model.sample_conditional(
            7, r=6.0, given=GIVEN, memory_budget_bytes=required - 1
        )


@pytest.mark.parametrize("family", ["equicorr", "student"])
def test_parameter_path_validation_happens_before_sampling(family):
    model = (
        EquicorrGaussianCopula(DIMENSION)
        if family == "equicorr"
        else StochasticStudentCopula(DIMENSION, R=CORRELATION)
    )
    invalid = np.ones(8)
    if family == "student":
        invalid[3] = 2.0
        message = "greater than 2"
    else:
        invalid[3] = -1.0 / (DIMENSION - 1.0)
        message = "finite and in"
    with pytest.raises(ValueError, match=message):
        model.sample_conditional(8, r=invalid, given=GIVEN)


@pytest.mark.validation
@pytest.mark.parametrize(
    "parameter",
    [-1.0 / (DIMENSION - 1.0) + 1e-4, 0.0, 0.94],
    ids=["near-negative-boundary", "independence", "strong-positive"],
)
def test_equicorr_extreme_parameter_regimes_match_one_free_oracle(parameter):
    model = EquicorrGaussianCopula(DIMENSION)
    sample = model.sample_conditional(
        16_000,
        r=parameter,
        given=POINT_GIVEN,
        rng=np.random.default_rng(20261015),
        n_threads=4,
    )
    pit = equicorr_conditional_cdf(
        sample[:, 2], np.array([parameter]), DIMENSION, POINT_GIVEN
    )[:, 0]
    assert_uniform_pit(pit, numerical_floor=0.006)


def _gas_case(family):
    history = np.random.default_rng(20261016).uniform(
        0.08, 0.92, size=(24, DIMENSION)
    )
    if family == "equicorr":
        model = EquicorrGaussianCopula(DIMENSION)
        last_parameter = 0.2
    else:
        model = StochasticStudentCopula(DIMENSION, R=CORRELATION)
        last_parameter = 6.0
    result = GASResult(
        log_likelihood=0.0,
        method="GAS",
        copula_name=model.name,
        success=True,
        params=gas_params(0.1, 0.15, 0.75),
        scaling="unit",
        r_last=last_parameter,
    )
    return model, history, result


@pytest.mark.validation
@pytest.mark.parametrize("horizon", ["current", "next"])
@pytest.mark.parametrize("family", ["equicorr", "student"])
def test_gas_point_state_prediction_matches_its_conditional_component(
        family, horizon):
    model, history, result = _gas_case(family)
    state = get_strategy_for_result(result).predictive_state(
        model, history, result, horizon=horizon
    )
    parameter = float(state.r[0])
    sample = api_predict(
        model,
        history,
        result,
        14_000,
        given=POINT_GIVEN,
        horizon=horizon,
        rng=np.random.default_rng(20261017),
    )
    if family == "equicorr":
        pit = equicorr_conditional_cdf(
            sample[:, 2], np.array([parameter]), DIMENSION, POINT_GIVEN
        )[:, 0]
    else:
        pit = student_conditional_cdf(
            sample[:, 2], np.array([parameter]), CORRELATION, POINT_GIVEN
        )[:, 0]
    assert_uniform_pit(pit, numerical_floor=0.006)
    for index, value in POINT_GIVEN.items():
        np.testing.assert_array_equal(sample[:, index], value)


@pytest.mark.parametrize("family", ["equicorr", "student"])
def test_gas_current_and_next_point_states_are_not_interchanged(family):
    model, history, result = _gas_case(family)
    strategy = get_strategy_for_result(result)
    current = strategy.predictive_state(
        model, history, result, horizon="current"
    )
    following = strategy.predictive_state(
        model, history, result, horizon="next"
    )
    assert float(current.r[0]) != pytest.approx(float(following.r[0]), abs=1e-8)


def test_factor_student_component_does_not_need_dense_materialization(monkeypatch):
    model = StochasticStudentCopula(
        DIMENSION,
        corr_mode="factor",
        factor_rank=2,
        factor_loadings=LOADINGS,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("factor conditional path materialized dense R")

    monkeypatch.setattr(model, "to_correlation_matrix", forbidden)
    sample = model.sample_conditional(
        23,
        r=np.linspace(3.0, 30.0, 23),
        given=GIVEN,
        rng=np.random.default_rng(20261024),
        n_threads=3,
    )
    assert sample.shape == (23, DIMENSION)
    assert model._R is None


@pytest.mark.validation
@pytest.mark.parametrize(
    "mode",
    ["fixed", "shrinkage", "cholesky", "factor", "factor-joint"],
)
def test_stochastic_student_mle_correlation_modes_feed_conditional_sampler(mode):
    rng = np.random.default_rng(20261025)
    df = 6.5
    latent = rng.multivariate_normal(
        np.zeros(DIMENSION), CORRELATION, size=240
    )
    latent *= np.sqrt(df / rng.chisquare(df, len(latent)))[:, None]
    from scipy.stats import t as t_dist

    training = t_dist.cdf(latent, df=df)
    kwargs = {}
    corr_mode = mode
    if mode == "fixed":
        kwargs["R"] = CORRELATION
    elif mode in {"shrinkage", "cholesky"}:
        kwargs["corr_base"] = CORRELATION
    else:
        corr_mode = "factor"
        kwargs.update(
            factor_rank=2,
            factor_loadings=LOADINGS,
            factor_estimation=("joint" if mode == "factor-joint" else "two-stage"),
        )
    model = StochasticStudentCopula(
        DIMENSION, corr_mode=corr_mode, **kwargs
    )
    # Joint loading estimation has substantially more parameters than the
    # other modes, and L-BFGS-B needs a larger cross-platform iteration budget
    # to reach the shared final-gradient validation gate reliably.
    maxiter = 240 if mode == "factor-joint" else 120
    result = model.fit(training, method="mle", maxiter=maxiter)
    assert result.success, result.message
    correlation = (
        model.to_correlation_matrix()
        if corr_mode == "factor"
        else model._R
    )
    sample = api_predict(
        model,
        training,
        result,
        18_000,
        given=POINT_GIVEN,
        rng=np.random.default_rng(20261026),
    )
    pit = student_conditional_cdf(
        sample[:, 2],
        np.array([result.copula_param]),
        correlation,
        POINT_GIVEN,
    )[:, 0]
    assert_uniform_pit(pit, numerical_floor=0.007)
