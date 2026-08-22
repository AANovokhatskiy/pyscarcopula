"""Conditional tests for static multivariate copulas."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest
from scipy.stats import norm
from scipy.stats import t as t_dist

from pyscarcopula import GaussianCopula, StudentCopula

from ._analytical_oracles import (
    gaussian_conditional_parameters,
    student_conditional_parameters,
)
from ._statistical_assertions import (
    assert_covariance_with_whitening,
    assert_mean_with_mc_error,
)


DIMENSION = 6
RANK = 2
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
GIVEN_LAYOUTS = {
    "two-scattered": {0: 0.17, 4: 0.83},
    "one-free": {0: 0.12, 1: 0.31, 2: 0.52, 3: 0.74, 5: 0.91},
}


def _training_data(family: str) -> np.ndarray:
    rng = np.random.default_rng(20261001)
    latent = rng.multivariate_normal(
        np.zeros(DIMENSION), CORRELATION, size=240
    )
    if family == "student":
        df = 6.5
        latent *= np.sqrt(df / rng.chisquare(df, size=len(latent)))[:, None]
        return t_dist.cdf(latent, df=df)
    return norm.cdf(latent)


@lru_cache(maxsize=None)
def _fitted_model(family: str, corr_mode: str):
    cls = GaussianCopula if family == "gaussian" else StudentCopula
    kwargs = {}
    if corr_mode == "fixed":
        kwargs["R"] = CORRELATION
    elif corr_mode in {"shrinkage", "cholesky"}:
        kwargs["corr_base"] = CORRELATION
    else:
        kwargs.update(
            factor_rank=RANK,
            factor_loadings=LOADINGS,
            factor_tile_size=3,
        )
    model = cls(DIMENSION, corr_mode=corr_mode, **kwargs)
    fit_kwargs = {"maxiter": 120} if (
        family == "student" or corr_mode in {"shrinkage", "cholesky"}
    ) else {}
    result = model.fit(_training_data(family), **fit_kwargs)
    assert result.success, result.message
    return model


def _assert_given_exact(samples, given):
    for index, value in given.items():
        np.testing.assert_array_equal(samples[:, index], value)


@pytest.mark.validation
@pytest.mark.parametrize("layout", GIVEN_LAYOUTS, ids=GIVEN_LAYOUTS.keys())
@pytest.mark.parametrize(
    "corr_mode", ["fixed", "shrinkage", "cholesky", "factor"]
)
def test_gaussian_all_correlation_modes_match_full_conditional_oracle(
        corr_mode, layout):
    model = _fitted_model("gaussian", corr_mode)
    given = GIVEN_LAYOUTS[layout]
    samples = model.sample_conditional(
        16_000,
        given,
        rng=np.random.default_rng(20261002),
        n_threads=3,
    )
    correlation = model.to_correlation_matrix()
    oracle = gaussian_conditional_parameters(correlation, given)
    latent = norm.ppf(samples[:, oracle.free_indices])
    assert_mean_with_mc_error(latent, oracle.mean, oracle.covariance)
    assert_covariance_with_whitening(latent, oracle.mean, oracle.covariance)
    _assert_given_exact(samples, given)


@pytest.mark.validation
@pytest.mark.parametrize("layout", GIVEN_LAYOUTS, ids=GIVEN_LAYOUTS.keys())
@pytest.mark.parametrize(
    "corr_mode", ["fixed", "shrinkage", "cholesky", "factor"]
)
def test_student_all_correlation_modes_match_full_conditional_oracle(
        corr_mode, layout):
    model = _fitted_model("student", corr_mode)
    given = GIVEN_LAYOUTS[layout]
    samples = model.sample_conditional(
        22_000,
        given,
        rng=np.random.default_rng(20261003),
        n_threads=3,
    )
    correlation = model.to_correlation_matrix()
    oracle = student_conditional_parameters(correlation, model.df, given)
    latent = oracle.latent_from_copula(samples)
    assert_mean_with_mc_error(latent, oracle.location, oracle.covariance)
    assert_covariance_with_whitening(
        latent,
        oracle.location,
        oracle.covariance,
        sigma=9.0,
        numerical_floor=0.025,
    )
    _assert_given_exact(samples, given)


@pytest.mark.parametrize("family", ["gaussian", "student"])
@pytest.mark.parametrize("corr_mode", ["fixed", "factor"])
def test_static_conditional_sampling_is_seed_exact_across_threads(
        family, corr_mode):
    model = _fitted_model(family, corr_mode)
    given = GIVEN_LAYOUTS["two-scattered"]
    sequential = model.sample_conditional(
        257, given, rng=np.random.default_rng(20261004), n_threads=1
    )
    parallel = model.sample_conditional(
        257, given, rng=np.random.default_rng(20261004), n_threads=4
    )
    np.testing.assert_array_equal(parallel, sequential)


@pytest.mark.parametrize("corr_mode", ["fixed", "factor"])
def test_gaussian_conditional_batches_preserve_seeded_stream(corr_mode):
    model = _fitted_model("gaussian", corr_mode)
    given = GIVEN_LAYOUTS["two-scattered"]
    first = np.vstack(list(model.sample_batches(
        19,
        batch_rows=6,
        given=given,
        rng=np.random.default_rng(20261005),
        n_threads=2,
    )))
    second = np.vstack(list(model.sample_batches(
        19,
        batch_rows=6,
        given=given,
        rng=np.random.default_rng(20261005),
        n_threads=2,
    )))
    np.testing.assert_array_equal(second, first)
    _assert_given_exact(first, given)


def test_dense_and_factor_models_with_same_matrix_have_same_oracle_target():
    given = GIVEN_LAYOUTS["two-scattered"]
    dense_gaussian = gaussian_conditional_parameters(CORRELATION, given)
    factor_gaussian = gaussian_conditional_parameters(
        _fitted_model("gaussian", "factor").to_correlation_matrix(), given
    )
    np.testing.assert_allclose(
        factor_gaussian.mean, dense_gaussian.mean, rtol=0.0, atol=2e-15
    )
    np.testing.assert_allclose(
        factor_gaussian.covariance,
        dense_gaussian.covariance,
        rtol=0.0,
        atol=2e-15,
    )


def test_static_api_exposes_documented_memory_asymmetry():
    gaussian = _fitted_model("gaussian", "fixed")
    student = _fitted_model("student", "fixed")
    with pytest.raises(MemoryError, match="sampling requires"):
        gaussian.sample_conditional(
            5,
            {0: 0.4},
            memory_budget_bytes=5 * DIMENSION * 8 - 1,
        )
    with pytest.raises(TypeError, match="memory_budget_bytes"):
        student.sample_conditional(
            5,
            {0: 0.4},
            memory_budget_bytes=5 * DIMENSION * 8,
        )


@pytest.mark.validation
def test_static_student_joint_factor_fit_feeds_conditional_sampler():
    model = StudentCopula(
        DIMENSION,
        corr_mode="factor",
        factor_rank=RANK,
        factor_loadings=LOADINGS,
        factor_estimation="joint",
    )
    result = model.fit(_training_data("student"), maxiter=120)
    assert result.success, result.message
    given = GIVEN_LAYOUTS["two-scattered"]
    sample = model.sample_conditional(
        22_000,
        given,
        rng=np.random.default_rng(20261006),
        n_threads=3,
    )
    oracle = student_conditional_parameters(
        model.to_correlation_matrix(), model.df, given
    )
    latent = oracle.latent_from_copula(sample)
    assert_mean_with_mc_error(latent, oracle.location, oracle.covariance)
    assert_covariance_with_whitening(
        latent,
        oracle.location,
        oracle.covariance,
        sigma=9.0,
        numerical_floor=0.025,
    )
