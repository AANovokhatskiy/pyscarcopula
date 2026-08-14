"""Stage-4 low-dimensional correlation and tail-regime matrix."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from pyscarcopula import GaussianCopula, StochasticStudentCopula

from ._analytical_oracles import gaussian_conditional_parameters
from ._multivariate_scar_oracle import student_conditional_cdf
from ._statistical_assertions import assert_uniform_pit


def _correlation_fixtures():
    dimension = 5
    identity = np.eye(dimension)
    positive = np.full((dimension, dimension), 0.55)
    np.fill_diagonal(positive, 1.0)
    negative = np.full((dimension, dimension), -0.18)
    np.fill_diagonal(negative, 1.0)
    ar = 0.67 ** np.abs(
        np.arange(dimension)[:, None] - np.arange(dimension)[None, :]
    )
    block = np.array([
        [1.00, 0.72, 0.65, 0.08, 0.05],
        [0.72, 1.00, 0.61, 0.06, 0.04],
        [0.65, 0.61, 1.00, 0.09, 0.07],
        [0.08, 0.06, 0.09, 1.00, 0.58],
        [0.05, 0.04, 0.07, 0.58, 1.00],
    ])
    rng = np.random.default_rng(20261030)
    raw = rng.normal(size=(dimension, dimension))
    spd = raw @ raw.T + 2.0 * np.eye(dimension)
    scale = np.sqrt(np.diag(spd))
    random_spd = spd / scale[:, None] / scale[None, :]
    near_singular = np.full((dimension, dimension), 0.985)
    np.fill_diagonal(near_singular, 1.0)
    return {
        "identity": identity,
        "equicorr-positive": positive,
        "equicorr-negative": negative,
        "ar1": ar,
        "block": block,
        "random-spd": random_spd,
        "near-singular": near_singular,
    }


CORRELATIONS = _correlation_fixtures()
ONE_FREE_GIVEN = {0: 0.02, 1: 0.27, 3: 0.78, 4: 0.98}


def _configured_gaussian(correlation):
    model = GaussianCopula(d=len(correlation), R=correlation)
    training = np.random.default_rng(20261029).uniform(
        0.05, 0.95, size=(40, len(correlation))
    )
    result = model.fit(training)
    assert result.success
    return model


@pytest.mark.validation
@pytest.mark.parametrize("fixture", CORRELATIONS, ids=CORRELATIONS.keys())
def test_gaussian_correlation_fixture_matrix_passes_one_free_pit(fixture):
    correlation = CORRELATIONS[fixture]
    model = _configured_gaussian(correlation)
    sample = model.sample_conditional(
        16_000,
        ONE_FREE_GIVEN,
        rng=np.random.default_rng(20261031),
        n_threads=4,
    )
    oracle = gaussian_conditional_parameters(correlation, ONE_FREE_GIVEN)
    latent = norm.ppf(sample[:, oracle.free_indices[0]])
    pit = norm.cdf(
        (latent - oracle.mean[0]) / np.sqrt(oracle.covariance[0, 0])
    )
    assert_uniform_pit(pit, numerical_floor=0.006)


@pytest.mark.validation
@pytest.mark.parametrize(
    "df", [2.001, 3.0, 5.0, 10.0, 30.0, 1000.0],
    ids=["df-2.001", "df-3", "df-5", "df-10", "df-30", "gaussian-limit"],
)
def test_student_df_and_tail_regimes_use_original_marginal_cdf(df):
    correlation = CORRELATIONS["ar1"]
    model = StochasticStudentCopula(5, R=correlation)
    sample = model.sample_conditional(
        18_000,
        r=df,
        given=ONE_FREE_GIVEN,
        rng=np.random.default_rng(20261032),
        n_threads=4,
    )
    pit = student_conditional_cdf(
        sample[:, 2], np.array([df]), correlation, ONE_FREE_GIVEN
    )[:, 0]
    assert_uniform_pit(pit, numerical_floor=0.007)


def _factor_loadings(dimension, rank):
    return np.random.default_rng(20261040 + rank).normal(
        scale=0.075, size=(dimension, rank)
    )


@pytest.mark.validation
@pytest.mark.parametrize("rank", [1, 3, 8])
@pytest.mark.parametrize("family", ["gaussian", "student"])
def test_factor_rank_matrix_matches_dense_conditional_target(family, rank):
    dimension = 10
    loadings = _factor_loadings(dimension, rank)
    given = {
        index: value
        for index, value in enumerate(np.linspace(0.08, 0.92, dimension))
        if index != 6
    }
    if family == "gaussian":
        model = GaussianCopula(
            dimension,
            corr_mode="factor",
            factor_rank=rank,
            factor_loadings=loadings,
        )
        sample = model.sample_conditional(
            16_000,
            given,
            rng=np.random.default_rng(20261033),
            n_threads=4,
        )
        oracle = gaussian_conditional_parameters(
            model.to_correlation_matrix(), given
        )
        latent = norm.ppf(sample[:, 6])
        pit = norm.cdf(
            (latent - oracle.mean[0]) / np.sqrt(oracle.covariance[0, 0])
        )
    else:
        model = StochasticStudentCopula(
            dimension,
            corr_mode="factor",
            factor_rank=rank,
            factor_loadings=loadings,
        )
        sample = model.sample_conditional(
            18_000,
            r=5.0,
            given=given,
            rng=np.random.default_rng(20261034),
            n_threads=4,
        )
        pit = student_conditional_cdf(
            sample[:, 6],
            np.array([5.0]),
            model.to_correlation_matrix(),
            given,
        )[:, 0]
    assert_uniform_pit(pit, numerical_floor=0.007)


@pytest.mark.validation
@pytest.mark.parametrize("family", ["gaussian", "student"])
def test_conditional_distribution_is_equivariant_to_variable_permutation(family):
    correlation = CORRELATIONS["block"]
    permutation = np.array([3, 0, 4, 1, 2])
    permuted = correlation[np.ix_(permutation, permutation)]
    given = {0: 0.13, 1: 0.31, 3: 0.77, 4: 0.94}
    original_given = {
        int(permutation[index]): value for index, value in given.items()
    }
    if family == "gaussian":
        model = _configured_gaussian(permuted)
        sample = model.sample_conditional(
            16_000, given, rng=np.random.default_rng(20261035)
        )
        oracle = gaussian_conditional_parameters(permuted, given)
        latent = norm.ppf(sample[:, 2])
        pit = norm.cdf(
            (latent - oracle.mean[0]) / np.sqrt(oracle.covariance[0, 0])
        )
        original = gaussian_conditional_parameters(correlation, original_given)
        assert oracle.mean[0] == pytest.approx(original.mean[0], abs=2e-14)
        assert oracle.covariance[0, 0] == pytest.approx(
            original.covariance[0, 0], abs=2e-14
        )
    else:
        model = StochasticStudentCopula(5, R=permuted)
        sample = model.sample_conditional(
            18_000, r=7.0, given=given, rng=np.random.default_rng(20261036)
        )
        pit = student_conditional_cdf(
            sample[:, 2], np.array([7.0]), permuted, given
        )[:, 0]
    assert_uniform_pit(pit, numerical_floor=0.007)
