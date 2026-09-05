"""Rank-deficient joint factors and scalable, tie-aware Kendall initialization."""

import numpy as np
import pytest
from scipy.stats import kendalltau, norm

from pyscarcopula._native import multivariate
from pyscarcopula.copula.multivariate.factor_estimation import (
    FactorLoadingParameterization,
)


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("case", ["zero", "deficient", "near_deficient", "full"])
def test_factor_anchor_qr_completes_deficient_basis(rank, case):
    dimension = 2 * rank + 1
    rng = np.random.default_rng(91)
    loadings = rng.normal(scale=0.12, size=(dimension, rank))
    if case == "zero":
        loadings[:] = 0.0
    elif case in ("deficient", "near_deficient"):
        loadings[:, -1] = 0.0 if case == "deficient" else 1e-14
        # Rotate the deficient factor away from a coordinate axis.
        rotation, _ = np.linalg.qr(rng.normal(size=(rank, rank)))
        loadings = loadings @ rotation

    parameterization, parameters = FactorLoadingParameterization.from_loadings(
        loadings, uniqueness_min=1e-8)
    restored = parameterization.loadings(parameters)
    assert np.all(np.isfinite(parameters))
    assert np.all(np.isfinite(restored))
    anchors = restored[parameterization.anchors]
    np.testing.assert_array_equal(np.triu(anchors, 1), 0.0)
    assert np.all(np.diag(anchors) > 0.0)
    assert np.all(np.sum(restored**2, axis=1) < 1 - 1e-8)
    # The only material change is the established 1e-6 positive anchor floor.
    np.testing.assert_allclose(
        restored @ restored.T, loadings @ loadings.T, rtol=0, atol=2e-12)

    repeated, repeated_parameters = FactorLoadingParameterization.from_loadings(
        loadings, uniqueness_min=1e-8)
    np.testing.assert_array_equal(repeated.anchors, parameterization.anchors)
    np.testing.assert_array_equal(repeated_parameters, parameters)


def test_crypto6_zero_second_factor_can_initialize_joint_coordinates():
    # Actual failing two-stage initialization from the 0.22.0 comparison.
    loadings = np.column_stack(([
        0.8165683370158384, 0.8405685144754449, 0.7849635986010791,
        0.7926737169375474, 0.8100086591, 0.7375423890,
    ], np.zeros(6)))
    parameterization, parameters = FactorLoadingParameterization.from_loadings(
        loadings, uniqueness_min=1e-8)
    restored = parameterization.loadings(parameters)
    np.testing.assert_allclose(
        restored @ restored.T, loadings @ loadings.T, rtol=0, atol=2e-12)
    assert np.all(np.isfinite(parameterization.pullback(
        parameters, np.ones_like(loadings))))


@pytest.mark.parametrize("rows", [1, 2, 17, 257, 2515])
@pytest.mark.parametrize("ties", [False, True])
def test_multivariate_kendall_matches_scipy_with_joint_and_marginal_ties(rows, ties):
    rng = np.random.default_rng(20260905)
    values = rng.normal(size=(rows, 5))
    if ties:
        values = np.round(values)
    values[:, 3] = values[:, 0]
    values[:, 4] = -values[:, 1]
    values = norm.cdf(values)
    _, diagnostics = multivariate.estimate_kendall_correlation(values)
    expected = np.eye(5)
    nonfinite = []
    for first in range(5):
        for second in range(first + 1, 5):
            # SciPy warns for fewer than two observations; the native contract
            # explicitly records these undefined pairs and substitutes zero.
            tau = kendalltau(values[:, first], values[:, second]).statistic if rows > 1 else np.nan
            if not np.isfinite(tau):
                nonfinite.append((first, second))
                tau = 0.0
            expected[first, second] = expected[second, first] = np.sin(np.pi * tau / 2)
    np.testing.assert_allclose(diagnostics["input_correlation"], expected, rtol=0, atol=4e-16)
    assert diagnostics["nonfinite_kendall_pairs"] == tuple(nonfinite)


def test_multivariate_kendall_handles_constant_and_signed_zero_columns():
    values = np.column_stack((
        [-0.0, 0.0, -0.0, 0.0, 0.0, -0.0],
        [0.0, 0.0, 1.0, 2.0, 2.0, 3.0],
        [3.0, 2.0, 2.0, 1.0, 0.0, 0.0],
    )) / 3.0
    _, diagnostics = multivariate.estimate_kendall_correlation(values)
    assert diagnostics["nonfinite_kendall_pairs"] == ((0, 1), (0, 2))
    assert diagnostics["input_correlation"][1, 2] == pytest.approx(
        np.sin(np.pi * kendalltau(values[:, 1], values[:, 2]).statistic / 2), abs=4e-16)
