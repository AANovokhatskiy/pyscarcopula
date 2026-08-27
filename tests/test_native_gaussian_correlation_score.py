from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from pyscarcopula._native import (
    _descriptors as _cpp_copula,
    static as static_likelihood,
)
from pyscarcopula._native._extension import load


def _correlation(d: int, kind: str, seed: int) -> np.ndarray:
    if kind == "independence":
        correlation = np.eye(d)
        correlation[1, 0] = correlation[0, 1] = 1e-4
        return correlation
    if kind == "strong":
        return np.full((d, d), 0.82) + np.eye(d) * 0.18
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(d, d))
    covariance = matrix @ matrix.T + 0.3 * np.eye(d)
    scale = np.sqrt(np.diag(covariance))
    return covariance / scale[:, None] / scale[None, :]


def _pairs(d: int):
    return [(i, j) for i in range(1, d) for j in range(i)]


def _finite_difference(evaluator, correlation, step=1e-6):
    result = []
    for i, j in _pairs(correlation.shape[0]):
        plus = correlation.copy()
        minus = correlation.copy()
        plus[i, j] += step
        plus[j, i] += step
        minus[i, j] -= step
        minus[j, i] -= step
        result.append((
            evaluator.gaussian_objective_and_gradient(plus)[0]
            - evaluator.gaussian_objective_and_gradient(minus)[0]
        ) / (2.0 * step))
    return np.asarray(result)


@pytest.mark.parametrize("d", [2, 3, 5])
@pytest.mark.parametrize("kind", ["independence", "random", "strong"])
def test_native_gaussian_score_matches_central_difference(d, kind):
    rng = np.random.default_rng(1000 + d)
    u = rng.uniform(0.002, 0.998, size=(180, d))
    correlation = _correlation(d, kind, seed=11 + d)
    evaluator = static_likelihood.prepare_gaussian(correlation, u)

    _, analytical = evaluator.gaussian_objective_and_gradient(correlation)
    numerical = _finite_difference(evaluator, correlation)

    np.testing.assert_allclose(
        analytical, numerical, rtol=3e-5, atol=3e-5)


def test_prepared_gaussian_rows_and_objective_match_scipy():
    rng = np.random.default_rng(91)
    u = rng.uniform(0.001, 0.999, size=(240, 4))
    correlation = _correlation(4, "random", seed=8)
    evaluator = static_likelihood.prepare_gaussian(correlation, u)
    rows = evaluator.log_pdf_rows(0.0)
    scores = norm.ppf(u)
    expected = (
        multivariate_normal.logpdf(scores, cov=correlation)
        - np.sum(norm.logpdf(scores), axis=1))

    np.testing.assert_allclose(rows, expected, rtol=5e-8, atol=5e-8)
    value, _ = evaluator.gaussian_objective_and_gradient(correlation)
    assert value == pytest.approx(
        -float(np.sum(expected)), rel=5e-9, abs=5e-7)


def test_prepared_gaussian_evaluator_owns_scores_across_trials():
    rng = np.random.default_rng(7)
    observations = rng.uniform(0.01, 0.99, size=(160, 3))
    original = observations.copy()
    initial = _correlation(3, "random", seed=1)
    trial = _correlation(3, "random", seed=2)
    prepared = static_likelihood.prepare_gaussian(initial, observations)
    observations[:] = np.nan

    actual = prepared.gaussian_objective_and_gradient(trial)
    reference = static_likelihood.prepare_gaussian(
        trial, original).gaussian_objective_and_gradient(trial)
    np.testing.assert_allclose(actual[0], reference[0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual[1], reference[1], rtol=0.0, atol=0.0)


def test_gaussian_score_is_permutation_equivariant():
    rng = np.random.default_rng(123)
    u = rng.uniform(0.005, 0.995, size=(210, 4))
    correlation = _correlation(4, "random", seed=14)
    permutation = np.array([2, 0, 3, 1])
    value, gradient = static_likelihood.prepare_gaussian(
        correlation, u).gaussian_objective_and_gradient(correlation)
    permuted_correlation = correlation[np.ix_(permutation, permutation)]
    permuted_value, permuted_gradient = static_likelihood.prepare_gaussian(
        permuted_correlation,
        u[:, permutation],
    ).gaussian_objective_and_gradient(permuted_correlation)

    original_by_pair = {
        tuple(sorted(pair)): score
        for pair, score in zip(_pairs(4), gradient)
    }
    expected = np.array([
        original_by_pair[tuple(sorted((permutation[i], permutation[j])))]
        for i, j in _pairs(4)
    ])
    assert permuted_value == pytest.approx(value, rel=2e-14, abs=2e-14)
    np.testing.assert_allclose(
        permuted_gradient, expected, rtol=2e-13, atol=2e-13)


def test_gaussian_score_parallel_reduction_is_deterministic():
    rng = np.random.default_rng(29)
    u = rng.uniform(0.001, 0.999, size=(900, 5))
    correlation = _correlation(5, "random", seed=33)
    sequential = static_likelihood.prepare_gaussian(
        correlation, u, n_threads=1)
    parallel = static_likelihood.prepare_gaussian(
        correlation, u, n_threads=4)

    seq_value, seq_gradient = sequential.gaussian_objective_and_gradient(
        correlation)
    par_value, par_gradient = parallel.gaussian_objective_and_gradient(
        correlation)
    np.testing.assert_allclose(par_value, seq_value, rtol=2e-15, atol=1e-11)
    np.testing.assert_allclose(
        par_gradient, seq_gradient, rtol=2e-14, atol=2e-11)
    repeated_value, repeated_gradient = parallel.gaussian_objective_and_gradient(
        correlation)
    assert repeated_value == par_value
    np.testing.assert_array_equal(repeated_gradient, par_gradient)


def test_native_gaussian_trial_rejects_invalid_spec_and_observations():
    module = load()
    u = np.full((20, 3), 0.5)
    correlation = np.eye(3)
    spec = _cpp_copula.make_gaussian_static_spec(module, correlation)
    native = module.StaticCopulaEvaluator(spec, u, 1)

    wrong_dimension = _cpp_copula.make_gaussian_static_spec(
        module, np.eye(2))
    result = dict(native.gaussian_objective_with_correlation_gradient(
        wrong_dimension))
    assert result["status"] != module.SCAR_OK
    assert np.isinf(result["negative_log_likelihood"])

    invalid_u = u.copy()
    invalid_u[3, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        module.StaticCopulaEvaluator(spec, invalid_u, 1)


def test_existing_static_joint_api_returns_gaussian_lower_triangle_score():
    rng = np.random.default_rng(55)
    u = rng.uniform(0.01, 0.99, size=(100, 3))
    correlation = _correlation(3, "random", seed=6)
    evaluator = static_likelihood.prepare_gaussian(correlation, u)
    value, scalar_gradient, correlation_gradient = (
        evaluator.objective_and_joint_gradient(0.0))
    prepared_value, prepared_gradient = (
        evaluator.gaussian_objective_and_gradient(correlation))

    assert value == prepared_value
    np.testing.assert_array_equal(scalar_gradient, np.array([0.0]))
    np.testing.assert_array_equal(correlation_gradient, prepared_gradient)
