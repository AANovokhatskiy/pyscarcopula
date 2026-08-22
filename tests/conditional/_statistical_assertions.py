"""Monte Carlo error-budget assertions shared by conditional tests."""

from __future__ import annotations

import numpy as np


def uniform_ecdf_deviation(values) -> float:
    sample = np.sort(np.asarray(values, dtype=np.float64).ravel())
    if not len(sample):
        raise ValueError("uniform sample must be non-empty")
    if np.any(~np.isfinite(sample)) or np.any((sample < 0.0) | (sample > 1.0)):
        raise ValueError("uniform sample must lie in [0, 1]")
    n = len(sample)
    upper = np.arange(1, n + 1, dtype=np.float64) / n
    lower = np.arange(0, n, dtype=np.float64) / n
    return float(max(np.max(upper - sample), np.max(sample - lower)))


def assert_uniform_pit(
        values,
        *,
        alpha: float = 1e-6,
        numerical_floor: float = 0.004) -> None:
    sample = np.asarray(values, dtype=np.float64).ravel()
    deviation = uniform_ecdf_deviation(sample)
    dkw = np.sqrt(np.log(2.0 / alpha) / (2.0 * len(sample)))
    bound = dkw + numerical_floor
    assert deviation <= bound, (
        f"PIT ECDF deviation {deviation:.6g} exceeds DKW budget {bound:.6g}"
    )
    mean_error = abs(float(np.mean(sample)) - 0.5)
    mean_bound = 6.0 / np.sqrt(12.0 * len(sample)) + numerical_floor
    assert mean_error <= mean_bound, (
        f"PIT mean error {mean_error:.6g} exceeds MC budget {mean_bound:.6g}"
    )


def assert_mean_with_mc_error(
        sample,
        expected_mean,
        expected_covariance,
        *,
        sigma: float = 6.0,
        numerical_floor: float = 2e-3) -> None:
    values = np.atleast_2d(np.asarray(sample, dtype=np.float64))
    mean = np.atleast_1d(np.asarray(expected_mean, dtype=np.float64))
    covariance = np.atleast_2d(
        np.asarray(expected_covariance, dtype=np.float64)
    )
    observed = np.mean(values, axis=0)
    standard_error = np.sqrt(np.diag(covariance) / len(values))
    bound = sigma * standard_error + numerical_floor
    error = np.abs(observed - mean)
    assert np.all(error <= bound), (
        f"mean error {error} exceeds Monte Carlo budget {bound}"
    )


def assert_covariance_with_whitening(
        sample,
        expected_mean,
        expected_covariance,
        *,
        sigma: float = 7.0,
        numerical_floor: float = 0.015) -> None:
    values = np.atleast_2d(np.asarray(sample, dtype=np.float64))
    mean = np.atleast_1d(np.asarray(expected_mean, dtype=np.float64))
    covariance = np.atleast_2d(
        np.asarray(expected_covariance, dtype=np.float64)
    )
    root = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(root, (values - mean).T).T
    observed = np.cov(whitened, rowvar=False, ddof=1)
    observed = np.atleast_2d(observed)
    error = np.abs(observed - np.eye(observed.shape[0]))
    asymptotic = sigma * np.sqrt(2.0 / max(len(values) - 1, 1))
    bound = asymptotic + numerical_floor
    assert float(np.max(error)) <= bound, (
        f"max whitened covariance error {np.max(error):.6g} "
        f"exceeds MC budget {bound:.6g}; covariance={observed}"
    )
