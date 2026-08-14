"""Conditional sampling helpers for multivariate copulas."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.stats import t as t_dist

from pyscarcopula._constants import CONDITIONAL_SAMPLE_EPS
from pyscarcopula._utils import clip_pseudo_observations


def validate_multivariate_given(given, d):
    """Normalize finite ``given`` values in the open unit interval.

    Values closer to a boundary than ``PSEUDO_OBS_EPS`` remain unchanged in
    the public result, but conditional quantile calculations identify them
    with the corresponding clipping-layer boundary.
    """
    if given is None:
        return {}
    if not isinstance(given, dict):
        raise TypeError("given must be a dict[int, float] or None")

    out = {}
    for key, value in given.items():
        if isinstance(key, (bool, np.bool_)) or not isinstance(
                key, (int, np.integer)):
            raise TypeError("given keys must be integers")
        idx = int(key)
        if idx < 0 or idx >= int(d):
            raise ValueError(
                f"given key must be in [0, {int(d) - 1}], got {key!r}")
        if (
                isinstance(value, (bool, np.bool_, str, bytes, complex,
                                   np.complexfloating))
                or not np.isscalar(value)):
            raise TypeError("given values must be numeric scalars")
        val = float(value)
        if not (0.0 < val < 1.0):
            raise ValueError(
                f"given[{idx}] must be in pseudo-observation space (0, 1), "
                f"got {val}")
        out[idx] = val
    return out


def as_path(values, n, name):
    """Return a scalar-or-length-n numeric value as a length-n path."""
    arr = np.atleast_1d(np.asarray(values, dtype=np.float64)).ravel()
    if arr.size == 1:
        return np.full(int(n), float(arr[0]), dtype=np.float64)
    if arr.size != int(n):
        raise ValueError(f"{name} must be scalar or array of length {n}, got {arr.size}")
    return arr.astype(np.float64, copy=False)


def fill_given(n, d, given):
    """Return an ``(n, d)`` array with all coordinates fixed by ``given``."""
    out = np.empty((int(n), int(d)), dtype=np.float64)
    for idx in range(int(d)):
        out[:, idx] = given[idx]
    return out


def equicorr_matrix(d, rho):
    rho = float(rho)
    return (1.0 - rho) * np.eye(int(d), dtype=np.float64) + rho * np.ones(
        (int(d), int(d)), dtype=np.float64)


def _partition_indices(d, given):
    given_idx = np.array(sorted(given), dtype=int)
    free_idx = np.array([idx for idx in range(int(d)) if idx not in given],
                        dtype=int)
    return given_idx, free_idx


def _given_quantile_inputs(given, given_idx):
    """Return common-boundary inputs for conditional inverse CDFs."""
    values = np.array(
        [given[idx] for idx in given_idx], dtype=np.float64)
    return clip_pseudo_observations(values)


def _finalize_conditional_sample(out, free_idx, given):
    """Clip sampled coordinates while preserving fixed values exactly."""
    if len(free_idx):
        out[:, free_idx] = np.clip(
            out[:, free_idx],
            CONDITIONAL_SAMPLE_EPS,
            1.0 - CONDITIONAL_SAMPLE_EPS,
        )
    for idx, value in given.items():
        out[:, idx] = value
    return out


def sample_gaussian_conditional(
        n, d, rho, given, rng=None, *, n_threads=1):
    """Sample a Gaussian copula conditional on fixed pseudo-observations.

    Conditional Gaussian quantiles use the same ``PSEUDO_OBS_EPS`` clipping
    policy as likelihood evaluation. Valid fixed values are returned exactly.
    Python owns validation, quantile conversion, and random-number generation;
    the conditional linear algebra is evaluated by the mandatory C++ backend.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = int(n)
    d = int(d)
    given = validate_multivariate_given(given, d)
    if not given:
        raise ValueError("sample_gaussian_conditional requires non-empty given")
    rho_path = as_path(rho, n, "rho")
    lower = -1.0 / (d - 1.0)
    if (
            np.any(~np.isfinite(rho_path))
            or np.any(rho_path <= lower)
            or np.any(rho_path >= 1.0)):
        raise ValueError(f"rho must be finite and in ({lower}, 1)")
    if len(given) == d:
        return fill_given(n, d, given)
    given_idx, free_idx = _partition_indices(d, given)
    z_given = norm.ppf(_given_quantile_inputs(given, given_idx))
    normal_draws = rng.standard_normal((n, len(free_idx)))

    from pyscarcopula.numerical import multivariate_native
    multivariate_native._validated_n_threads(n_threads)

    # Equicorrelation is closed under conditioning.  Work in the span of the
    # all-ones vector and its orthogonal complement, avoiding a dense R or
    # Schur complement for arbitrarily large d.
    n_given = len(given_idx)
    n_free = len(free_idx)
    denominator = 1.0 + (n_given - 1.0) * rho_path
    conditional_mean = (
        rho_path / denominator * float(np.sum(z_given)))
    alpha = 1.0 - rho_path
    parallel_eigenvalue = (
        alpha * (1.0 + (d - 1.0) * rho_path) / denominator)
    if np.any(parallel_eigenvalue <= 0.0):
        raise ValueError(
            "rho produces a non-positive conditional covariance")
    row_means = normal_draws.mean(axis=1, keepdims=True)
    z_free = (
        np.sqrt(alpha)[:, None] * (normal_draws - row_means)
        + np.sqrt(parallel_eigenvalue)[:, None] * row_means
        + conditional_mean[:, None]
    )
    out = np.empty((n, d), dtype=np.float64)
    out[:, free_idx] = norm.cdf(z_free)

    return _finalize_conditional_sample(out, free_idx, given)


def sample_gaussian_copula_conditional(
        n, R, given, rng=None, *, n_threads=1):
    """Sample a Gaussian copula with an arbitrary correlation matrix
    conditionally on fixed pseudo-observations.

    Same clipping and validation policy as ``sample_gaussian_conditional``,
    which is the equicorrelation special case.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = int(n)
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R must be a square correlation matrix")
    d = R.shape[0]
    given = validate_multivariate_given(given, d)
    if not given:
        raise ValueError(
            "sample_gaussian_copula_conditional requires non-empty given")
    if len(given) == d:
        return fill_given(n, d, given)

    given_idx, free_idx = _partition_indices(d, given)
    z_given = norm.ppf(_given_quantile_inputs(given, given_idx))
    normal_draws = rng.standard_normal((n, len(free_idx)))
    from pyscarcopula.numerical import multivariate_native
    z_free = multivariate_native.gaussian_conditional_latent(
        R, given_idx, z_given, normal_draws, n_threads=n_threads)
    out = np.empty((n, d), dtype=np.float64)
    out[:, free_idx] = norm.cdf(z_free)

    return _finalize_conditional_sample(out, free_idx, given)


def sample_student_conditional(
        n, R_path, df, given, rng=None, *, n_threads=1):
    """Sample a Student-t copula conditional on fixed pseudo-observations.

    Conditional Student quantiles use the same ``PSEUDO_OBS_EPS`` clipping
    policy as likelihood evaluation. Valid fixed values are returned exactly.
    Python owns validation, quantile conversion, and random-number generation;
    the conditional linear algebra is evaluated by the mandatory C++ backend.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = int(n)
    R_arr = np.asarray(R_path, dtype=np.float64)
    if R_arr.ndim == 2:
        d = R_arr.shape[0]
    elif R_arr.ndim == 3:
        if len(R_arr) != n:
            raise ValueError(
                f"R_path length {len(R_arr)} does not match n={n}")
        d = R_arr.shape[1]
    else:
        raise ValueError("R_path must be a matrix or a length-n matrix path")

    given = validate_multivariate_given(given, d)
    if not given:
        raise ValueError("sample_student_conditional requires non-empty given")
    df_path = as_path(df, n, "df")
    if (
            np.any(~np.isfinite(df_path))
            or np.any(df_path <= 2.0)):
        raise ValueError("df must be finite and greater than 2")
    if len(given) == d:
        return fill_given(n, d, given)
    given_idx, free_idx = _partition_indices(d, given)
    given_inputs = _given_quantile_inputs(given, given_idx)
    given_latent = np.empty((n, len(given_idx)), dtype=np.float64)
    normal_draws = np.empty((n, len(free_idx)), dtype=np.float64)
    chi_square_draws = np.empty(n, dtype=np.float64)
    if R_arr.ndim == 2:
        for df_val in np.unique(df_path):
            mask = df_path == df_val
            rows = np.where(mask)[0]
            m = int(np.sum(mask))
            given_latent[rows] = t_dist.ppf(
                given_inputs, df=float(df_val))
            normal_draws[rows] = rng.standard_normal(
                (m, len(free_idx)))
            chi_square_draws[rows] = rng.chisquare(
                float(df_val) + len(given_idx), size=m)
    else:
        for row in range(n):
            given_latent[row] = t_dist.ppf(
                given_inputs, df=float(df_path[row]))
            normal_draws[row] = rng.standard_normal(
                (1, len(free_idx)))[0]
            chi_square_draws[row] = rng.chisquare(
                float(df_path[row]) + len(given_idx), size=1)[0]

    from pyscarcopula.numerical import multivariate_native
    x_free = multivariate_native.student_conditional_latent(
        R_arr,
        given_idx,
        given_latent,
        df_path,
        normal_draws,
        chi_square_draws,
        n_threads=n_threads,
    )
    out = np.empty((n, d), dtype=np.float64)
    out[:, free_idx] = t_dist.cdf(x_free, df=df_path[:, None])

    return _finalize_conditional_sample(out, free_idx, given)


def sample_factor_gaussian_conditional(
        n,
        correlation,
        given,
        rng=None,
        *,
        n_threads=1):
    """Sample a factor Gaussian copula without a dense Schur complement."""
    from pyscarcopula.copula.multivariate.factor_correlation import (
        PreparedFactorCorrelation,
    )

    if not isinstance(correlation, PreparedFactorCorrelation):
        raise TypeError(
            "correlation must be a PreparedFactorCorrelation")
    if isinstance(n, (bool, np.bool_)) or not isinstance(
            n, (int, np.integer)):
        raise TypeError("n must be an integer")
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    if rng is None:
        rng = np.random.default_rng()

    d = correlation.dimension
    given = validate_multivariate_given(given, d)
    if not given:
        raise ValueError(
            "sample_factor_gaussian_conditional requires non-empty given")
    if len(given) == d:
        return fill_given(n, d, given)

    given_idx, free_idx = _partition_indices(d, given)
    given_latent = norm.ppf(
        _given_quantile_inputs(given, given_idx))
    given_loadings = correlation.loadings[given_idx]
    given_uniqueness = correlation.uniqueness[given_idx]
    small_precision = (
        np.eye(correlation.rank, dtype=np.float64)
        + given_loadings.T
        @ (given_loadings / given_uniqueness[:, None])
    )
    small_cholesky = np.linalg.cholesky(small_precision)
    conditional_factor_root = np.linalg.solve(
        small_cholesky.T,
        np.eye(correlation.rank, dtype=np.float64),
    )
    projected = (
        given_latent / given_uniqueness) @ given_loadings
    conditional_factor_mean = np.linalg.solve(
        small_precision, projected)

    factor_draws = (
        rng.standard_normal((n, correlation.rank))
        @ conditional_factor_root.T
        + conditional_factor_mean[None, :]
    )
    residual_draws = rng.standard_normal((n, d))
    latent = correlation.transform_normal_draws(
        factor_draws,
        residual_draws,
        n_threads=n_threads,
    )
    out = np.empty((n, d), dtype=np.float64)
    out[:, free_idx] = norm.cdf(latent[:, free_idx])
    return _finalize_conditional_sample(out, free_idx, given)


def sample_factor_student_conditional(
        n,
        correlation,
        df,
        given,
        rng=None,
        *,
        n_threads=1):
    """Sample a factor Student copula without a dense Schur complement.

    Conditioning only factorizes the ``(rank, rank)`` matrix associated with
    the fixed coordinates. The native factor operator transforms the
    row-wise innovations, so the full correlation and conditional covariance
    are never materialized.
    """
    from pyscarcopula.copula.multivariate.factor_correlation import (
        PreparedFactorCorrelation,
    )

    if not isinstance(correlation, PreparedFactorCorrelation):
        raise TypeError(
            "correlation must be a PreparedFactorCorrelation")
    if isinstance(n, (bool, np.bool_)) or not isinstance(
            n, (int, np.integer)):
        raise TypeError("n must be an integer")
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    if rng is None:
        rng = np.random.default_rng()

    d = correlation.dimension
    given = validate_multivariate_given(given, d)
    if not given:
        raise ValueError(
            "sample_factor_student_conditional requires non-empty given")
    if len(given) == d:
        return fill_given(n, d, given)

    df_path = as_path(df, n, "df")
    if (
            not np.all(np.isfinite(df_path))
            or np.any(df_path <= 2.0)):
        raise ValueError("df must be finite and greater than 2")
    given_idx, free_idx = _partition_indices(d, given)
    given_inputs = _given_quantile_inputs(given, given_idx)

    loadings = correlation.loadings
    uniqueness = correlation.uniqueness
    given_loadings = loadings[given_idx]
    given_uniqueness = uniqueness[given_idx]
    weighted_given_loadings = (
        given_loadings / given_uniqueness[:, None])
    small_precision = (
        np.eye(correlation.rank, dtype=np.float64)
        + given_loadings.T @ weighted_given_loadings)
    small_cholesky = np.linalg.cholesky(small_precision)
    conditional_factor_root = np.linalg.solve(
        small_cholesky.T,
        np.eye(correlation.rank, dtype=np.float64),
    )

    given_latent = np.empty(
        (n, len(given_idx)), dtype=np.float64)
    chi_square_draws = np.empty(n, dtype=np.float64)
    for df_value in np.unique(df_path):
        rows = np.flatnonzero(df_path == df_value)
        given_latent[rows] = t_dist.ppf(
            given_inputs, df=float(df_value))
        chi_square_draws[rows] = rng.chisquare(
            float(df_value) + len(given_idx),
            size=len(rows),
        )

    projected = (
        given_latent / given_uniqueness[None, :]
    ) @ given_loadings
    conditional_factor_mean = np.linalg.solve(
        small_precision, projected.T).T
    delta = (
        np.einsum(
            "ij,ij,j->i",
            given_latent,
            given_latent,
            1.0 / given_uniqueness,
            optimize=False,
        )
        - np.einsum(
            "ij,ij->i",
            projected,
            conditional_factor_mean,
            optimize=False,
        )
    )
    radial_scale = np.sqrt(
        (df_path + np.maximum(delta, 0.0)) / chi_square_draws)

    factor_draws = rng.standard_normal((n, correlation.rank))
    factor_draws = (
        factor_draws @ conditional_factor_root.T
    ) * radial_scale[:, None] + conditional_factor_mean
    residual_draws = rng.standard_normal((n, d))
    residual_draws *= radial_scale[:, None]
    latent = correlation.transform_normal_draws(
        factor_draws,
        residual_draws,
        n_threads=n_threads,
    )

    out = np.empty((n, d), dtype=np.float64)
    out[:, free_idx] = t_dist.cdf(
        latent[:, free_idx], df=df_path[:, None])
    return _finalize_conditional_sample(out, free_idx, given)


__all__ = [
    "as_path",
    "equicorr_matrix",
    "fill_given",
    "sample_gaussian_conditional",
    "sample_gaussian_copula_conditional",
    "sample_factor_gaussian_conditional",
    "sample_factor_student_conditional",
    "sample_student_conditional",
    "validate_multivariate_given",
]
