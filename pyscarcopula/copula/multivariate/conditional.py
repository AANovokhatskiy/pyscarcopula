"""Conditional sampling helpers for multivariate copulas."""

from __future__ import annotations

import numpy as np

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


def _partition_indices(d, given):
    given_idx = np.array(sorted(given), dtype=int)
    free_idx = np.array([idx for idx in range(int(d)) if idx not in given],
                        dtype=int)
    return given_idx, free_idx


def _assemble_native_conditional_sample(
        n, d, free_values, free_idx, given):
    """Assemble already-finalized native free coordinates with fixed values."""
    out = np.empty((int(n), int(d)), dtype=np.float64)
    out[:, free_idx] = free_values
    for idx, value in given.items():
        out[:, idx] = value
    return out


def sample_gaussian_conditional(
        n, d, rho, given, rng=None, *, n_threads=1):
    """Sample a Gaussian copula conditional on fixed pseudo-observations.

    Conditional Gaussian quantiles use the same ``PSEUDO_OBS_EPS`` clipping
    policy as likelihood evaluation. Valid fixed values are returned exactly.
    Python owns validation and random-number generation; C++ owns quantile
    conversion and the scalar-or-row equicorrelation transform.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = int(n)
    d = int(d)
    given = validate_multivariate_given(given, d)
    if not given:
        raise ValueError("sample_gaussian_conditional requires non-empty given")
    rho_path = as_path(rho, n, "rho")
    from pyscarcopula._native import multivariate as multivariate_native
    multivariate_native.validate_equicorrelation_path(rho_path, d, n)
    if len(given) == d:
        return fill_given(n, d, given)
    given_idx, free_idx = _partition_indices(d, given)
    normal_draws = rng.standard_normal((n, len(free_idx)))

    free_values = (
        multivariate_native.equicorr_gaussian_conditional_from_uniforms(
            rho_path,
            d,
            given_idx,
            np.array([given[idx] for idx in given_idx], dtype=np.float64),
            normal_draws,
            n_threads=n_threads,
        )
    )
    return _assemble_native_conditional_sample(
        n, d, free_values, free_idx, given)


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
    normal_draws = rng.standard_normal((n, len(free_idx)))
    from pyscarcopula._native import multivariate as multivariate_native
    free_values = multivariate_native.gaussian_conditional_from_uniforms(
        R,
        given_idx,
        np.array([given[idx] for idx in given_idx], dtype=np.float64),
        normal_draws,
        n_threads=n_threads,
    )
    return _assemble_native_conditional_sample(
        n, d, free_values, free_idx, given)


def sample_student_conditional(
        n, R_path, df, given, rng=None, *, n_threads=1):
    """Sample a Student-t copula conditional on fixed pseudo-observations.

    Conditional Student quantiles use the same ``PSEUDO_OBS_EPS`` clipping
    policy as likelihood evaluation. Valid fixed values are returned exactly.
    Python owns validation and random-number generation. Quantile conversion,
    conditional algebra, and the final copula transform use the mandatory C++
    backend.
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
    normal_draws = np.empty((n, len(free_idx)), dtype=np.float64)
    chi_square_uniforms = np.empty(n, dtype=np.float64)
    if R_arr.ndim == 2:
        for df_val in np.unique(df_path):
            mask = df_path == df_val
            rows = np.where(mask)[0]
            m = int(np.sum(mask))
            normal_draws[rows] = rng.standard_normal(
                (m, len(free_idx)))
            chi_square_uniforms[rows] = rng.uniform(0.0, 1.0, size=m)
    else:
        for row in range(n):
            normal_draws[row] = rng.standard_normal(
                (1, len(free_idx)))[0]
            chi_square_uniforms[row] = rng.uniform(0.0, 1.0, size=1)[0]

    from pyscarcopula._native import multivariate as multivariate_native
    free_values = multivariate_native.student_conditional_from_normal_uniforms(
        R_arr,
        given_idx,
        np.array([given[idx] for idx in given_idx], dtype=np.float64),
        df_path,
        normal_draws,
        chi_square_uniforms,
        n_threads=n_threads,
    )
    return _assemble_native_conditional_sample(
        n, d, free_values, free_idx, given)


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
    factor_draws = rng.standard_normal((n, correlation.rank))
    residual_draws = rng.standard_normal((n, d))
    from pyscarcopula._native import multivariate as multivariate_native
    free_values = (
        multivariate_native.factor_gaussian_conditional_from_uniforms(
            correlation,
            given_idx,
            np.array([given[idx] for idx in given_idx], dtype=np.float64),
            factor_draws,
            residual_draws,
            n_threads=n_threads,
        )
    )
    return _assemble_native_conditional_sample(
        n, d, free_values, free_idx, given)


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
    chi_square_uniforms = rng.uniform(0.0, 1.0, size=n)
    factor_draws = rng.standard_normal((n, correlation.rank))
    residual_draws = rng.standard_normal((n, d))
    from pyscarcopula._native import multivariate as multivariate_native
    free_values = (
        multivariate_native.factor_student_conditional_from_normal_uniforms(
            correlation,
            given_idx,
            np.array([given[idx] for idx in given_idx], dtype=np.float64),
            df_path,
            factor_draws,
            residual_draws,
            chi_square_uniforms,
            n_threads=n_threads,
        )
    )
    return _assemble_native_conditional_sample(
        n, d, free_values, free_idx, given)


__all__ = [
    "as_path",
    "fill_given",
    "sample_gaussian_conditional",
    "sample_gaussian_copula_conditional",
    "sample_factor_gaussian_conditional",
    "sample_factor_student_conditional",
    "sample_student_conditional",
    "validate_multivariate_given",
]
