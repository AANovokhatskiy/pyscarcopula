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
        try:
            idx = int(key)
        except Exception as exc:
            raise TypeError("given keys must be integers") from exc
        if idx < 0 or idx >= int(d):
            raise ValueError(
                f"given key must be in [0, {int(d) - 1}], got {key!r}")
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


def sample_gaussian_conditional(n, d, rho, given, rng=None):
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
    if len(given) == d:
        return fill_given(n, d, given)

    rho_path = as_path(rho, n, "rho")
    given_idx, free_idx = _partition_indices(d, given)
    z_given = norm.ppf(_given_quantile_inputs(given, given_idx))
    normal_draws = np.empty((n, len(free_idx)), dtype=np.float64)
    for rho_val in np.unique(rho_path):
        mask = rho_path == rho_val
        rows = np.where(mask)[0]
        m = int(np.sum(mask))
        normal_draws[rows] = rng.standard_normal((m, len(free_idx)))

    rho_input = np.atleast_1d(np.asarray(rho, dtype=np.float64)).ravel()
    correlations = (
        equicorr_matrix(d, rho_path[0])
        if rho_input.size == 1
        else np.stack([
            equicorr_matrix(d, rho_val) for rho_val in rho_path])
    )
    from pyscarcopula.numerical import multivariate_native
    z_free = multivariate_native.gaussian_conditional_latent(
        correlations, given_idx, z_given, normal_draws)
    out = np.empty((n, d), dtype=np.float64)
    out[:, free_idx] = norm.cdf(z_free)

    return _finalize_conditional_sample(out, free_idx, given)


def sample_student_conditional(n, R_path, df, given, rng=None):
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
    if len(given) == d:
        return fill_given(n, d, given)

    df_path = as_path(df, n, "df")
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
    )
    out = np.empty((n, d), dtype=np.float64)
    out[:, free_idx] = t_dist.cdf(x_free, df=df_path[:, None])

    return _finalize_conditional_sample(out, free_idx, given)


__all__ = [
    "as_path",
    "equicorr_matrix",
    "fill_given",
    "sample_gaussian_conditional",
    "sample_student_conditional",
    "validate_multivariate_given",
]
