"""Conditional sampling helpers for experimental multivariate copulas."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.stats import t as t_dist


_EPS = 1e-12


def validate_multivariate_given(given, d):
    """Normalize ``given`` to ``{int: float}`` for a d-variate copula."""
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


def _stable_cholesky(matrix):
    matrix = 0.5 * (matrix + matrix.T)
    jitter = 0.0
    eye = np.eye(matrix.shape[0], dtype=np.float64)
    for _ in range(6):
        try:
            return np.linalg.cholesky(matrix + jitter * eye)
        except np.linalg.LinAlgError:
            jitter = 1e-12 if jitter == 0.0 else jitter * 10.0
    vals, vecs = np.linalg.eigh(matrix)
    vals = np.maximum(vals, 1e-12)
    return np.linalg.cholesky(vecs @ np.diag(vals) @ vecs.T)


def _partition_indices(d, given):
    given_idx = np.array(sorted(given), dtype=int)
    free_idx = np.array([idx for idx in range(int(d)) if idx not in given],
                        dtype=int)
    return given_idx, free_idx


def sample_gaussian_conditional(n, d, rho, given, rng=None):
    """Sample a Gaussian copula conditional on arbitrary fixed coordinates."""
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
    z_given = norm.ppf([given[idx] for idx in given_idx])

    out = np.empty((n, d), dtype=np.float64)
    for idx, value in given.items():
        out[:, idx] = value

    for rho_val in np.unique(rho_path):
        mask = rho_path == rho_val
        rows = np.where(mask)[0]
        m = int(np.sum(mask))
        R = equicorr_matrix(d, rho_val)
        R_gg = R[np.ix_(given_idx, given_idx)]
        R_fg = R[np.ix_(free_idx, given_idx)]
        R_gf = R[np.ix_(given_idx, free_idx)]
        R_ff = R[np.ix_(free_idx, free_idx)]

        solved_mean = np.linalg.solve(R_gg, z_given)
        mean = R_fg @ solved_mean
        solved_cov = np.linalg.solve(R_gg, R_gf)
        cov = R_ff - R_fg @ solved_cov
        L = _stable_cholesky(cov)

        z_free = rng.standard_normal((m, len(free_idx))) @ L.T + mean
        out[np.ix_(rows, free_idx)] = norm.cdf(z_free)

    return np.clip(out, _EPS, 1.0 - _EPS)


def _student_conditional_block(R, df, given, given_idx, free_idx, m, rng):
    x_given = t_dist.ppf([given[idx] for idx in given_idx], df=df)
    R_gg = R[np.ix_(given_idx, given_idx)]
    R_fg = R[np.ix_(free_idx, given_idx)]
    R_gf = R[np.ix_(given_idx, free_idx)]
    R_ff = R[np.ix_(free_idx, free_idx)]

    solved_x = np.linalg.solve(R_gg, x_given)
    loc = R_fg @ solved_x
    delta = float(x_given @ solved_x)
    solved_cov = np.linalg.solve(R_gg, R_gf)
    schur = R_ff - R_fg @ solved_cov

    cond_df = float(df) + len(given_idx)
    scale = ((float(df) + delta) / cond_df) * schur
    L = _stable_cholesky(scale)
    z = rng.standard_normal((int(m), len(free_idx))) @ L.T
    chi2 = rng.chisquare(cond_df, size=int(m))
    x_free = loc + np.sqrt(cond_df / chi2)[:, None] * z
    return t_dist.cdf(x_free, df=df)


def sample_student_conditional(n, R_path, df, given, rng=None):
    """Sample a Student-t copula conditional on arbitrary fixed coordinates."""
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

    out = np.empty((n, d), dtype=np.float64)
    for idx, value in given.items():
        out[:, idx] = value

    if R_arr.ndim == 2:
        for df_val in np.unique(df_path):
            mask = df_path == df_val
            rows = np.where(mask)[0]
            out[np.ix_(rows, free_idx)] = _student_conditional_block(
                R_arr, float(df_val), given, given_idx, free_idx,
                int(np.sum(mask)), rng)
    else:
        for row in range(n):
            out[row, free_idx] = _student_conditional_block(
                R_arr[row], float(df_path[row]), given, given_idx, free_idx,
                1, rng)[0]

    return np.clip(out, _EPS, 1.0 - _EPS)


__all__ = [
    "as_path",
    "equicorr_matrix",
    "fill_given",
    "sample_gaussian_conditional",
    "sample_student_conditional",
    "validate_multivariate_given",
]
