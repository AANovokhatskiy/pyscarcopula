"""Independent closed-form conditional-distribution oracles.

Only NumPy/SciPy mathematics is used here.  Importing production conditional
sampling helpers is intentionally forbidden so these functions can detect a
shared implementation error in the library kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import norm
from scipy.stats import t as t_dist


_OPEN_EPS = 1e-12


@dataclass(frozen=True)
class GaussianConditional:
    given_indices: np.ndarray
    free_indices: np.ndarray
    mean: np.ndarray
    covariance: np.ndarray

    def whiten(self, latent: np.ndarray) -> np.ndarray:
        root = np.linalg.cholesky(self.covariance)
        centered = np.asarray(latent, dtype=np.float64) - self.mean
        return solve_triangular(root, centered.T, lower=True).T


@dataclass(frozen=True)
class StudentConditional:
    given_indices: np.ndarray
    free_indices: np.ndarray
    location: np.ndarray
    shape: np.ndarray
    conditional_df: float
    original_df: float

    @property
    def covariance(self) -> np.ndarray:
        return (
            self.conditional_df
            / (self.conditional_df - 2.0)
            * self.shape
        )

    def latent_from_copula(self, sample: np.ndarray) -> np.ndarray:
        values = np.asarray(sample, dtype=np.float64)
        return t_dist.ppf(
            np.clip(values[..., self.free_indices], _OPEN_EPS, 1.0 - _OPEN_EPS),
            df=self.original_df,
        )

    def whiten(self, latent: np.ndarray) -> np.ndarray:
        root = np.linalg.cholesky(self.shape)
        centered = np.asarray(latent, dtype=np.float64) - self.location
        return solve_triangular(root, centered.T, lower=True).T


def _correlation(value) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("R must be a square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("R must be finite")
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0):
        raise ValueError("R must be symmetric")
    np.linalg.cholesky(matrix)
    return matrix


def _partition(d: int, given: Mapping[int, float]):
    given_indices = np.array(sorted(given), dtype=np.int64)
    free_indices = np.array(
        [index for index in range(d) if index not in given],
        dtype=np.int64,
    )
    if not len(given_indices) or not len(free_indices):
        raise ValueError("oracle requires both given and free coordinates")
    values = np.array(
        [given[int(index)] for index in given_indices],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(values)) or np.any((values <= 0.0) | (values >= 1.0)):
        raise ValueError("given values must be finite and in (0, 1)")
    return given_indices, free_indices, values


def gaussian_conditional_parameters(
        R, given: Mapping[int, float]) -> GaussianConditional:
    """Closed-form latent Gaussian law for free coordinates given U_G."""

    matrix = _correlation(R)
    given_idx, free_idx, values = _partition(len(matrix), given)
    z_given = norm.ppf(values)
    r_gg = matrix[np.ix_(given_idx, given_idx)]
    r_fg = matrix[np.ix_(free_idx, given_idx)]
    solved_values = np.linalg.solve(r_gg, z_given)
    solved_cross = np.linalg.solve(r_gg, r_fg.T)
    mean = r_fg @ solved_values
    covariance = matrix[np.ix_(free_idx, free_idx)] - r_fg @ solved_cross
    covariance = 0.5 * (covariance + covariance.T)
    np.linalg.cholesky(covariance)
    return GaussianConditional(given_idx, free_idx, mean, covariance)


def student_conditional_parameters(
        R, df: float, given: Mapping[int, float]) -> StudentConditional:
    """Closed-form latent multivariate-t conditional law."""

    matrix = _correlation(R)
    df = float(df)
    if not np.isfinite(df) or df <= 2.0:
        raise ValueError("df must be finite and greater than 2")
    given_idx, free_idx, values = _partition(len(matrix), given)
    x_given = t_dist.ppf(values, df=df)
    r_gg = matrix[np.ix_(given_idx, given_idx)]
    r_fg = matrix[np.ix_(free_idx, given_idx)]
    solved_values = np.linalg.solve(r_gg, x_given)
    solved_cross = np.linalg.solve(r_gg, r_fg.T)
    delta = float(x_given @ solved_values)
    location = r_fg @ solved_values
    schur = matrix[np.ix_(free_idx, free_idx)] - r_fg @ solved_cross
    conditional_df = df + len(given_idx)
    shape = ((df + delta) / conditional_df) * schur
    shape = 0.5 * (shape + shape.T)
    np.linalg.cholesky(shape)
    return StudentConditional(
        given_idx,
        free_idx,
        location,
        shape,
        conditional_df,
        df,
    )


def gaussian_copula_parameter_from_state(state):
    """Independent form of the public bivariate Gaussian SCAR link."""

    state = np.asarray(state, dtype=np.float64)
    return 0.9999 * np.tanh(state / 4.0)


def gaussian_copula_log_density(u, rho):
    """Bivariate Gaussian copula log-density from first principles."""

    values = np.asarray(u, dtype=np.float64)
    if values.shape[-1] != 2:
        raise ValueError("u must have final dimension 2")
    if np.any(~np.isfinite(values)) or np.any((values <= 0.0) | (values >= 1.0)):
        raise ValueError("u must be finite and in (0, 1)")
    rho = np.asarray(rho, dtype=np.float64)
    if np.any(~np.isfinite(rho)) or np.any(np.abs(rho) >= 1.0):
        raise ValueError("rho must be finite and in (-1, 1)")
    z1 = norm.ppf(values[..., 0])
    z2 = norm.ppf(values[..., 1])
    denominator = 1.0 - rho * rho
    exponent = (
        2.0 * rho * z1 * z2
        - rho * rho * (z1 * z1 + z2 * z2)
    ) / (2.0 * denominator)
    return -0.5 * np.log(denominator) + exponent


def gaussian_conditional_cdf(free_u, given_u, rho):
    """H(U_free | U_given) for a bivariate Gaussian copula."""

    free_u = np.asarray(free_u, dtype=np.float64)
    given_u = np.asarray(given_u, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    z_free = norm.ppf(np.clip(free_u, _OPEN_EPS, 1.0 - _OPEN_EPS))
    z_given = norm.ppf(np.clip(given_u, _OPEN_EPS, 1.0 - _OPEN_EPS))
    return norm.cdf(
        (z_free - rho * z_given) / np.sqrt(1.0 - rho * rho)
    )


def gaussian_conditional_inverse(q, given_u, rho):
    """Closed-form inverse H for a bivariate Gaussian copula."""

    q = np.asarray(q, dtype=np.float64)
    given_u = np.asarray(given_u, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    z_given = norm.ppf(np.clip(given_u, _OPEN_EPS, 1.0 - _OPEN_EPS))
    latent = (
        rho * z_given
        + np.sqrt(1.0 - rho * rho)
        * norm.ppf(np.clip(q, _OPEN_EPS, 1.0 - _OPEN_EPS))
    )
    return norm.cdf(latent)
