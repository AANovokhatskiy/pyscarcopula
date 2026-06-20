"""Static correlation parameterizations for multivariate copulas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import kendalltau


@dataclass(frozen=True)
class CorrelationPreprocessingResult:
    """Result of converting an input matrix to an SPD correlation matrix."""

    correlation: np.ndarray
    input_correlation: np.ndarray
    source: str
    projection_applied: bool
    min_eigenvalue_before: float
    min_eigenvalue_after: float
    nonfinite_kendall_pairs: tuple[tuple[int, int], ...] = ()

    def diagnostics(self) -> dict:
        """Return serialization-friendly correlation preprocessing metadata."""
        return {
            "corr_initialization_source": self.source,
            "corr_projection_applied": self.projection_applied,
            "corr_min_eigenvalue_before": self.min_eigenvalue_before,
            "corr_min_eigenvalue_after": self.min_eigenvalue_after,
            "corr_nonfinite_kendall_pairs": self.nonfinite_kendall_pairs,
        }


def sigmoid(x: float | np.ndarray) -> np.ndarray:
    """Numerically stable logistic transform."""
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x_arr, dtype=np.float64)
    positive = x_arr >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-x_arr[positive]))
    exp_x = np.exp(x_arr[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def logit(p: float | np.ndarray) -> np.ndarray:
    """Inverse logistic transform with open-interval clipping."""
    p_arr = np.asarray(p, dtype=np.float64)
    p_clip = np.clip(p_arr, 1e-12, 1.0 - 1e-12)
    return np.log(p_clip) - np.log1p(-p_clip)


def project_to_corr(R: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Project a finite square matrix to an SPD correlation matrix."""
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R must be a square matrix")
    if not np.all(np.isfinite(R)):
        raise ValueError("R must contain only finite values")

    sym = 0.5 * (R + R.T)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.maximum(vals, float(eps))
    out = vecs @ np.diag(vals) @ vecs.T
    diag = np.sqrt(np.maximum(np.diag(out), float(eps)))
    out = out / np.outer(diag, diag)
    out = 0.5 * (out + out.T)
    np.fill_diagonal(out, 1.0)
    validate_corr_matrix(out, eps=eps)
    return out


def preprocess_correlation_matrix(
        R: np.ndarray,
        *,
        source: str,
        eps: float = 1e-8,
        nonfinite_kendall_pairs=()) -> CorrelationPreprocessingResult:
    """Project a finite matrix and report whether SPD correction was needed."""
    input_correlation = np.asarray(R, dtype=np.float64)
    if (
            input_correlation.ndim != 2
            or input_correlation.shape[0] != input_correlation.shape[1]):
        raise ValueError("R must be a square matrix")
    if not np.all(np.isfinite(input_correlation)):
        raise ValueError("R must contain only finite values")

    symmetric = 0.5 * (input_correlation + input_correlation.T)
    min_before = float(np.min(np.linalg.eigvalsh(symmetric)))
    correlation = project_to_corr(input_correlation, eps=eps)
    min_after = float(np.min(np.linalg.eigvalsh(correlation)))
    projection_applied = (
        min_before <= float(eps)
        or not np.allclose(
            input_correlation,
            correlation,
            rtol=0.0,
            atol=10.0 * np.finfo(np.float64).eps,
        )
    )
    return CorrelationPreprocessingResult(
        correlation=correlation,
        input_correlation=input_correlation.copy(),
        source=str(source),
        projection_applied=bool(projection_applied),
        min_eigenvalue_before=min_before,
        min_eigenvalue_after=min_after,
        nonfinite_kendall_pairs=tuple(
            (int(i), int(j)) for i, j in nonfinite_kendall_pairs),
    )


def estimate_kendall_correlation(
        observations: np.ndarray,
        *,
        eps: float = 1e-8) -> CorrelationPreprocessingResult:
    """Estimate an SPD correlation matrix through pairwise Kendall tau.

    A non-finite pairwise statistic represents unavailable dependence
    information and is mapped to zero dependence while retaining unit
    diagonal.
    """
    observations = np.asarray(observations, dtype=np.float64)
    if observations.ndim != 2:
        raise ValueError(
            "observations must have shape (n_observations, dimension)")
    if observations.shape[1] < 2:
        raise ValueError("observations must contain at least two variables")

    d = observations.shape[1]
    correlation = np.eye(d, dtype=np.float64)
    nonfinite_pairs = []
    for i in range(d):
        for j in range(i + 1, d):
            kendall_result = kendalltau(
                observations[:, i],
                observations[:, j],
            )
            tau = float(
                kendall_result.statistic
                if hasattr(kendall_result, "statistic")
                else kendall_result[0]
            )
            if not np.isfinite(tau):
                tau = 0.0
                nonfinite_pairs.append((i, j))
            value = float(np.sin(0.5 * np.pi * tau))
            correlation[i, j] = value
            correlation[j, i] = value

    return preprocess_correlation_matrix(
        correlation,
        source="kendall",
        eps=eps,
        nonfinite_kendall_pairs=nonfinite_pairs,
    )


def validate_corr_matrix(R: np.ndarray, eps: float = 1e-8) -> None:
    """Validate that ``R`` is a finite SPD correlation matrix."""
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R must be a square matrix")
    if not np.all(np.isfinite(R)):
        raise ValueError("R must contain only finite values")
    if not np.allclose(R, R.T, atol=eps, rtol=0.0):
        raise ValueError("R must be symmetric")
    if not np.allclose(np.diag(R), 1.0, atol=eps, rtol=0.0):
        raise ValueError("R must have unit diagonal")
    try:
        np.linalg.cholesky(R)
    except np.linalg.LinAlgError as exc:
        raise ValueError("R must be positive definite") from exc


def _make_shrinkage_corr_from_validated(
        alpha_raw: float, R0: np.ndarray) -> np.ndarray:
    """Build a shrinkage correlation from an already validated base."""
    R0 = np.asarray(R0, dtype=np.float64)
    alpha = float(sigmoid(alpha_raw))
    out = alpha * R0 + (1.0 - alpha) * np.eye(R0.shape[0])
    out = 0.5 * (out + out.T)
    np.fill_diagonal(out, 1.0)
    return out


def make_shrinkage_corr(alpha_raw: float, R0: np.ndarray) -> np.ndarray:
    """Return ``alpha * R0 + (1 - alpha) * I`` with ``alpha=sigmoid(raw)``."""
    R0 = project_to_corr(R0)
    out = _make_shrinkage_corr_from_validated(alpha_raw, R0)
    validate_corr_matrix(out)
    return out


def cholesky_corr_n_params(d: int) -> int:
    """Number of lower off-diagonal entries for a ``d`` by ``d`` matrix."""
    d = int(d)
    if d < 2:
        raise ValueError("d must be >= 2")
    return d * (d - 1) // 2


def pack_cholesky_corr(R: np.ndarray) -> np.ndarray:
    """Pack row-major lower off-diagonal entries from a unit-diagonal Cholesky."""
    R = project_to_corr(R)
    L = np.linalg.cholesky(R)
    diag = np.diag(L)
    if np.any(diag <= 0.0):
        raise ValueError("Cholesky diagonal must be positive")
    L_unit = L / diag[:, None]
    d = R.shape[0]
    params = np.empty(cholesky_corr_n_params(d), dtype=np.float64)
    pos = 0
    for i in range(1, d):
        for j in range(i):
            params[pos] = L_unit[i, j]
            pos += 1
    return params


def _corr_from_cholesky_params(params: np.ndarray, d: int) -> np.ndarray:
    """Build a correlation matrix from finite Cholesky parameters."""
    d = int(d)
    expected = cholesky_corr_n_params(d)
    params = np.asarray(params, dtype=np.float64).reshape(-1)
    if params.size != expected:
        raise ValueError(
            f"expected {expected} Cholesky correlation parameters, "
            f"got {params.size}")
    if not np.all(np.isfinite(params)):
        raise ValueError("Cholesky correlation parameters must be finite")

    L = np.eye(d, dtype=np.float64)
    pos = 0
    for i in range(1, d):
        for j in range(i):
            L[i, j] = params[pos]
            pos += 1
    sigma = L @ L.T
    diag = np.sqrt(np.diag(sigma))
    R = sigma / np.outer(diag, diag)
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    return R


def unpack_cholesky_corr(params: np.ndarray, d: int) -> np.ndarray:
    """Unpack row-major lower off-diagonal entries to an SPD correlation."""
    R = _corr_from_cholesky_params(params, d)
    validate_corr_matrix(R)
    return R


def _corr_gradient_to_raw_params(
        corr_mode: str,
        params: np.ndarray,
        R: np.ndarray,
        corr_gradient: np.ndarray,
        corr_base: np.ndarray | None = None) -> np.ndarray:
    """Map derivatives over symmetric ``R[i, j]`` to raw parameters."""
    corr_mode = str(corr_mode).lower()
    params = np.asarray(params, dtype=np.float64).reshape(-1)
    R = np.asarray(R, dtype=np.float64)
    d = R.shape[0]
    expected = cholesky_corr_n_params(d)
    corr_gradient = np.asarray(
        corr_gradient, dtype=np.float64).reshape(-1)
    if R.shape != (d, d) or corr_gradient.size != expected:
        raise ValueError("correlation gradient shape does not match R")
    if not np.all(np.isfinite(corr_gradient)):
        raise ValueError("correlation gradient must contain only finite values")

    if corr_mode == "shrinkage":
        if params.size != 1 or corr_base is None:
            raise ValueError("shrinkage gradient requires one parameter and base")
        corr_base = np.asarray(corr_base, dtype=np.float64)
        if corr_base.shape != R.shape:
            raise ValueError("corr_base shape does not match R")
        alpha = float(sigmoid(params[0]))
        factor = alpha * (1.0 - alpha)
        gradient = 0.0
        pos = 0
        for i in range(1, d):
            for j in range(i):
                gradient += (
                    corr_gradient[pos] * factor * corr_base[i, j])
                pos += 1
        return np.array([gradient], dtype=np.float64)

    if corr_mode != "cholesky" or params.size != expected:
        raise ValueError("unsupported correlation gradient parameterization")

    A = np.eye(d, dtype=np.float64)
    pos = 0
    for i in range(1, d):
        for j in range(i):
            A[i, j] = params[pos]
            pos += 1
    sigma = A @ A.T
    sigma_diag = np.diag(sigma)
    scales = np.sqrt(sigma_diag)

    matrix_gradient = np.zeros((d, d), dtype=np.float64)
    pos = 0
    for i in range(1, d):
        for j in range(i):
            value = 0.5 * corr_gradient[pos]
            matrix_gradient[i, j] = value
            matrix_gradient[j, i] = value
            pos += 1

    sigma_gradient = matrix_gradient / np.outer(scales, scales)
    diagonal_correction = np.sum(matrix_gradient * R, axis=1) / sigma_diag
    sigma_gradient[np.diag_indices(d)] -= diagonal_correction
    A_gradient = 2.0 * sigma_gradient @ A

    raw_gradient = np.empty(expected, dtype=np.float64)
    pos = 0
    for i in range(1, d):
        for j in range(i):
            raw_gradient[pos] = A_gradient[i, j]
            pos += 1
    return raw_gradient
