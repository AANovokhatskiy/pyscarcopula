"""Static correlation parameterizations for multivariate copulas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
    """Numerically stable native logistic transform."""
    from pyscarcopula._native import multivariate as multivariate_native
    return multivariate_native.correlation_logistic(x)


def logit(p: float | np.ndarray) -> np.ndarray:
    """Native inverse logistic transform with open-interval clipping."""
    from pyscarcopula._native import multivariate as multivariate_native
    return multivariate_native.correlation_logit(p)


def project_to_corr(R: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Project a finite square matrix to an SPD correlation matrix in C++."""
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R must be a square matrix")
    if not np.all(np.isfinite(R)):
        raise ValueError("R must contain only finite values")

    from pyscarcopula._native import multivariate as multivariate_native
    correlation, _ = multivariate_native.preprocess_correlation(
        R, eigenvalue_floor=eps)
    return correlation


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

    from pyscarcopula._native import multivariate as multivariate_native
    correlation, native = multivariate_native.preprocess_correlation(
        input_correlation, eigenvalue_floor=eps)
    return CorrelationPreprocessingResult(
        correlation=correlation,
        input_correlation=native["input_correlation"],
        source=str(source),
        projection_applied=native["projection_applied"],
        min_eigenvalue_before=native["min_eigenvalue_before"],
        min_eigenvalue_after=native["min_eigenvalue_after"],
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

    from pyscarcopula._native import multivariate as multivariate_native
    correlation, native = multivariate_native.estimate_kendall_correlation(
        observations, eigenvalue_floor=eps)
    return CorrelationPreprocessingResult(
        correlation=correlation,
        input_correlation=native["input_correlation"],
        source="kendall",
        projection_applied=native["projection_applied"],
        min_eigenvalue_before=native["min_eigenvalue_before"],
        min_eigenvalue_after=native["min_eigenvalue_after"],
        nonfinite_kendall_pairs=native["nonfinite_kendall_pairs"],
    )


def validate_corr_matrix(R: np.ndarray, eps: float = 1e-8) -> None:
    """Validate a finite SPD correlation matrix in the native kernel."""
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R must be a square matrix")
    if not np.all(np.isfinite(R)):
        raise ValueError("R must contain only finite values")
    from pyscarcopula._native import multivariate as multivariate_native
    multivariate_native.validate_correlation(R, tolerance=eps)


def _make_shrinkage_corr_from_validated(
        alpha_raw: float, R0: np.ndarray) -> np.ndarray:
    """Build a shrinkage correlation from a validated base in C++."""
    R0 = np.asarray(R0, dtype=np.float64)
    from pyscarcopula._native import multivariate as multivariate_native
    return multivariate_native.make_shrinkage_correlation(alpha_raw, R0)


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
    """Pack native row-major unit-diagonal Cholesky coordinates."""
    from pyscarcopula._native import multivariate as multivariate_native
    return multivariate_native.pack_cholesky_correlation(R)


def _corr_from_cholesky_params(params: np.ndarray, d: int) -> np.ndarray:
    """Build a correlation matrix from native Cholesky parameters."""
    d = int(d)
    expected = cholesky_corr_n_params(d)
    params = np.asarray(params, dtype=np.float64).reshape(-1)
    if params.size != expected:
        raise ValueError(
            f"expected {expected} Cholesky correlation parameters, "
            f"got {params.size}")
    if not np.all(np.isfinite(params)):
        raise ValueError("Cholesky correlation parameters must be finite")

    from pyscarcopula._native import multivariate as multivariate_native
    return multivariate_native.unpack_cholesky_correlation(params, d)


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
    from pyscarcopula._native import multivariate as multivariate_native
    return multivariate_native.correlation_gradient_to_raw(
        corr_mode,
        np.asarray(params, dtype=np.float64).reshape(-1),
        R,
        np.asarray(corr_gradient, dtype=np.float64).reshape(-1),
        corr_base,
    )


def _shrinkage_raw_corr_direction(
        params: np.ndarray,
        corr_base: np.ndarray) -> np.ndarray:
    """Return lower-triangle ``dR/draw`` for shrinkage correlation."""
    from pyscarcopula._native import multivariate as multivariate_native
    return multivariate_native.shrinkage_raw_correlation_direction(
        np.asarray(params, dtype=np.float64).reshape(-1), corr_base)
