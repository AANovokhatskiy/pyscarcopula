"""Native copula density evaluation for SCAR-MC trajectory blocks."""

from __future__ import annotations

import numpy as np

from pyscarcopula.numerical import _cpp_copula, _cpp_extension
from pyscarcopula.numerical._cpp_extension import CppError


def supported(copula) -> bool:
    return _cpp_copula.supported_for_mc(copula)


def _log_pdf_trajectory_grid_result(
        copula, u, latent_paths, *, n_threads=1):
    observations = np.ascontiguousarray(np.asarray(u, dtype=np.float64))
    paths = np.ascontiguousarray(
        np.asarray(latent_paths, dtype=np.float64))
    expected_d = int(getattr(copula, "d", 2))
    if observations.ndim != 2 or observations.shape[1] != expected_d:
        raise ValueError(
            f"u must have shape (T, {expected_d}), got {observations.shape}")
    if paths.ndim != 2 or paths.shape[0] != len(observations):
        raise ValueError(
            "latent_paths must have shape (T, n_trajectories)")
    if paths.shape[1] == 0:
        raise ValueError("latent_paths must contain at least one trajectory")
    if not np.all(np.isfinite(observations)):
        raise ValueError("u must contain only finite values")
    if not np.all(np.isfinite(paths)):
        raise ValueError("latent_paths must contain only finite values")

    module = _cpp_extension.load()
    spec = _cpp_copula.make_mc_spec(module, copula, u=observations)
    from pyscarcopula.numerical.multivariate_native import (
        _validated_n_threads,
    )
    result = dict(module.copula_log_pdf_trajectory_grid(
        spec, observations, paths, _validated_n_threads(n_threads)))
    if result["status"] != module.SCAR_OK:
        raise CppError(
            "C++ SCAR-MC trajectory density failed with "
            f"status={result['status']}, "
            f"failure_index={result['failure_index']}")
    values = np.asarray(result["log_pdf"], dtype=np.float64)
    if values.shape != paths.shape or np.any(~np.isfinite(values)):
        raise CppError("C++ SCAR-MC trajectory density returned invalid values")
    return result, values


def log_pdf_trajectory_grid(
        copula, u, latent_paths, *, n_threads=1) -> np.ndarray:
    """Evaluate fixed trajectories without exposing kernel diagnostics."""
    return _log_pdf_trajectory_grid_result(
        copula, u, latent_paths, n_threads=n_threads)[1]


def log_pdf_trajectory_grid_info(
        copula, u, latent_paths, *, n_threads=1):
    """Return trajectory values and read-only parallel diagnostics."""
    result, values = _log_pdf_trajectory_grid_result(
        copula, u, latent_paths, n_threads=n_threads)
    diagnostics = {
        "n_threads_requested": int(result.get("n_threads_requested", 1)),
        "parallel_blocks": int(result.get("parallel_blocks", 0)),
    }
    return values, diagnostics
