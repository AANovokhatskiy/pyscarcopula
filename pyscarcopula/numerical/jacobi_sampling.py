"""Compatibility facade for native Jacobi trajectory sampling."""

from __future__ import annotations

from pyscarcopula._native import jacobi as jacobi_native


DEFAULT_LAMPERTI_CHUNK_OBSERVATIONS = (
    jacobi_native.DEFAULT_LAMPERTI_CHUNK_OBSERVATIONS)


def normalize_jacobi_sampling_method(value) -> str:
    return jacobi_native.normalize_sampling_method(value)


def normalize_lamperti_boundary(value) -> str:
    return jacobi_native.normalize_lamperti_boundary(value)


def normalize_lamperti_engine(value) -> str:
    return jacobi_native.normalize_lamperti_engine(value)


def validate_lamperti_eps(value) -> float:
    return jacobi_native.validate_lamperti_eps(value)


def sample_jacobi_lamperti_trajectory(
        kappa,
        m,
        xi,
        n,
        *,
        rng=None,
        substeps=8,
        boundary="reflect",
        eps=1e-10,
        engine="native",
        chunk_observations=DEFAULT_LAMPERTI_CHUNK_OBSERVATIONS,
        memory_budget_bytes=None,
        return_diagnostics=False):
    """Preserve the public API while `_native` owns the sampling facade."""
    result = jacobi_native.sample_lamperti_trajectory(
        kappa,
        m,
        xi,
        n,
        rng=rng,
        substeps=substeps,
        boundary=boundary,
        eps=eps,
        engine=engine,
        chunk_observations=chunk_observations,
        memory_budget_bytes=memory_budget_bytes,
        return_diagnostics=return_diagnostics,
    )
    if not return_diagnostics:
        return result
    path, diagnostics = result
    requested_engine = str(engine).strip().lower()
    if requested_engine in {"python", "numba"}:
        diagnostics = dict(diagnostics)
        diagnostics["sampling_engine"] = requested_engine
    return path, diagnostics


__all__ = [
    "normalize_jacobi_sampling_method",
    "normalize_lamperti_boundary",
    "normalize_lamperti_engine",
    "sample_jacobi_lamperti_trajectory",
    "validate_lamperti_eps",
]
