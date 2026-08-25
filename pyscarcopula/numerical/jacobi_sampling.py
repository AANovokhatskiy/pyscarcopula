"""Continuous-path sampling helpers for the Jacobi latent process."""

from __future__ import annotations

import numpy as np

from pyscarcopula._native import jacobi as jacobi_native

from pyscarcopula.numerical._arrays import (
    validate_float64_allocation,
    validate_positive_int,
)
from pyscarcopula.numerical.jacobi_tm import (
    DEFAULT_JACOBI_MEMORY_BUDGET_BYTES,
    _jacobi_stationary_shape,
    _validate_nonnegative_int,
)


_LAMPERTI_BOUNDARIES = frozenset({"reflect", "clip"})
_LAMPERTI_ENGINES = frozenset({"native", "numba", "python"})
DEFAULT_LAMPERTI_CHUNK_OBSERVATIONS = 4096


def normalize_jacobi_sampling_method(value) -> str:
    """Return a supported unconditional Jacobi sampling backend."""
    if not isinstance(value, str):
        raise TypeError("sampling_method must be a string")
    method = value.strip().lower()
    if method not in {"tm_grid", "lamperti_euler"}:
        raise ValueError(
            "sampling_method must be 'tm_grid' or 'lamperti_euler'")
    return method


def normalize_lamperti_boundary(value) -> str:
    """Return a supported Lamperti boundary policy."""
    if not isinstance(value, str):
        raise TypeError("lamperti_boundary must be a string")
    boundary = value.strip().lower()
    if boundary not in _LAMPERTI_BOUNDARIES:
        raise ValueError(
            "lamperti_boundary must be 'reflect' or 'clip'")
    return boundary


def normalize_lamperti_engine(value) -> str:
    """Normalize legacy engine labels to the mandatory native engine."""
    if not isinstance(value, str):
        raise TypeError("lamperti_engine must be a string")
    engine = value.strip().lower()
    if engine not in _LAMPERTI_ENGINES:
        raise ValueError("lamperti_engine must be 'native'")
    return "native"


def validate_lamperti_eps(value) -> float:
    """Validate the interior epsilon used only for drift evaluation."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("lamperti_eps must be a finite real number")
    try:
        eps = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "lamperti_eps must be a finite real number") from exc
    if not np.isfinite(eps):
        raise ValueError("lamperti_eps must be finite")
    if not (0.0 < eps < 0.5):
        raise ValueError("lamperti_eps must be in (0, 0.5)")
    return eps


def _effective_chunk_observations(
        *, n, substeps, requested, memory_budget_bytes):
    if n <= 1:
        validate_float64_allocation(
            (n,),
            name="Lamperti-Euler Jacobi path",
            memory_budget_bytes=memory_budget_bytes,
        )
        return 0

    # One chunk keeps the Python path, Python and C++ innovation buffers, and
    # the C++ and NumPy tau result buffers live at the pybind boundary.
    elements_per_interval = 2 * substeps + 2
    # The smallest executable chunk contains one full observation interval.
    validate_float64_allocation(
        (n + elements_per_interval,),
        name="Lamperti-Euler native chunk peak",
        memory_budget_bytes=memory_budget_bytes,
    )
    max_elements = memory_budget_bytes // np.dtype(np.float64).itemsize
    max_chunk = (max_elements - n) // elements_per_interval
    effective = min(requested, n - 1, max_chunk)
    validate_float64_allocation(
        (n + effective * elements_per_interval,),
        name="Lamperti-Euler native chunk peak",
        memory_budget_bytes=memory_budget_bytes,
    )
    return int(effective)


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
    """Sample a Jacobi path with a substepped Lamperti--Euler scheme.

    The observation horizon is one, matching the Jacobi transition-matrix
    backend: adjacent observations are separated by ``1 / (n - 1)``.  This is
    an approximate SDE sampler and is intentionally independent of the
    likelihood transition matrix.
    """
    n = _validate_nonnegative_int(n, "n")
    substeps = validate_positive_int(substeps, "lamperti_substeps")
    boundary = normalize_lamperti_boundary(boundary)
    eps = validate_lamperti_eps(eps)
    engine = normalize_lamperti_engine(engine)
    chunk_observations = validate_positive_int(
        chunk_observations, "lamperti_chunk_observations")
    shapes = _jacobi_stationary_shape(kappa, m, xi)
    if shapes is None:
        raise ValueError("invalid Jacobi parameters")
    kappa = float(kappa)
    m = float(m)
    xi = float(xi)
    alpha, beta = shapes

    if memory_budget_bytes is None:
        memory_budget_bytes = DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
    # Validate the budget type even for n=0 and select the largest safe chunk
    # before the RNG is advanced.
    validate_float64_allocation(
        (0,),
        name="Lamperti-Euler memory budget",
        memory_budget_bytes=memory_budget_bytes,
    )
    effective_chunk = _effective_chunk_observations(
        n=n,
        substeps=substeps,
        requested=chunk_observations,
        memory_budget_bytes=int(memory_budget_bytes),
    )
    diagnostics = {
        "sampling_method": "lamperti_euler",
        "sampling_engine": engine,
        "boundary_policy": boundary,
        "substeps": substeps,
        "drift_eps": eps,
        "stationary_alpha": alpha,
        "stationary_beta": beta,
        "stationary_boundary_singular": bool(
            alpha < 1.0 or beta < 1.0),
        "chunk_observations_requested": chunk_observations,
        "chunk_observations": effective_chunk,
        "n": n,
        "boundary_interventions": 0,
        "boundary_intervention_rate": 0.0,
        "euler_steps": max(n - 1, 0) * substeps,
    }
    if n == 0:
        path = np.empty(0, dtype=np.float64)
        return (path, diagnostics) if return_diagnostics else path
    if rng is None:
        rng = np.random.default_rng()

    path = np.empty(n, dtype=np.float64)
    path[0] = float(rng.beta(alpha, beta))
    if n == 1:
        return (path, diagnostics) if return_diagnostics else path

    y = float(jacobi_native.lamperti(
        np.array([path[0]], dtype=np.float64), xi)[0])
    interventions = 0
    output_offset = 1
    while output_offset < n:
        block = min(effective_chunk, n - output_offset)
        innovations = np.asarray(
            rng.standard_normal((block, substeps)),
            dtype=np.float64,
        )
        native = jacobi_native.sample_lamperti_chunk_fixed_draws(
            kappa,
            m,
            xi,
            y,
            innovations,
            n_obs=n,
            substeps=substeps,
            boundary=boundary,
            interior_eps=eps,
        )
        used = int(native["normal_draws_used"])
        if used != block * substeps:
            raise RuntimeError(
                "native Lamperti sampler returned an invalid draw count")
        path[output_offset:output_offset + block] = native["tau"]
        y = float(native["final_lamperti_value"])
        interventions += int(native["boundary_interventions"])
        output_offset += block

    diagnostics["boundary_interventions"] = interventions
    diagnostics["boundary_intervention_rate"] = (
        interventions / diagnostics["euler_steps"])
    if return_diagnostics:
        return path, diagnostics
    return path


__all__ = [
    "normalize_jacobi_sampling_method",
    "normalize_lamperti_boundary",
    "normalize_lamperti_engine",
    "sample_jacobi_lamperti_trajectory",
    "validate_lamperti_eps",
]
