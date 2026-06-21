"""Thin Python adapter for the compiled GAS evaluator.

This module owns input validation, Python/C++ object conversion, and status
translation. All GAS recursion, score, likelihood, prediction, and h-path
mathematics are implemented by ``GasEvaluator`` in the C++ extension.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyscarcopula.numerical import _cpp_copula, _cpp_extension
from pyscarcopula.numerical._cpp_extension import (
    CppError,
    CppUnavailable,
    CppUnsupported,
)


@dataclass(frozen=True)
class GasFilterOutput:
    g_path: np.ndarray
    r_path: np.ndarray
    score_path: np.ndarray
    log_likelihood: float


@dataclass(frozen=True)
class GasUpdateOutput:
    g_next: float
    r: float
    r_next: float
    log_likelihood: float
    score: float


@dataclass(frozen=True)
class GasStateOutput:
    g: float
    parameter: float


def available() -> bool:
    """Return whether the compiled GAS evaluator can be imported."""
    return _cpp_extension.available()


def require_available() -> None:
    """Raise ``CppUnavailable`` unless the compiled evaluator can load."""
    _cpp_extension.load()


def ensure_supported(copula) -> None:
    """Raise ``CppUnsupported`` for copulas outside the native GAS set."""
    _cpp_copula.ensure_supported_for_gas(copula)


def supported(copula) -> bool:
    """Return whether ``copula`` is supported and the extension is available."""
    return _cpp_copula.supported_for_gas(copula)


def _finite_float(value, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite float") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite float")
    return result


def _observations(
    u,
    copula,
    *,
    allow_single_row: bool = False,
) -> np.ndarray:
    try:
        obs = np.asarray(u, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("u must be convertible to a float64 array") from exc
    expected_dim = int(getattr(copula, "d", 2))
    if allow_single_row and obs.ndim == 1 and obs.shape == (expected_dim,):
        obs = obs.reshape(1, expected_dim)
    if obs.ndim != 2 or obs.shape[1] != expected_dim:
        raise ValueError(
            f"u must have shape (T, {expected_dim}), got {obs.shape}")
    if len(obs) == 0:
        raise ValueError("u must contain at least one observation")
    if np.any(~np.isfinite(obs)):
        raise ValueError("u must contain only finite values")
    return np.ascontiguousarray(obs)


def _scaling_name(scaling) -> str:
    value = str(scaling).lower()
    if value not in {"unit", "fisher"}:
        raise ValueError("scaling must be 'unit' or 'fisher'")
    return value


def _params(module, omega, gamma, beta):
    out = module.GasParams()
    out.omega = _finite_float(omega, "omega")
    out.gamma = _finite_float(gamma, "gamma")
    out.beta = _finite_float(beta, "beta")
    return out


def _config(
    module,
    scaling,
    score_eps,
    *,
    g_clip=50.0,
    score_clip=100.0,
    fisher_floor=1e-6,
    stationary_beta_tol=1e-8,
):
    scaling = _scaling_name(scaling)
    out = module.GasConfig()
    out.scaling = (
        module.GasScaling.Unit
        if scaling == "unit"
        else module.GasScaling.Fisher
    )
    out.score_eps = _finite_float(score_eps, "score_eps")
    out.g_clip = _finite_float(g_clip, "g_clip")
    out.score_clip = _finite_float(score_clip, "score_clip")
    out.fisher_floor = _finite_float(fisher_floor, "fisher_floor")
    out.stationary_beta_tol = _finite_float(
        stationary_beta_tol, "stationary_beta_tol")
    if out.score_eps <= 0.0:
        raise ValueError("score_eps must be positive")
    if out.g_clip <= 0.0:
        raise ValueError("g_clip must be positive")
    if out.score_clip <= 0.0:
        raise ValueError("score_clip must be positive")
    if out.fisher_floor <= 0.0:
        raise ValueError("fisher_floor must be positive")
    if not 0.0 <= out.stationary_beta_tol < 1.0:
        raise ValueError("stationary_beta_tol must be in [0, 1)")
    return out


def _inputs(omega, gamma, beta, u, copula, scaling, score_eps):
    module = _cpp_extension.load()
    obs = _observations(u, copula)
    scaling = _scaling_name(scaling)
    spec = _cpp_copula.make_gas_spec(
        module,
        copula,
        u=obs,
        use_student_cache=scaling == "unit",
    )
    return (
        module,
        _params(module, omega, gamma, beta),
        spec,
        obs,
        _config(module, scaling, score_eps),
    )


def _status_name(status: int) -> str:
    return {
        0: "ok",
        1: "null_pointer",
        2: "invalid_size",
        3: "invalid_family",
        4: "invalid_rotation",
        5: "invalid_transform",
        6: "invalid_parameter",
        7: "numerical_failure",
    }.get(int(status), "unknown")


def _raise_status(result, operation: str) -> None:
    status = int(result["status"])
    if status == 0:
        return
    failure_index = int(result.get("failure_index", -1))
    detail = (
        f"C++ GAS {operation} failed: status={status} "
        f"({_status_name(status)})"
    )
    if failure_index >= 0:
        detail += f", observation={failure_index}"
    if status in (2, 6):
        raise ValueError(detail)
    if status in (3, 4, 5):
        raise CppUnsupported(detail)
    if status == 7:
        raise FloatingPointError(detail)
    raise CppError(detail)


def filter_result(
    omega,
    gamma,
    beta,
    u,
    copula,
    scaling="unit",
    score_eps=1e-4,
) -> GasFilterOutput:
    """Run the compiled GAS filter and return paths, scores, and ``logL``."""
    module, params, spec, obs, config = _inputs(
        omega, gamma, beta, u, copula, scaling, score_eps)
    result = module.GasEvaluator().filter(
        params, spec, obs, config)
    _raise_status(result, "filter")
    output = GasFilterOutput(
        g_path=np.asarray(result["g_path"], dtype=np.float64),
        r_path=np.asarray(result["r_path"], dtype=np.float64),
        score_path=np.asarray(result["score_path"], dtype=np.float64),
        log_likelihood=float(result["log_likelihood"]),
    )
    if (
        np.any(~np.isfinite(output.g_path))
        or np.any(~np.isfinite(output.r_path))
        or np.any(~np.isfinite(output.score_path))
        or not np.isfinite(output.log_likelihood)
    ):
        raise FloatingPointError(
            "C++ GAS filter returned non-finite values with status=ok")
    return output


def initial_state(
    omega,
    gamma,
    beta,
    copula,
    scaling="unit",
    score_eps=1e-4,
) -> GasStateOutput:
    """Return the compiled initial GAS state and transformed parameter."""
    module = _cpp_extension.load()
    params = _params(module, omega, gamma, beta)
    spec = _cpp_copula.make_gas_spec(module, copula)
    config = _config(module, scaling, score_eps)
    result = module.GasEvaluator().initial_state(
        params, spec, config)
    _raise_status(result, "initial_state")
    output = GasStateOutput(
        g=float(result["g"]),
        parameter=float(result["parameter"]),
    )
    if not np.all(np.isfinite([output.g, output.parameter])):
        raise FloatingPointError(
            "C++ GAS initial_state returned non-finite values with status=ok")
    return output


def filter(
    omega,
    gamma,
    beta,
    u,
    copula,
    scaling="unit",
    score_eps=1e-4,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run the compiled GAS filter and return its tuple-shaped public result."""
    result = filter_result(
        omega, gamma, beta, u, copula, scaling, score_eps)
    return result.g_path, result.r_path, result.log_likelihood


def log_likelihood(
    omega,
    gamma,
    beta,
    u,
    copula,
    scaling="unit",
    score_eps=1e-4,
) -> float:
    """Evaluate ``logL`` without allocating full GAS trajectories."""
    module, params, spec, obs, config = _inputs(
        omega, gamma, beta, u, copula, scaling, score_eps)
    result = module.GasEvaluator().log_likelihood(
        params, spec, obs, config)
    _raise_status(result, "log_likelihood")
    value = float(result["log_likelihood"])
    if not np.isfinite(value):
        raise FloatingPointError(
            "C++ GAS log_likelihood returned a non-finite value")
    return value


def negative_log_likelihood(
    omega,
    gamma,
    beta,
    u,
    copula,
    scaling="unit",
    score_eps=1e-4,
) -> float:
    """Evaluate ``-logL`` without allocating full GAS trajectories."""
    module, params, spec, obs, config = _inputs(
        omega, gamma, beta, u, copula, scaling, score_eps)
    result = module.GasEvaluator().negative_log_likelihood(
        params, spec, obs, config)
    _raise_status(result, "negative_log_likelihood")
    value = float(result["log_likelihood"])
    if not np.isfinite(value):
        raise FloatingPointError(
            "C++ GAS negative_log_likelihood returned a non-finite value")
    return value


def update_one(
    omega,
    gamma,
    beta,
    g,
    observation,
    copula,
    scaling="unit",
    score_eps=1e-4,
) -> GasUpdateOutput:
    """Apply one compiled score update to one observation."""
    module = _cpp_extension.load()
    obs = _observations(
        observation, copula, allow_single_row=True)
    if len(obs) != 1:
        raise ValueError("observation must contain exactly one row")
    params = _params(module, omega, gamma, beta)
    scaling = _scaling_name(scaling)
    spec = _cpp_copula.make_gas_spec(
        module,
        copula,
        u=obs,
        use_student_cache=scaling == "unit",
    )
    config = _config(module, scaling, score_eps)
    result = module.GasEvaluator().update_observation(
        params,
        spec,
        _finite_float(g, "g"),
        obs,
        config,
    )
    _raise_status(result, "update_one")
    output = GasUpdateOutput(
        g_next=float(result["g_next"]),
        r=float(result["r"]),
        r_next=float(result["r_next"]),
        log_likelihood=float(result["log_likelihood"]),
        score=float(result["score"]),
    )
    if not np.all(np.isfinite([
        output.g_next,
        output.r,
        output.r_next,
        output.log_likelihood,
        output.score,
    ])):
        raise FloatingPointError(
            "C++ GAS update_one returned non-finite values with status=ok")
    return output


def predict_parameter(
    omega,
    gamma,
    beta,
    u,
    copula,
    scaling="unit",
    score_eps=1e-4,
    horizon="next",
) -> float:
    """Return the compiled GAS parameter for ``current`` or ``next`` timing."""
    module, params, spec, obs, config = _inputs(
        omega, gamma, beta, u, copula, scaling, score_eps)
    if horizon in (0, "0"):
        horizon = "current"
    elif horizon in (1, "1"):
        horizon = "next"
    else:
        horizon = str(horizon).lower()
    if horizon not in {"current", "next"}:
        raise ValueError("horizon must be 'current' or 'next'")
    result = module.GasEvaluator().predict_parameter(
        params, spec, obs, config, horizon == "next")
    _raise_status(result, "predict_parameter")
    value = float(result["parameter"])
    if not np.isfinite(value):
        raise FloatingPointError(
            "C++ GAS predict_parameter returned a non-finite value")
    return value


def h_path(
    omega,
    gamma,
    beta,
    u,
    copula,
    scaling="unit",
    score_eps=1e-4,
) -> np.ndarray:
    """Evaluate ``h(u2 | u1; Psi(g_t))`` using the compiled GAS path."""
    module, params, spec, obs, config = _inputs(
        omega, gamma, beta, u, copula, scaling, score_eps)
    result = module.GasEvaluator().h_path(
        params, spec, obs, config)
    _raise_status(result, "h_path")
    values = np.asarray(result["values"], dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise FloatingPointError(
            "C++ GAS h_path returned non-finite values with status=ok")
    return values


__all__ = [
    "CppError",
    "CppUnavailable",
    "CppUnsupported",
    "GasFilterOutput",
    "GasStateOutput",
    "GasUpdateOutput",
    "available",
    "require_available",
    "ensure_supported",
    "supported",
    "filter",
    "filter_result",
    "initial_state",
    "log_likelihood",
    "negative_log_likelihood",
    "update_one",
    "predict_parameter",
    "h_path",
]
