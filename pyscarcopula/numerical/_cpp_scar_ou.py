"""SCAR-OU adapters for the optional C++ extension.

The public package uses this module only when ``backend='cpp'`` or
``backend='auto'`` selects a supported C++ path.  The extension implements the
SCAR-TM-OU likelihood, analytical gradient, selected grid forward passes, and
pointwise copula h/h_inverse kernels.

Supported SCAR-TM-OU likelihood families are Clayton, Gumbel, Frank, and Joe
with the softplus transform, plus the bivariate Gaussian copula.  Frank is
available only without rotation for SCAR-TM-OU.  Grid forward calls use
``transition_method='auto'``, ``'matrix'``, or ``'local'``; direct
``'spectral'`` forward/state calls are intentionally not implemented because
they require a grid posterior state.
"""

from __future__ import annotations

import numpy as np

from pyscarcopula.numerical.auto_tm import AutoTMConfig, select_auto_backend
from pyscarcopula.numerical._arrays import as_float64_array
from pyscarcopula.numerical._transition_methods import (
    normalize_ou_transition_method,
)
from pyscarcopula.numerical import _cpp_copula, _cpp_extension
from pyscarcopula.numerical._cpp_extension import (
    CppError,
    CppUnavailable,
    CppUnsupported,
)


def available() -> bool:
    """Return whether the compiled extension can be imported."""
    return _cpp_extension.available()


def ensure_supported(copula) -> None:
    """Raise CppUnsupported if ``copula`` cannot use C++ SCAR-TM-OU kernels."""
    _cpp_copula.ensure_supported_for_scar_ou(copula)


def supported(copula) -> bool:
    """Return whether C++ SCAR-TM-OU kernels support ``copula``."""
    return _cpp_copula.supported_for_scar_ou(copula)


def supported_copula_ops(copula) -> bool:
    """Return whether C++ pointwise h/h_inverse kernels support ``copula``."""
    return _cpp_copula.supported_for_copula_ops(copula)


def _pair_and_r(first, second, r):
    first = np.atleast_1d(np.asarray(first, dtype=np.float64)).reshape(-1)
    second = np.atleast_1d(np.asarray(second, dtype=np.float64)).reshape(-1)
    r = np.atleast_1d(np.asarray(r, dtype=np.float64)).reshape(-1)
    sizes = [arr.size for arr in (first, second, r)]
    n = max(sizes)
    values = []
    for arr, name in ((first, "first"), (second, "second"), (r, "r")):
        if arr.size == n:
            values.append(arr)
        elif arr.size == 1:
            values.append(np.full(n, float(arr[0]), dtype=np.float64))
        else:
            raise ValueError(
                f"{name} must have size 1 or match the broadcast size {n}"
            )
    pair = np.column_stack((values[0], values[1])).astype(np.float64, copy=False)
    return np.ascontiguousarray(pair), np.ascontiguousarray(values[2])


def copula_h(copula, u_conditioned, u_given, r) -> np.ndarray:
    """Evaluate ``h(u_conditioned | u_given)`` with C++ copula kernels.

    Inputs are broadcast as one-dimensional arrays.  The pybind boundary
    rejects non-finite values before entering the C++ numerical kernels.
    """
    module = _cpp_extension.load()
    pair, r_arr = _pair_and_r(u_conditioned, u_given, r)
    spec = _cpp_copula.make_copula_ops_spec(module, copula)
    values = np.asarray(module.copula_h(spec, pair, r_arr), dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise CppError("C++ copula_h returned non-finite values")
    return values


def copula_h_inverse(copula, q, u_given, r) -> np.ndarray:
    """Evaluate ``h^{-1}(q | u_given)`` with C++ copula kernels.

    Inputs are broadcast as one-dimensional arrays.  Non-finite inputs and
    non-finite C++ results are converted to Python exceptions.
    """
    module = _cpp_extension.load()
    pair, r_arr = _pair_and_r(q, u_given, r)
    spec = _cpp_copula.make_copula_ops_spec(module, copula)
    values = np.asarray(
        module.copula_h_inverse(spec, pair, r_arr), dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise CppError("C++ copula_h_inverse returned non-finite values")
    return values


def _params(module, kappa, mu, nu):
    params = module.OuParams()
    params.kappa = float(kappa)
    params.mu = float(mu)
    params.nu = float(nu)
    return params


def _config(module, cfg: AutoTMConfig):
    out = module.OuNumericalConfig()
    out.K = int(cfg.K)
    out.grid_range = float(cfg.grid_range)
    out.adaptive = bool(cfg.adaptive)
    out.pts_per_sigma = int(cfg.pts_per_sigma)
    out.max_K = 0 if cfg.max_K is None else int(cfg.max_K)
    out.r_gh = float(cfg.r_gh)
    out.gh_order = int(cfg.gh_order)
    out.auto_small_kdt = float(cfg.small_kdt)
    out.spectral_basis_order = int(cfg.basis_order)
    out.spectral_quad_order = 0 if cfg.quad_order is None else int(cfg.quad_order)
    return out


def _inputs(kappa, mu, nu, u, copula, config):
    module = _cpp_extension.load()
    cfg = config or AutoTMConfig()
    obs = as_float64_array(u)
    return (
        module,
        _params(module, kappa, mu, nu),
        _cpp_copula.make_spec(module, copula),
        obs,
        _config(module, cfg),
        normalize_ou_transition_method(cfg.transition_method),
    )


def _call_loglik(evaluator, method, params, spec, u, config):
    if method == "auto":
        return evaluator.loglik_auto(params, spec, u, config)
    if method == "spectral":
        return evaluator.loglik_spectral(params, spec, u, config)
    if method == "local":
        return evaluator.loglik_local_gh(params, spec, u, config)
    if method == "matrix":
        return evaluator.loglik_matrix(params, spec, u, config)
    raise ValueError(f"Unsupported transition_method: {method}")


def _kappa_dt(kappa: float, n_obs: int) -> float:
    if n_obs <= 1:
        return float(kappa)
    return float(kappa) / float(n_obs - 1)


def _result_info(result, method: str, kappa, n_obs: int,
                 cfg: AutoTMConfig) -> dict:
    backend = _backend_name(result["backend"])
    info = {
        "backend": backend,
        "status": int(result["status"]),
        "transition_method": method,
        "engine": "cpp",
        "kappa_dt": _kappa_dt(kappa, n_obs),
        "n_obs": int(n_obs),
        "basis_order": int(cfg.basis_order),
    }
    fallback_chain = [
        _backend_name(value)
        for value in result.get("fallback_chain", [])
    ]
    if fallback_chain:
        info["fallback_chain"] = fallback_chain

    fallback_from = int(result.get("fallback_from", -1))
    if fallback_from >= 0:
        info["fallback_from"] = _backend_name(fallback_from)

    matrix_reason = _matrix_fallback_reason_name(
        result.get("matrix_fallback_reason", 0))
    if matrix_reason is not None:
        info["matrix_fallback_reason"] = matrix_reason

    if method == "auto" and not fallback_chain:
        selected = select_auto_backend(float(kappa), n_obs, cfg)
        info["selected_backend"] = selected
        if selected == "spectral" and backend in {"matrix", "local"}:
            info["fallback_from"] = "spectral"
            info["fallback_chain"] = ["spectral"]
        if selected == "spectral" and backend == "local":
            info["fallback_from"] = "matrix"
            info.setdefault("fallback_chain", []).append("matrix")
            info["matrix_fallback_reason"] = "unknown"
    elif method == "auto":
        info["selected_backend"] = select_auto_backend(float(kappa), n_obs, cfg)
    return info


def loglik(kappa, mu, nu, u, copula,
           config: AutoTMConfig | None = None) -> tuple[float, dict]:
    """Evaluate SCAR-TM-OU log-likelihood using the C++ backend.

    ``config.transition_method`` may be ``'auto'``, ``'spectral'``,
    ``'matrix'``, or ``'local'``.  In ``'auto'`` mode the C++ path mirrors the
    Python fallback order: spectral, then matrix, then local GH when needed.
    Non-zero C++ status codes are raised as :class:`CppError`.
    """
    cfg_py = config or AutoTMConfig()
    module, params, spec, obs, cfg, method = _inputs(
        kappa, mu, nu, u, copula, config)
    result = _call_loglik(
        module.ScarOuEvaluator(), method, params, spec, obs, cfg)
    info = _result_info(result, method, kappa, len(obs), cfg_py)
    if info["status"] != 0:
        raise CppError(
            "C++ SCAR-OU loglik failed: "
            f"status={info['status']} ({_status_name(info['status'])})"
        )
    return float(result["log_likelihood"]), info


def neg_loglik(kappa, mu, nu, u, copula,
               config: AutoTMConfig | None = None) -> float:
    """Evaluate negative SCAR-TM-OU log-likelihood with C++ kernels."""
    value, _ = neg_loglik_info(kappa, mu, nu, u, copula, config)
    return value


def neg_loglik_info(kappa, mu, nu, u, copula,
                    config: AutoTMConfig | None = None):
    """Evaluate negative log-likelihood and return C++ backend diagnostics."""
    value, info = loglik(kappa, mu, nu, u, copula, config)
    if not np.isfinite(value):
        return 1e10, info
    return -float(value), info


def neg_loglik_with_grad(kappa, mu, nu, u, copula,
                         config: AutoTMConfig | None = None):
    """Evaluate negative log-likelihood and analytical gradient in C++.

    The returned gradient is with respect to ``(kappa, mu, nu)`` and follows
    the same sign convention as the Python optimizer objective.
    """
    value, grad, _ = neg_loglik_with_grad_info(
        kappa, mu, nu, u, copula, config)
    return value, grad


def neg_loglik_with_grad_info(kappa, mu, nu, u, copula,
                              config: AutoTMConfig | None = None):
    """Evaluate negative log-likelihood, gradient, and C++ diagnostics."""
    cfg_py = config or AutoTMConfig()
    module, params, spec, obs, cfg, method = _inputs(
        kappa, mu, nu, u, copula, config)
    evaluator = module.ScarOuEvaluator()
    if method == "auto":
        result = evaluator.neg_loglik_with_grad_auto(params, spec, obs, cfg)
    elif method == "spectral":
        result = evaluator.neg_loglik_with_grad_spectral(params, spec, obs, cfg)
    elif method == "local":
        result = evaluator.neg_loglik_with_grad_local_gh(params, spec, obs, cfg)
    elif method == "matrix":
        result = evaluator.neg_loglik_with_grad_matrix(params, spec, obs, cfg)
    else:
        raise ValueError(f"Unsupported transition_method: {method}")

    info = _result_info(result, method, kappa, len(obs), cfg_py)
    status = info["status"]
    if status != 0:
        raise CppError(
            "C++ SCAR-OU gradient failed: "
            f"status={status} ({_status_name(status)}), "
            f"backend={info['backend']}"
        )
    return (
        float(result["neg_log_likelihood"]),
        np.asarray(result["neg_gradient"], dtype=np.float64),
        info,
    )


def _call_vector(evaluator, prefix, method, params, spec, u, config):
    if method == "auto":
        return getattr(evaluator, f"{prefix}_auto")(params, spec, u, config)
    if method == "local":
        return getattr(evaluator, f"{prefix}_local_gh")(params, spec, u, config)
    if method == "matrix":
        return getattr(evaluator, f"{prefix}_matrix")(params, spec, u, config)
    if method == "spectral":
        raise CppUnsupported(
            f"C++ {prefix} does not support transition_method='spectral'"
        )
    raise ValueError(f"Unsupported transition_method: {method}")


def _vector_result(result):
    status = int(result["status"])
    if status != 0:
        raise CppError(
            "C++ SCAR-OU forward call failed: "
            f"status={status} ({_status_name(status)})"
        )
    return np.asarray(result["values"], dtype=np.float64)


def predictive_mean(kappa, mu, nu, u, copula,
                    config: AutoTMConfig | None = None) -> np.ndarray:
    """Return the grid-filtered predictive mean of the copula parameter.

    Supported transition methods are ``'auto'``, ``'matrix'``, and ``'local'``.
    Use a grid method for forward quantities even if the fitted likelihood used
    the Hermite spectral backend.
    """
    module, params, spec, obs, cfg, method = _inputs(
        kappa, mu, nu, u, copula, config)
    result = _call_vector(
        module.ScarOuEvaluator(), "predictive_mean",
        method, params, spec, obs, cfg)
    return _vector_result(result)


def mixture_h(kappa, mu, nu, u, copula,
              config: AutoTMConfig | None = None) -> np.ndarray:
    """Return the SCAR-TM mixture h-function from C++ grid filtering.

    Supported transition methods are ``'auto'``, ``'matrix'``, and ``'local'``.
    The output is clipped to the same open-unit interval guard as the Python
    path.
    """
    module, params, spec, obs, cfg, method = _inputs(
        kappa, mu, nu, u, copula, config)
    result = _call_vector(
        module.ScarOuEvaluator(), "mixture_h",
        method, params, spec, obs, cfg)
    return np.clip(_vector_result(result), 1e-6, 1.0 - 1e-6)


def state_distribution(kappa, mu, nu, u, copula,
                       config: AutoTMConfig | None = None,
                       horizon: str = "current") -> tuple[np.ndarray, np.ndarray]:
    """Return the C++ grid posterior or one-step-ahead state distribution.

    ``horizon='current'`` returns the posterior state after the observations;
    ``horizon='next'`` advances it one transition step.  The C++ state path is
    grid-based and therefore does not implement direct ``'spectral'`` mode.
    """
    horizon = str(horizon).lower()
    if horizon not in ("current", "next"):
        raise ValueError("horizon must be 'current' or 'next'")

    module, params, spec, obs, cfg, method = _inputs(
        kappa, mu, nu, u, copula, config)
    if method == "auto":
        result = module.ScarOuEvaluator().state_distribution_auto(
            params, spec, obs, cfg, horizon == "next")
    elif method == "local":
        result = module.ScarOuEvaluator().state_distribution_local_gh(
            params, spec, obs, cfg, horizon == "next")
    elif method == "matrix":
        result = module.ScarOuEvaluator().state_distribution_matrix(
            params, spec, obs, cfg, horizon == "next")
    elif method == "spectral":
        raise CppUnsupported(
            "C++ state_distribution does not support transition_method='spectral'"
        )
    else:
        raise ValueError(f"Unsupported transition_method: {method}")

    status = int(result["status"])
    if status != 0:
        raise CppError(
            "C++ SCAR-OU state_distribution failed: "
            f"status={status} ({_status_name(status)})")
    return (
        np.asarray(result["z_grid"], dtype=np.float64),
        np.asarray(result["prob"], dtype=np.float64),
    )


def _backend_name(value: int) -> str:
    return {
        0: "spectral",
        1: "local",
        2: "matrix",
    }.get(int(value), "unknown")


def _status_name(value: int) -> str:
    return {
        0: "ok",
        1: "null_pointer",
        2: "invalid_size",
        3: "invalid_family",
        4: "invalid_rotation",
        5: "invalid_transform",
        6: "invalid_parameter",
        7: "numerical_failure",
    }.get(int(value), "unknown")


def _matrix_fallback_reason_name(value: int):
    return {
        0: None,
        1: "failed",
        2: "capped",
    }.get(int(value), "unknown")
