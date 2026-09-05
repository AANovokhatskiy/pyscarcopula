"""SCAR-OU adapters for the bundled C++ extension.

The extension is the production numerical engine for SCAR-TM-OU likelihood,
gradient, grid-forward/state operations, and pointwise copula operations.
When a likelihood uses the spectral transition, posterior quantities are
reconstructed explicitly on a native grid.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pyscarcopula._utils import clip_h_function_values
from pyscarcopula.numerical._scar_ou_config import (
    AutoTMConfig,
    normalize_grid_method,
    select_auto_backend,
    validate_cpp_config,
)
from pyscarcopula.numerical._arrays import (
    as_float64_array,
    as_float64_scalar,
    as_pseudo_observation_array,
)
from pyscarcopula.numerical._transition_methods import (
    normalize_ou_transition_method,
)
from pyscarcopula._native import _descriptors, _extension, pair
from pyscarcopula._native.errors import (
    NativeError,
    NativeUnavailable,
    NativeUnsupported,
    FailureContext,
    NATIVE_ADAPTER_STATUS_EXCEPTION_POLICY,
    raise_for_status as _raise_native_status,
)


def _raise_native_result(result, operation: str, *, backend=None) -> None:
    _raise_native_status(
        result,
        operation,
        prefix="C++ SCAR-OU",
        context=FailureContext(backend=backend),
        exception_policy=NATIVE_ADAPTER_STATUS_EXCEPTION_POLICY,
    )


def require_available() -> None:
    """Raise ``NativeUnavailable`` unless the compiled extension can load."""
    _extension.load()


def ensure_supported(copula) -> None:
    """Raise NativeUnsupported if ``copula`` cannot use C++ SCAR-TM-OU kernels."""
    _descriptors.ensure_supported_for_scar_ou(copula)


def supported(copula) -> bool:
    """Return whether C++ SCAR-TM-OU kernels support ``copula``."""
    return _descriptors.supported_for_scar_ou(copula)


def supported_copula_ops(copula) -> bool:
    """Return whether C++ pointwise h/h_inverse kernels support ``copula``."""
    return _descriptors.supported_for_copula_ops(copula)


def copula_h(copula, u_conditioned, u_given, r) -> np.ndarray:
    """Evaluate ``h(u_conditioned | u_given)`` with C++ copula kernels.

    Inputs are broadcast as one-dimensional arrays.  The pybind boundary
    rejects non-finite values before entering the C++ numerical kernels.
    """
    return pair.h(copula, u_conditioned, u_given, r)


def copula_h_inverse(copula, q, u_given, r) -> np.ndarray:
    """Evaluate ``h^{-1}(q | u_given)`` with C++ copula kernels.

    Inputs are broadcast as one-dimensional arrays.  Non-finite inputs and
    non-finite C++ results are converted to Python exceptions.
    """
    return pair.h_inverse(copula, q, u_given, r)


def _params(module, kappa, mu, nu):
    params = module.OuParams()
    params.kappa = as_float64_scalar(kappa, name="kappa")
    params.mu = as_float64_scalar(mu, name="mu")
    params.nu = as_float64_scalar(nu, name="nu")
    return params


def validate_trajectory_parameters(kappa, mu, nu, count):
    """Reject invalid OU parameters before NumPy advances its generator."""
    module = _extension.load()
    status = module.ou_validate_trajectory_parameters(
        _params(module, kappa, mu, nu), int(count))
    _raise_native_status(status, "OU trajectory parameter validation")


def sample_trajectory(kappa, mu, nu, standard_normals):
    """Transform raw standard normals into a stationary exact OU trajectory."""
    module = _extension.load()
    result = module.ou_sample_trajectory(
        _params(module, kappa, mu, nu),
        np.ascontiguousarray(standard_normals, dtype=np.float64))
    _raise_native_result(result, "trajectory sampling")
    return np.asarray(result["values"], dtype=np.float64)


def sample_trajectory_block(
        kappa, mu, nu, total_count, previous_state, initialize,
        standard_normals):
    """Advance a bounded OU block using the complete path's time step."""
    module = _extension.load()
    result = module.ou_sample_trajectory_block(
        _params(module, kappa, mu, nu), int(total_count),
        float(previous_state), bool(initialize),
        as_float64_array(standard_normals, name="standard_normals"))
    _raise_native_result(result, "trajectory block sampling")
    return np.asarray(result["values"], dtype=np.float64)


def sample_stationary_fixed_draws(kappa, mu, nu, standard_normals):
    """Transform raw standard normals into stationary OU states in C++."""
    draws = np.ascontiguousarray(
        np.asarray(standard_normals, dtype=np.float64).ravel())
    module = _extension.load()
    result = module.ou_sample_stationary(
        _params(module, kappa, mu, nu), draws)
    _raise_native_result(result, "stationary state sampling")
    values = np.asarray(result["values"], dtype=np.float64)
    if values.size != draws.size:
        raise NativeError(
            "C++ SCAR-OU stationary sampler violated its fixed-draw contract")
    return values


def sample_state_distribution_fixed_draws(
        z_grid, probability, selection_uniforms, jitter_uniforms, *,
        mode="histogram"):
    """Sample a native OU grid/histogram state from raw uniform draws."""
    normalized = "histogram" if mode is None else str(mode).lower()
    if normalized not in {"grid", "histogram"}:
        raise ValueError("predictive_r_mode must be 'grid' or 'histogram'")
    selection = np.ascontiguousarray(
        np.asarray(selection_uniforms, dtype=np.float64).ravel())
    jitter = np.ascontiguousarray(
        np.asarray(jitter_uniforms, dtype=np.float64).ravel())
    result = _extension.load().ou_sample_state_distribution(
        np.ascontiguousarray(np.asarray(z_grid, dtype=np.float64).ravel()),
        np.ascontiguousarray(np.asarray(probability, dtype=np.float64).ravel()),
        selection,
        jitter,
        normalized == "histogram",
    )
    _raise_native_result(result, "fixed-draw state distribution sampling")
    selection_used = int(result["selection_draws_used"])
    jitter_used = int(result["jitter_draws_used"])
    if selection_used != selection.size or jitter_used != jitter.size:
        raise NativeError(
            "C++ SCAR-OU state sampler violated its fixed-draw contract")
    return np.asarray(result["values"], dtype=np.float64), {
        "selection_draws_used": selection_used,
        "jitter_draws_used": jitter_used,
    }


def condition_state(copula, z_grid, probability, observation):
    """Bayes-reweight one OU state distribution in the native backend."""
    module = _extension.load()
    spec = _descriptors.make_copula_ops_spec(module, copula)
    obs = np.ascontiguousarray(observation, dtype=np.float64)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    result = module.ou_condition_state(
        spec,
        np.ascontiguousarray(np.asarray(z_grid, dtype=np.float64).ravel()),
        np.ascontiguousarray(np.asarray(probability, dtype=np.float64).ravel()),
        obs,
    )
    _raise_native_result(result, "state conditioning")
    return (
        np.asarray(result["z_grid"], dtype=np.float64),
        np.asarray(result["prob"], dtype=np.float64),
    )


def trajectory_from_innovations(x0, mu, rho, sigma_cond, innovations):
    """Evaluate the native OU recurrence from a supplied initial state."""
    result = _extension.load().ou_trajectory_from_innovations(
        float(x0), float(mu), float(rho), float(sigma_cond),
        np.ascontiguousarray(innovations, dtype=np.float64))
    _raise_native_result(result, "trajectory recurrence")
    return np.asarray(result["values"], dtype=np.float64)


def _initialization_result(result, operation):
    _raise_native_result(result, operation)
    return np.asarray(result["values"], dtype=np.float64), dict(result)


def _parameter_vector(result, operation):
    _raise_native_result(result, operation)
    return np.asarray(result["values"], dtype=np.float64)


def _vector(values, name):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def to_log_stationary(values):
    """Map a physical OU block to native log-stationary coordinates."""
    array = _vector(values, "physical OU parameters")
    return _parameter_vector(
        _extension.load().ou_to_log_stationary(array.tolist()),
        "OU log-stationary forward transform")


def from_log_stationary(values):
    """Map native log-stationary coordinates to a physical OU block."""
    array = _vector(values, "log-stationary OU parameters")
    return _parameter_vector(
        _extension.load().ou_from_log_stationary(array.tolist()),
        "OU log-stationary inverse transform")


def gradient_to_log_stationary(physical, gradient):
    """Pull a physical OU gradient back to log-stationary coordinates."""
    physical_array = _vector(physical, "physical OU parameters")
    gradient_array = _vector(gradient, "OU gradient")
    return _parameter_vector(
        _extension.load().ou_gradient_to_log_stationary(
            physical_array.tolist(), gradient_array.tolist()),
        "OU log-stationary gradient pullback")


def gradient_from_log_stationary(physical, gradient):
    """Map a log-stationary gradient back to physical OU coordinates."""
    physical_array = _vector(physical, "physical OU parameters")
    gradient_array = _vector(gradient, "OU optimizer gradient")
    return _parameter_vector(
        _extension.load().ou_gradient_from_log_stationary(
            physical_array.tolist(), gradient_array.tolist()),
        "OU physical gradient conversion")


def project_optimizer_block(values, lower, upper):
    """Project the three-coordinate OU optimizer block in native code."""
    array = _vector(values, "OU optimizer values")
    lower_array = _vector(lower, "OU optimizer lower bounds")
    upper_array = _vector(upper, "OU optimizer upper bounds")
    return _parameter_vector(
        _extension.load().ou_project_optimizer_block(
            array.tolist(), lower_array.tolist(), upper_array.tolist()),
        "OU optimizer projection")


def _scaled(operation, values, scale, label):
    value_array = _vector(values, label)
    scale_array = _vector(scale, "optimizer scale")
    return _parameter_vector(
        operation(value_array.tolist(), scale_array.tolist()), label)


def scaled_to_physical(values, scale):
    return _scaled(
        _extension.load().optimizer_scaled_to_physical,
        values, scale, "scaled-to-physical parameter conversion")


def physical_to_scaled(values, scale):
    return _scaled(
        _extension.load().physical_to_optimizer_scaled,
        values, scale, "physical-to-scaled parameter conversion")


def gradient_to_scaled(gradient, scale):
    return _scaled(
        _extension.load().gradient_to_optimizer_scaled,
        gradient, scale, "scaled-gradient pullback")


def gradient_from_scaled(gradient, scale):
    return _scaled(
        _extension.load().gradient_from_optimizer_scaled,
        gradient, scale, "physical-gradient conversion")


def initial_kappa(count, rho_target=0.96, kappa_min=0.01, kappa_max=100.0):
    result = _extension.load().ou_initial_kappa(
        int(count), float(rho_target), float(kappa_min), float(kappa_max))
    _raise_native_result(result, "initial kappa selection")
    return float(result["value"])


def default_initial_point(mu):
    return _initialization_result(
        _extension.load().ou_default_initial_point(float(mu)),
        "default initial point")


def heuristic_initial_point(
        count, mu, rho_target=0.95, sigma_fraction=0.3):
    return _initialization_result(
        _extension.load().ou_heuristic_initial_point(
            int(count), float(mu), float(rho_target), float(sigma_fraction)),
        "heuristic initial point")


def stochastic_student_initial_point(
        count, theta_mle, mu, static_log_likelihood,
        rho_target=0.96, nu=0.1):
    return _initialization_result(
        _extension.load().ou_stochastic_student_initial_point(
            int(count), float(theta_mle), float(mu),
            float(static_log_likelihood), float(rho_target), float(nu)),
        "stochastic Student initial point")


def strength_aware_initial_point(
        observations, theta_mle, mu, static_log_likelihood, **options):
    module = _extension.load()
    config = module.OuInitializationConfig()
    for name, value in options.items():
        setattr(config, name, float(value))
    return _initialization_result(
        module.ou_strength_aware_initial_point(
            np.ascontiguousarray(observations, dtype=np.float64),
            float(theta_mle), float(mu), float(static_log_likelihood), config),
        "strength-aware initial point")


def hermite_rule(quad_order, basis_order):
    """Return the same native Hermite rule used by the spectral evaluator."""
    result = _extension.load().ou_hermite_rule(int(quad_order), int(basis_order))
    _raise_native_result(result, "Hermite rule construction")
    return tuple(np.asarray(result[key], dtype=np.float64)
                 for key in ("nodes", "weights", "basis"))


def default_quad_order(basis_order):
    result = _extension.load().ou_default_quad_order(int(basis_order))
    _raise_native_result(result, "default quadrature-order selection")
    return int(result["order"])


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
    out.n_threads = int(cfg.n_threads)
    out.corr_gradient_block_bytes = int(cfg.corr_gradient_block_bytes)
    grid_method = normalize_grid_method(cfg.grid_method)
    out.grid_method = {
        "auto": module.OuGridMethod.Auto,
        "dense": module.OuGridMethod.Dense,
        "sparse": module.OuGridMethod.Sparse,
    }[grid_method]
    return out


def _inputs(kappa, mu, nu, u, copula, config):
    module = _extension.load()
    cfg = config or AutoTMConfig()
    obs = as_pseudo_observation_array(u)
    method = normalize_ou_transition_method(cfg.transition_method)
    validate_cpp_config(cfg, transition_method=method)
    if obs.ndim != 2:
        raise ValueError("u must have 2D shape (n_obs, dimension)")

    student_dim = getattr(copula, "d", None)
    if student_dim is not None:
        if isinstance(student_dim, (bool, np.bool_)) or not isinstance(
                student_dim, (int, np.integer)):
            raise ValueError("Student dimension d must be an integer")
        student_dim = int(student_dim)
        if student_dim < 2:
            raise ValueError(
                f"Student dimension d must be at least 2, got {student_dim}")

    return (
        module,
        _params(module, kappa, mu, nu),
        _descriptors.make_spec(module, copula, obs),
        obs,
        _config(module, cfg),
        method,
    )


def _prepared_inputs(u, copula, config):
    module = _extension.load()
    cfg = config or AutoTMConfig()
    from pyscarcopula.copula.multivariate.equicorr_prepared import (
        EquicorrPreparedData,
    )
    prepared_input = isinstance(u, EquicorrPreparedData)
    if prepared_input:
        if int(getattr(copula, "d", -1)) != u.dimension:
            raise ValueError(
                "prepared dimension does not match copula dimension")
        obs = u
    else:
        obs = as_pseudo_observation_array(u)
    method = normalize_ou_transition_method(cfg.transition_method)
    validate_cpp_config(cfg, transition_method=method)
    if not prepared_input and obs.ndim != 2:
        raise ValueError("u must have 2D shape (n_obs, dimension)")

    student_dim = getattr(copula, "d", None)
    if student_dim is not None:
        if isinstance(student_dim, (bool, np.bool_)) or not isinstance(
                student_dim, (int, np.integer)):
            raise ValueError("Student dimension d must be an integer")
        student_dim = int(student_dim)
        if student_dim < 2:
            raise ValueError(
                f"Student dimension d must be at least 2, got {student_dim}")

    return (
        module,
        _descriptors.make_spec(
            module, copula, None if prepared_input else obs),
        obs,
        _config(module, cfg),
        method,
        cfg,
    )


class PreparedScarOuObjective:
    """Prepared C++ SCAR-TM-OU objective for repeated fit evaluations.

    The functional wrappers below intentionally stay stateless. Optimizer
    loops that evaluate the same observations many times should use this
    object so the native evaluator can reuse its copied observations, Student
    PPF cache, and grid/spectral gradient workspaces.
    """

    def __init__(self, u, copula, config: AutoTMConfig | None = None):
        (
            self.module,
            self.spec,
            self.obs,
            self.config,
            self.method,
            self.cfg_py,
        ) = _prepared_inputs(u, copula, config)
        if hasattr(self.obs, "sum_z") and hasattr(self.obs, "sum_z2"):
            self._native = self.module.PreparedScarOuEvaluator(
                self.spec,
                self.obs.sum_z,
                self.obs.sum_z2,
                self.config,
                self.method,
            )
        else:
            self._native = self.module.PreparedScarOuEvaluator(
                self.spec, self.obs, self.config, self.method)
        self._student_corr_version = getattr(
            copula, "_corr_cache_version", None)

    def update_copula(self, copula, *, force: bool = False) -> None:
        if getattr(copula, "d", None) is None:
            return
        if not hasattr(copula, "_L_inv") or not hasattr(copula, "_log_det"):
            return
        corr_version = getattr(copula, "_corr_cache_version", None)
        if not force and corr_version == self._student_corr_version:
            return
        if getattr(copula, "corr_mode", None) == "factor":
            # The native in-place updater accepts a dense inverse Cholesky
            # factor only. Rebuild on low-rank correlation changes without
            # materializing a dense matrix; unchanged versions still reuse it.
            spec = _descriptors.make_spec(self.module, copula, self.obs)
            native = self.module.PreparedScarOuEvaluator(
                spec, self.obs, self.config, self.method)
            self.spec = spec
            self._native = native
            self._student_corr_version = corr_version
            return
        self._native.update_student_factor(
            np.asarray(copula._L_inv, dtype=np.float64).reshape(-1),
            float(copula._log_det),
        )
        self._student_corr_version = corr_version

    def neg_loglik_info(self, kappa, mu, nu):
        params = _params(self.module, kappa, mu, nu)
        result = self._native.loglik(params)
        info = _result_info(result, self.method, kappa, len(self.obs), self.cfg_py)
        _raise_native_result(
            result, "prepared loglik", backend=info["backend"])
        value = -float(result["log_likelihood"])
        if not np.isfinite(value):
            raise FloatingPointError(
                "C++ prepared SCAR-OU objective returned a non-finite value")
        return value, info

    def neg_loglik_with_grad_info(self, kappa, mu, nu):
        params = _params(self.module, kappa, mu, nu)
        result = self._native.neg_loglik_with_grad(params)
        info = _result_info(result, self.method, kappa, len(self.obs), self.cfg_py)
        self._raise_if_failed(info, "C++ prepared SCAR-OU gradient failed")
        return (
            float(result["neg_log_likelihood"]),
            np.asarray(result["neg_gradient"], dtype=np.float64),
            info,
        )

    def neg_loglik_with_grad_and_corr_info(self, kappa, mu, nu):
        params = _params(self.module, kappa, mu, nu)
        result = self._native.neg_loglik_with_grad_and_corr(params)
        info = _result_info(result, self.method, kappa, len(self.obs), self.cfg_py)
        self._raise_if_failed(
            info, "C++ prepared SCAR-OU correlation gradient failed")
        return (
            float(result["neg_log_likelihood"]),
            np.asarray(result["neg_gradient"], dtype=np.float64),
            np.asarray(result["neg_corr_gradient"], dtype=np.float64),
            info,
        )

    def neg_loglik_with_grad_and_corr_directional_info(
            self, kappa, mu, nu, corr_direction):
        params = _params(self.module, kappa, mu, nu)
        direction = np.ascontiguousarray(
            np.asarray(corr_direction, dtype=np.float64).reshape(-1))
        result = self._native.neg_loglik_with_grad_and_corr_directional(
            params, direction)
        info = _result_info(result, self.method, kappa, len(self.obs), self.cfg_py)
        self._raise_if_failed(
            info, "C++ prepared SCAR-OU directional correlation gradient failed")
        return (
            float(result["neg_log_likelihood"]),
            np.asarray(result["neg_gradient"], dtype=np.float64),
            np.asarray(result["neg_corr_gradient"], dtype=np.float64),
            info,
        )

    def predictive_mean(self, kappa, mu, nu) -> np.ndarray:
        params = _params(self.module, kappa, mu, nu)
        result = self._native.predictive_mean(params)
        return _vector_result(result)

    def mixture_h(self, kappa, mu, nu) -> np.ndarray:
        params = _params(self.module, kappa, mu, nu)
        result = self._native.mixture_h(params)
        return clip_h_function_values(_vector_result(result))

    def mixture_h_pair(
            self, kappa, mu, nu) -> tuple[np.ndarray, np.ndarray]:
        """Return both h-directions from one prepared native forward pass."""
        params = _params(self.module, kappa, mu, nu)
        result = self._native.mixture_h_pair(params)
        values = _vector_result(result)
        first, second = np.split(values, 2)
        return (
            clip_h_function_values(first),
            clip_h_function_values(second),
        )

    def state_distribution(
            self, kappa, mu, nu,
            horizon: str = "current") -> tuple[np.ndarray, np.ndarray]:
        horizon = str(horizon).lower()
        if horizon not in ("current", "next"):
            raise ValueError("horizon must be 'current' or 'next'")
        params = _params(self.module, kappa, mu, nu)
        result = self._native.state_distribution(params, horizon == "next")
        _raise_native_result(result, "prepared state_distribution")
        return (
            np.asarray(result["z_grid"], dtype=np.float64),
            np.asarray(result["prob"], dtype=np.float64),
        )

    @staticmethod
    def _raise_if_failed(info: dict, message: str) -> None:
        operation = message.removeprefix("C++ ").removesuffix(" failed")
        _raise_native_result(info, operation, backend=info["backend"])


def prepare_objective(u, copula, config: AutoTMConfig | None = None):
    """Prepare a native SCAR-TM-OU objective for repeated evaluations."""
    return PreparedScarOuObjective(u, copula, config)


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
    from pyscarcopula._native import model_policy

    return model_policy.ou_kappa_dt(kappa, n_obs)


def _result_info(result, method: str, kappa, n_obs: int,
                 cfg: AutoTMConfig) -> dict:
    backend = _backend_name(result["backend"])
    info = {
        "backend": backend,
        "status": int(result["status"]),
        "transition_method": method,
        "engine": "cpp",
        "kappa_dt": (
            np.nan if n_obs < 2 else _kappa_dt(kappa, n_obs)),
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

    if n_obs < 2:
        return info

    if "K_effective" in result:
        info.update(K_requested=int(result["K_requested"]),
                    K_effective=int(result["K_effective"]),
                    grid_was_capped=bool(result["grid_was_capped"]))

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
    ``'matrix'``, or ``'local'``. In ``'auto'`` mode the native evaluator
    tries spectral, matrix, and local GH paths as required by numerical
    diagnostics. Non-zero C++ status codes are raised as :class:`NativeError`.
    """
    cfg_py = config or AutoTMConfig()
    module, params, spec, obs, cfg, method = _inputs(
        kappa, mu, nu, u, copula, config)
    result = _call_loglik(
        module.ScarOuEvaluator(), method, params, spec, obs, cfg)
    info = _result_info(result, method, kappa, len(obs), cfg_py)
    _raise_native_result(result, "loglik", backend=info["backend"])
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
        raise FloatingPointError(
            "C++ OU loglik returned a non-finite value with status=ok")
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
    _raise_native_result(result, "gradient", backend=info["backend"])
    return (
        float(result["neg_log_likelihood"]),
        np.asarray(result["neg_gradient"], dtype=np.float64),
        info,
    )


def neg_loglik_with_grad_and_corr(kappa, mu, nu, u, copula,
                                  config: AutoTMConfig | None = None):
    """Return negative likelihood and gradients over OU and current ``R``."""
    value, ou_grad, corr_grad, _ = neg_loglik_with_grad_and_corr_info(
        kappa, mu, nu, u, copula, config)
    return value, ou_grad, corr_grad


def neg_loglik_with_grad_and_corr_info(
        kappa, mu, nu, u, copula,
        config: AutoTMConfig | None = None):
    """Evaluate analytical OU and static-correlation gradients in C++.

    The correlation gradient follows row-major lower-triangle order and is
    taken with respect to symmetric off-diagonal entries of the current
    correlation matrix.
    """
    cfg_py = config or AutoTMConfig()
    module, params, spec, obs, cfg, method = _inputs(
        kappa, mu, nu, u, copula, config)
    evaluator = module.ScarOuEvaluator()
    if method == "auto":
        result = evaluator.neg_loglik_with_grad_and_corr_auto(
            params, spec, obs, cfg)
    elif method == "spectral":
        result = evaluator.neg_loglik_with_grad_and_corr_spectral(
            params, spec, obs, cfg)
    elif method == "local":
        result = evaluator.neg_loglik_with_grad_and_corr_local_gh(
            params, spec, obs, cfg)
    elif method == "matrix":
        result = evaluator.neg_loglik_with_grad_and_corr_matrix(
            params, spec, obs, cfg)
    else:
        raise ValueError(f"Unsupported transition_method: {method}")

    info = _result_info(result, method, kappa, len(obs), cfg_py)
    _raise_native_result(
        result, "correlation gradient", backend=info["backend"])
    return (
        float(result["neg_log_likelihood"]),
        np.asarray(result["neg_gradient"], dtype=np.float64),
        np.asarray(result["neg_corr_gradient"], dtype=np.float64),
        info,
    )


def neg_loglik_with_grad_and_corr_directional_info(
        kappa, mu, nu, u, copula, corr_direction,
        config: AutoTMConfig | None = None):
    """Evaluate OU gradient and one directional static-correlation gradient."""
    cfg_py = config or AutoTMConfig()
    module, params, spec, obs, cfg, method = _inputs(
        kappa, mu, nu, u, copula, config)
    direction = np.ascontiguousarray(
        np.asarray(corr_direction, dtype=np.float64).reshape(-1))
    evaluator = module.ScarOuEvaluator()
    if method == "auto":
        result = evaluator.neg_loglik_with_grad_and_corr_directional_auto(
            params, spec, obs, cfg, direction)
    elif method == "spectral":
        result = evaluator.neg_loglik_with_grad_and_corr_directional_spectral(
            params, spec, obs, cfg, direction)
    elif method == "local":
        result = evaluator.neg_loglik_with_grad_and_corr_directional_local_gh(
            params, spec, obs, cfg, direction)
    elif method == "matrix":
        result = evaluator.neg_loglik_with_grad_and_corr_directional_matrix(
            params, spec, obs, cfg, direction)
    else:
        raise ValueError(f"Unsupported transition_method: {method}")

    info = _result_info(result, method, kappa, len(obs), cfg_py)
    _raise_native_result(
        result,
        "directional correlation gradient",
        backend=info["backend"],
    )
    return (
        float(result["neg_log_likelihood"]),
        np.asarray(result["neg_gradient"], dtype=np.float64),
        np.asarray(result["neg_corr_gradient"], dtype=np.float64),
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
        raise NativeUnsupported(
            f"C++ {prefix} does not support transition_method='spectral'"
        )
    raise ValueError(f"Unsupported transition_method: {method}")


def _grid_config(config: AutoTMConfig | None) -> AutoTMConfig:
    """Return a native grid reconstruction config for posterior quantities."""
    cfg = config or AutoTMConfig()
    if normalize_ou_transition_method(cfg.transition_method) == "spectral":
        return replace(cfg, transition_method="auto")
    return cfg


def _vector_result(result):
    _raise_native_result(result, "forward call")
    return np.asarray(result["values"], dtype=np.float64)


def predictive_mean(kappa, mu, nu, u, copula,
                    config: AutoTMConfig | None = None) -> np.ndarray:
    """Return the grid-filtered predictive mean of the copula parameter.

    All public transition methods are accepted. ``'spectral'`` requests use
    native auto-grid reconstruction for this posterior quantity.
    """
    module, params, spec, obs, cfg, method = _inputs(
        kappa, mu, nu, u, copula, _grid_config(config))
    result = _call_vector(
        module.ScarOuEvaluator(), "predictive_mean",
        method, params, spec, obs, cfg)
    return _vector_result(result)


def forward_rosenblatt(kappa, mu, nu, u, copula,
                       config: AutoTMConfig | None = None) -> np.ndarray:
    """Return the fully native bivariate SCAR-OU Rosenblatt transform.

    Grid construction, emission evaluation, filtering, h-function mixing,
    and final Rosenblatt clipping are performed inside C++.
    """
    module, params, spec, obs, cfg, method = _inputs(
        kappa, mu, nu, u, copula, _grid_config(config))
    if obs.shape[1] != 2:
        raise ValueError(
            "SCAR-OU bivariate Rosenblatt transform requires u.shape[1] == 2")
    result = _call_vector(
        module.ScarOuEvaluator(), "forward_rosenblatt",
        method, params, spec, obs, cfg)
    values = _vector_result(result)
    expected_size = 2 * len(obs)
    if values.size != expected_size:
        raise NativeError(
            "C++ SCAR-OU Rosenblatt result has invalid size: "
            f"expected {expected_size}, got {values.size}")
    return values.reshape(len(obs), 2)


def gaussian_rosenblatt(kappa, mu, nu, u, copula,
                        config: AutoTMConfig | None = None) -> np.ndarray:
    """Return the native dynamic equicorrelation Gaussian Rosenblatt path.

    C++ performs OU grid construction, streaming predictive filtering,
    prefix-density reweighting, conditional Gaussian CDF evaluation, and
    final output clipping. No ``T x K`` state-weight matrix crosses the
    binding.
    """
    module, params, spec, obs, cfg, method = _inputs(
        kappa, mu, nu, u, copula, _grid_config(config))
    result = _call_vector(
        module.ScarOuEvaluator(), "gaussian_rosenblatt",
        method, params, spec, obs, cfg)
    values = _vector_result(result)
    expected_size = int(obs.shape[0]) * int(obs.shape[1])
    if values.size != expected_size:
        raise NativeError(
            "C++ SCAR-OU Gaussian Rosenblatt result has invalid size: "
            f"expected {expected_size}, got {values.size}")
    return values.reshape(obs.shape)


def student_rosenblatt(kappa, mu, nu, u, copula,
                       config: AutoTMConfig | None = None) -> np.ndarray:
    """Return the native dynamic multivariate Student Rosenblatt path.

    C++ performs Student PPF-cache interpolation, OU filtering,
    prefix-density reweighting, conditional Student CDF evaluation, and
    final output clipping without materializing a ``T x K`` weight matrix.
    """
    module, params, spec, obs, cfg, method = _inputs(
        kappa, mu, nu, u, copula, _grid_config(config))
    result = _call_vector(
        module.ScarOuEvaluator(), "student_rosenblatt",
        method, params, spec, obs, cfg)
    values = _vector_result(result)
    expected_size = int(obs.shape[0]) * int(obs.shape[1])
    if values.size != expected_size:
        raise NativeError(
            "C++ SCAR-OU Student Rosenblatt result has invalid size: "
            f"expected {expected_size}, got {values.size}")
    return values.reshape(obs.shape)


def mixture_h(kappa, mu, nu, u, copula,
              config: AutoTMConfig | None = None) -> np.ndarray:
    """Return the SCAR-TM mixture h-function from C++ grid filtering.

    All public transition methods are accepted. ``'spectral'`` requests use
    native auto-grid reconstruction. The output is clipped to the open-unit
    interval guard.
    """
    module, params, spec, obs, cfg, method = _inputs(
        kappa, mu, nu, u, copula, _grid_config(config))
    result = _call_vector(
        module.ScarOuEvaluator(), "mixture_h",
        method, params, spec, obs, cfg)
    return clip_h_function_values(_vector_result(result))


def mixture_h_pair(kappa, mu, nu, u, copula,
                   config: AutoTMConfig | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Return both SCAR-TM h-directions from one C++ grid filter pass."""
    module, params, spec, obs, cfg, method = _inputs(
        kappa, mu, nu, u, copula, _grid_config(config))
    result = _call_vector(
        module.ScarOuEvaluator(), "mixture_h_pair",
        method, params, spec, obs, cfg)
    values = _vector_result(result)
    first, second = np.split(values, 2)
    return (
        clip_h_function_values(first),
        clip_h_function_values(second),
    )


def state_distribution(kappa, mu, nu, u, copula,
                       config: AutoTMConfig | None = None,
                       horizon: str = "current") -> tuple[np.ndarray, np.ndarray]:
    """Return the C++ grid posterior or one-step-ahead state distribution.

    ``horizon='current'`` returns the posterior state after the observations;
    ``horizon='next'`` advances it one transition step. ``'spectral'`` requests
    use native auto-grid reconstruction.
    """
    horizon = str(horizon).lower()
    if horizon not in ("current", "next"):
        raise ValueError("horizon must be 'current' or 'next'")

    module, params, spec, obs, cfg, method = _inputs(
        kappa, mu, nu, u, copula, _grid_config(config))
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
        raise NativeUnsupported(
            "C++ state_distribution does not support transition_method='spectral'"
        )
    else:
        raise ValueError(f"Unsupported transition_method: {method}")

    _raise_native_result(result, "state_distribution")
    return (
        np.asarray(result["z_grid"], dtype=np.float64),
        np.asarray(result["prob"], dtype=np.float64),
    )


def smoothed_state_distribution(
        kappa, mu, nu, u, copula,
        config: AutoTMConfig | None = None
        ) -> tuple[np.ndarray, np.ndarray]:
    """Return the native forward-backward posterior on every grid state.

    The returned pair is ``(z_grid, weights)`` with ``weights.shape == (T, K)``.
    Grid construction, Student emissions, both filtering passes, and posterior
    normalization are performed in C++.
    """
    module, params, spec, obs, cfg, method = _inputs(
        kappa, mu, nu, u, copula, _grid_config(config))
    evaluator = module.ScarOuEvaluator()
    if method == "auto":
        result = evaluator.smoothed_state_distribution_auto(
            params, spec, obs, cfg)
    elif method == "local":
        result = evaluator.smoothed_state_distribution_local_gh(
            params, spec, obs, cfg)
    elif method == "matrix":
        result = evaluator.smoothed_state_distribution_matrix(
            params, spec, obs, cfg)
    else:
        raise ValueError(f"Unsupported transition_method: {method}")

    _raise_native_result(result, "state smoothing")
    z_grid = np.asarray(result["z_grid"], dtype=np.float64)
    weights = np.asarray(result["weights"], dtype=np.float64)
    if z_grid.ndim != 1 or weights.shape != (len(obs), len(z_grid)):
        raise NativeError(
            "C++ SCAR-OU state smoothing returned inconsistent dimensions"
        )
    return z_grid, weights


def _backend_name(value: int) -> str:
    return {
        0: "spectral",
        1: "local",
        2: "matrix",
    }.get(int(value), "unknown")


def _matrix_fallback_reason_name(value: int):
    return {
        0: None,
        1: "failed",
        2: "capped",
    }.get(int(value), "unknown")
