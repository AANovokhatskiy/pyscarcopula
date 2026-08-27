"""Typed Python adapter for the native Jacobi domain core."""

from __future__ import annotations

import math

import numpy as np

from pyscarcopula._native._extension import load
from pyscarcopula._native.errors import raise_for_status
from pyscarcopula.numerical._arrays import (
    validate_float64_allocation,
    validate_positive_int,
)
from pyscarcopula.numerical._transition_methods import (
    normalize_jacobi_stationarity_correction,
    normalize_jacobi_strategy_transition_method,
    normalize_jacobi_transition_storage,
)


MAX_JACOBI_ORDER = 2048
DEFAULT_JACOBI_MEMORY_BUDGET_BYTES = 1024 ** 3
DEFAULT_LAMPERTI_CHUNK_OBSERVATIONS = 4096
_LAMPERTI_BOUNDARIES = frozenset({"reflect", "clip"})
_LAMPERTI_ENGINES = frozenset({"native", "numba", "python"})


def _params(kappa, m, xi):
    module = load()
    params = module.JacobiParams()
    params.kappa = float(kappa)
    params.m = float(m)
    params.xi = float(xi)
    return params


def _raise(result, operation):
    raise_for_status(result, operation, prefix="C++ Jacobi")


def _memory_budget(value):
    return (
        DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
        if value is None else int(value))


def _validate_order(value, name):
    value = validate_positive_int(value, name)
    if value > MAX_JACOBI_ORDER:
        raise ValueError(
            f"{name} must be <= {MAX_JACOBI_ORDER}; larger Jacobi grids "
            "are disabled to prevent unsafe quadratic allocations")
    return value


def _validate_nonnegative_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)):
        raise TypeError(f"{name} must be a non-negative integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def normalize_sampling_method(value):
    """Normalize the public unconditional Jacobi sampling backend."""
    if not isinstance(value, str):
        raise TypeError("sampling_method must be a string")
    method = value.strip().lower()
    if method not in {"tm_grid", "lamperti_euler"}:
        raise ValueError(
            "sampling_method must be 'tm_grid' or 'lamperti_euler'")
    return method


def normalize_lamperti_boundary(value):
    """Normalize the native Lamperti boundary policy."""
    if not isinstance(value, str):
        raise TypeError("lamperti_boundary must be a string")
    boundary = value.strip().lower()
    if boundary not in _LAMPERTI_BOUNDARIES:
        raise ValueError(
            "lamperti_boundary must be 'reflect' or 'clip'")
    return boundary


def normalize_lamperti_engine(value):
    """Normalize historical engine labels to the mandatory native engine."""
    if not isinstance(value, str):
        raise TypeError("lamperti_engine must be a string")
    engine = value.strip().lower()
    if engine not in _LAMPERTI_ENGINES:
        raise ValueError("lamperti_engine must be 'native'")
    return "native"


def validate_lamperti_eps(value):
    """Validate the interior epsilon used for native drift evaluation."""
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


def raw_to_physical(raw):
    values = np.asarray(raw, dtype=np.float64)
    if values.shape != (3,):
        raise ValueError(f"raw Jacobi parameters must have shape (3,), got {values.shape}")
    result = load().jacobi_raw_to_physical(values.tolist())
    _raise(result, "raw-to-physical transform")
    return np.asarray(result["values"], dtype=np.float64)


def physical_to_raw(values, tau_eps):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (3,):
        raise ValueError(
            f"physical Jacobi parameters must have shape (3,), got {values.shape}")
    result = load().jacobi_physical_to_raw(
        _params(*values), float(tau_eps))
    _raise(result, "physical-to-raw transform")
    return np.asarray(result["values"], dtype=np.float64)


def raw_bounds(kappa_bounds, xi_bounds, tau_eps):
    module = load()
    bounds = module.JacobiParameterBounds()
    bounds.kappa_lower, bounds.kappa_upper = map(float, kappa_bounds)
    bounds.xi_lower, bounds.xi_upper = map(float, xi_bounds)
    bounds.tau_eps = float(tau_eps)
    result = module.jacobi_raw_bounds(bounds)
    _raise(result, "raw optimizer bounds")
    return (
        np.asarray(result["lower"], dtype=np.float64),
        np.asarray(result["upper"], dtype=np.float64),
    )


def stationary_shape(kappa, m, xi):
    result = load().jacobi_stationary_shape(_params(kappa, m, xi))
    if int(result["status"]) != 0:
        return None
    return float(result["alpha"]), float(result["beta"])


def shape_is_supported(kappa, m, xi, stationary_shape_max):
    limit = (
        math.inf if stationary_shape_max is None
        else float(stationary_shape_max))
    status = load().jacobi_validate_params(_params(kappa, m, xi), limit)
    return int(status) == 0


def copula_supported(copula):
    """Return whether the pair copula has the native Jacobi kernel contract."""
    from pyscarcopula._native import _descriptors

    return _descriptors.supported_for_copula_ops(copula)


def _copula_spec(copula):
    from pyscarcopula._native import _descriptors

    module = load()
    return module, _descriptors.make_copula_ops_spec(module, copula)


def tau_to_parameter(copula, tau, *, theta_cap=None):
    """Map Jacobi tau values through the native prepared pair contract."""
    module, spec = _copula_spec(copula)
    values = np.ascontiguousarray(
        np.atleast_1d(np.asarray(tau, dtype=np.float64)).ravel())
    result = np.asarray(
        module.copula_tau_to_param(spec, values), dtype=np.float64)
    if np.any(~np.isfinite(result)):
        raise FloatingPointError(
            "native tau_to_param produced non-finite sampling parameters")
    if theta_cap is not None:
        result = np.minimum(result, float(theta_cap))
    return result


def parameter_to_tau(copula, parameter):
    """Map pair parameters to Kendall tau through the native pair contract."""
    module, spec = _copula_spec(copula)
    values = np.ascontiguousarray(
        np.atleast_1d(np.asarray(parameter, dtype=np.float64)).ravel())
    result = np.asarray(
        module.copula_param_to_tau(spec, values), dtype=np.float64)
    if np.any(~np.isfinite(result)):
        raise FloatingPointError(
            "native param_to_tau produced non-finite values")
    return result


def validate_copula_mapping(copula):
    """Fail before numerical work when the native tau mapping is unavailable."""
    try:
        tau_to_parameter(copula, np.array([0.5], dtype=np.float64))
    except NotImplementedError:
        raise
    except Exception as exc:
        raise ValueError(
            f"{type(copula).__name__} does not provide a usable "
            "tau_to_param mapping") from exc


def estimate_workspace(
        *, quad_order, basis_order=1, n_obs=0, gradient=False,
        matrix=True, gh_order=1, memory_budget_bytes):
    module = load()
    config = module.JacobiNumericalConfig()
    config.quad_order = int(quad_order)
    config.basis_order = int(basis_order)
    config.gh_order = int(gh_order)
    config.n_obs = int(n_obs)
    config.gradient = bool(gradient)
    config.matrix = bool(matrix)
    config.memory_budget_bytes = int(memory_budget_bytes)
    result = module.jacobi_estimate_workspace(config)
    if not bool(result["within_budget"]) and int(result["bytes"]) > 0:
        raise MemoryError(
            "Jacobi numerical workspace requires an estimated "
            f"{int(result['bytes'])} bytes, exceeding memory_budget_bytes="
            f"{int(result['budget_bytes'])}")
    _raise(result, "workspace preflight")
    return int(result["bytes"])


def estimate_sampling_workspace(
        *, n, quad_order, basis_order, gh_order=1, memory_budget_bytes):
    module = load()
    config = module.JacobiNumericalConfig()
    config.quad_order = int(quad_order)
    config.basis_order = int(basis_order)
    config.gh_order = int(gh_order)
    config.memory_budget_bytes = int(memory_budget_bytes)
    result = module.jacobi_estimate_sampling_workspace(int(n), config)
    if not bool(result["within_budget"]) and int(result["bytes"]) > 0:
        raise MemoryError(
            "Jacobi sampling workspace requires an estimated "
            f"{int(result['bytes'])} bytes, exceeding memory_budget_bytes="
            f"{int(result['budget_bytes'])}")
    _raise(result, "sampling workspace preflight")
    return int(result["bytes"])


def gauss_hermite_rule(
        order, memory_budget_bytes=1024**3):
    estimate_workspace(
        quad_order=1,
        basis_order=1,
        gh_order=order,
        matrix=False,
        memory_budget_bytes=memory_budget_bytes,
    )
    result = load().jacobi_gauss_hermite_rule(
        int(order), int(memory_budget_bytes))
    _raise(result, "Gauss-Hermite rule")
    return (
        np.asarray(result["nodes"], dtype=np.float64),
        np.asarray(result["weights"], dtype=np.float64),
    )


def jacobi_rule(alpha, beta, quad_order, basis_order, memory_budget_bytes):
    estimate_workspace(
        quad_order=quad_order,
        basis_order=basis_order,
        gh_order=1,
        matrix=False,
        memory_budget_bytes=memory_budget_bytes,
    )
    result = load().jacobi_build_rule(
        float(alpha),
        float(beta),
        int(quad_order),
        int(basis_order),
        int(memory_budget_bytes),
    )
    _raise(result, "Gauss-Jacobi basis rule")
    return (
        np.asarray(result["tau"], dtype=np.float64),
        np.asarray(result["weights"], dtype=np.float64),
        np.asarray(result["basis"], dtype=np.float64),
    )


def fixed_tau_rule(kappa, m, xi, quad_order, memory_budget_bytes):
    result = load().jacobi_build_fixed_rule(
        _params(kappa, m, xi), int(quad_order), int(memory_budget_bytes))
    _raise(result, "fixed-tau stationary rule")
    return (
        np.asarray(result["tau"], dtype=np.float64),
        np.asarray(result["weights"], dtype=np.float64),
        np.asarray(result["weight_derivatives"], dtype=np.float64),
    )


def fixed_shape_rule(alpha, beta, quad_order, memory_budget_bytes):
    alpha = float(alpha)
    beta = float(beta)
    total = alpha + beta
    if not np.isfinite(total) or alpha <= 0.0 or beta <= 0.0:
        raise ValueError("alpha and beta must be finite and positive")
    return fixed_tau_rule(
        0.5 * total,
        alpha / total,
        1.0,
        quad_order,
        memory_budget_bytes,
    )


def lamperti(tau, xi):
    values = np.asarray(tau, dtype=np.float64)
    result = load().jacobi_lamperti_values(values, float(xi))
    _raise(result, "Lamperti transform")
    return np.asarray(result["values"], dtype=np.float64).reshape(values.shape)


def inverse_lamperti(values, xi):
    values = np.asarray(values, dtype=np.float64)
    result = load().jacobi_inverse_lamperti_values(values, float(xi))
    _raise(result, "inverse Lamperti transform")
    return np.asarray(result["values"], dtype=np.float64).reshape(values.shape)


def lamperti_drift(tau, kappa, m, xi, interior_eps=0.0):
    values = np.asarray(tau, dtype=np.float64)
    result = load().jacobi_lamperti_drift_values(
        _params(kappa, m, xi), values, float(interior_eps))
    _raise(result, "Lamperti drift")
    return np.asarray(result["values"], dtype=np.float64).reshape(values.shape)


_METHOD_NAMES = {
    0: "auto",
    1: "spectral_matrix",
    2: "local",
    3: "local_fixed",
    4: "spectral_coeff",
}
_STORAGE_NAMES = {0: "dense", 1: "sparse"}
_CORRECTION_NAMES = {0: "none", 1: "mh", 2: "ipfp"}


def resolve_dt(n_obs):
    result = load().jacobi_resolve_dt(int(n_obs))
    _raise(result, "transition time-grid resolution")
    return float(result["value"])


def transition_powers(kappa, xi, n_obs, basis_order):
    result = load().jacobi_transition_powers(
        _params(kappa, 0.5, xi), int(n_obs), int(basis_order))
    _raise(result, "spectral transition powers")
    return np.asarray(result["values"], dtype=np.float64)


def default_quad_order(basis_order):
    result = load().jacobi_default_quad_order(int(basis_order))
    _raise(result, "default quadrature-order selection")
    return int(result["value"])


def estimate_sparse_workspace(
        *, quad_order, gh_order, correction="none",
        memory_budget_bytes=1024**3):
    config = _transition_config(
        n_obs=2,
        quad_order=quad_order,
        basis_order=1,
        gh_order=gh_order,
        method="local",
        storage="sparse",
        correction=correction,
        memory_budget_bytes=memory_budget_bytes,
    )
    result = load().jacobi_estimate_sparse_workspace(config)
    if int(result["status"]) == 2 and not result["within_budget"]:
        raise MemoryError(
            "sparse Jacobi transition workspace requires an estimated "
            f"{int(result['bytes'])} bytes, exceeding "
            f"memory_budget_bytes={int(result['budget_bytes'])}")
    _raise(result, "sparse transition workspace preflight")
    return int(result["bytes"])


def estimate_sparse_storage(
        *, quad_order, gh_order, correction="none",
        memory_budget_bytes=1024**3):
    """Return the retained sparse-storage estimate used by legacy callers."""
    config = _transition_config(
        n_obs=2,
        quad_order=quad_order,
        basis_order=1,
        gh_order=gh_order,
        method="local",
        storage="sparse",
        correction=correction,
        memory_budget_bytes=memory_budget_bytes,
    )
    result = load().jacobi_estimate_sparse_storage(config)
    if int(result["status"]) == 2 and not result["within_budget"]:
        raise MemoryError(
            "sparse Jacobi transition workspace requires an estimated "
            f"{int(result['bytes'])} bytes, exceeding "
            f"memory_budget_bytes={int(result['budget_bytes'])}")
    _raise(result, "sparse transition storage preflight")
    return int(result["bytes"])


def _transition_config(
        *, n_obs, quad_order, basis_order, gh_order, method,
        storage="dense", correction="none", clip_negative=False,
        negative_mass_tol=1e-5, return_grad=False,
        memory_budget_bytes=1024**3, ipfp_tolerance=1e-15,
        ipfp_max_iterations=10_000, theta_cap=None,
        stationary_shape_max=None):
    module = load()
    numerical = module.JacobiNumericalConfig()
    numerical.quad_order = int(quad_order)
    numerical.basis_order = int(basis_order)
    numerical.gh_order = int(gh_order)
    numerical.n_obs = int(n_obs)
    numerical.matrix = storage == "dense" and method != "spectral_coeff"
    numerical.gradient = bool(return_grad)
    numerical.memory_budget_bytes = int(memory_budget_bytes)
    numerical.theta_cap = (
        math.nan if theta_cap is None else float(theta_cap))
    numerical.stationary_shape_max = (
        math.inf if stationary_shape_max is None
        else float(stationary_shape_max))

    config = module.JacobiTransitionConfig()
    config.numerical = numerical
    config.method = {
        "auto": module.JacobiTransitionMethod.Auto,
        "spectral_matrix": module.JacobiTransitionMethod.SpectralMatrix,
        "local": module.JacobiTransitionMethod.Local,
        "local_fixed": module.JacobiTransitionMethod.LocalFixed,
        "spectral_coeff": module.JacobiTransitionMethod.SpectralCoeff,
    }[method]
    config.storage = {
        "dense": module.JacobiTransitionStorage.Dense,
        "sparse": module.JacobiTransitionStorage.Sparse,
    }[storage]
    config.correction = {
        "none": module.JacobiStationarityCorrection.None_,
        "mh": module.JacobiStationarityCorrection.MetropolisHastings,
        "ipfp": module.JacobiStationarityCorrection.IpFp,
    }[correction]
    config.negative_mass_tolerance = float(negative_mass_tol)
    config.clip_negative = bool(clip_negative)
    config.derivatives = bool(return_grad)
    config.ipfp_tolerance = float(ipfp_tolerance)
    config.ipfp_max_iterations = int(ipfp_max_iterations)
    return config


def _evaluator_raise(result, operation):
    diagnostics = dict(result.get("diagnostics", {}))
    estimated = int(diagnostics.get("estimated_workspace_bytes", 0))
    budget = int(diagnostics.get("memory_budget_bytes", 0))
    if int(result["status"]) == 2 and estimated > budget:
        raise MemoryError(
            "Jacobi numerical workspace requires an estimated "
            f"{estimated} bytes, exceeding memory_budget_bytes={budget}; "
            "reduce quad_order, basis_order, or the observation count, or "
            "increase memory_budget_bytes")
    _raise(result, operation)


def _evaluator_config(
        module, *, n_obs, basis_order=32, quad_order=None,
        theta_cap=None, transition_method="auto", storage="dense",
        correction="none", clip_negative=False,
        negative_mass_tol=1e-5, gh_order=5,
        memory_budget_bytes=1024**3, fd_rel_step=1e-5,
        stationary_shape_max=None):
    """Build the native evaluator policy shared by pair and vine adapters."""
    if quad_order is None:
        quad_order = default_quad_order(basis_order)
    memory_budget_bytes = _memory_budget(memory_budget_bytes)
    transition = _transition_config(
        n_obs=int(n_obs),
        quad_order=quad_order,
        basis_order=basis_order,
        gh_order=gh_order,
        method=str(transition_method),
        storage=storage,
        correction=correction,
        clip_negative=clip_negative,
        negative_mass_tol=negative_mass_tol,
        memory_budget_bytes=memory_budget_bytes,
        theta_cap=theta_cap,
        stationary_shape_max=stationary_shape_max,
    )
    config = module.JacobiEvaluatorConfig()
    config.transition = transition
    config.finite_difference_relative_step = float(fd_rel_step)
    return config


class PreparedScarJacobiEvaluator:
    """Prepared native Jacobi objective/filter/state facade.

    One instance owns the immutable copula descriptor and observations.  Its
    native peer caches the latest transition, transformed grid, emissions,
    filter, and smoother for repeated calls at the same physical parameters.
    """

    def __init__(
            self, u, copula, *, basis_order=32, quad_order=None,
            theta_cap=None, transition_method="auto", storage="dense",
            correction="none", clip_negative=False,
            negative_mass_tol=1e-5, gh_order=5,
            memory_budget_bytes=1024**3, fd_rel_step=1e-5,
            stationary_shape_max=None):
        from pyscarcopula._native import _descriptors

        module = load()
        observations = np.ascontiguousarray(u, dtype=np.float64)
        if (
                observations.ndim != 2
                or observations.shape[1] != 2
                or len(observations) < 1):
            raise ValueError(
                "u must be a 2D float64 array with shape (n, 2), n >= 1")
        config = _evaluator_config(
            module,
            n_obs=len(observations),
            basis_order=basis_order,
            quad_order=quad_order,
            theta_cap=theta_cap,
            transition_method=transition_method,
            storage=storage,
            correction=correction,
            clip_negative=clip_negative,
            negative_mass_tol=negative_mass_tol,
            gh_order=gh_order,
            memory_budget_bytes=memory_budget_bytes,
            fd_rel_step=fd_rel_step,
            stationary_shape_max=stationary_shape_max,
        )
        spec = _descriptors.make_copula_ops_spec(module, copula)
        self._module = module
        self._native = module.PreparedScarJacobiEvaluator(
            spec, observations, config)

    @property
    def preparation_count(self):
        return int(self._native.preparation_count)

    def filter(self, kappa, m, xi):
        result = self._native.filter(_params(kappa, m, xi))
        _evaluator_raise(result, "prepared filter/smoother")
        return {
            "tau": np.asarray(result["tau"], dtype=np.float64),
            "theta": np.asarray(result["theta"], dtype=np.float64),
            "emissions": np.asarray(result["emissions"], dtype=np.float64),
            "predicted": np.asarray(result["predicted"], dtype=np.float64),
            "filtered": np.asarray(result["filtered"], dtype=np.float64),
            "smoothed": np.asarray(result["smoothed"], dtype=np.float64),
            "scales": np.asarray(result["scales"], dtype=np.float64),
            "current_probability": np.asarray(
                result["current_probability"], dtype=np.float64),
            "next_probability": np.asarray(
                result["next_probability"], dtype=np.float64),
            "diagnostics": _transition_diagnostics(result["diagnostics"]),
        }

    def loglik(self, kappa, m, xi):
        result = self._native.loglik(_params(kappa, m, xi))
        _evaluator_raise(result, "prepared log-likelihood")
        return float(result["log_likelihood"])

    def neg_loglik(self, kappa, m, xi):
        result = self._native.loglik(_params(kappa, m, xi))
        _evaluator_raise(result, "prepared objective")
        return float(result["objective"])

    def neg_loglik_with_grad(self, kappa, m, xi):
        result = self._native.neg_loglik_with_grad(_params(kappa, m, xi))
        _evaluator_raise(result, "prepared objective gradient")
        return (
            float(result["objective"]),
            np.asarray(result["gradient"], dtype=np.float64),
        )

    def predictive_mean(self, kappa, m, xi):
        result = self._native.predictive_mean(_params(kappa, m, xi))
        _evaluator_raise(result, "prepared predictive mean")
        return np.asarray(result["values"], dtype=np.float64)

    def mixture_h(self, kappa, m, xi):
        result = self._native.mixture_h(_params(kappa, m, xi))
        _evaluator_raise(result, "prepared mixture-h")
        return np.asarray(result["values"], dtype=np.float64)

    def mixture_h_pair(self, kappa, m, xi):
        result = self._native.mixture_h_pair(_params(kappa, m, xi))
        _evaluator_raise(result, "prepared mixture-h pair")
        return (
            np.asarray(result["first"], dtype=np.float64),
            np.asarray(result["second"], dtype=np.float64),
        )

    def rosenblatt(self, kappa, m, xi, *, gaussian=False):
        operation = (
            self._native.gaussian_rosenblatt if gaussian
            else self._native.rosenblatt)
        result = operation(_params(kappa, m, xi))
        _evaluator_raise(result, "prepared Rosenblatt residual")
        return np.column_stack((
            np.asarray(result["first"], dtype=np.float64),
            np.asarray(result["second"], dtype=np.float64),
        ))

    def state_distribution(self, kappa, m, xi, *, horizon="current"):
        normalized = str(horizon).lower()
        if normalized not in {"current", "next"}:
            raise ValueError("horizon must be 'current' or 'next'")
        native_horizon = {
            "current": self._module.JacobiStateHorizon.Current,
            "next": self._module.JacobiStateHorizon.Next,
        }[normalized]
        result = self._native.state_distribution(
            _params(kappa, m, xi), native_horizon)
        _evaluator_raise(result, "prepared state distribution")
        return (
            np.asarray(result["tau"], dtype=np.float64),
            np.asarray(result["probability"], dtype=np.float64),
        )

    def condition_state(
            self, tau, probability, observation, *, horizon="current"):
        normalized = str(horizon).lower()
        if normalized not in {"current", "next"}:
            raise ValueError("horizon must be 'current' or 'next'")
        native_horizon = {
            "current": self._module.JacobiStateHorizon.Current,
            "next": self._module.JacobiStateHorizon.Next,
        }[normalized]
        result = self._native.condition_state(
            np.asarray(tau, dtype=np.float64),
            np.asarray(probability, dtype=np.float64),
            np.asarray(observation, dtype=np.float64),
            native_horizon,
        )
        _evaluator_raise(result, "prepared state conditioning")
        return (
            np.asarray(result["tau"], dtype=np.float64),
            np.asarray(result["probability"], dtype=np.float64),
        )


def _transition_raise(result, operation):
    diagnostics = dict(result.get("diagnostics", {}))
    estimated = int(diagnostics.get("estimated_workspace_bytes", 0))
    budget = int(diagnostics.get("memory_budget_bytes", 0))
    if int(result["status"]) == 2 and estimated > budget:
        raise MemoryError(
            "Jacobi numerical workspace requires an estimated "
            f"{estimated} bytes, exceeding memory_budget_bytes={budget}; "
            "reduce quad_order, basis_order, or the observation count, or "
            "increase memory_budget_bytes")
    native_operation = int(result.get("failure_operation", -1))
    if int(result["status"]) == 7 and native_operation == 3:
        raise FloatingPointError(
            "sparse IPFP support cannot reach every stationary node")
    if int(result["status"]) == 7 and native_operation == 5:
        raise FloatingPointError(
            "sparse IPFP did not converge within max_iterations")
    if int(result["status"]) == 7 and native_operation == 6:
        raise FloatingPointError(
            "spectral transition contains material negative probability mass")
    _raise(result, operation)


def _transition_diagnostics(values):
    diagnostics = dict(values)
    diagnostics["transition_method_requested"] = _METHOD_NAMES[
        int(diagnostics.pop("method_requested"))]
    diagnostics["transition_method"] = _METHOD_NAMES[
        int(diagnostics.pop("method_used"))]
    diagnostics["transition_storage"] = _STORAGE_NAMES[
        int(diagnostics.pop("storage"))]
    diagnostics["correction"] = _CORRECTION_NAMES[
        int(diagnostics["correction"])]
    return diagnostics


def sample_grid_trajectory_fixed_draws(
        kappa, m, xi, uniforms, *, quad_order, basis_order, gh_order,
        method, storage="dense", correction="none", clip_negative=False,
        negative_mass_tol=1e-5, memory_budget_bytes=1024**3):
    """Run the complete dense/sparse TM-grid trajectory in native C++."""
    draws = np.ascontiguousarray(
        np.asarray(uniforms, dtype=np.float64).ravel())
    if draws.size == 0:
        return np.empty(0, dtype=np.float64), {
            "draws_used": 0,
            "transition_method_requested": str(method),
            "transition_method": "not_built",
            "transition_storage": str(storage),
            "correction": str(correction),
        }
    config = _transition_config(
        n_obs=draws.size,
        quad_order=quad_order,
        basis_order=basis_order,
        gh_order=gh_order,
        method=method,
        storage=storage,
        correction=correction,
        clip_negative=clip_negative,
        negative_mass_tol=negative_mass_tol,
        memory_budget_bytes=memory_budget_bytes,
    )
    result = load().jacobi_sample_grid_trajectory(
        _params(kappa, m, xi), config, draws)
    _transition_raise(result, "fixed-draw grid trajectory sampling")
    used = int(result["draws_used"])
    if used != draws.size:
        raise RuntimeError(
            "C++ Jacobi grid sampler violated its fixed-draw contract: "
            f"used {used} of {draws.size} uniforms")
    diagnostics = _transition_diagnostics(result["diagnostics"])
    diagnostics["draws_used"] = used
    return np.asarray(result["tau"], dtype=np.float64), diagnostics


def sample_lamperti_chunk_fixed_draws(
        kappa, m, xi, initial_lamperti_value, normal_draws, *,
        n_obs, substeps, boundary, interior_eps):
    """Advance complete Lamperti intervals using caller-owned normals."""
    module = load()
    config = module.JacobiLampertiSamplingConfig()
    config.n_obs = int(n_obs)
    config.substeps = int(substeps)
    config.interior_eps = float(interior_eps)
    config.boundary = {
        "reflect": module.JacobiBoundaryPolicy.Reflect,
        "clip": module.JacobiBoundaryPolicy.Clip,
    }[boundary]
    draws = np.ascontiguousarray(
        np.asarray(normal_draws, dtype=np.float64))
    result = module.jacobi_sample_lamperti_chunk(
        _params(kappa, m, xi), config, float(initial_lamperti_value), draws)
    _raise(result, "fixed-draw Lamperti-Euler chunk")
    used = int(result["normal_draws_used"])
    if used != draws.size:
        raise RuntimeError(
            "C++ Lamperti sampler violated its fixed-draw contract: "
            f"used {used} of {draws.size} normals")
    return {
        "tau": np.asarray(result["tau"], dtype=np.float64),
        "final_lamperti_value": float(result["final_lamperti_value"]),
        "normal_draws_used": used,
        "euler_steps": int(result["euler_steps"]),
        "boundary_interventions": int(result["boundary_interventions"]),
    }


def _sampling_memory_error(exc):
    raise MemoryError(
        f"{exc}; reduce quad_order, basis_order, or the observation "
        "count, or increase memory_budget_bytes") from exc


def _validate_sampling_workspace(
        *, n, quad_order, basis_order, gh_order, memory_budget_bytes):
    try:
        native_peak_bytes = estimate_sampling_workspace(
            n=n,
            quad_order=quad_order,
            basis_order=basis_order,
            gh_order=gh_order,
            memory_budget_bytes=memory_budget_bytes,
        )
        boundary_bytes = validate_float64_allocation(
            (3, n), name="Jacobi fixed-draw boundary buffers")
        required_bytes = native_peak_bytes + boundary_bytes
        validate_float64_allocation(
            (required_bytes // np.dtype(np.float64).itemsize,),
            name="Jacobi sampling workspace",
            memory_budget_bytes=memory_budget_bytes,
        )
        return required_bytes
    except MemoryError as exc:
        _sampling_memory_error(exc)


def _legacy_grid_sampling_diagnostics(
        native, *, sampling_method, requested_method, storage, correction, n):
    """Preserve frozen public diagnostics while native C++ owns execution."""
    method_used = native["transition_method"]
    if n == 1:
        return {
            "transition_method_requested": requested_method,
            "sampling_transition_method_requested": sampling_method,
            "transition_method": "stationary_only",
            "n": 1,
        }
    if storage == "sparse":
        diagnostics = {
            "dt": native["dt"],
            "alpha": native["alpha"],
            "beta": native["beta"],
            "gh_order": native["gh_order"],
            "transition_method": sampling_method,
            "correction": correction,
            "nnz": int(native["nnz"]),
            "max_width": int(native["max_width"]),
            "retained_bytes": int(native["retained_bytes"]),
            "dense_bytes": int(native["dense_bytes"]),
            "stationary_error": native["stationary_error"],
        }
        if sampling_method == "local":
            diagnostics["max_row_sum_error"] = native[
                "max_row_sum_error"]
        if correction == "mh":
            for name in (
                    "mean_accepted_off_diagonal_mass",
                    "mean_proposed_off_diagonal_mass",
                    "acceptance_mass_ratio",
                    "min_row_acceptance_ratio",
                    "mean_stay_probability",
                    "max_stay_probability",
                    "reverse_missing_edge_fraction",
                    "detailed_balance_error"):
                diagnostics[name] = native[name]
        elif correction == "ipfp":
            diagnostics.update({
                "ipfp_iterations": int(native["ipfp_iterations"]),
                "ipfp_stationary_residual": native[
                    "ipfp_stationary_residual"],
                "ipfp_kl_divergence": native["ipfp_kl_divergence"],
                "ipfp_max_probability_change": native[
                    "ipfp_max_probability_change"],
                "mean_stay_probability": native["mean_stay_probability"],
                "max_stay_probability": native["max_stay_probability"],
            })
        diagnostics.update({
            "transition_method_requested": requested_method,
            "sampling_transition_method_requested": sampling_method,
            "model_transition_method_requested": requested_method,
            "transition_storage": "sparse",
            "n": n,
        })
        return diagnostics
    if method_used == "spectral_matrix":
        diagnostics = {
            "dt": native["dt"],
            "alpha": native["alpha"],
            "beta": native["beta"],
            "raw_min_entry": native["raw_min_entry"],
            "raw_negative_mass": native["raw_negative_mass"],
            "max_row_sum_error_before_normalization": native[
                "max_row_sum_error_before_normalization"],
            "stationary_error": native["stationary_error"],
            "clipped_negative": native["clipped_negative"],
            "transition_method_requested": sampling_method,
            "transition_method": method_used,
            "probability_cleanup_applied": native[
                "probability_cleanup_applied"],
            "probability_cleanup_negative_mass": native[
                "probability_cleanup_negative_mass"],
            "probability_min_entry_before_cleanup": native[
                "probability_min_entry_before_cleanup"],
        }
    else:
        diagnostics = {
            "dt": native["dt"],
            "alpha": native["alpha"],
            "beta": native["beta"],
            "gh_order": native["gh_order"],
            "min_entry": native["min_entry"],
            "max_row_sum_error": native["max_row_sum_error"],
            "stationary_error": native["stationary_error"],
            "transition_method_requested": sampling_method,
            "transition_method": method_used,
        }
        if int(native["spectral_status"]) != 0:
            diagnostics["spectral_error"] = (
                "FloatingPointError: C++ Jacobi spectral transition failed")
    diagnostics["sampling_transition_method_requested"] = sampling_method
    diagnostics["model_transition_method_requested"] = requested_method
    diagnostics["n"] = n
    return diagnostics


def sample_grid_trajectory(
        kappa, m, xi, n, *, rng=None, basis_order=32, quad_order=None,
        transition_method="auto", clip_negative=False,
        negative_mass_tol=1e-5, gh_order=5, transition_storage="dense",
        stationarity_correction="none", memory_budget_bytes=None,
        return_diagnostics=False):
    """Run the public fixed-draw TM-grid protocol through the native facade."""
    n = _validate_nonnegative_int(n, "n")
    stationarity_correction = normalize_jacobi_stationarity_correction(
        stationarity_correction)
    transition_storage = normalize_jacobi_transition_storage(
        transition_storage)
    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        if return_diagnostics:
            return empty, {
                "transition_method_requested": str(transition_method),
                "transition_method": "not_built",
                "n": 0,
            }
        return empty
    if stationary_shape(kappa, m, xi) is None:
        raise ValueError("invalid Jacobi parameters")
    basis_order = _validate_order(basis_order, "basis_order")
    if quad_order is None:
        quad_order = default_quad_order(basis_order)
    quad_order = _validate_order(quad_order, "quad_order")
    gh_order = _validate_order(gh_order, "gh_order")
    if basis_order > quad_order:
        raise ValueError("quad_order must be >= basis_order")

    requested_method = normalize_jacobi_strategy_transition_method(
        transition_method)
    sampling_method = (
        "auto" if requested_method == "spectral_coeff"
        else requested_method)
    use_sparse_sampling = (
        n > 1
        and (
            sampling_method == "local"
            or (
                sampling_method == "local_fixed"
                and transition_storage == "sparse"
            )
        )
    )
    sampling_storage = "sparse" if use_sparse_sampling else "dense"
    if stationarity_correction != "none" and sampling_method != "local":
        raise ValueError(
            "stationarity_correction currently requires "
            "transition_method='local'")
    validate_float64_allocation(
        (0,), name="Jacobi sampling memory budget",
        memory_budget_bytes=memory_budget_bytes)
    budget = _memory_budget(memory_budget_bytes)
    if sampling_storage == "sparse" and n > 1:
        try:
            sparse_workspace_bytes = estimate_sparse_workspace(
                quad_order=quad_order,
                gh_order=gh_order,
                correction=stationarity_correction,
                memory_budget_bytes=budget,
            )
            retained = estimate_sparse_storage(
                quad_order=quad_order,
                gh_order=gh_order,
                correction=stationarity_correction,
                memory_budget_bytes=budget,
            )
            native_peak_bytes = max(
                sparse_workspace_bytes,
                retained + (2 * quad_order + n) * 8,
            )
            boundary_bytes = validate_float64_allocation(
                (3, n), name="sparse Jacobi fixed-draw boundary buffers")
            required_bytes = native_peak_bytes + boundary_bytes
            validate_float64_allocation(
                (required_bytes // np.dtype(np.float64).itemsize,),
                name="sparse Jacobi sampling workspace",
                memory_budget_bytes=budget,
            )
        except MemoryError as exc:
            _sampling_memory_error(exc)
    else:
        _validate_sampling_workspace(
            n=n,
            quad_order=quad_order,
            basis_order=basis_order,
            gh_order=gh_order,
            memory_budget_bytes=budget,
        )
    if rng is None:
        rng = np.random.default_rng()
    uniforms = np.asarray(rng.random(n), dtype=np.float64)
    path, diagnostics = sample_grid_trajectory_fixed_draws(
        kappa,
        m,
        xi,
        uniforms,
        basis_order=basis_order,
        quad_order=quad_order,
        method=sampling_method,
        storage=sampling_storage,
        correction=stationarity_correction,
        clip_negative=clip_negative,
        negative_mass_tol=negative_mass_tol,
        gh_order=gh_order,
        memory_budget_bytes=budget,
    )
    if return_diagnostics:
        diagnostics = _legacy_grid_sampling_diagnostics(
            diagnostics,
            sampling_method=sampling_method,
            requested_method=requested_method,
            storage=sampling_storage,
            correction=stationarity_correction,
            n=n,
        )
        return path, diagnostics
    return path


def _effective_lamperti_chunk(
        *, n, substeps, requested, memory_budget_bytes):
    if n <= 1:
        validate_float64_allocation(
            (n,),
            name="Lamperti-Euler Jacobi path",
            memory_budget_bytes=memory_budget_bytes,
        )
        return 0
    elements_per_interval = 2 * substeps + 2
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


def sample_lamperti_trajectory(
        kappa, m, xi, n, *, rng=None, substeps=8, boundary="reflect",
        eps=1e-10, engine="native",
        chunk_observations=DEFAULT_LAMPERTI_CHUNK_OBSERVATIONS,
        memory_budget_bytes=None, return_diagnostics=False):
    """Run chunked Lamperti--Euler sampling through fixed-draw native calls."""
    n = _validate_nonnegative_int(n, "n")
    substeps = validate_positive_int(substeps, "lamperti_substeps")
    boundary = normalize_lamperti_boundary(boundary)
    eps = validate_lamperti_eps(eps)
    engine = normalize_lamperti_engine(engine)
    chunk_observations = validate_positive_int(
        chunk_observations, "lamperti_chunk_observations")
    shapes = stationary_shape(kappa, m, xi)
    if shapes is None:
        raise ValueError("invalid Jacobi parameters")
    kappa = float(kappa)
    m = float(m)
    xi = float(xi)
    alpha, beta = shapes

    validate_float64_allocation(
        (0,), name="Lamperti-Euler memory budget",
        memory_budget_bytes=memory_budget_bytes)
    budget = _memory_budget(memory_budget_bytes)
    effective_chunk = _effective_lamperti_chunk(
        n=n,
        substeps=substeps,
        requested=chunk_observations,
        memory_budget_bytes=budget,
    )
    diagnostics = {
        "sampling_method": "lamperti_euler",
        "sampling_engine": engine,
        "boundary_policy": boundary,
        "substeps": substeps,
        "drift_eps": eps,
        "stationary_alpha": alpha,
        "stationary_beta": beta,
        "stationary_boundary_singular": bool(alpha < 1.0 or beta < 1.0),
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

    y = float(lamperti(np.array([path[0]], dtype=np.float64), xi)[0])
    interventions = 0
    output_offset = 1
    while output_offset < n:
        block = min(effective_chunk, n - output_offset)
        innovations = np.asarray(
            rng.standard_normal((block, substeps)), dtype=np.float64)
        native = sample_lamperti_chunk_fixed_draws(
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
    return (path, diagnostics) if return_diagnostics else path


def sample_prepared_sparse_trajectory_fixed_draws(
        tau, weights, indices, probabilities, counts, uniforms):
    """Sample a caller-prepared sparse transition wholly in native C++."""
    draws = np.ascontiguousarray(
        np.asarray(uniforms, dtype=np.float64).ravel())
    result = load().jacobi_sample_prepared_sparse_trajectory(
        np.ascontiguousarray(np.asarray(tau, dtype=np.float64).ravel()),
        np.ascontiguousarray(np.asarray(weights, dtype=np.float64).ravel()),
        np.ascontiguousarray(np.asarray(indices, dtype=np.int64)),
        np.ascontiguousarray(np.asarray(probabilities, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(counts, dtype=np.int64).ravel()),
        draws,
    )
    _raise(result, "fixed-draw prepared sparse trajectory sampling")
    used = int(result["draws_used"])
    if used != draws.size:
        raise RuntimeError(
            "C++ sparse Jacobi sampler violated its fixed-draw contract")
    return np.asarray(result["tau"], dtype=np.float64), used


def sample_state_distribution_fixed_draws(
        copula, tau, probability, selection_draws, jitter_draws, *,
        mode="grid", theta_cap=None):
    """Sample a native filtered/conditioned state and map tau to theta."""
    from pyscarcopula._native import _descriptors

    module = load()
    normalized = str(mode).lower()
    native_mode = {
        "grid": module.JacobiStateSamplingMode.Grid,
        "histogram": module.JacobiStateSamplingMode.Histogram,
    }.get(normalized)
    if native_mode is None:
        raise ValueError("predictive_r_mode must be 'grid' or 'histogram'")
    spec = _descriptors.make_copula_ops_spec(module, copula)
    selection = np.ascontiguousarray(
        np.asarray(selection_draws, dtype=np.float64).ravel())
    jitter = np.ascontiguousarray(
        np.asarray(jitter_draws, dtype=np.float64).ravel())
    result = module.jacobi_sample_state_distribution(
        spec,
        np.ascontiguousarray(np.asarray(tau, dtype=np.float64).ravel()),
        np.ascontiguousarray(
            np.asarray(probability, dtype=np.float64).ravel()),
        selection,
        jitter,
        native_mode,
        math.nan if theta_cap is None else float(theta_cap),
    )
    _raise(result, "fixed-draw filtered-state sampling")
    selection_used = int(result["selection_draws_used"])
    jitter_used = int(result["jitter_draws_used"])
    if selection_used != selection.size or jitter_used != jitter.size:
        raise RuntimeError(
            "C++ Jacobi state sampler violated its fixed-draw contract")
    return (
        np.asarray(result["tau"], dtype=np.float64),
        np.asarray(result["parameters"], dtype=np.float64),
        {
            "selection_draws_used": selection_used,
            "jitter_draws_used": jitter_used,
        },
    )


def state_histogram_cells(tau, indices):
    """Return native midpoint cells for selected Jacobi grid indices."""
    result = load().jacobi_state_histogram_cells(
        np.ascontiguousarray(np.asarray(tau, dtype=np.float64).ravel()),
        np.ascontiguousarray(np.asarray(indices, dtype=np.int64).ravel()),
    )
    _raise(result, "Jacobi predictive histogram cells")
    return (
        np.asarray(result["left"], dtype=np.float64),
        np.asarray(result["right"], dtype=np.float64),
    )


def dense_transition(
        kappa, m, xi, *, n_obs, quad_order, basis_order,
        gh_order, method, raw_backend=None, clip_negative=False,
        negative_mass_tol=1e-5, return_grad=False,
        memory_budget_bytes=1024**3):
    module = load()
    config = _transition_config(
        n_obs=n_obs,
        quad_order=quad_order,
        basis_order=basis_order,
        gh_order=gh_order,
        method=method,
        clip_negative=clip_negative,
        negative_mass_tol=negative_mass_tol,
        return_grad=return_grad,
        memory_budget_bytes=memory_budget_bytes,
    )
    function = {
        "spectral": module.jacobi_build_spectral_transition,
        "local": module.jacobi_build_local_transition,
        "local_fixed": module.jacobi_build_fixed_transition,
        None: module.jacobi_build_dense_transition,
    }[raw_backend]
    result = function(_params(kappa, m, xi), config)
    _transition_raise(result, "dense transition construction")
    derivatives = np.asarray(result["derivatives"], dtype=np.float64)
    if derivatives.size == 0:
        derivatives = None
    return (
        np.asarray(result["tau"], dtype=np.float64),
        np.asarray(result["weights"], dtype=np.float64),
        np.asarray(result["probabilities"], dtype=np.float64),
        derivatives,
        np.asarray(result["spectral_powers"], dtype=np.float64),
        _transition_diagnostics(result["diagnostics"]),
    )


def coefficient_transition(
        kappa, m, xi, *, n_obs, quad_order, basis_order,
        memory_budget_bytes=1024**3):
    config = _transition_config(
        n_obs=n_obs,
        quad_order=quad_order,
        basis_order=basis_order,
        gh_order=1,
        method="spectral_coeff",
        storage="dense",
        memory_budget_bytes=memory_budget_bytes,
    )
    result = load().jacobi_build_coefficient_transition(
        _params(kappa, m, xi), config)
    _transition_raise(result, "spectral-coefficient transition setup")
    return (
        np.asarray(result["tau"], dtype=np.float64),
        np.asarray(result["weights"], dtype=np.float64),
        np.asarray(result["basis"], dtype=np.float64),
        np.asarray(result["spectral_powers"], dtype=np.float64),
        _transition_diagnostics(result["diagnostics"]),
    )


def apply_coefficient_transition(powers, coefficients):
    result = load().jacobi_apply_coefficient_transition(
        np.asarray(powers, dtype=np.float64),
        np.asarray(coefficients, dtype=np.float64),
    )
    _raise(result, "spectral-coefficient propagation")
    return np.asarray(result["values"], dtype=np.float64)


def sparse_transition(
        kappa, m, xi, *, n_obs, quad_order, basis_order,
        gh_order, method, correction="none", return_grad=False,
        memory_budget_bytes=1024**3):
    config = _transition_config(
        n_obs=n_obs,
        quad_order=quad_order,
        basis_order=basis_order,
        gh_order=gh_order,
        method=method,
        storage="sparse",
        correction=correction,
        return_grad=return_grad,
        memory_budget_bytes=memory_budget_bytes,
    )
    result = load().jacobi_build_sparse_transition(
        _params(kappa, m, xi), config)
    _transition_raise(result, "sparse transition construction")
    derivatives = np.asarray(result["derivatives"], dtype=np.float64)
    if derivatives.size == 0:
        derivatives = None
    return (
        np.asarray(result["tau"], dtype=np.float64),
        np.asarray(result["weights"], dtype=np.float64),
        np.asarray(result["indices"], dtype=np.intp),
        np.asarray(result["probabilities"], dtype=np.float64),
        np.asarray(result["counts"], dtype=np.intp),
        derivatives,
        _transition_diagnostics(result["diagnostics"]),
    )


def sparse_left_multiply(indices, probabilities, counts, values):
    result = load().jacobi_sparse_left_multiply(
        np.asarray(indices, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
        np.asarray(counts, dtype=np.int64),
        np.asarray(values, dtype=np.float64),
    )
    _raise(result, "sparse transition matvec")
    return np.asarray(result["values"], dtype=np.float64)


def sparse_to_dense(indices, probabilities, counts):
    """Materialize sparse transition rows through the native boundary."""
    result = load().jacobi_sparse_to_dense(
        np.asarray(indices, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
        np.asarray(counts, dtype=np.int64),
    )
    _raise(result, "sparse transition dense materialization")
    return np.asarray(result["values"], dtype=np.float64)


def sparse_full_horizon_diagnostics(
        kappa, m, xi, tau, weights, indices, probabilities, counts, steps):
    result = load().jacobi_sparse_full_horizon_diagnostics(
        _params(kappa, m, xi),
        np.asarray(tau, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        np.asarray(indices, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
        np.asarray(counts, dtype=np.int64),
        int(steps),
    )
    _raise(result, "full-horizon transition diagnostics")
    return dict(result["diagnostics"])


def select_sparse_order(
        kappa, m, xi, *, n_obs, quad_orders, basis_order, gh_order,
        max_full_horizon_tv, max_relative_variance_error,
        max_conditional_mean_rmse, max_lag_one_correlation_error,
        memory_budget_bytes=1024**3, require_pass=False):
    module = load()
    config = _transition_config(
        n_obs=n_obs,
        quad_order=quad_orders[0],
        basis_order=min(basis_order, quad_orders[0]),
        gh_order=gh_order,
        method="local",
        storage="sparse",
        memory_budget_bytes=memory_budget_bytes,
    )
    thresholds = module.JacobiAdaptiveThresholds()
    thresholds.max_full_horizon_tv = float(max_full_horizon_tv)
    thresholds.max_relative_variance_error = float(
        max_relative_variance_error)
    thresholds.max_conditional_mean_rmse = float(
        max_conditional_mean_rmse)
    thresholds.max_lag_one_correlation_error = float(
        max_lag_one_correlation_error)
    result = module.jacobi_select_sparse_order(
        _params(kappa, m, xi),
        config,
        [int(order) for order in quad_orders],
        thresholds,
        bool(require_pass),
    )
    if int(result["status"]) == 7 and int(
            result.get("failure_operation", -1)) == 6:
        raise RuntimeError(
            "no sparse Jacobi quad_order satisfied the full-horizon gates")
    if int(result["status"]) == 2 and int(
            result.get("failure_operation", -1)) == 7:
        raise MemoryError(
            "memory budget prevented every sparse Jacobi candidate")
    _raise(result, "adaptive sparse-order selection")
    transition = result["transition"]
    diagnostics = _transition_diagnostics(transition["diagnostics"])
    candidates = []
    for raw in result["candidates"]:
        record = {
            "quad_order": int(raw["quad_order"]),
            "passed": bool(raw["passed"]),
            "memory_limited": bool(raw["memory_limited"]),
        }
        if int(raw["status"]) == 0:
            record["retained_bytes"] = int(raw["retained_bytes"])
            record.update(dict(raw["diagnostics"]))
        candidates.append(record)
    return (
        np.asarray(transition["tau"], dtype=np.float64),
        np.asarray(transition["weights"], dtype=np.float64),
        np.asarray(transition["indices"], dtype=np.intp),
        np.asarray(transition["probabilities"], dtype=np.float64),
        np.asarray(transition["counts"], dtype=np.intp),
        diagnostics,
        {
            "selected_quad_order": int(result["selected_quad_order"]),
            "passed": bool(result["passed"]),
            "exhausted": bool(result["exhausted"]),
            "candidates": candidates,
        },
    )


__all__ = [
    "DEFAULT_JACOBI_MEMORY_BUDGET_BYTES",
    "DEFAULT_LAMPERTI_CHUNK_OBSERVATIONS",
    "MAX_JACOBI_ORDER",
    "PreparedScarJacobiEvaluator",
    "apply_coefficient_transition",
    "coefficient_transition",
    "copula_supported",
    "estimate_sampling_workspace",
    "estimate_sparse_storage",
    "estimate_sparse_workspace",
    "estimate_workspace",
    "fixed_tau_rule",
    "fixed_shape_rule",
    "gauss_hermite_rule",
    "inverse_lamperti",
    "jacobi_rule",
    "lamperti",
    "lamperti_drift",
    "normalize_lamperti_boundary",
    "normalize_lamperti_engine",
    "normalize_sampling_method",
    "dense_transition",
    "default_quad_order",
    "parameter_to_tau",
    "resolve_dt",
    "select_sparse_order",
    "sparse_full_horizon_diagnostics",
    "sparse_left_multiply",
    "sparse_to_dense",
    "sparse_transition",
    "transition_powers",
    "physical_to_raw",
    "raw_bounds",
    "raw_to_physical",
    "sample_grid_trajectory_fixed_draws",
    "sample_grid_trajectory",
    "sample_lamperti_chunk_fixed_draws",
    "sample_lamperti_trajectory",
    "sample_prepared_sparse_trajectory_fixed_draws",
    "sample_state_distribution_fixed_draws",
    "state_histogram_cells",
    "shape_is_supported",
    "stationary_shape",
    "tau_to_parameter",
    "validate_copula_mapping",
    "validate_lamperti_eps",
]
