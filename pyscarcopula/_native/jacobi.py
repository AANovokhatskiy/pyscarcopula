"""Typed Python adapter for the native Jacobi domain core."""

from __future__ import annotations

import math

import numpy as np

from pyscarcopula._native._extension import load
from pyscarcopula._native.errors import raise_for_status


def _params(kappa, m, xi):
    module = load()
    params = module.JacobiParams()
    params.kappa = float(kappa)
    params.m = float(m)
    params.xi = float(xi)
    return params


def _raise(result, operation):
    raise_for_status(result, operation, prefix="C++ Jacobi")


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
        from pyscarcopula.numerical import _cpp_copula

        module = load()
        observations = np.ascontiguousarray(u, dtype=np.float64)
        if (
                observations.ndim != 2
                or observations.shape[1] != 2
                or len(observations) < 1):
            raise ValueError(
                "u must be a 2D float64 array with shape (n, 2), n >= 1")
        if quad_order is None:
            quad_order = default_quad_order(basis_order)
        method = str(transition_method)
        transition = _transition_config(
            n_obs=len(observations),
            quad_order=quad_order,
            basis_order=basis_order,
            gh_order=gh_order,
            method=method,
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
        spec = _cpp_copula.make_copula_ops_spec(module, copula)
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
    "PreparedScarJacobiEvaluator",
    "apply_coefficient_transition",
    "coefficient_transition",
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
    "dense_transition",
    "default_quad_order",
    "resolve_dt",
    "select_sparse_order",
    "sparse_full_horizon_diagnostics",
    "sparse_left_multiply",
    "sparse_transition",
    "transition_powers",
    "physical_to_raw",
    "raw_bounds",
    "raw_to_physical",
    "shape_is_supported",
    "stationary_shape",
]
