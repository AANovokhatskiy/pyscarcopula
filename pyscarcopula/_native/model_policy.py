"""Thin adapters for C++-owned model domains and initialization policy."""

from __future__ import annotations

from pyscarcopula._native import _descriptors, _extension
from pyscarcopula._native.errors import raise_for_status

import numpy as np


def _checked(result, operation):
    raise_for_status(result, operation, prefix="C++ model policy")
    return result


def public_bounds(copula):
    module = _extension.load()
    if hasattr(copula, "_native_pair_family"):
        spec = _descriptors.make_pair_model_policy_spec(module, copula)
    else:
        spec = _descriptors.make_multivariate_model_policy_spec(
            module, copula)
    result = _checked(
        module.model_public_parameter_bounds(spec),
        "public parameter bounds")
    return list(zip(result["lower"], result["upper"]))


def latent_bounds(process, *, gamma_bound=None, beta_bound=None):
    module = _extension.load()
    result = {
        "ou": module.model_ou_parameter_bounds,
        "jacobi": module.model_jacobi_parameter_bounds,
    }.get(process)
    if result is None:
        if process != "gas":
            raise ValueError(f"unknown latent process: {process!r}")
        native = module.model_gas_parameter_bounds(
            float(gamma_bound), float(beta_bound))
    else:
        native = result()
    native = _checked(native, f"{process} parameter bounds")
    return tuple(native["lower"]), tuple(native["upper"])


def ou_scaled_optimizer_bounds(scale):
    values = np.asarray(scale, dtype=np.float64)
    if values.shape != (3,):
        raise ValueError("OU optimizer scale must have shape (3,)")
    result = _checked(
        _extension.load().model_ou_scaled_optimizer_bounds(
            values.tolist()),
        "scaled OU optimizer bounds")
    return (
        np.asarray(result["lower"], dtype=np.float64),
        np.asarray(result["upper"], dtype=np.float64),
    )


def ou_log_stationary_optimizer_bounds(scale_bounds=None):
    module = _extension.load()
    if scale_bounds is None:
        lower = upper = None
    else:
        lower, upper = scale_bounds
    parameter = _checked(
        module.model_ou_log_stationary_parameter_bounds_for_scale(
            lower, upper),
        "OU log-stationary optimizer bounds")
    result = _checked(
        module.model_ou_log_stationary_result_bounds(),
        "OU log-stationary result bounds")
    return (
        (tuple(parameter["lower"]), tuple(parameter["upper"])),
        (tuple(result["lower"]), tuple(result["upper"])),
    )


def default_gas_limits():
    result = _checked(
        _extension.load().model_default_gas_parameter_bounds(),
        "default GAS parameter bounds")
    return float(result["upper"][1]), float(result["upper"][2])


def stationary_scale_bounds():
    result = _checked(
        _extension.load().model_stationary_scale_bounds(),
        "stationary-scale bounds")
    return result["lower"][0], result["upper"][0]


def normalize_positive_bounds(value, name):
    if value is None:
        lower = upper = None
    else:
        if len(value) != 2:
            raise ValueError(f"{name} must be a (lower, upper) pair")
        lower, upper = value
    result = _extension.load().model_normalize_positive_bounds(lower, upper)
    if int(result["status"]) != 0:
        raise ValueError(f"{name} must satisfy 0 < lower < upper")
    return result["lower"][0], result["upper"][0]


def normalize_optional_positive_bounds(value, name):
    if len(value) != 2:
        raise ValueError(f"{name} must be a (lower, upper) pair")
    lower, upper = value
    resolved = normalize_positive_bounds(value, name)
    return (
        None if lower is None else resolved[0],
        None if upper is None else resolved[1],
    )


def student_fit_policy(dimension, *, stochastic):
    result = _checked(
        _extension.load().model_student_fit_policy(
            int(dimension), bool(stochastic)),
        "Student fit parameter policy")
    bounds = (
        result["lower"] if result["has_lower"] else None,
        result["upper"] if result["has_upper"] else None,
    )
    return float(result["initial"]), bounds


def equicorr_fit_policy():
    result = _checked(
        _extension.load().model_equicorr_fit_policy(),
        "equicorrelation fit parameter policy")
    return (
        float(result["initial"]),
        (float(result["lower"]), float(result["upper"])),
    )


def default_pair_mle_parameter(copula):
    module = _extension.load()
    spec = _descriptors.make_copula_ops_spec(module, copula)
    result = _checked(
        module.model_default_pair_mle_parameter(spec),
        "default pair MLE parameter")
    return float(result["value"])


def pair_mle_initial_parameter(copula, tau):
    """Return a native MLE start; exact Kendall limits require finite bounds."""
    module = _extension.load()
    spec = _descriptors.make_copula_ops_spec(module, copula)
    result = _checked(
        module.model_pair_mle_initial_parameter(spec, float(tau)),
        "pair MLE initial parameter (exact Kendall limits require finite bounds)")
    return float(result["value"])


def gas_default_initial_point(mu):
    result = _checked(
        _extension.load().model_gas_default_initial_point(float(mu)),
        "default GAS initial point")
    return np.asarray(result["values"], dtype=np.float64)


def optimizer_unit_scale(parameters):
    """Return native generic optimizer scaling ``max(abs(x), 1)``."""
    values = np.asarray(parameters, dtype=np.float64).reshape(-1)
    result = _checked(
        _extension.load().model_optimizer_unit_scale(values.tolist()),
        "optimizer unit scale")
    return np.asarray(result["values"], dtype=np.float64)


def project_optimizer_point(parameters, lower, upper):
    """Project an optimizer point onto caller-supplied native bounds."""
    values = np.asarray(parameters, dtype=np.float64).reshape(-1)
    lower_values = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper_values = np.asarray(upper, dtype=np.float64).reshape(-1)
    result = _checked(
        _extension.load().model_project_optimizer_point(
            values.tolist(), lower_values.tolist(), upper_values.tolist()),
        "optimizer initial-point projection")
    return np.asarray(result["values"], dtype=np.float64)


def ou_kappa_dt(kappa, n_obs):
    result = _checked(
        _extension.load().model_ou_kappa_dt(float(kappa), int(n_obs)),
        "OU kappa-dt policy")
    return float(result["value"])


def ou_auto_backend(kappa, n_obs, small_kdt):
    result = _checked(
        _extension.load().model_ou_auto_backend(
            float(kappa), int(n_obs), float(small_kdt)),
        "OU automatic backend policy")
    return {0: "spectral", 1: "local", 2: "matrix"}[int(result["backend"])]


def ou_adaptive_spectral_basis_order(kappa, n_obs):
    result = _checked(
        _extension.load().model_ou_adaptive_spectral_basis_order(
            float(kappa), int(n_obs)),
        "OU adaptive spectral basis policy")
    return int(result["order"])


def ou_resolve_quad_order(basis_order, explicit_quad_order=None):
    result = _checked(
        _extension.load().model_ou_resolve_quad_order(
            int(basis_order),
            None if explicit_quad_order is None else int(explicit_quad_order)),
        "OU quadrature-order policy")
    return int(result["order"])


def _require_native_numerical_failure(error):
    if int(getattr(error, "status", -1)) != 7:
        raise error


def optimizer_failure_objective(error, fail_value=None):
    """Resolve a structured numerical failure through the C++ policy."""
    _require_native_numerical_failure(error)
    operation = _extension.load().model_optimizer_failure_objective
    result = _checked(
        operation() if fail_value is None else operation(float(fail_value)),
        "optimizer failure objective")
    return float(result["objective"])


def optimizer_failure_evaluation(
        parameters, initial_parameters, fail_value, *, directional_gradient):
    """Return the C++-owned penalty and gradient for a rejected evaluation."""
    parameters = np.asarray(parameters, dtype=np.float64).reshape(-1)
    initial_parameters = np.asarray(
        initial_parameters, dtype=np.float64).reshape(-1)
    result = _checked(
        _extension.load().model_optimizer_failure_evaluation(
            parameters.tolist(), initial_parameters.tolist(),
            float(fail_value), bool(directional_gradient)),
        "optimizer failure evaluation")
    return (
        float(result["objective"]),
        np.asarray(result["gradient"], dtype=np.float64),
    )


def optimizer_numerical_failure_evaluation(
        error, parameters, initial_parameters, fail_value, *,
        directional_gradient):
    """Resolve only a structured C++ numerical-failure exception."""
    _require_native_numerical_failure(error)
    return optimizer_failure_evaluation(
        parameters, initial_parameters, fail_value,
        directional_gradient=directional_gradient)


def optimizer_numerical_failure_evaluation_for_size(
        error, gradient_size, fail_value):
    """Resolve a structured failure when only the gradient shape is known."""
    _require_native_numerical_failure(error)
    result = _checked(
        _extension.load().model_optimizer_failure_evaluation_for_size(
            int(gradient_size), float(fail_value)),
        "optimizer failure evaluation by gradient size")
    return (
        float(result["objective"]),
        np.asarray(result["gradient"], dtype=np.float64),
    )
