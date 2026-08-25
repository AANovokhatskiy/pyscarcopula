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


__all__ = [
    "estimate_sampling_workspace",
    "estimate_workspace",
    "fixed_tau_rule",
    "fixed_shape_rule",
    "gauss_hermite_rule",
    "inverse_lamperti",
    "jacobi_rule",
    "lamperti",
    "lamperti_drift",
    "physical_to_raw",
    "raw_bounds",
    "raw_to_physical",
    "shape_is_supported",
    "stationary_shape",
]
