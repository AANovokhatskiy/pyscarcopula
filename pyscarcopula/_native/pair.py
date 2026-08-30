"""Shared native adapter for built-in bivariate copula operations."""

from __future__ import annotations

import numpy as np

from pyscarcopula._utils import broadcast
from pyscarcopula._native import _descriptors, _extension
from pyscarcopula._native.errors import NativeError
from pyscarcopula.numerical._arrays import as_float64_array


def supported(copula) -> bool:
    return _descriptors.supported_for_copula_ops(copula)


def _module_and_spec(copula, *, unrotated=False):
    module = _extension.load()
    spec = _descriptors.make_copula_ops_spec(module, copula)
    if unrotated:
        spec.rotation = module.Rotation.R0
    return module, spec


def _vector(values, *, name="values") -> np.ndarray:
    return np.ascontiguousarray(
        np.atleast_1d(as_float64_array(values, name=name)).ravel()
    )


def _array(values, *, name) -> np.ndarray:
    return np.ascontiguousarray(as_float64_array(values, name=name))


def _pair_and_r(first, second, r):
    if r is None:
        r = 0.0
    first_arr, second_arr, r_arr = broadcast(first, second, r)
    pair = np.column_stack((first_arr, second_arr))
    return np.ascontiguousarray(pair), np.ascontiguousarray(r_arr)


def _finite(values, operation):
    out = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(out)):
        raise NativeError(f"C++ {operation} returned non-finite values")
    return out


def transform(copula, x) -> np.ndarray:
    module, spec = _module_and_spec(copula)
    return _finite(
        module.copula_transform(spec, _vector(x, name="x")),
        "copula_transform",
    )


def inverse_transform(copula, r) -> np.ndarray:
    module, spec = _module_and_spec(copula)
    return _finite(
        module.copula_inverse_transform(spec, _vector(r, name="r")),
        "copula_inverse_transform",
    )


def dtransform(copula, x) -> np.ndarray:
    module, spec = _module_and_spec(copula)
    return _finite(
        module.copula_dtransform(spec, _vector(x, name="x")),
        "copula_dtransform",
    )


def tau_to_param(copula, tau) -> np.ndarray:
    module, spec = _module_and_spec(copula)
    return _finite(
        module.copula_tau_to_param(spec, _vector(tau, name="tau")),
        "copula_tau_to_param",
    )


def param_to_tau(copula, r) -> np.ndarray:
    module, spec = _module_and_spec(copula)
    return _finite(
        module.copula_param_to_tau(spec, _vector(r, name="r")),
        "copula_param_to_tau",
    )


def log_pdf(copula, u1, u2, r, *, unrotated=False) -> np.ndarray:
    module, spec = _module_and_spec(copula, unrotated=unrotated)
    pair, r_arr = _pair_and_r(u1, u2, r)
    return _finite(
        module.copula_log_pdf(spec, pair, r_arr),
        "copula_log_pdf",
    )


def pdf(copula, u1, u2, r, *, unrotated=False) -> np.ndarray:
    module, spec = _module_and_spec(copula, unrotated=unrotated)
    pair, r_arr = _pair_and_r(u1, u2, r)
    return _finite(module.copula_pdf(spec, pair, r_arr), "copula_pdf")


def dlog_pdf_dr(copula, u1, u2, r, *, unrotated=False) -> np.ndarray:
    module, spec = _module_and_spec(copula, unrotated=unrotated)
    pair, r_arr = _pair_and_r(u1, u2, r)
    return _finite(
        module.copula_dlog_pdf_dr(spec, pair, r_arr),
        "copula_dlog_pdf_dr",
    )


def h(copula, u_conditioned, u_given, r, *, unrotated=False) -> np.ndarray:
    module, spec = _module_and_spec(copula, unrotated=unrotated)
    pair, r_arr = _pair_and_r(u_conditioned, u_given, r)
    return _finite(module.copula_h(spec, pair, r_arr), "copula_h")


def h_pair(copula, u, v, r, *, unrotated=False):
    module, spec = _module_and_spec(copula, unrotated=unrotated)
    pair, r_arr = _pair_and_r(u, v, r)
    first, second = module.copula_h_pair(spec, pair, r_arr)
    return (
        _finite(first, "copula_h_pair"),
        _finite(second, "copula_h_pair"),
    )


def h_inverse(copula, q, u_given, r, *, unrotated=False) -> np.ndarray:
    module, spec = _module_and_spec(copula, unrotated=unrotated)
    pair, r_arr = _pair_and_r(q, u_given, r)
    return _finite(
        module.copula_h_inverse(spec, pair, r_arr),
        "copula_h_inverse",
    )


def sample_from_uniforms(copula, uniforms, r) -> np.ndarray:
    """Apply the native fixed-uniform pair sampling transform."""
    module, spec = _module_and_spec(copula)
    draws = _array(uniforms, name="uniforms")
    if draws.ndim != 2 or draws.shape[1] != 2:
        raise ValueError(
            f"uniforms must have shape (n, 2), got {draws.shape}")
    parameters = _vector(0.0 if r is None else r)
    values = module.copula_sample_from_uniforms(spec, draws, parameters)
    return _finite(values, "copula_sample_from_uniforms")


def sample_from_rng_draws(copula, draws, auxiliary, r) -> np.ndarray:
    """Apply the native family-specific transform to caller-owned RNG draws."""
    module, spec = _module_and_spec(copula)
    primary = _array(draws, name="draws")
    extra = _array(auxiliary, name="auxiliary")
    if primary.ndim != 2 or primary.shape[1] != 2:
        raise ValueError(
            f"draws must have shape (n, 2), got {primary.shape}")
    if extra.ndim != 2:
        raise ValueError(
            f"auxiliary must be a 2D array, got {extra.shape}")
    parameters = _vector(0.0 if r is None else r)
    values = module.copula_sample_from_rng_draws(
        spec, primary, extra, parameters)
    return _finite(values, "copula_sample_from_rng_draws")


def conditional_sample_from_uniforms(
        copula, uniforms, r, *, given_coordinate, given_value) -> np.ndarray:
    """Apply the native fixed-uniform inverse-h conditional transform."""
    module, spec = _module_and_spec(copula)
    draws = _vector(uniforms, name="uniforms")
    parameters = _vector(0.0 if r is None else r)
    values = module.copula_conditional_sample_from_uniforms(
        spec,
        draws,
        parameters,
        int(given_coordinate),
        float(given_value),
    )
    return _finite(values, "copula_conditional_sample_from_uniforms")


def pdf_grid(copula, u, x_grid) -> np.ndarray:
    module, spec = _module_and_spec(copula)
    values = module.copula_pdf_grid(
        spec,
        _array(u, name="u"),
        _vector(x_grid, name="x_grid"),
    )
    return _finite(values, "copula_pdf_grid")


def pdf_parameter_grid(copula, u, r_grid) -> np.ndarray:
    module, spec = _module_and_spec(copula)
    values = module.copula_pdf_parameter_grid(
        spec,
        _array(u, name="u"),
        _vector(r_grid, name="r_grid"),
    )
    return _finite(values, "copula_pdf_parameter_grid")


def h_parameter_grid(copula, u, r_grid) -> np.ndarray:
    module, spec = _module_and_spec(copula)
    values = module.copula_h_parameter_grid(
        spec,
        _array(u, name="u"),
        _vector(r_grid, name="r_grid"),
    )
    return _finite(values, "copula_h_parameter_grid")


def pdf_and_grad_grid(copula, u, x_grid):
    module, spec = _module_and_spec(copula)
    pdf_values, grad_values = module.copula_pdf_and_grad_grid(
        spec,
        _array(u, name="u"),
        _vector(x_grid, name="x_grid"),
    )
    return (
        _finite(pdf_values, "copula_pdf_and_grad_grid"),
        _finite(grad_values, "copula_pdf_and_grad_grid"),
    )
