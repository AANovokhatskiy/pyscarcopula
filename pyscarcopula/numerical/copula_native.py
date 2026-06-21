"""Shared native adapter for built-in bivariate copula operations."""

from __future__ import annotations

import numpy as np

from pyscarcopula._utils import broadcast
from pyscarcopula.numerical import _cpp_copula, _cpp_extension
from pyscarcopula.numerical._cpp_extension import CppError


def available() -> bool:
    return _cpp_extension.available()


def supported(copula) -> bool:
    return _cpp_copula.supported_for_copula_ops(copula)


def _module_and_spec(copula, *, unrotated=False):
    module = _cpp_extension.load()
    spec = _cpp_copula.make_copula_ops_spec(module, copula)
    if unrotated:
        spec.rotation = module.Rotation.R0
    return module, spec


def _vector(values) -> np.ndarray:
    return np.ascontiguousarray(
        np.atleast_1d(np.asarray(values, dtype=np.float64)).ravel()
    )


def _pair_and_r(first, second, r):
    if r is None:
        r = 0.0
    first_arr, second_arr, r_arr = broadcast(first, second, r)
    pair = np.column_stack((first_arr, second_arr))
    return np.ascontiguousarray(pair), np.ascontiguousarray(r_arr)


def _finite(values, operation):
    out = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(out)):
        raise CppError(f"C++ {operation} returned non-finite values")
    return out


def transform(copula, x) -> np.ndarray:
    module, spec = _module_and_spec(copula)
    return _finite(
        module.copula_transform(spec, _vector(x)),
        "copula_transform",
    )


def inverse_transform(copula, r) -> np.ndarray:
    module, spec = _module_and_spec(copula)
    return _finite(
        module.copula_inverse_transform(spec, _vector(r)),
        "copula_inverse_transform",
    )


def dtransform(copula, x) -> np.ndarray:
    module, spec = _module_and_spec(copula)
    return _finite(
        module.copula_dtransform(spec, _vector(x)),
        "copula_dtransform",
    )


def tau_to_param(copula, tau) -> np.ndarray:
    module, spec = _module_and_spec(copula)
    return _finite(
        module.copula_tau_to_param(spec, _vector(tau)),
        "copula_tau_to_param",
    )


def param_to_tau(copula, r) -> np.ndarray:
    module, spec = _module_and_spec(copula)
    return _finite(
        module.copula_param_to_tau(spec, _vector(r)),
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


def pdf_grid(copula, u, x_grid) -> np.ndarray:
    module, spec = _module_and_spec(copula)
    values = module.copula_pdf_grid(
        spec,
        np.ascontiguousarray(np.asarray(u, dtype=np.float64)),
        _vector(x_grid),
    )
    return _finite(values, "copula_pdf_grid")


def pdf_parameter_grid(copula, u, r_grid) -> np.ndarray:
    module, spec = _module_and_spec(copula)
    values = module.copula_pdf_parameter_grid(
        spec,
        np.ascontiguousarray(np.asarray(u, dtype=np.float64)),
        _vector(r_grid),
    )
    return _finite(values, "copula_pdf_parameter_grid")


def h_parameter_grid(copula, u, r_grid) -> np.ndarray:
    module, spec = _module_and_spec(copula)
    values = module.copula_h_parameter_grid(
        spec,
        np.ascontiguousarray(np.asarray(u, dtype=np.float64)),
        _vector(r_grid),
    )
    return _finite(values, "copula_h_parameter_grid")


def pdf_and_grad_grid(copula, u, x_grid):
    module, spec = _module_and_spec(copula)
    pdf_values, grad_values = module.copula_pdf_and_grad_grid(
        spec,
        np.ascontiguousarray(np.asarray(u, dtype=np.float64)),
        _vector(x_grid),
    )
    return (
        _finite(pdf_values, "copula_pdf_and_grad_grid"),
        _finite(grad_values, "copula_pdf_and_grad_grid"),
    )
