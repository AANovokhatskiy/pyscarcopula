"""Hermite-rule utilities and native spectral SCAR-OU adapters."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import roots_hermitenorm

from pyscarcopula.numerical._arrays import validate_positive_int
from pyscarcopula.numerical._scar_ou_config import AutoTMConfig


_validate_positive_int = validate_positive_int


@lru_cache(maxsize=32)
def standard_normal_hermite_rule(quad_order: int, basis_order: int):
    """Return nodes, normal weights and orthonormal probabilists basis."""
    quad_order = _validate_positive_int(quad_order, "quad_order")
    basis_order = _validate_positive_int(basis_order, "basis_order")
    if quad_order < basis_order:
        raise ValueError("quad_order must be >= basis_order")
    z, raw_w = roots_hermitenorm(quad_order)
    weights = raw_w / np.sqrt(2.0 * np.pi)
    basis = np.empty((quad_order, basis_order), dtype=np.float64)
    basis[:, 0] = 1.0
    if basis_order > 1:
        basis[:, 1] = z
    for n in range(1, basis_order - 1):
        basis[:, n + 1] = (
            z * basis[:, n] - np.sqrt(float(n)) * basis[:, n - 1]
        ) / np.sqrt(float(n + 1))
    return z, weights, basis


def default_quad_order(basis_order: int) -> int:
    basis_order = _validate_positive_int(basis_order, "basis_order")
    return max(2 * basis_order + 16, 48)


def default_block_size(quad_order: int, max_elements: int = 1_000_000) -> int:
    quad_order = _validate_positive_int(quad_order, "quad_order")
    return max(1, int(max_elements) // quad_order)


def _config(basis_order, quad_order):
    return AutoTMConfig(
        transition_method="spectral",
        basis_order=basis_order,
        quad_order=quad_order,
    )


def hermite_loglik(
        kappa, mu, nu, u, copula, basis_order=32, quad_order=None,
        block_size=None):
    """Return native spectral log-likelihood."""
    del block_size
    from pyscarcopula.numerical import _cpp_scar_ou
    return _cpp_scar_ou.loglik(
        kappa, mu, nu, u, copula,
        _config(basis_order, quad_order))[0]


def hermite_loglik_with_grad(
        kappa, mu, nu, u, copula, basis_order=32, quad_order=None,
        block_size=None):
    """Return native spectral negative log-likelihood and gradient."""
    del block_size
    from pyscarcopula.numerical import _cpp_scar_ou
    return _cpp_scar_ou.neg_loglik_with_grad(
        kappa, mu, nu, u, copula,
        _config(basis_order, quad_order))


def hermite_neg_loglik(*args, **kwargs):
    return -hermite_loglik(*args, **kwargs)


__all__ = [
    "standard_normal_hermite_rule",
    "default_quad_order",
    "default_block_size",
    "hermite_loglik",
    "hermite_loglik_with_grad",
    "hermite_neg_loglik",
]
