"""Hermite spectral likelihood for SCAR-OU models."""

from __future__ import annotations

from functools import lru_cache
from inspect import signature

import numpy as np

from pyscarcopula.numerical._arrays import as_float64_array


def _validate_positive_int(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _batch_method_accepts_t_index(method):
    try:
        return "t_index" in signature(method).parameters
    except (TypeError, ValueError):
        return False


def _call_batch_method(method, u, x_grid, t_index):
    if _batch_method_accepts_t_index(method):
        return method(u, x_grid, t_index=t_index)
    return method(u, x_grid)


def _call_batch_method_with_cache(method, u, x_grid, t_index, cache):
    if cache is None:
        return _call_batch_method(method, u, x_grid, t_index)
    try:
        if _batch_method_accepts_t_index(method):
            return method(u, x_grid, t_index=t_index, cache=cache)
        return method(u, x_grid, cache=cache)
    except TypeError:
        return _call_batch_method(method, u, x_grid, t_index)


@lru_cache(maxsize=32)
def standard_normal_hermite_rule(quad_order: int, basis_order: int):
    """Return nodes, normal weights and orthonormal probabilists basis.

    ``numpy.polynomial.hermite_e.hermegauss`` integrates against
    ``exp(-z**2/2)``.  Dividing weights by ``sqrt(2*pi)`` gives expectation
    under the standard normal law.
    """
    quad_order = _validate_positive_int(quad_order, "quad_order")
    basis_order = _validate_positive_int(basis_order, "basis_order")
    if quad_order < basis_order:
        raise ValueError("quad_order must be >= basis_order")

    z, raw_w = np.polynomial.hermite_e.hermegauss(quad_order)
    w = raw_w / np.sqrt(2.0 * np.pi)

    basis = np.empty((quad_order, basis_order), dtype=np.float64)
    basis[:, 0] = 1.0
    if basis_order > 1:
        basis[:, 1] = z
    for n in range(1, basis_order - 1):
        basis[:, n + 1] = (
            z * basis[:, n] - np.sqrt(float(n)) * basis[:, n - 1]
        ) / np.sqrt(float(n + 1))

    return z, w, basis


def default_quad_order(basis_order: int) -> int:
    """Conservative quadrature order for projected multiplication."""
    basis_order = _validate_positive_int(basis_order, "basis_order")
    return max(2 * basis_order + 16, 48)


def default_block_size(quad_order: int, max_elements: int = 1_000_000) -> int:
    """Bound copula emission blocks by roughly ``max_elements`` doubles."""
    quad_order = _validate_positive_int(quad_order, "quad_order")
    return max(1, int(max_elements) // quad_order)


def _project_multiply(coeff, fi_row, basis, weights):
    """Project ``fi_row(z) * sum_n coeff_n psi_n(z)`` to Hermite coeffs."""
    values = basis @ coeff
    return basis.T @ (weights * fi_row * values)


def _project_multiply_with_grad(
        coeff, dcoeff, fi_row, dfi_dx_row, dx_dalpha, basis, weights):
    """Project multiplication and its gradient w.r.t. OU parameters."""
    values = basis @ coeff
    dvalues = basis @ dcoeff.T
    out = basis.T @ (weights * fi_row * values)

    dout = np.empty((3, coeff.shape[0]), dtype=np.float64)
    for p in range(3):
        dfi = dfi_dx_row * dx_dalpha[p]
        integrand = weights * (dfi * values + fi_row * dvalues[:, p])
        dout[p] = basis.T @ integrand
    return out, dout


def hermite_loglik(
        kappa,
        mu,
        nu,
        u,
        copula,
        basis_order=32,
        quad_order=None,
        block_size=None):
    """Evaluate the SCAR-OU log-likelihood in a truncated Hermite basis.

    The OU state is standardized as ``Z=(X-mu)/sigma`` under the stationary
    Gaussian law.  In the orthonormal probabilists-Hermite basis, the OU
    transition is diagonal with entries ``rho**n``.  Observation factors are
    applied as projected multiplication operators using Gauss-Hermite
    quadrature.

    Returns ``-np.inf`` on numerical failure.
    """
    kappa = float(kappa)
    mu = float(mu)
    nu = float(nu)
    if kappa <= 0.0 or nu <= 0.0:
        return -np.inf

    u = as_float64_array(u)
    n_obs = len(u)
    if n_obs < 1:
        return -np.inf

    basis_order = _validate_positive_int(basis_order, "basis_order")
    if quad_order is None:
        quad_order = default_quad_order(basis_order)
    quad_order = _validate_positive_int(quad_order, "quad_order")
    if block_size is None:
        block_size = default_block_size(quad_order)
    block_size = _validate_positive_int(block_size, "block_size")

    sigma = nu / np.sqrt(2.0 * kappa)
    if not np.isfinite(sigma) or sigma <= 0.0:
        return -np.inf

    if n_obs > 1:
        dt = 1.0 / (n_obs - 1)
    else:
        dt = 1.0
    rho = np.exp(-kappa * dt)
    powers = rho ** np.arange(basis_order, dtype=np.float64)

    z, weights, basis = standard_normal_hermite_rule(
        quad_order, basis_order)
    x_grid = mu + sigma * z
    emission_cache = None
    prepare_cache = getattr(copula, "prepare_emission_cache", None)
    if prepare_cache is not None:
        emission_cache = prepare_cache(u)

    coeff = np.zeros(basis_order, dtype=np.float64)
    coeff[0] = 1.0
    log_scale = 0.0

    # Backward recursion over k=T-1,...,1:
    #   m_{k-1} = P(g_k * m_k), and P is diagonal in this basis.
    for stop in range(n_obs, 1, -block_size):
        start = max(1, stop - block_size)
        fi_block = _call_batch_method_with_cache(
            copula.copula_grid_batch,
            u[start:stop],
            x_grid,
            start,
            emission_cache,
        )
        for local in range(stop - start - 1, -1, -1):
            next_coeff = _project_multiply(
                coeff, fi_block[local], basis, weights)
            coeff = powers * next_coeff
            scale = np.max(np.abs(coeff))
            if not np.isfinite(scale) or scale <= 0.0:
                return -np.inf
            coeff /= scale
            log_scale += np.log(scale)

    fi0 = _call_batch_method_with_cache(
        copula.copula_grid_batch, u[:1], x_grid, 0, emission_cache)[0]
    final_coeff = _project_multiply(coeff, fi0, basis, weights)
    likelihood_scaled = final_coeff[0]
    if not np.isfinite(likelihood_scaled) or likelihood_scaled <= 0.0:
        return -np.inf
    return float(np.log(likelihood_scaled) + log_scale)


def hermite_loglik_with_grad(
        kappa,
        mu,
        nu,
        u,
        copula,
        basis_order=32,
        quad_order=None,
        block_size=None):
    """Evaluate Hermite spectral log-likelihood and analytical gradient.

    Returns
    -------
    tuple
        ``(neg_log_likelihood, neg_gradient)`` for compatibility with
        ``scipy.optimize.minimize(jac=True)``.  The gradient is with respect to
        physical parameters ``(kappa, mu, nu)``.
    """
    fail = 1e10, np.zeros(3, dtype=np.float64)
    kappa = float(kappa)
    mu = float(mu)
    nu = float(nu)
    if kappa <= 0.0 or nu <= 0.0:
        return fail

    u = as_float64_array(u)
    n_obs = len(u)
    if n_obs < 1:
        return fail

    basis_order = _validate_positive_int(basis_order, "basis_order")
    if quad_order is None:
        quad_order = default_quad_order(basis_order)
    quad_order = _validate_positive_int(quad_order, "quad_order")
    if block_size is None:
        block_size = default_block_size(quad_order)
    block_size = _validate_positive_int(block_size, "block_size")

    sigma = nu / np.sqrt(2.0 * kappa)
    if not np.isfinite(sigma) or sigma <= 0.0:
        return fail

    dt = 1.0 / (n_obs - 1) if n_obs > 1 else 1.0
    rho = np.exp(-kappa * dt)
    powers = rho ** np.arange(basis_order, dtype=np.float64)
    dpowers_dkappa = -dt * np.arange(
        basis_order, dtype=np.float64) * powers

    z, weights, basis = standard_normal_hermite_rule(
        quad_order, basis_order)
    x_grid = mu + sigma * z
    emission_cache = None
    prepare_cache = getattr(copula, "prepare_emission_cache", None)
    if prepare_cache is not None:
        emission_cache = prepare_cache(u)

    dx_dalpha = np.empty((3, quad_order), dtype=np.float64)
    dx_dalpha[0] = -0.5 * sigma / kappa * z
    dx_dalpha[1] = 1.0
    dx_dalpha[2] = sigma / nu * z

    coeff = np.zeros(basis_order, dtype=np.float64)
    coeff[0] = 1.0
    dcoeff = np.zeros((3, basis_order), dtype=np.float64)
    log_scale = 0.0
    dlog_scale = np.zeros(3, dtype=np.float64)

    try:
        for stop in range(n_obs, 1, -block_size):
            start = max(1, stop - block_size)
            fi_block, dfi_block = _call_batch_method_with_cache(
                copula.pdf_and_grad_on_grid_batch,
                u[start:stop],
                x_grid,
                start,
                emission_cache,
            )
            for local in range(stop - start - 1, -1, -1):
                projected, dprojected = _project_multiply_with_grad(
                    coeff, dcoeff, fi_block[local], dfi_block[local],
                    dx_dalpha, basis, weights)
                raw = powers * projected
                draw = powers[np.newaxis, :] * dprojected
                draw[0] += dpowers_dkappa * projected

                idx = int(np.argmax(np.abs(raw)))
                scale = abs(raw[idx])
                if not np.isfinite(scale) or scale <= 0.0:
                    return fail
                sign = 1.0 if raw[idx] >= 0.0 else -1.0
                dscale = sign * draw[:, idx]

                coeff = raw / scale
                dcoeff = (
                    draw * scale - raw[np.newaxis, :] * dscale[:, np.newaxis]
                ) / (scale * scale)
                log_scale += np.log(scale)
                dlog_scale += dscale / scale

        fi0, dfi0 = _call_batch_method_with_cache(
            copula.pdf_and_grad_on_grid_batch,
            u[:1],
            x_grid,
            0,
            emission_cache,
        )
        final_coeff, dfinal_coeff = _project_multiply_with_grad(
            coeff, dcoeff, fi0[0], dfi0[0], dx_dalpha, basis, weights)
    except Exception:
        return fail

    likelihood_scaled = final_coeff[0]
    if not np.isfinite(likelihood_scaled) or likelihood_scaled <= 0.0:
        return fail

    log_likelihood = np.log(likelihood_scaled) + log_scale
    grad = dfinal_coeff[:, 0] / likelihood_scaled + dlog_scale
    return -float(log_likelihood), -grad


def hermite_neg_loglik(*args, **kwargs):
    """Minus log-likelihood wrapper for optimizers."""
    value = hermite_loglik(*args, **kwargs)
    if not np.isfinite(value):
        return 1e10
    return -value
