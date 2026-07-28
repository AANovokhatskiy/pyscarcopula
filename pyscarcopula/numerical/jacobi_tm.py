"""Jacobi spectral likelihood for copulas driven by Kendall's tau."""

from __future__ import annotations

import numpy as np
from scipy.special import betaln, digamma, eval_jacobi, roots_jacobi

from pyscarcopula._utils import clip_h_function_values
from pyscarcopula.numerical._arrays import (
    validate_float64_allocation,
    validate_positive_int,
)
from pyscarcopula.numerical import copula_native
from pyscarcopula.numerical._transition_methods import (
    normalize_jacobi_matrix_transition_method,
)


_validate_positive_int = validate_positive_int
MAX_JACOBI_ORDER = 2048
DEFAULT_JACOBI_MEMORY_BUDGET_BYTES = 1024 ** 3


def _validate_jacobi_order(value, name):
    value = _validate_positive_int(value, name)
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


def _raise_jacobi_memory_error(exc):
    raise MemoryError(
        f"{exc}; reduce quad_order, basis_order, or the observation "
        "count, or increase memory_budget_bytes") from exc


def _validate_jacobi_workspace(
        *, quad_order, basis_order=1, n_obs=0, gradient=False,
        matrix=True, memory_budget_bytes=None):
    """Preflight a conservative upper bound for simultaneous float64 arrays."""
    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    basis_order = _validate_jacobi_order(basis_order, "basis_order")
    if basis_order > quad_order:
        raise ValueError("quad_order must be >= basis_order")
    if isinstance(n_obs, (bool, np.bool_)) or not isinstance(
            n_obs, (int, np.integer)):
        raise TypeError("n_obs must be a non-negative integer")
    n_obs = int(n_obs)
    if n_obs < 0:
        raise ValueError("n_obs must be non-negative")

    k = quad_order
    b = basis_order
    if memory_budget_bytes is None:
        memory_budget_bytes = DEFAULT_JACOBI_MEMORY_BUDGET_BYTES

    rule_elements = 2 * k * b + b * b + 12 * k + 4 * b
    if matrix:
        # Covers the spectral kernel plus transition and retained emissions.
        elements = rule_elements + 3 * k * k + 5 * n_obs * k
        if gradient:
            # Base, three derivatives and a temporary perturbed setup.
            elements += 4 * k * k + 4 * n_obs * k
    else:
        elements = rule_elements + 5 * n_obs * k
    try:
        return validate_float64_allocation(
            (elements,),
            name="Jacobi numerical workspace",
            memory_budget_bytes=memory_budget_bytes,
        )
    except MemoryError as exc:
        _raise_jacobi_memory_error(exc)


def _validate_jacobi_sampling_workspace(
        *, n, quad_order, basis_order, memory_budget_bytes=None):
    """Preflight transition construction, in-place CDF, and the tau path."""
    n = _validate_nonnegative_int(n, "n")
    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    basis_order = _validate_jacobi_order(basis_order, "basis_order")
    if basis_order > quad_order:
        raise ValueError("quad_order must be >= basis_order")
    if memory_budget_bytes is None:
        memory_budget_bytes = DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
    k = quad_order
    b = basis_order
    transition_elements = 3 * k * k if n > 1 else 0
    elements = (
        transition_elements
        + 2 * k * b
        + b * b
        + 12 * k
        + 4 * b
        + n
    )
    try:
        return validate_float64_allocation(
            (elements,),
            name="Jacobi sampling workspace",
            memory_budget_bytes=memory_budget_bytes,
        )
    except MemoryError as exc:
        _raise_jacobi_memory_error(exc)


def _jacobi_stationary_shape(kappa, m, xi):
    kappa = float(kappa)
    m = float(m)
    xi = float(xi)
    if (
            not np.all(np.isfinite([kappa, m, xi]))
            or kappa <= 0.0
            or xi <= 0.0
            or not (0.0 < m < 1.0)):
        return None
    alpha = 2.0 * kappa * m / (xi * xi)
    beta = 2.0 * kappa * (1.0 - m) / (xi * xi)
    if alpha <= 0.0 or beta <= 0.0:
        return None
    return alpha, beta


def default_quad_order(basis_order: int) -> int:
    """Conservative quadrature order for Jacobi projected multiplication."""
    basis_order = _validate_jacobi_order(basis_order, "basis_order")
    quad_order = max(2 * basis_order + 16, 48)
    return _validate_jacobi_order(quad_order, "quad_order")


def _normal_hermite_rule(order):
    order = _validate_jacobi_order(order, "gh_order")
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    return nodes.astype(np.float64), (weights / np.sqrt(np.pi)).astype(np.float64)


def _fixed_tau_rule(alpha, beta, quad_order):
    """Return a parameter-independent tau grid and beta stationary masses."""
    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    if quad_order < 2:
        raise ValueError("quad_order must be >= 2")
    eps = 0.5 / (quad_order + 1.0)
    tau = np.linspace(eps, 1.0 - eps, quad_order, dtype=np.float64)
    width = np.empty(quad_order, dtype=np.float64)
    width[1:-1] = tau[2:] - tau[:-2]
    width[1:-1] *= 0.5
    width[0] = tau[1] - tau[0]
    width[-1] = tau[-1] - tau[-2]

    log_density = (
        (float(alpha) - 1.0) * np.log(tau)
        + (float(beta) - 1.0) * np.log1p(-tau)
        - betaln(float(alpha), float(beta))
    )
    log_mass = log_density + np.log(width)
    log_mass -= np.max(log_mass)
    weights = np.exp(log_mass)
    weights /= np.sum(weights)
    return tau, weights


def _fixed_tau_weight_derivatives(kappa, m, xi, tau, weights):
    """Derivatives of normalized stationary beta masses."""
    alpha, beta = _jacobi_stationary_shape(kappa, m, xi)
    xi2 = float(xi) * float(xi)
    dalpha = np.array([
        2.0 * float(m) / xi2,
        2.0 * float(kappa) / xi2,
        -4.0 * float(kappa) * float(m) / (xi2 * float(xi)),
    ], dtype=np.float64)
    dbeta = np.array([
        2.0 * (1.0 - float(m)) / xi2,
        -2.0 * float(kappa) / xi2,
        -4.0 * float(kappa) * (1.0 - float(m)) / (xi2 * float(xi)),
    ], dtype=np.float64)

    dlog_dalpha = np.log(tau) - digamma(alpha) + digamma(alpha + beta)
    dlog_dbeta = np.log1p(-tau) - digamma(beta) + digamma(alpha + beta)
    dweights = np.empty((3, len(tau)), dtype=np.float64)
    for p in range(3):
        score = dalpha[p] * dlog_dalpha + dbeta[p] * dlog_dbeta
        dweights[p] = weights * (score - np.sum(weights * score))
    return dweights


def jacobi_rule(
        alpha, beta, quad_order, basis_order, memory_budget_bytes=None):
    """Return tau nodes, probability weights, and orthonormal Jacobi basis.

    Parameters
    ----------
    alpha, beta : float
        Stationary beta-shape parameters for tau on ``(0, 1)``.
    quad_order : int
        Number of Gauss-Jacobi quadrature nodes.
    basis_order : int
        Number of basis functions, including the constant mode.

    Returns
    -------
    tuple
        ``(tau, weights, basis)`` where ``weights`` sum to one and
        ``basis.T @ diag(weights) @ basis`` is numerically the identity.
    """
    alpha = float(alpha)
    beta = float(beta)
    if not np.isfinite(alpha) or not np.isfinite(beta):
        raise ValueError("alpha and beta must be finite")
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("alpha and beta must be positive")

    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    basis_order = _validate_jacobi_order(basis_order, "basis_order")
    if quad_order < basis_order:
        raise ValueError("quad_order must be >= basis_order")
    _validate_jacobi_workspace(
        quad_order=quad_order,
        basis_order=basis_order,
        matrix=False,
        memory_budget_bytes=memory_budget_bytes,
    )

    # scipy's roots_jacobi(a, b) uses weight (1-x)**a * (1+x)**b.
    # For tau=(x+1)/2 and Beta(alpha, beta), this is a=beta-1,
    # b=alpha-1.
    x, raw_weights = roots_jacobi(quad_order, beta - 1.0, alpha - 1.0)
    tau = 0.5 * (x + 1.0)
    weights = raw_weights / np.sum(raw_weights)

    poly = np.empty((quad_order, basis_order), dtype=np.float64)
    for n in range(basis_order):
        poly[:, n] = eval_jacobi(n, beta - 1.0, alpha - 1.0, x)

    gram = poly.T @ (weights[:, np.newaxis] * poly)
    chol = np.linalg.cholesky(gram)
    basis = poly @ np.linalg.inv(chol.T)

    return tau.astype(np.float64), weights.astype(np.float64), basis


def _jacobi_powers(kappa, xi, n_obs, basis_order):
    dt = 1.0 / (n_obs - 1) if n_obs > 1 else 1.0
    return _jacobi_transition_powers(kappa, xi, dt, basis_order)


def _jacobi_transition_powers(kappa, xi, dt, basis_order):
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    n = np.arange(basis_order, dtype=np.float64)
    eig = n * float(kappa) + 0.5 * float(xi) ** 2 * n * (n - 1.0)
    return np.exp(-eig * dt)


def _resolve_dt(dt, n_obs):
    if dt is None:
        if n_obs is None:
            raise ValueError("either dt or n_obs must be provided")
        n_obs = int(n_obs)
        dt = 1.0 / (n_obs - 1) if n_obs > 1 else 1.0
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    return dt


def jacobi_spectral_transition_matrix(
        kappa,
        m,
        xi,
        *,
        dt=None,
        n_obs=None,
        basis_order=32,
        quad_order=None,
        clip_negative=False,
        memory_budget_bytes=None,
        return_diagnostics=False):
    """Build a node-space transition matrix from the Jacobi spectral density.

    The truncated transition density is

    ``p_dt(y | x) = pi(y) * sum_n exp(-lambda_n * dt) q_n(x) q_n(y)``.

    With Gauss-Jacobi nodes and probability weights for ``pi``, the returned
    row-stochastic mass matrix is

    ``T[i, j] = w[j] * sum_n exp(-lambda_n * dt) q_n(tau[i]) q_n(tau[j])``.

    Negative entries are possible when the spectral series is truncated,
    especially for small ``dt``.  By default they are left untouched and
    reported in diagnostics.  Set ``clip_negative=True`` only for exploratory
    node-space filtering with explicit renormalization.
    """
    shapes = _jacobi_stationary_shape(kappa, m, xi)
    if shapes is None:
        raise ValueError("invalid Jacobi parameters")
    alpha, beta = shapes

    basis_order = _validate_jacobi_order(basis_order, "basis_order")
    if quad_order is None:
        quad_order = default_quad_order(basis_order)
    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    _validate_jacobi_workspace(
        quad_order=quad_order,
        basis_order=basis_order,
        matrix=True,
        memory_budget_bytes=memory_budget_bytes,
    )

    dt = _resolve_dt(dt, n_obs)

    tau, weights, basis = jacobi_rule(
        alpha, beta, quad_order, basis_order,
        memory_budget_bytes=memory_budget_bytes)
    powers = _jacobi_transition_powers(kappa, xi, dt, basis_order)

    kernel = (basis * powers[np.newaxis, :]) @ basis.T
    transition = kernel * weights[np.newaxis, :]
    raw_min = float(np.min(transition))
    raw_negative_mass = float(-np.sum(transition[transition < 0.0]))

    if clip_negative:
        transition = np.where(transition > 0.0, transition, 0.0)

    row_sums = np.sum(transition, axis=1)
    valid = np.isfinite(row_sums) & (row_sums > 0.0)
    if not np.all(valid):
        raise FloatingPointError("invalid transition row normalization")
    transition = transition / row_sums[:, np.newaxis]

    diagnostics = {
        "dt": float(dt),
        "alpha": float(alpha),
        "beta": float(beta),
        "raw_min_entry": raw_min,
        "raw_negative_mass": raw_negative_mass,
        "max_row_sum_error_before_normalization": float(
            np.max(np.abs(row_sums - 1.0))),
        "stationary_error": float(
            np.max(np.abs(weights @ transition - weights))),
        "clipped_negative": bool(clip_negative),
    }

    if return_diagnostics:
        return tau, weights, transition, diagnostics
    return tau, weights, transition


def _jacobi_lamperti(tau, xi):
    tau = np.asarray(tau, dtype=np.float64)
    tau = np.clip(tau, 0.0, 1.0)
    return (2.0 / float(xi)) * np.arcsin(np.sqrt(tau))


def _jacobi_lamperti_inverse(y, xi):
    y = np.asarray(y, dtype=np.float64)
    xi = float(xi)
    y = np.clip(y, 0.0, np.pi / xi)
    s = np.sin(0.5 * xi * y)
    return s * s


def _jacobi_lamperti_drift_from_tau(tau, kappa, m, xi):
    tau = np.asarray(tau, dtype=np.float64)
    kappa = float(kappa)
    m = float(m)
    xi = float(xi)
    denom = np.sqrt(np.maximum(tau * (1.0 - tau), 1e-300))
    return (
        kappa * (m - tau) / (xi * denom)
        - xi * (1.0 - 2.0 * tau) / (4.0 * denom)
    )


def _add_interpolated_mass(row, tau_grid, y, weight, xi):
    tau_y = float(_jacobi_lamperti_inverse(y, xi))
    if tau_y <= tau_grid[0]:
        row[0] += weight
        return
    if tau_y >= tau_grid[-1]:
        row[-1] += weight
        return

    right = int(np.searchsorted(tau_grid, tau_y, side="right"))
    left = right - 1
    width = tau_grid[right] - tau_grid[left]
    if width <= 0.0:
        row[left] += weight
        return
    lam = (tau_y - tau_grid[left]) / width
    row[left] += weight * (1.0 - lam)
    row[right] += weight * lam


def _add_interpolated_mass_with_grad(
        row, drow, tau_grid, tau_y, dtau_y, weight):
    if tau_y <= tau_grid[0]:
        row[0] += weight
        return
    if tau_y >= tau_grid[-1]:
        row[-1] += weight
        return

    right = int(np.searchsorted(tau_grid, tau_y, side="right"))
    left = right - 1
    width = tau_grid[right] - tau_grid[left]
    if width <= 0.0:
        row[left] += weight
        return
    lam = (tau_y - tau_grid[left]) / width
    dlam = dtau_y / width
    row[left] += weight * (1.0 - lam)
    row[right] += weight * lam
    drow[:, left] -= weight * dlam
    drow[:, right] += weight * dlam


def jacobi_fixed_grid_transition_matrix(
        kappa,
        m,
        xi,
        *,
        dt=None,
        n_obs=None,
        quad_order=128,
        gh_order=5,
        return_grad=False,
        memory_budget_bytes=None,
        return_diagnostics=False):
    """Build local-GH Jacobi transition on a fixed tau grid.

    The tau nodes are independent of model parameters.  This backend is used
    for analytical-gradient optimization, because all parameter sensitivities
    of the discrete likelihood are explicit: beta initial masses, Lamperti
    drift, inverse Lamperti map, and linear interpolation weights.
    """
    shapes = _jacobi_stationary_shape(kappa, m, xi)
    if shapes is None:
        raise ValueError("invalid Jacobi parameters")
    alpha, beta = shapes
    dt = _resolve_dt(dt, n_obs)
    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    gh_order = _validate_jacobi_order(gh_order, "gh_order")
    _validate_jacobi_workspace(
        quad_order=quad_order,
        matrix=True,
        gradient=return_grad,
        memory_budget_bytes=memory_budget_bytes,
    )

    tau, weights = _fixed_tau_rule(alpha, beta, quad_order)
    gh_nodes, gh_weights = _normal_hermite_rule(gh_order)

    transition = np.zeros((quad_order, quad_order), dtype=np.float64)
    dtransition = np.zeros((3, quad_order, quad_order), dtype=np.float64)
    y_grid = _jacobi_lamperti(tau, xi)
    drift = _jacobi_lamperti_drift_from_tau(tau, kappa, m, xi)
    denom = np.sqrt(np.maximum(tau * (1.0 - tau), 1e-300))
    asin_sqrt = np.arcsin(np.sqrt(tau))
    y_min = 0.0
    y_max = np.pi / float(xi)

    d_y_grid = np.zeros((3, quad_order), dtype=np.float64)
    d_y_grid[2] = -2.0 * asin_sqrt / (float(xi) * float(xi))

    d_drift = np.empty((3, quad_order), dtype=np.float64)
    d_drift[0] = (float(m) - tau) / (float(xi) * denom)
    d_drift[1] = float(kappa) / (float(xi) * denom)
    d_drift[2] = (
        -float(kappa) * (float(m) - tau) / (float(xi) ** 2 * denom)
        - (1.0 - 2.0 * tau) / (4.0 * denom)
    )

    offsets = np.sqrt(2.0 * dt) * gh_nodes
    for i in range(quad_order):
        center = y_grid[i] + drift[i] * dt
        dcenter = d_y_grid[:, i] + d_drift[:, i] * dt
        for offset, weight in zip(offsets, gh_weights):
            y_next = center + offset
            if y_next <= y_min:
                transition[i, 0] += weight
                continue
            if y_next >= y_max:
                transition[i, -1] += weight
                continue

            phase = 0.5 * float(xi) * y_next
            tau_y = np.sin(phase) ** 2
            dtau_y = 0.5 * np.sin(float(xi) * y_next) * (
                np.array([0.0, 0.0, 1.0], dtype=np.float64) * y_next
                + float(xi) * dcenter
            )
            _add_interpolated_mass_with_grad(
                transition[i], dtransition[:, i], tau, tau_y, dtau_y, weight)

    row_sums = np.sum(transition, axis=1)
    valid = np.isfinite(row_sums) & (row_sums > 0.0)
    if not np.all(valid):
        raise FloatingPointError("invalid fixed-grid transition row normalization")
    transition /= row_sums[:, np.newaxis]
    for p in range(3):
        drow_sum = np.sum(dtransition[p], axis=1)
        dtransition[p] = (
            dtransition[p] * row_sums[:, np.newaxis]
            - transition * row_sums[:, np.newaxis] * drow_sum[:, np.newaxis]
        ) / (row_sums[:, np.newaxis] * row_sums[:, np.newaxis])

    diagnostics = {
        "dt": dt,
        "alpha": float(alpha),
        "beta": float(beta),
        "gh_order": int(gh_order),
        "min_entry": float(np.min(transition)),
        "max_row_sum_error": float(
            np.max(np.abs(np.sum(transition, axis=1) - 1.0))),
        "stationary_error": float(
            np.max(np.abs(weights @ transition - weights))),
        "transition_method": "local_fixed",
    }
    if return_grad and return_diagnostics:
        return tau, weights, transition, dtransition, diagnostics
    if return_grad:
        return tau, weights, transition, dtransition
    if return_diagnostics:
        return tau, weights, transition, diagnostics
    return tau, weights, transition


def jacobi_local_transition_matrix(
        kappa,
        m,
        xi,
        *,
        dt=None,
        n_obs=None,
        quad_order=128,
        basis_order=1,
        gh_order=5,
        memory_budget_bytes=None,
        return_diagnostics=False):
    """Build a local transition matrix for Jacobi diffusion.

    The transition is approximated in the Lamperti coordinate

    ``y = 2 / xi * asin(sqrt(tau))``,

    where the diffusion coefficient is one.  A one-step local Gaussian
    approximation is then mapped back to ``tau`` and linearly interpolated on
    the Jacobi quadrature grid.  Rows are source nodes and columns are target
    nodes; rows are nonnegative and normalized.
    """
    shapes = _jacobi_stationary_shape(kappa, m, xi)
    if shapes is None:
        raise ValueError("invalid Jacobi parameters")
    alpha, beta = shapes

    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    # ``basis_order`` is accepted so callers can use the same constructor
    # signature as the spectral matrix path; only the grid order matters here.
    basis_order = _validate_jacobi_order(basis_order, "basis_order")
    gh_order = _validate_jacobi_order(gh_order, "gh_order")
    _validate_jacobi_workspace(
        quad_order=quad_order,
        basis_order=basis_order,
        matrix=True,
        memory_budget_bytes=memory_budget_bytes,
    )

    dt = _resolve_dt(dt, n_obs)

    tau, weights, _ = jacobi_rule(
        alpha, beta, quad_order, basis_order=1,
        memory_budget_bytes=memory_budget_bytes)
    y_grid = _jacobi_lamperti(tau, xi)
    drift = _jacobi_lamperti_drift_from_tau(tau, kappa, m, xi)
    gh_nodes, gh_weights = _normal_hermite_rule(gh_order)
    offsets = np.sqrt(2.0 * dt) * gh_nodes

    transition = np.zeros((quad_order, quad_order), dtype=np.float64)
    y_min = 0.0
    y_max = np.pi / float(xi)
    for i in range(quad_order):
        center = y_grid[i] + drift[i] * dt
        for offset, weight in zip(offsets, gh_weights):
            y_next = np.clip(center + offset, y_min, y_max)
            _add_interpolated_mass(
                transition[i], tau, y_next, float(weight), xi)

    row_sums = np.sum(transition, axis=1)
    valid = np.isfinite(row_sums) & (row_sums > 0.0)
    if not np.all(valid):
        raise FloatingPointError("invalid local GH transition row normalization")
    transition /= row_sums[:, np.newaxis]

    diagnostics = {
        "dt": dt,
        "alpha": float(alpha),
        "beta": float(beta),
        "gh_order": int(gh_order),
        "min_entry": float(np.min(transition)),
        "max_row_sum_error": float(
            np.max(np.abs(np.sum(transition, axis=1) - 1.0))),
        "stationary_error": float(
            np.max(np.abs(weights @ transition - weights))),
    }
    if return_diagnostics:
        return tau, weights, transition, diagnostics
    return tau, weights, transition


def _probability_transition_matrix(transition, negative_mass_tol):
    """Return a finite row-stochastic matrix with no signed probability mass."""
    transition = np.asarray(transition, dtype=np.float64)
    negative_mass_tol = float(negative_mass_tol)
    if not np.isfinite(negative_mass_tol) or negative_mass_tol < 0.0:
        raise ValueError("negative_mass_tol must be finite and non-negative")
    if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
        raise ValueError("transition must be a square matrix")
    if np.any(~np.isfinite(transition)):
        raise FloatingPointError("transition contains non-finite values")

    negative = transition < 0.0
    negative_mass = float(-np.sum(transition[negative]))
    min_entry = float(np.min(transition))
    if (
            min_entry < -negative_mass_tol
            or negative_mass > negative_mass_tol):
        raise FloatingPointError(
            "spectral transition contains material negative probability mass")

    cleaned = bool(np.any(negative))
    if cleaned:
        transition = np.where(transition > 0.0, transition, 0.0)

    row_sums = np.sum(transition, axis=1)
    if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= 0.0):
        raise FloatingPointError("invalid transition row normalization")
    transition = transition / row_sums[:, np.newaxis]
    return transition, {
        "probability_cleanup_applied": cleaned,
        "probability_cleanup_negative_mass": negative_mass,
        "probability_min_entry_before_cleanup": min_entry,
    }


def jacobi_transition_matrix(
        kappa,
        m,
        xi,
        *,
        dt=None,
        n_obs=None,
        basis_order=32,
        quad_order=None,
        transition_method="auto",
        clip_negative=False,
        negative_mass_tol=1e-5,
        gh_order=5,
        memory_budget_bytes=None,
        return_diagnostics=False):
    """Build a Jacobi transition matrix using the requested backend.

    ``spectral_matrix`` uses the truncated exact spectral density on the
    Jacobi quadrature grid.  ``local`` uses a local Gaussian step in the
    Lamperti coordinate.  ``auto`` tries the spectral matrix first and falls
    back to ``local`` when truncation creates material negative mass.
    """
    method_requested = normalize_jacobi_matrix_transition_method(
        transition_method)
    dt = _resolve_dt(dt, n_obs)
    if quad_order is None:
        quad_order = default_quad_order(basis_order)
    basis_order = _validate_jacobi_order(basis_order, "basis_order")
    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    gh_order = _validate_jacobi_order(gh_order, "gh_order")
    _validate_jacobi_workspace(
        quad_order=quad_order,
        basis_order=basis_order,
        matrix=True,
        memory_budget_bytes=memory_budget_bytes,
    )

    if method_requested == "local_fixed":
        tau, weights, transition, diagnostics = (
            jacobi_fixed_grid_transition_matrix(
                kappa,
                m,
                xi,
                dt=dt,
                quad_order=quad_order,
                gh_order=gh_order,
                memory_budget_bytes=memory_budget_bytes,
                return_diagnostics=True,
            )
        )
        diagnostics = dict(diagnostics)
        diagnostics["transition_method_requested"] = method_requested
        if return_diagnostics:
            return tau, weights, transition, diagnostics
        return tau, weights, transition

    spectral_error = None
    if method_requested in {"auto", "spectral_matrix"}:
        try:
            tau, weights, transition, diagnostics = (
                jacobi_spectral_transition_matrix(
                    kappa,
                    m,
                    xi,
                    dt=dt,
                    basis_order=basis_order,
                    quad_order=quad_order,
                    clip_negative=clip_negative,
                    memory_budget_bytes=memory_budget_bytes,
                    return_diagnostics=True,
                )
            )
        except Exception as exc:
            if method_requested != "auto":
                raise
            spectral_error = exc
        else:
            diagnostics = dict(diagnostics)
            diagnostics["transition_method_requested"] = method_requested
            diagnostics["transition_method"] = "spectral_matrix"
            has_bad_negative_mass = (
                diagnostics["raw_min_entry"] < -float(negative_mass_tol)
                or diagnostics["raw_negative_mass"] > float(negative_mass_tol)
            )
            if (
                    method_requested == "spectral_matrix"
                    and has_bad_negative_mass
                    and not clip_negative):
                raise FloatingPointError(
                    "spectral transition contains material negative "
                    "probability mass")
            if method_requested != "auto" or not has_bad_negative_mass:
                transition, probability_diagnostics = (
                    _probability_transition_matrix(
                        transition,
                        0.0 if clip_negative else negative_mass_tol,
                    )
                )
                diagnostics.update(probability_diagnostics)
                diagnostics["stationary_error"] = float(
                    np.max(np.abs(weights @ transition - weights)))
                if return_diagnostics:
                    return tau, weights, transition, diagnostics
                return tau, weights, transition

    tau, weights, transition, diagnostics = jacobi_local_transition_matrix(
        kappa,
        m,
        xi,
        dt=dt,
        quad_order=quad_order,
        basis_order=basis_order,
            gh_order=gh_order,
            memory_budget_bytes=memory_budget_bytes,
            return_diagnostics=True,
    )
    diagnostics = dict(diagnostics)
    diagnostics["transition_method_requested"] = method_requested
    diagnostics["transition_method"] = "local"
    if spectral_error is not None:
        diagnostics["spectral_error"] = (
            f"{type(spectral_error).__name__}: {spectral_error}"
        )
    if return_diagnostics:
        return tau, weights, transition, diagnostics
    return tau, weights, transition


def sample_jacobi_grid_trajectory(
        kappa,
        m,
        xi,
        n,
        *,
        rng=None,
        basis_order=32,
        quad_order=None,
        transition_method="auto",
        clip_negative=False,
        negative_mass_tol=1e-5,
        gh_order=5,
        memory_budget_bytes=None,
        return_diagnostics=False):
    """Sample the discrete Jacobi Markov model used by matrix likelihoods.

    The returned values are quadrature-grid atoms. The transition matrix is
    converted to row-wise CDFs in place, so sampling does not retain a second
    ``K x K`` array.
    """
    n = _validate_nonnegative_int(n, "n")
    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        if return_diagnostics:
            return empty, {
                "transition_method_requested": str(transition_method),
                "transition_method": "not_built",
                "n": 0,
            }
        return empty

    shapes = _jacobi_stationary_shape(kappa, m, xi)
    if shapes is None:
        raise ValueError("invalid Jacobi parameters")
    alpha, beta = shapes
    basis_order = _validate_jacobi_order(basis_order, "basis_order")
    if quad_order is None:
        quad_order = default_quad_order(basis_order)
    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    gh_order = _validate_jacobi_order(gh_order, "gh_order")
    _validate_jacobi_sampling_workspace(
        n=n,
        quad_order=quad_order,
        basis_order=basis_order,
        memory_budget_bytes=memory_budget_bytes,
    )
    if rng is None:
        rng = np.random.default_rng()

    requested_method = str(transition_method).lower()
    sampling_method = (
        "auto" if requested_method == "spectral_coeff"
        else requested_method)

    if n == 1:
        if sampling_method == "local_fixed":
            tau_grid, stationary_prob = _fixed_tau_rule(
                alpha, beta, quad_order)
        else:
            tau_grid, stationary_prob, _ = jacobi_rule(
                alpha,
                beta,
                quad_order,
                basis_order=1,
                memory_budget_bytes=memory_budget_bytes,
            )
        stationary_cdf = np.cumsum(stationary_prob)
        stationary_cdf[-1] = 1.0
        index = int(np.searchsorted(
            stationary_cdf, rng.random(), side="right"))
        path = np.array([tau_grid[index]], dtype=np.float64)
        if return_diagnostics:
            return path, {
                "transition_method_requested": requested_method,
                "sampling_transition_method_requested": sampling_method,
                "transition_method": "stationary_only",
                "n": 1,
            }
        return path

    tau_grid, stationary_prob, transition, diagnostics = (
        jacobi_transition_matrix(
            kappa,
            m,
            xi,
            n_obs=n,
            basis_order=basis_order,
            quad_order=quad_order,
            transition_method=sampling_method,
            clip_negative=clip_negative,
            negative_mass_tol=negative_mass_tol,
            gh_order=gh_order,
            memory_budget_bytes=memory_budget_bytes,
            return_diagnostics=True,
        )
    )
    stationary_cdf = np.cumsum(stationary_prob)
    stationary_cdf[-1] = 1.0
    np.cumsum(transition, axis=1, out=transition)
    transition[:, -1] = 1.0

    # Reuse the uniform buffer as the returned tau path.
    path = np.asarray(rng.random(n), dtype=np.float64)
    index = int(np.searchsorted(
        stationary_cdf, path[0], side="right"))
    path[0] = tau_grid[index]
    for t in range(1, n):
        index = int(np.searchsorted(
            transition[index], path[t], side="right"))
        path[t] = tau_grid[index]

    if return_diagnostics:
        diagnostics = dict(diagnostics)
        diagnostics["sampling_transition_method_requested"] = sampling_method
        diagnostics["model_transition_method_requested"] = requested_method
        diagnostics["n"] = n
        return path, diagnostics
    return path


def _theta_grid(copula, tau, theta_cap=None):
    if copula_native.supported(copula):
        theta = copula_native.tau_to_param(copula, tau)
    else:
        theta = np.asarray(copula.tau_to_param(tau), dtype=np.float64)
    if theta_cap is not None:
        theta = np.minimum(theta, float(theta_cap))
    if np.any(~np.isfinite(theta)):
        raise FloatingPointError("tau_to_param produced non-finite values")
    return theta


def _emission_grid(u, copula, tau, theta_cap=None):
    u = np.asarray(u, dtype=np.float64)
    theta = _theta_grid(copula, tau, theta_cap=theta_cap)
    if copula_native.supported(copula):
        return copula_native.pdf_parameter_grid(copula, u, theta), theta

    n_obs = len(u)
    n_grid = len(tau)
    out = np.empty((n_obs, n_grid), dtype=np.float64)
    u1_grid = np.empty(n_grid, dtype=np.float64)
    u2_grid = np.empty(n_grid, dtype=np.float64)
    for t in range(n_obs):
        u1_grid.fill(u[t, 0])
        u2_grid.fill(u[t, 1])
        out[t] = copula.pdf(u1_grid, u2_grid, theta)
    return out, theta


def _finite_unit_grid_values(values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if np.all(finite):
        return clip_h_function_values(values)
    if not np.any(finite):
        return np.full(values.shape, 0.5, dtype=np.float64)

    idx = np.arange(values.size, dtype=np.float64)
    filled = values.copy()
    filled[~finite] = np.interp(idx[~finite], idx[finite], values[finite])
    return clip_h_function_values(filled)


def _h_values_on_theta(copula, u_row, theta, u1_grid=None, u2_grid=None):
    n_grid = len(theta)
    if u1_grid is None:
        u1_grid = np.empty(n_grid, dtype=np.float64)
    if u2_grid is None:
        u2_grid = np.empty(n_grid, dtype=np.float64)
    u2_grid.fill(u_row[1])
    u1_grid.fill(u_row[0])
    return _finite_unit_grid_values(copula.h(u2_grid, u1_grid, theta))


def _h_grid_on_theta(copula, u, theta):
    if copula_native.supported(copula):
        values = copula_native.h_parameter_grid(copula, u, theta)
        return np.vstack([
            _finite_unit_grid_values(row) for row in values
        ])

    n_grid = len(theta)
    u1_grid = np.empty(n_grid, dtype=np.float64)
    u2_grid = np.empty(n_grid, dtype=np.float64)
    return np.vstack([
        _h_values_on_theta(copula, row, theta, u1_grid, u2_grid)
        for row in u
    ])


def _h_pair_grids_on_theta(copula, u, theta):
    """Return h(u2 | u1) and h(u1 | u2) grids for one orientation."""
    n_grid = len(theta)
    first = []
    second = []
    for row in np.asarray(u, dtype=np.float64):
        u1_grid = np.full(n_grid, row[0], dtype=np.float64)
        u2_grid = np.full(n_grid, row[1], dtype=np.float64)
        first_given_second, second_given_first = copula.h_pair(
            u1_grid, u2_grid, theta)
        first.append(_finite_unit_grid_values(second_given_first))
        second.append(_finite_unit_grid_values(first_given_second))
    return np.vstack(first), np.vstack(second)


def _matrix_setup(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order,
        quad_order,
        theta_cap,
        transition_method,
        clip_negative,
        negative_mass_tol,
        gh_order,
        memory_budget_bytes=None,
        return_transition_diagnostics=False):
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 2 or u.shape[1] != 2 or len(u) < 1:
        return None
    basis_order = _validate_jacobi_order(basis_order, "basis_order")
    if quad_order is None:
        quad_order = default_quad_order(basis_order)
    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    _validate_jacobi_workspace(
        quad_order=quad_order,
        basis_order=basis_order,
        n_obs=len(u),
        matrix=True,
        memory_budget_bytes=memory_budget_bytes,
    )

    transition_result = jacobi_transition_matrix(
        kappa,
        m,
        xi,
        n_obs=len(u),
        basis_order=basis_order,
        quad_order=quad_order,
        transition_method=transition_method,
        clip_negative=clip_negative,
        negative_mass_tol=negative_mass_tol,
        gh_order=gh_order,
        memory_budget_bytes=memory_budget_bytes,
        return_diagnostics=return_transition_diagnostics,
    )
    if return_transition_diagnostics:
        tau, weights, transition, transition_diagnostics = transition_result
    else:
        tau, weights, transition = transition_result
    fi_grid, theta = _emission_grid(u, copula, tau, theta_cap=theta_cap)
    setup = u, tau, weights, transition, fi_grid, theta
    if return_transition_diagnostics:
        return setup + (transition_diagnostics,)
    return setup


def _normalize_prob_mass(prob, *, negative_tol=1e-12):
    prob = np.asarray(prob, dtype=np.float64)
    if np.any(~np.isfinite(prob)):
        return None
    if np.min(prob) < -negative_tol:
        return None
    prob = np.where(prob > 0.0, prob, 0.0)
    total = np.sum(prob)
    if not np.isfinite(total) or total <= 0.0:
        return None
    return prob / total


def _normalize_prob_mass_with_derivatives(prob, dprob):
    """Normalize probability mass and propagate its parameter derivatives."""
    prob = np.asarray(prob, dtype=np.float64)
    dprob = np.asarray(dprob, dtype=np.float64)
    if (
            prob.ndim != 1
            or dprob.ndim != 2
            or dprob.shape[1] != prob.size
            or np.any(~np.isfinite(prob))
            or np.any(~np.isfinite(dprob))
            or np.any(prob < 0.0)):
        return None

    total = float(np.sum(prob))
    if not np.isfinite(total) or total <= 0.0:
        return None
    dtotal = np.sum(dprob, axis=1)
    normalized = prob / total
    dnormalized = (
        dprob * total
        - prob[np.newaxis, :] * dtotal[:, np.newaxis]
    ) / (total * total)
    if (
            np.any(~np.isfinite(normalized))
            or np.any(~np.isfinite(dnormalized))):
        return None
    return normalized, dnormalized, total, dtotal


def _advance_matrix_posterior(predicted, fi_row):
    weighted = predicted * fi_row
    scale = float(np.sum(weighted))
    if not np.isfinite(scale) or scale <= 0.0:
        return None, None
    posterior = _normalize_prob_mass(weighted / scale)
    if posterior is None:
        return None, None
    return posterior, scale


def _iter_matrix_filter(weights, transition, fi_grid):
    predicted = weights.copy()
    for t in range(fi_grid.shape[0]):
        posterior, scale = _advance_matrix_posterior(predicted, fi_grid[t])
        if posterior is None:
            raise FloatingPointError("Jacobi matrix filter update failed")
        yield t, predicted, posterior, scale
        if t < fi_grid.shape[0] - 1:
            predicted = _normalize_prob_mass(posterior @ transition)
            if predicted is None:
                raise FloatingPointError("Jacobi matrix prediction failed")


def _matrix_filter_setup(*args, **kwargs):
    try:
        setup = _matrix_setup(*args, **kwargs)
    except Exception:
        return None
    return setup


def jacobi_matrix_loglik(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        transition_method="auto",
        clip_negative=False,
        negative_mass_tol=1e-5,
        gh_order=5,
        memory_budget_bytes=None):
    """Evaluate log-likelihood by filtering probability masses on tau nodes."""
    resolved_quad_order = (
        default_quad_order(basis_order)
        if quad_order is None else quad_order)
    _validate_jacobi_workspace(
        quad_order=resolved_quad_order,
        basis_order=basis_order,
        n_obs=len(u),
        matrix=True,
        memory_budget_bytes=memory_budget_bytes,
    )
    setup = _matrix_filter_setup(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order,
        quad_order,
        theta_cap,
        transition_method,
        clip_negative,
        negative_mass_tol,
        gh_order,
        memory_budget_bytes,
    )
    if setup is None:
        return -np.inf

    _, _, weights, transition, fi_grid, _ = setup
    log_likelihood = 0.0
    try:
        for _, _, _, scale in _iter_matrix_filter(
                weights, transition, fi_grid):
            log_likelihood += np.log(scale)
    except FloatingPointError:
        return -np.inf

    return float(log_likelihood)


def jacobi_matrix_neg_loglik(*args, **kwargs):
    """Minus matrix-filter log-likelihood wrapper for optimizers."""
    value = jacobi_matrix_loglik(*args, **kwargs)
    if not np.isfinite(value):
        return 1e10
    return -value


def _matrix_neg_loglik_from_derivatives(
        weights,
        transition,
        fi_grid,
        dweights,
        dtransition,
        dfi_grid):
    predicted = weights.copy()
    dpredicted = dweights.copy()
    log_likelihood = 0.0
    grad = np.zeros(3, dtype=np.float64)

    for t in range(fi_grid.shape[0]):
        fi_row = fi_grid[t]
        weighted = predicted * fi_row
        dweighted = (
            dpredicted * fi_row[np.newaxis, :]
            + predicted[np.newaxis, :] * dfi_grid[:, t, :]
        )
        normalized = _normalize_prob_mass_with_derivatives(
            weighted, dweighted)
        if normalized is None:
            return 1e10, np.zeros(3, dtype=np.float64)
        posterior, dposterior, scale, dscale = normalized
        log_likelihood += np.log(scale)
        grad += dscale / scale

        if t < fi_grid.shape[0] - 1:
            raw_next_predicted = posterior @ transition
            raw_next_dpredicted = np.empty_like(dpredicted)
            for p in range(3):
                raw_next_dpredicted[p] = (
                    dposterior[p] @ transition
                    + posterior @ dtransition[p]
                )
            normalized = _normalize_prob_mass_with_derivatives(
                raw_next_predicted, raw_next_dpredicted)
            if normalized is None:
                return 1e10, np.zeros(3, dtype=np.float64)
            predicted, dpredicted, _, _ = normalized

    if not np.isfinite(log_likelihood) or np.any(~np.isfinite(grad)):
        return 1e10, np.zeros(3, dtype=np.float64)
    return -float(log_likelihood), -grad


def _matrix_setup_fd_derivatives(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order,
        quad_order,
        theta_cap,
        transition_method,
        clip_negative,
        negative_mass_tol,
        gh_order,
        fd_rel_step,
        memory_budget_bytes=None):
    resolved_quad_order = (
        default_quad_order(basis_order)
        if quad_order is None else quad_order)
    _validate_jacobi_workspace(
        quad_order=resolved_quad_order,
        basis_order=basis_order,
        n_obs=len(u),
        matrix=True,
        gradient=True,
        memory_budget_bytes=memory_budget_bytes,
    )
    base = _matrix_setup(
        kappa, m, xi, u, copula, basis_order, quad_order, theta_cap,
        transition_method, clip_negative, negative_mass_tol, gh_order,
        memory_budget_bytes,
        return_transition_diagnostics=True)
    if base is None:
        return None

    _, tau, weights, transition, fi_grid, _, transition_diagnostics = base
    derivative_transition_method = str(
        transition_diagnostics.get("transition_method", transition_method))
    params = np.array([kappa, m, xi], dtype=np.float64)
    dweights = np.empty((3, len(weights)), dtype=np.float64)
    dtransition = np.empty((3,) + transition.shape, dtype=np.float64)
    dfi_grid = np.empty((3,) + fi_grid.shape, dtype=np.float64)

    for p in range(3):
        step = float(fd_rel_step) * max(abs(params[p]), 1.0)
        if p == 1:
            step = min(step, 0.49 * params[p], 0.49 * (1.0 - params[p]))
        else:
            step = min(step, 0.49 * params[p])
        if not np.isfinite(step) or step <= 0.0:
            return None

        plus = params.copy()
        minus = params.copy()
        plus[p] += step
        minus[p] -= step
        plus_setup = _matrix_setup(
            plus[0], plus[1], plus[2], u, copula, basis_order, quad_order,
            theta_cap, derivative_transition_method, clip_negative,
            negative_mass_tol, gh_order, memory_budget_bytes)
        minus_setup = _matrix_setup(
            minus[0], minus[1], minus[2], u, copula, basis_order, quad_order,
            theta_cap, derivative_transition_method, clip_negative,
            negative_mass_tol, gh_order, memory_budget_bytes)
        if plus_setup is None or minus_setup is None:
            return None

        _, _, weights_p, transition_p, fi_grid_p, _ = plus_setup
        _, _, weights_m, transition_m, fi_grid_m, _ = minus_setup
        denom = 2.0 * step
        if (weights_p.shape != weights.shape
                or weights_m.shape != weights.shape
                or transition_p.shape != transition.shape
                or transition_m.shape != transition.shape
                or fi_grid_p.shape != fi_grid.shape
                or fi_grid_m.shape != fi_grid.shape):
            return None
        dweights[p] = (weights_p - weights_m) / denom
        dtransition[p] = (transition_p - transition_m) / denom
        dfi_grid[p] = (fi_grid_p - fi_grid_m) / denom

    return tau, weights, transition, fi_grid, dweights, dtransition, dfi_grid


def jacobi_matrix_neg_loglik_with_grad(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        transition_method="local_fixed",
        clip_negative=False,
        negative_mass_tol=1e-5,
        gh_order=5,
        fd_rel_step=1e-5,
        memory_budget_bytes=None):
    """Evaluate Jacobi matrix negative log-likelihood and gradient.

    Returns ``(neg_log_likelihood, neg_gradient)`` with derivatives with
    respect to physical parameters ``(kappa, m, xi)``.

    ``local_fixed`` uses explicit transition and stationary-weight
    derivatives.  Moving-grid backends such as ``spectral_matrix`` use a
    semi-analytical gradient: finite differences for setup-level arrays
    ``weights``, ``transition`` and ``fi_grid``, followed by analytical
    differentiation of the filtering recursion.
    """
    fail = 1e10, np.zeros(3, dtype=np.float64)
    method = normalize_jacobi_matrix_transition_method(transition_method)

    shapes = _jacobi_stationary_shape(kappa, m, xi)
    if shapes is None:
        return fail
    alpha, beta = shapes

    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 2 or u.shape[1] != 2 or len(u) < 1:
        return fail

    basis_order = _validate_jacobi_order(basis_order, "basis_order")
    if quad_order is None:
        quad_order = default_quad_order(basis_order)
    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    _validate_jacobi_workspace(
        quad_order=quad_order,
        basis_order=basis_order,
        n_obs=len(u),
        matrix=True,
        gradient=True,
        memory_budget_bytes=memory_budget_bytes,
    )

    if method == "local_fixed":
        try:
            tau, weights, transition, dtransition = (
                jacobi_fixed_grid_transition_matrix(
                    kappa,
                    m,
                    xi,
                    n_obs=len(u),
                    quad_order=quad_order,
                    gh_order=gh_order,
                    return_grad=True,
                    memory_budget_bytes=memory_budget_bytes,
                )
            )
            dweights = _fixed_tau_weight_derivatives(
                kappa, m, xi, tau, weights)
            fi_grid, _ = _emission_grid(u, copula, tau, theta_cap=theta_cap)
            dfi_grid = np.zeros((3,) + fi_grid.shape, dtype=np.float64)
        except Exception:
            return fail
    else:
        try:
            setup = _matrix_setup_fd_derivatives(
                kappa,
                m,
                xi,
                u,
                copula,
                basis_order,
                quad_order,
                theta_cap,
                method,
                clip_negative,
                negative_mass_tol,
                gh_order,
                fd_rel_step,
                memory_budget_bytes,
            )
        except Exception:
            return fail
        if setup is None:
            return fail
        _, weights, transition, fi_grid, dweights, dtransition, dfi_grid = setup

    return _matrix_neg_loglik_from_derivatives(
        weights, transition, fi_grid, dweights, dtransition, dfi_grid)


def jacobi_matrix_forward_predictive_mean(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        transition_method="auto",
        clip_negative=False,
        negative_mass_tol=1e-5,
        gh_order=5,
        memory_budget_bytes=None):
    """Return node-space E[theta(tau_k) | u_{1:k-1}]."""
    setup = _matrix_setup(
        kappa, m, xi, u, copula, basis_order, quad_order, theta_cap,
        transition_method, clip_negative, negative_mass_tol, gh_order,
        memory_budget_bytes)
    if setup is None:
        raise ValueError("invalid Jacobi parameters or observations")
    _, _, weights, transition, fi_grid, theta = setup

    out = np.empty(fi_grid.shape[0], dtype=np.float64)
    for t, predicted, _, _ in _iter_matrix_filter(
            weights, transition, fi_grid):
        out[t] = np.sum(predicted * theta)
    return out


def jacobi_matrix_forward_mixture_h(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        transition_method="auto",
        clip_negative=False,
        negative_mass_tol=1e-5,
        gh_order=5,
        memory_budget_bytes=None):
    """Return node-space E[h(u2 | u1; theta(tau_k)) | u_{1:k-1}]."""
    setup = _matrix_setup(
        kappa, m, xi, u, copula, basis_order, quad_order, theta_cap,
        transition_method, clip_negative, negative_mass_tol, gh_order,
        memory_budget_bytes)
    if setup is None:
        raise ValueError("invalid Jacobi parameters or observations")
    u, _, weights, transition, fi_grid, theta = setup

    n_obs = len(u)
    out = np.empty(n_obs, dtype=np.float64)
    h_grid = _h_grid_on_theta(copula, u, theta)
    for t, predicted, _, _ in _iter_matrix_filter(
            weights, transition, fi_grid):
        out[t] = np.sum(predicted * h_grid[t])
    return clip_h_function_values(out)


def jacobi_matrix_forward_mixture_h_pair(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        transition_method="auto",
        clip_negative=False,
        negative_mass_tol=1e-5,
        gh_order=5,
        memory_budget_bytes=None):
    """Return both h-directions from one Jacobi matrix filter pass."""
    setup = _matrix_setup(
        kappa, m, xi, u, copula, basis_order, quad_order, theta_cap,
        transition_method, clip_negative, negative_mass_tol, gh_order,
        memory_budget_bytes)
    if setup is None:
        raise ValueError("invalid Jacobi parameters or observations")
    u, _, weights, transition, fi_grid, theta = setup

    first = np.empty(len(u), dtype=np.float64)
    second = np.empty(len(u), dtype=np.float64)
    first_grid, second_grid = _h_pair_grids_on_theta(copula, u, theta)
    for t, predicted, _, _ in _iter_matrix_filter(
            weights, transition, fi_grid):
        first[t] = np.sum(predicted * first_grid[t])
        second[t] = np.sum(predicted * second_grid[t])
    return clip_h_function_values(first), clip_h_function_values(second)


def jacobi_matrix_state_distribution(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        transition_method="auto",
        clip_negative=False,
        negative_mass_tol=1e-5,
        gh_order=5,
        memory_budget_bytes=None,
        horizon="current"):
    """Return a node-space tau distribution at current or next horizon."""
    horizon = str(horizon).lower()
    if horizon not in ("current", "next"):
        raise ValueError("horizon must be 'current' or 'next'")

    setup = _matrix_setup(
        kappa, m, xi, u, copula, basis_order, quad_order, theta_cap,
        transition_method, clip_negative, negative_mass_tol, gh_order,
        memory_budget_bytes)
    if setup is None:
        raise ValueError("invalid Jacobi parameters or observations")
    _, tau, weights, transition, fi_grid, _ = setup

    posterior = None
    for _, _, posterior, _ in _iter_matrix_filter(
            weights, transition, fi_grid):
        pass

    prob = posterior
    if horizon == "next":
        prob = _normalize_prob_mass(prob @ transition)
        if prob is None:
            raise FloatingPointError("Jacobi matrix prediction failed")

    return tau.copy(), prob


def _setup(
        kappa, m, xi, u, copula, basis_order, quad_order, theta_cap,
        memory_budget_bytes=None):
    shapes = _jacobi_stationary_shape(kappa, m, xi)
    if shapes is None:
        return None
    alpha, beta = shapes

    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 2 or u.shape[1] != 2 or len(u) < 1:
        return None

    basis_order = _validate_jacobi_order(basis_order, "basis_order")
    if quad_order is None:
        quad_order = default_quad_order(basis_order)
    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    _validate_jacobi_workspace(
        quad_order=quad_order,
        basis_order=basis_order,
        n_obs=len(u),
        matrix=False,
        memory_budget_bytes=memory_budget_bytes,
    )

    tau, weights, basis = jacobi_rule(
        alpha, beta, quad_order, basis_order,
        memory_budget_bytes=memory_budget_bytes)
    powers = _jacobi_powers(kappa, xi, len(u), basis_order)
    fi_grid, theta = _emission_grid(u, copula, tau, theta_cap=theta_cap)
    return u, tau, weights, basis, powers, fi_grid, theta


def _project_update(coeff, fi_row, weights, basis):
    predicted = basis @ coeff
    raw = basis.T @ (weights * fi_row * predicted)
    scale = raw[0]
    if not np.isfinite(scale) or scale <= 0.0:
        return None, None
    return raw / scale, float(scale)


def _iter_coeff_filter(powers, fi_grid, weights, basis):
    coeff = np.zeros(basis.shape[1], dtype=np.float64)
    coeff[0] = 1.0
    for t in range(fi_grid.shape[0]):
        predicted_coeff = powers * coeff
        coeff, scale = _project_update(
            predicted_coeff, fi_grid[t], weights, basis)
        if coeff is None:
            raise FloatingPointError("Jacobi filter update failed")
        yield t, predicted_coeff, coeff, scale


def jacobi_loglik(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        memory_budget_bytes=None):
    """Evaluate the Jacobi-diffusion copula log-likelihood.

    The latent state is Kendall's tau on ``(0, 1)``.  Observation emissions are
    evaluated at ``copula.tau_to_param(tau)``.
    """
    resolved_quad_order = (
        default_quad_order(basis_order)
        if quad_order is None else quad_order)
    _validate_jacobi_workspace(
        quad_order=resolved_quad_order,
        basis_order=basis_order,
        n_obs=len(u),
        matrix=False,
        memory_budget_bytes=memory_budget_bytes,
    )
    try:
        setup = _setup(
            kappa, m, xi, u, copula, basis_order, quad_order, theta_cap,
            memory_budget_bytes)
    except Exception:
        return -np.inf
    if setup is None:
        return -np.inf

    _, _, weights, basis, powers, fi_grid, _ = setup
    log_likelihood = 0.0

    try:
        for _, _, _, scale in _iter_coeff_filter(
                powers, fi_grid, weights, basis):
            log_likelihood += np.log(scale)
    except FloatingPointError:
        return -np.inf

    return float(log_likelihood)


def jacobi_neg_loglik(*args, **kwargs):
    """Minus log-likelihood wrapper for optimizers."""
    value = jacobi_loglik(*args, **kwargs)
    if not np.isfinite(value):
        return 1e10
    return -value


def jacobi_forward_predictive_mean(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        memory_budget_bytes=None):
    """Return E[theta(tau_k) | u_{1:k-1}] for each observation."""
    setup = _setup(
        kappa, m, xi, u, copula, basis_order, quad_order, theta_cap,
        memory_budget_bytes)
    if setup is None:
        raise ValueError("invalid Jacobi parameters or observations")
    _, _, weights, basis, powers, fi_grid, theta = setup

    out = np.empty(fi_grid.shape[0], dtype=np.float64)
    for t, predicted_coeff, _, _ in _iter_coeff_filter(
            powers, fi_grid, weights, basis):
        density_ratio = basis @ predicted_coeff
        out[t] = np.sum(weights * theta * density_ratio)
    return out


def jacobi_forward_mixture_h(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        memory_budget_bytes=None):
    """Return E[h(u2 | u1; theta(tau_k)) | u_{1:k-1}]."""
    setup = _setup(
        kappa, m, xi, u, copula, basis_order, quad_order, theta_cap,
        memory_budget_bytes)
    if setup is None:
        raise ValueError("invalid Jacobi parameters or observations")
    u, _, weights, basis, powers, fi_grid, theta = setup

    n_obs = len(u)
    out = np.empty(n_obs, dtype=np.float64)
    h_grid = _h_grid_on_theta(copula, u, theta)
    for t, predicted_coeff, _, _ in _iter_coeff_filter(
            powers, fi_grid, weights, basis):
        density_ratio = basis @ predicted_coeff
        out[t] = np.sum(weights * h_grid[t] * density_ratio)
    return clip_h_function_values(out)


def jacobi_forward_mixture_h_pair(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        memory_budget_bytes=None):
    """Return both h-directions from one Jacobi coefficient filter pass."""
    setup = _setup(
        kappa, m, xi, u, copula, basis_order, quad_order, theta_cap,
        memory_budget_bytes)
    if setup is None:
        raise ValueError("invalid Jacobi parameters or observations")
    u, _, weights, basis, powers, fi_grid, theta = setup

    first = np.empty(len(u), dtype=np.float64)
    second = np.empty(len(u), dtype=np.float64)
    first_grid, second_grid = _h_pair_grids_on_theta(copula, u, theta)
    for t, predicted_coeff, _, _ in _iter_coeff_filter(
            powers, fi_grid, weights, basis):
        density_ratio = basis @ predicted_coeff
        first[t] = np.sum(weights * first_grid[t] * density_ratio)
        second[t] = np.sum(weights * second_grid[t] * density_ratio)
    return clip_h_function_values(first), clip_h_function_values(second)


def _coeff_to_prob(coeff, weights, basis):
    density_ratio = basis @ coeff
    prob = weights * density_ratio
    prob = np.where(np.isfinite(prob) & (prob > 0.0), prob, 0.0)
    total = np.sum(prob)
    if total <= 0.0:
        return np.full(len(weights), 1.0 / len(weights), dtype=np.float64)
    return prob / total


def jacobi_state_distribution(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        memory_budget_bytes=None,
        horizon="current"):
    """Return a discrete tau distribution at the current or next horizon."""
    horizon = str(horizon).lower()
    if horizon not in ("current", "next"):
        raise ValueError("horizon must be 'current' or 'next'")

    setup = _setup(
        kappa, m, xi, u, copula, basis_order, quad_order, theta_cap,
        memory_budget_bytes)
    if setup is None:
        raise ValueError("invalid Jacobi parameters or observations")
    _, tau, weights, basis, powers, fi_grid, _ = setup

    coeff = None
    for _, _, coeff, _ in _iter_coeff_filter(powers, fi_grid, weights, basis):
        pass

    if horizon == "next":
        coeff = powers * coeff

    return tau.copy(), _coeff_to_prob(coeff, weights, basis)
