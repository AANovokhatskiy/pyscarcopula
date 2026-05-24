"""
pyscarcopula.numerical.tm_gradient — Analytical gradient of TM log-likelihood.

Key insight (from the paper, Section 3.4):
  Working in normalised coordinates xi = z / sigma (fixed grid),
  several parameter dependencies cancel:

    T_w[i,j] depends on kappa ONLY through rho = exp(-kappa*dt).
              Does NOT depend on nu (or mu).

    p0[j] * trap_w[j] is completely independent of (kappa, mu, nu).

    fi[t,j] = c(u_1t, u_2t; Psi(sigma*xi_j + mu))
              depends on kappa and nu through sigma, and on mu directly.

This makes the gradient computation efficient: only 3 chain-rule terms.

Contents:
  - tm_loglik_with_grad: neg logL + analytical gradient (3 components)
  - _build_Tw_and_grad_dense: T_w and dT_w/drho in xi-coordinates (dense)
  - _build_Tw_and_grad_sparse: same, sparse CSR version (numba-accelerated)
"""

import numpy as np
from numba import njit
from scipy.sparse import csr_matrix

from pyscarcopula.numerical.tm_grid import (
    _local_rule,
    _select_matrix_grid_method,
    _select_transition_method,
)
from pyscarcopula.numerical._transition_methods import (
    normalize_ou_transition_method,
)


def _build_Tw_and_grad_dense(xi, rho, base_w, K):
    """
    Build T_w and dT_w/drho (dense) in xi-coordinates.

    T_w[i,j] = base_w[j]/(sqrt(1-rho^2)*sqrt(2pi)) * exp(-0.5*q^2/(1-rho^2))
    where q = xi_j - rho*xi_i.
    """
    omr2 = 1.0 - rho ** 2
    q = xi[np.newaxis, :] - rho * xi[:, np.newaxis]
    gauss = np.exp(-0.5 * q ** 2 / omr2) / (np.sqrt(omr2) * np.sqrt(2.0 * np.pi))
    T_w = gauss * base_w[np.newaxis, :]

    dlog_drho = (rho / omr2
                 + q * xi[:, np.newaxis] / omr2
                 - rho * q ** 2 / omr2 ** 2)
    dTw_drho = dlog_drho * T_w

    return T_w, dTw_drho


@njit(cache=True)
def _build_Tw_and_grad_sparse_core(xi, rho, base_w, K, band):
    """
    Numba-accelerated COO builder for sparse T_w and dT_w/drho.

    Returns (rows, cols, t_vals, g_vals) as flat arrays.
    Caller wraps into scipy CSR.
    """
    omr2 = 1.0 - rho ** 2
    inv_omr2 = 1.0 / omr2
    coeff = 1.0 / (np.sqrt(omr2) * np.sqrt(2.0 * np.pi))

    d_xi = xi[1] - xi[0]
    xi0 = xi[0]
    inv_dxi = 1.0 / d_xi

    centers_idx = (rho * xi - xi0) * inv_dxi

    # Count total nnz
    total = 0
    for i in range(K):
        i_lo = max(0, int(np.floor(centers_idx[i])) - band)
        i_hi = min(K, int(np.ceil(centers_idx[i])) + band + 1)
        w = i_hi - i_lo
        if w > 0:
            total += w

    rows = np.empty(total, dtype=np.int32)
    cols = np.empty(total, dtype=np.int32)
    t_vals = np.empty(total, dtype=np.float64)
    g_vals = np.empty(total, dtype=np.float64)

    ptr = 0
    for i in range(K):
        i_lo = max(0, int(np.floor(centers_idx[i])) - band)
        i_hi = min(K, int(np.ceil(centers_idx[i])) + band + 1)

        for j in range(i_lo, i_hi):
            q = xi[j] - rho * xi[i]
            gauss = coeff * np.exp(-0.5 * q * q * inv_omr2)
            tw = gauss * base_w[j]

            rows[ptr] = i
            cols[ptr] = j
            t_vals[ptr] = tw

            dlog = (rho * inv_omr2
                    + q * xi[i] * inv_omr2
                    - rho * q * q * inv_omr2 * inv_omr2)
            g_vals[ptr] = dlog * tw
            ptr += 1

    return rows[:ptr], cols[:ptr], t_vals[:ptr], g_vals[:ptr]


def _build_Tw_and_grad_sparse(xi, rho, base_w, K, band):
    """
    Sparse version of T_w and dT_w/drho in xi-coordinates.

    Uses numba-accelerated COO construction + scipy CSR conversion.
    Returns T_w (CSR), dTw_drho (CSR).
    """
    rows, cols, t_vals, g_vals = _build_Tw_and_grad_sparse_core(
        xi, rho, base_w, K, band)

    shape = (K, K)
    T_sp = csr_matrix((t_vals, (rows, cols)), shape=shape)
    G_sp = csr_matrix((g_vals, (rows, cols)), shape=shape)
    return T_sp, G_sp


def _build_local_Tw_and_grad(xi, rho, K, transition_method, gh_order):
    """
    Local interpolation operator and dT/drho in normalised coordinates.

    Rows are source grid points, columns are target grid points.  With fixed
    grid and fixed interpolation cells, only the interpolation coordinate
    lambda changes with rho.  Clamped boundary nodes have zero derivative in
    this first local-gradient implementation.
    """
    gh_nodes, gh_weights = _local_rule(transition_method, gh_order)
    s = np.sqrt(1.0 - rho ** 2)
    offsets = np.sqrt(2.0) * s * gh_nodes
    if s > 0.0:
        doffsets_drho = -np.sqrt(2.0) * rho / s * gh_nodes
    else:
        doffsets_drho = np.zeros_like(gh_nodes)
    q = len(offsets)

    xi0 = xi[0]
    xi_last = xi[-1]
    d_xi = xi[1] - xi[0]

    rows = np.empty(K * q * 2, dtype=np.int32)
    cols = np.empty(K * q * 2, dtype=np.int32)
    vals = np.empty(K * q * 2, dtype=np.float64)
    grad_vals = np.empty(K * q * 2, dtype=np.float64)

    ptr = 0
    for i in range(K):
        center = rho * xi[i]
        dcenter_drho = xi[i]

        for offset, doffset_drho, weight in zip(
                offsets, doffsets_drho, gh_weights):
            y = center + offset

            if y <= xi0:
                rows[ptr] = i
                cols[ptr] = 0
                vals[ptr] = weight
                grad_vals[ptr] = 0.0
                ptr += 1
                continue
            if y >= xi_last:
                rows[ptr] = i
                cols[ptr] = K - 1
                vals[ptr] = weight
                grad_vals[ptr] = 0.0
                ptr += 1
                continue

            left = int(np.floor((y - xi0) / d_xi))
            if left >= K - 1:
                rows[ptr] = i
                cols[ptr] = K - 1
                vals[ptr] = weight
                grad_vals[ptr] = 0.0
                ptr += 1
                continue

            lam = (y - xi[left]) / d_xi
            dlam_drho = (dcenter_drho + doffset_drho) / d_xi

            rows[ptr] = i
            cols[ptr] = left
            vals[ptr] = weight * (1.0 - lam)
            grad_vals[ptr] = -weight * dlam_drho
            ptr += 1

            rows[ptr] = i
            cols[ptr] = left + 1
            vals[ptr] = weight * lam
            grad_vals[ptr] = weight * dlam_drho
            ptr += 1

    shape = (K, K)
    T_sp = csr_matrix((vals[:ptr], (rows[:ptr], cols[:ptr])), shape=shape)
    G_sp = csr_matrix(
        (grad_vals[:ptr], (rows[:ptr], cols[:ptr])), shape=shape)
    return T_sp, G_sp


@njit(cache=True)
def _build_local_stencil_core(xi, rho, gh_nodes, gh_weights):
    """Build fixed-width local interpolation stencil and dT/drho values."""
    K = len(xi)
    q = len(gh_nodes)
    width = q * 2
    s = np.sqrt(1.0 - rho ** 2)

    xi0 = xi[0]
    xi_last = xi[K - 1]
    d_xi = xi[1] - xi[0]

    cols = np.empty((K, width), dtype=np.int32)
    vals = np.zeros((K, width), dtype=np.float64)
    grad_vals = np.zeros((K, width), dtype=np.float64)

    for i in range(K):
        center = rho * xi[i]
        dcenter_drho = xi[i]

        for ell in range(q):
            node = gh_nodes[ell]
            weight = gh_weights[ell]
            offset = np.sqrt(2.0) * s * node
            if s > 0.0:
                doffset_drho = -np.sqrt(2.0) * rho / s * node
            else:
                doffset_drho = 0.0
            y = center + offset
            pos = 2 * ell

            if y <= xi0:
                cols[i, pos] = 0
                cols[i, pos + 1] = 0
                vals[i, pos] = weight
                continue
            if y >= xi_last:
                cols[i, pos] = K - 1
                cols[i, pos + 1] = K - 1
                vals[i, pos] = weight
                continue

            left = int(np.floor((y - xi0) / d_xi))
            if left >= K - 1:
                cols[i, pos] = K - 1
                cols[i, pos + 1] = K - 1
                vals[i, pos] = weight
                continue

            lam = (y - xi[left]) / d_xi
            dlam_drho = (dcenter_drho + doffset_drho) / d_xi

            cols[i, pos] = left
            cols[i, pos + 1] = left + 1
            vals[i, pos] = weight * (1.0 - lam)
            vals[i, pos + 1] = weight * lam
            grad_vals[i, pos] = -weight * dlam_drho
            grad_vals[i, pos + 1] = weight * dlam_drho

    return cols, vals, grad_vals


def _build_local_stencil_and_grad(xi, rho, transition_method, gh_order):
    """Build compact local interpolation stencil and dT/drho values."""
    gh_nodes, gh_weights = _local_rule(transition_method, gh_order)
    return _build_local_stencil_core(xi, rho, gh_nodes, gh_weights)


@njit(cache=True)
def _local_stencil_matvec(cols, vals, v):
    K = cols.shape[0]
    width = cols.shape[1]
    out = np.empty(K, dtype=np.float64)
    for i in range(K):
        acc = 0.0
        for j in range(width):
            acc += vals[i, j] * v[cols[i, j]]
        out[i] = acc
    return out


def _make_local_stencil_matvec(cols, vals):
    """Create a matvec closure for compact local interpolation stencils."""
    def _matvec(v):
        return _local_stencil_matvec(cols, vals, v)
    return _matvec


def _make_fast_matvec(mat):
    """Create a matvec closure that pre-binds the matrix."""
    def _matvec(v):
        return mat @ v
    return _matvec


def tm_loglik_with_grad(kappa, mu, nu, u, copula, K=300, grid_range=5.0,
                        grid_method='auto', adaptive=True, pts_per_sigma=4,
                        transition_method='matrix', max_K=None,
                        r_gh=3.0, gh_order=5):
    """
    Transfer matrix log-likelihood with analytical gradient.

    Uses normalised coordinates xi = z / sigma so that:
    - T_w depends on kappa only through rho = exp(-kappa*dt)
    - p0 * w is parameter-independent
    - fi depends on (kappa, nu) through sigma and on mu directly

    Parameters
    ----------
    kappa, mu, nu : float
        OU process parameters (kappa > 0, nu > 0).
    u : ndarray (n, 2)
        Pseudo-observations.
    copula : CopulaProtocol
    K, grid_range, grid_method, adaptive, pts_per_sigma,
    transition_method, max_K, r_gh, gh_order : grid params

    Returns
    -------
    neg_logL : float
        Minus log-likelihood (1e10 on failure).
    neg_grad : ndarray (3,)
        Minus gradient w.r.t. (kappa, mu, nu). Zero on failure.
    """
    FAIL = 1e10, np.zeros(3)

    if kappa <= 0 or nu <= 0:
        return FAIL

    try:
        transition_method = normalize_ou_transition_method(transition_method)
    except ValueError:
        return FAIL
    if transition_method in {'auto', 'spectral'}:
        from pyscarcopula.numerical.auto_tm import (
            AutoTMConfig,
            auto_neg_loglik_with_grad,
        )
        return auto_neg_loglik_with_grad(
            kappa, mu, nu, u, copula,
            AutoTMConfig(
                transition_method=transition_method,
                K=K,
                grid_range=grid_range,
                grid_method=grid_method,
                adaptive=adaptive,
                pts_per_sigma=pts_per_sigma,
                max_K=max_K,
                gh_order=gh_order,
                r_gh=r_gh,
            ),
        )
    if transition_method not in {'matrix', 'local'}:
        return FAIL

    n = len(u)
    if n < 2:
        return FAIL

    dt = 1.0 / (n - 1)
    rho = np.exp(-kappa * dt)
    sigma2 = 0.5 * nu ** 2 / kappa
    sigma = np.sqrt(sigma2)
    sigma_c = sigma * np.sqrt(1.0 - rho ** 2)

    if sigma <= 0 or sigma_c <= 0:
        return FAIL

    # ── fixed normalised grid ────────────────────────────────────
    if adaptive:
        dz_target = sigma_c / pts_per_sigma
        K_min = int(np.ceil(2.0 * grid_range * sigma / dz_target)) + 1
        K_adaptive = max(K, K_min)
    else:
        K_adaptive = K

    if max_K is not None:
        K_eff = min(K_adaptive, max_K)
        K_eff = max(K_eff, min(K, max_K))
    else:
        K_eff = K_adaptive
    adaptive_was_capped = K_eff < K_adaptive

    xi = np.linspace(-grid_range, grid_range, K_eff)
    d_xi = xi[1] - xi[0]
    r_kernel_grid = np.sqrt(1.0 - rho ** 2) / d_xi
    transition_method = _select_transition_method(
        transition_method,
        r_kernel_grid,
        r_gh,
        adaptive_was_capped,
    )

    base_w = np.full(K_eff, d_xi)
    base_w[0] *= 0.5
    base_w[-1] *= 0.5

    # p0 * trap_w is constant (sigma cancels in xi-coordinates)
    pw_const = np.exp(-0.5 * xi ** 2) / np.sqrt(2.0 * np.pi) * base_w

    # ── build T_w and dT_w/drho ──────────────────────────────────
    try:
        if transition_method == 'local':
            cols, vals, grad_vals = _build_local_stencil_and_grad(
                xi, rho, transition_method, gh_order)
            matvec = _make_local_stencil_matvec(cols, vals)
            dTw_matvec = _make_local_stencil_matvec(cols, grad_vals)
        else:
            half_width_xi = 5.0 * np.sqrt(1.0 - rho ** 2)
            band = int(np.ceil(half_width_xi / d_xi))

            if grid_method == 'auto':
                grid_method = _select_matrix_grid_method(band, K_eff)

            if grid_method == 'sparse':
                T_w, dTw_drho = _build_Tw_and_grad_sparse(
                    xi, rho, base_w, K_eff, band)
            else:
                T_w, dTw_drho = _build_Tw_and_grad_dense(
                    xi, rho, base_w, K_eff)
            matvec = _make_fast_matvec(T_w)
            dTw_matvec = _make_fast_matvec(dTw_drho)
    except Exception:
        return FAIL

    drho_dkappa = -dt * rho

    # ── copula density and its derivative on grid ─────────────────
    x_grid = sigma * xi + mu

    # Use batch method (fused numba kernel if available, else loop fallback)
    fi, dfi_dx = copula.pdf_and_grad_on_grid_batch(u, x_grid)

    # dx/dalpha:  x = sigma*xi + mu
    d_sigma_dkappa = -0.5 * nu ** 2 / kappa ** 2 / (2.0 * sigma)
    d_sigma_dnu = nu / (kappa * 2.0 * sigma)

    dx_dalpha = np.zeros((3, K_eff))
    dx_dalpha[0] = d_sigma_dkappa * xi   # dx/dkappa
    dx_dalpha[1] = 1.0                    # dx/dmu
    dx_dalpha[2] = d_sigma_dnu * xi       # dx/dnu

    # ── fast matvec: bypass scipy dispatch for sparse matrices ────
    # ══════════════════════════════════════════════════════════════
    # BACKWARD PASS — compute and store beta[t] and c_vals[t]
    # ══════════════════════════════════════════════════════════════

    beta = [None] * n
    beta[n - 1] = np.ones(K_eff)
    c_vals = np.empty(n - 1)
    cumul_logc = 0.0

    for t in range(n - 2, -1, -1):
        v = fi[t + 1] * beta[t + 1]
        b = matvec(v)
        mx = np.max(np.abs(b))
        if mx <= 0:
            return FAIL
        c_vals[t] = mx
        cumul_logc += np.log(mx)
        b /= mx
        beta[t] = b

    # Log-likelihood
    Z0 = np.sum(fi[0] * pw_const * beta[0])
    if Z0 <= 0:
        return FAIL
    logL = np.log(Z0) + cumul_logc
    neg_logL = -logL

    # ══════════════════════════════════════════════════════════════
    # GRADIENT via recursive d_beta propagation
    # ══════════════════════════════════════════════════════════════

    d_beta = np.zeros((3, K_eff))

    for t in range(n - 2, -1, -1):
        target = fi[t + 1] * beta[t + 1]
        inv_c = 1.0 / c_vals[t]

        new_d_beta = np.empty((3, K_eff))
        for k in range(3):
            dfi_k = dfi_dx[t + 1] * dx_dalpha[k]
            d_target_k = dfi_k * beta[t + 1] + fi[t + 1] * d_beta[k]

            contrib = matvec(d_target_k)
            if k == 0:
                contrib += dTw_matvec(target) * drho_dkappa

            new_d_beta[k] = contrib * inv_c

        d_beta = new_d_beta

    # ── Assemble gradient ────────────────────────────────────────
    grad = np.zeros(3)
    for k in range(3):
        dfi_k_0 = dfi_dx[0] * dx_dalpha[k]
        num = np.sum((dfi_k_0 * beta[0] + fi[0] * d_beta[k]) * pw_const)
        grad[k] = num / Z0

    return neg_logL, -grad
