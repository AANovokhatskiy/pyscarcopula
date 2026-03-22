"""
Ornstein-Uhlenbeck latent process for stochastic copula models.

Supports fitting grid_methods:
  SCAR-P-OU   — MC without importance sampling (p-sampler)
  SCAR-M-OU   — MC with efficient importance sampling (m-sampler)
  SCAR-TM-OU  — transfer matrix (deterministic quadrature)

Features:
  - Analytical gradient of TM log-likelihood (xi-coordinates)
  - Smart initial point via GAS moment matching
  - Coarse-grid multi-start initialization

MLE is handled directly by BivariateCopula._fit_mle.
"""

import numpy as np
from numba import njit
from scipy.optimize import minimize, Bounds
from scipy.signal import savgol_filter
from scipy.sparse import csr_matrix


# ══════════════════════════════════════════════════════════════════
# Numba kernels for OU process (pure numerical, no copula calls)
# ══════════════════════════════════════════════════════════════════

@njit(cache=True)
def _ou_init_state(mu, n_tr):
    return np.full(n_tr, mu)


@njit(cache=True)
def _ou_stationary_state(theta, mu, nu, n_tr):
    sigma2 = nu ** 2 / (2.0 * theta)
    return np.random.normal(mu, np.sqrt(sigma2), n_tr)


@njit(cache=True)
def _ou_sample_paths_exact(theta, mu, nu, dwt, x0):
    """
    Exact OU discretization (for P-sampler, no EIS).
    x_{i+1} = mu + rho*(x_i - mu) + sigma_c * eps_i
    where eps_i = dwt[i] * sqrt(1/dt) rescaled to N(0,1).
    dwt: (T, n_tr), x0: (n_tr,).
    Returns xt: (T, n_tr).
    """
    T, n_tr = dwt.shape
    dt = 1.0 / (T - 1)
    rho = np.exp(-theta * dt)
    sigma2_cond = nu ** 2 / (2.0 * theta) * (1.0 - rho ** 2)
    sigma_cond = np.sqrt(sigma2_cond)
    # dwt[i] ~ N(0, dt), need N(0, 1): eps = dwt / sqrt(dt)
    scale = sigma_cond / np.sqrt(dt)

    xt = np.empty((T, n_tr))
    xt[0] = x0
    for i in range(1, T):
        xt[i] = mu + rho * (xt[i - 1] - mu) + scale * dwt[i - 1]
    return xt


@njit(cache=True)
def _ou_sample_paths(theta, mu, nu, a1t, a2t, dwt, x0):
    """
    Generate OU trajectories (possibly modified by EIS params a1t, a2t).
    dwt: (T, n_tr), x0: (n_tr,).
    Returns xt: (T, n_tr).
    """
    T, n_tr = dwt.shape
    dt = 1.0 / (T - 1)
    D = nu ** 2 / 2.0
    xt = np.empty((T, n_tr))
    xt[0] = x0

    Mx0 = np.mean(x0)
    Dx0 = np.var(x0)

    Ito_sum = np.zeros(n_tr)

    for i in range(1, T):
        t = i * dt
        a1, a2 = a1t[i], a2t[i]

        sigma2 = D / theta * (1.0 - np.exp(-2.0 * theta * t)) + Dx0 * np.exp(-2.0 * theta * t)
        p = 1.0 - 2.0 * a2 * sigma2

        if i == 1:
            pm1 = 1.0
        else:
            tm1 = t - dt
            sigma2m1 = D / theta * (1.0 - np.exp(-2.0 * theta * tm1)) + Dx0 * np.exp(-2.0 * theta * tm1)
            pm1 = 1.0 - 2.0 * a2t[i - 1] * sigma2m1

        xs = (Mx0 - mu) * np.exp(-theta * t) + mu
        xsw = (xs + a1 * sigma2) / p
        st = np.exp(-theta * t) / np.sqrt(p)
        det_part = xsw + st * x0 - st * (Mx0 + a1t[0] * Dx0) / np.sqrt(1.0 - 2.0 * a2t[0] * Dx0)

        Ito_sum = (Ito_sum * np.sqrt(pm1 / p) + nu / np.sqrt(p) * dwt[i - 1]) * np.exp(-theta * dt)
        xt[i] = det_part + Ito_sum

    return xt


@njit(cache=True)
def _log_norm_ou(theta, mu, nu, a1, a2, dt, x0):
    """Log normalizing factor g2 for EIS."""
    D = nu ** 2 / 2.0
    sigma2 = D / theta * (1.0 - np.exp(-2.0 * theta * dt))
    xs = (x0 - mu) * np.exp(-theta * dt) + mu
    res = (a1 ** 2 * sigma2 + 2.0 * a1 * xs + 2.0 * a2 * xs ** 2) / \
          (2.0 - 4.0 * a2 * sigma2) - 0.5 * np.log(1.0 - 2.0 * a2 * sigma2)
    return res


@njit(cache=True)
def _log_mean_exp(log_vals):
    """Numerically stable log(mean(exp(log_vals)))."""
    xc = np.max(log_vals)
    return np.log(np.mean(np.exp(log_vals - xc))) + xc


# ══════════════════════════════════════════════════════════════════
# P-sampler (SCAR-P-OU) — no importance sampling
# ══════════════════════════════════════════════════════════════════

def _p_sampler_loglik(theta, mu, nu, u, dwt, copula, stationary):
    """
    P-sampler log-likelihood (exact OU discretization).
    Returns minus log-likelihood.
    """
    T, n_tr = dwt.shape

    if stationary:
        x0 = _ou_stationary_state(theta, mu, nu, n_tr)
    else:
        x0 = _ou_init_state(mu, n_tr)

    z = np.zeros(T)
    xt = _ou_sample_paths(theta, mu, nu, z, z, dwt, x0)

    if np.isnan(np.sum(xt)):
        return 1e10

    # For each t: vectorized copula log_pdf over all n_tr trajectories
    copula_log = np.zeros(n_tr)
    for t in range(T):
        r_vals = copula.transform(xt[t])         # (n_tr,)
        u1 = np.full(n_tr, u[t, 0])
        u2 = np.full(n_tr, u[t, 1])
        copula_log += copula.log_pdf(u1, u2, r_vals)

    return -_log_mean_exp(copula_log)


# ══════════════════════════════════════════════════════════════════
# M-sampler (SCAR-M-OU) — with EIS
# ══════════════════════════════════════════════════════════════════

def _m_sampler_loglik(theta, mu, nu, u, dwt, a1t, a2t, copula, stationary):
    """
    M-sampler log-likelihood with EIS auxiliary params.
    Returns minus log-likelihood.
    """
    T, n_tr = dwt.shape
    dt = 1.0 / (T - 1)

    if stationary:
        x0 = _ou_stationary_state(theta, mu, nu, n_tr)
    else:
        x0 = _ou_init_state(mu, n_tr)

    xt = _ou_sample_paths(theta, mu, nu, a1t, a2t, dwt, x0)

    if np.isnan(np.sum(xt)):
        return 1e10

    # Normalizing factors
    norm_log = np.zeros((T, n_tr))
    for i in range(T - 1, 0, -1):
        norm_log[i] = _log_norm_ou(theta, mu, nu, a1t[i], a2t[i], dt, xt[i - 1])
    norm_log[0] = _log_norm_ou(theta, mu, nu, a1t[0], a2t[0], dt, x0)

    # Copula log-likelihood with IS correction
    log_lik = np.zeros(n_tr)
    for t in range(T):
        r_vals = copula.transform(xt[t])
        u1 = np.full(n_tr, u[t, 0])
        u2 = np.full(n_tr, u[t, 1])
        c_vals = copula.log_pdf(u1, u2, r_vals)
        g_vals = a1t[t] * xt[t] + a2t[t] * xt[t] ** 2
        log_lik += c_vals + norm_log[t] - g_vals

    return -_log_mean_exp(log_lik)


# ══════════════════════════════════════════════════════════════════
# EIS auxiliary parameters (for SCAR-M-OU)
# ══════════════════════════════════════════════════════════════════

def _eis_find_auxiliary(alpha, u, M_iterations, dwt, copula, stationary):
    """Find EIS auxiliary params a1t, a2t via backward regression."""
    from pyscarcopula.utils import linear_least_squares

    theta, mu, nu = alpha
    T = len(u)
    n_tr = dwt.shape[1]
    dt = 1.0 / (T - 1)
    t_data = np.linspace(0, 1, T)

    a1t = np.zeros(T)
    a2t = np.zeros(T)

    for j in range(M_iterations):
        if stationary:
            x0 = _ou_stationary_state(theta, mu, nu, n_tr)
        else:
            x0 = _ou_init_state(mu, n_tr)

        xt = _ou_sample_paths(theta, mu, nu, a1t, a2t, dwt, x0)

        if np.isnan(np.sum(xt)):
            return np.zeros(T), np.zeros(T)

        a_data = np.zeros((T, 3))
        a_data[-1] = np.array([0.0, np.mean(a1t), min(np.mean(a2t), 0.0)])

        for i in range(T - 1, 0, -1):
            # Vectorized copula log-pdf for all trajectories at time i
            r_vals = copula.transform(xt[i])
            u1 = np.full(n_tr, u[i, 0])
            u2 = np.full(n_tr, u[i, 1])
            copula_log = copula.log_pdf(u1, u2, r_vals)

            norm_log = _log_norm_ou(theta, mu, nu,
                                    a_data[i][1], a_data[i][2],
                                    dt, xt[i - 1])

            A = np.column_stack((np.ones(n_tr), xt[i], xt[i] ** 2))
            b = copula_log + norm_log

            sigma2 = nu ** 2 / (2.0 * theta) * (1.0 - np.exp(-2.0 * theta * t_data[i]))
            ub = max(1.0 / (2.0 * sigma2) - 0.001, 0.0) if sigma2 > 0 else 0.0

            try:
                lr = linear_least_squares(A, b, 0.0, pseudo_inverse=True)
                if np.isnan(np.sum(lr)):
                    a_data[i - 1] = a_data[i]
                else:
                    a_data[i - 1] = lr
                    a_data[i - 1, 2] = min(a_data[i - 1, 2], ub)
            except Exception:
                a_data[i - 1] = a_data[i]

        a1_hat = a_data[:, 1]
        a2_hat = a_data[:, 2]

        # Smooth with Savitzky-Golay
        wl = max(T // 10, 5)
        if wl % 2 == 0:
            wl += 1
        d = min(2, wl - 1)
        a1t = savgol_filter(a1_hat, wl, d)
        a2t = savgol_filter(a2_hat, wl, d)

        # Enforce a2 upper bounds
        for k in range(1, T):
            t = k * dt
            sigma2 = nu ** 2 / (2.0 * theta) * (1.0 - np.exp(-2.0 * theta * t))
            if sigma2 > 0:
                ub = max(1.0 / (2.0 * sigma2) - 0.01, 0.0)
                a2t[k] = min(a2t[k], ub)

    return a1t, a2t


# ══════════════════════════════════════════════════════════════════
# Transfer matrix infrastructure (shared by all TM functions)
# ══════════════════════════════════════════════════════════════════

class _TMGrid:
    """
    Precomputed transfer-matrix grid and operator for the OU process.

    Encapsulates:
      - adaptive grid z with trapezoidal weights
      - stationary density p0
      - transfer operator (dense or sparse matrix)
      - grid_method selection logic

    All TM-based functions (_tm_loglik, _tm_forward_Jk, etc.) share
    this infrastructure through a common forward/backward pass.
    """

    __slots__ = (
        'z', 'K', 'dz', 'trap_w', 'p0', 'rho', 'sigma', 'sigma_cond',
        'mu', '_T_op', '_grid_method', '_band',
    )

    def __init__(self, theta, mu, nu, n, K=300, grid_range=5.0, 
                 grid_method='auto', adaptive=True, pts_per_sigma = 2):
        """
        Build grid and transfer operator.

        Parameters
        ----------
        theta, mu, nu : float
            OU parameters.
        n : int
            Number of observations (determines dt).
        K : int
            Minimum grid size.
        grid_range : float
            Grid extent in units of stationary sigma.
        grid_method : str
            'auto', 'dense', or 'sparse'.
        adaptive : bool
            Use adaptive grid size. If True then K can be extended to K_eff depending on initial data
        """
        dt = 1.0 / (n - 1)
        rho = np.exp(-theta * dt)
        sigma = np.sqrt(0.5 * nu ** 2 / theta)
        sigma_cond = sigma * np.sqrt(1.0 - rho ** 2)

        self.rho = rho
        self.sigma = sigma
        self.sigma_cond = sigma_cond
        self.mu = mu

        # ── adaptive grid: at least pts_per_sigma points per sigma_cond ──────
        if adaptive:
            dz_target = sigma_cond / pts_per_sigma
            K_min = int(np.ceil(2.0 * grid_range * sigma / dz_target)) + 1
            K_eff = max(K, K_min)
        else:
            K_eff = K

        z = np.linspace(-grid_range * sigma, grid_range * sigma, K_eff)
        dz = z[1] - z[0]

        trap_w = np.full(K_eff, dz)
        trap_w[0] *= 0.5
        trap_w[-1] *= 0.5

        self.z = z
        self.K = K_eff
        self.dz = dz
        self.trap_w = trap_w

        # ── stationary density ───────────────────────────────────
        self.p0 = (np.exp(-0.5 * (z / sigma) ** 2)
                    / (sigma * np.sqrt(2.0 * np.pi)))

        # ── choose and build transfer operator ───────────────────
        half_width = 5.0 * sigma_cond
        self._band = int(np.ceil(half_width / dz))

        if grid_method == 'auto':
            if self._band >= K_eff // 4:
                grid_method = 'dense'
            else:
                grid_method = 'sparse'
        self._grid_method = grid_method

        if grid_method == 'sparse':
            self._T_op = _build_sparse_T_vectorized(
                z, rho, sigma_cond, trap_w, K_eff, self._band)
        else:
            self._T_op = _build_dense_T(z, rho, sigma_cond, trap_w, K_eff)

    # ── matrix-vector products ───────────────────────────────────

    def matvec(self, v):
        """T_trap @ v  (transition kernel applied to vector)."""
        return self._T_op @ v

    def rmatvec(self, v):
        """T_trap.T @ v  (adjoint/transpose application)."""
        return self._T_op.T @ v

    # ── copula density on grid ───────────────────────────────────

    def copula_grid(self, u, copula):
        """
        Evaluate copula density on grid for all observations.

        Returns (n, K) array: fi_grid[t, j] = c(u1t, u2t; Psi(z_j + mu)).
        """
        x_grid = self.z + self.mu
        return copula.copula_grid_batch(u, x_grid)

    # ── generic backward pass (used by _tm_loglik) ───────────────

    def backward_pass(self, fi_grid):
        """
        Backward message pass with normalization.

        Parameters
        ----------
        fi_grid : (n, K)

        Returns
        -------
        log_scale : float
        msg : (K,) or None on failure.
        """
        n = fi_grid.shape[0]
        msg = np.ones(self.K)
        log_scale = 0.0

        for i in range(n - 1, 0, -1):
            msg = self.matvec(fi_grid[i] * msg)
            mx = np.max(np.abs(msg))
            if mx > 0:
                log_scale += np.log(mx)
                msg /= mx
            else:
                return log_scale, None

        return log_scale, msg

    # ── generic forward pass (used by all forward functions) ─────

    def forward_pass(self, fi_grid, callback):
        """
        Forward message pass with per-step callback.

        At each step k the callback receives:
            callback(k, alpha, trap_w, is_last)
        where alpha is the (unnormalized) forward message *before*
        observing u_k (i.e., the predictive density).

        The callback can return False to stop early.

        Parameters
        ----------
        fi_grid : (n, K)
        callback : callable(k, alpha, trap_w, is_last) -> bool
        """
        n = fi_grid.shape[0]
        alpha = self.p0.copy()

        for k in range(n):
            is_last = (k == n - 1)
            cont = callback(k, alpha, self.trap_w, is_last)
            if cont is False:
                break

            # Update: absorb observation u_k and propagate
            if not is_last:
                source = fi_grid[k] * alpha * self.trap_w
                alpha = self.rmatvec(source)
                mx = np.max(np.abs(alpha))
                if mx > 0:
                    alpha /= mx
                else:
                    # Propagation collapsed — let callback handle tail
                    for kk in range(k + 1, n):
                        callback(kk, None, self.trap_w, kk == n - 1)
                    break

    def forward_weights(self, fi_grid):
        """
        Forward pass returning normalized predictive weights.

        At each step k, returns the predictive distribution p(x_k | u_{1:k-1})
        as normalized weights on the grid.

        Parameters
        ----------
        fi_grid : (n, K) — copula density on grid for each observation

        Returns
        -------
        weights : (n, K) — normalized predictive weights, rows sum to 1.
            weights[k, j] = p(x_k = z_j | u_{1:k-1})
        """
        n = fi_grid.shape[0]
        K = self.K
        weights = np.zeros((n, K))

        def _cb(k, alpha, trap_w, is_last):
            if alpha is None:
                weights[k] = 1.0 / K
                return True
            raw_w = alpha * trap_w
            total = np.sum(raw_w)
            if total > 0:
                weights[k] = raw_w / total
            else:
                weights[k] = 1.0 / K
            return True

        self.forward_pass(fi_grid, _cb)
        return weights


# ══════════════════════════════════════════════════════════════════
# Transfer operator builders
# ══════════════════════════════════════════════════════════════════

def _build_dense_T(z, rho, sigma_cond, trap_w, K):
    """Dense K×K transfer matrix with trapezoidal weights baked in."""
    means = rho * z
    diff = z[np.newaxis, :] - means[:, np.newaxis]
    T_mat = (np.exp(-0.5 * (diff / sigma_cond) ** 2)
             / (sigma_cond * np.sqrt(2.0 * np.pi)))
    return T_mat * trap_w[np.newaxis, :]


def _build_sparse_T_vectorized(z, rho, sigma_cond, trap_w, K, band):
    """
    Sparse banded transfer matrix (vectorized construction).

    For each row j, only columns within [i_lo, i_hi) around the
    kernel center rho*z[j] are stored.  This version avoids the
    Python-level per-row loop using vectorized index computation.
    """
    z0 = z[0]
    dz = z[1] - z[0]
    inv_dz = 1.0 / dz
    coeff = 1.0 / (sigma_cond * np.sqrt(2.0 * np.pi))

    # Centre of kernel for each row j
    centers = rho * z                          # (K,)
    i_centers = (centers - z0) * inv_dz        # fractional index

    # Band limits per row
    i_lo = np.maximum(0, np.floor(i_centers).astype(np.intp) - band)
    i_hi = np.minimum(K, np.ceil(i_centers).astype(np.intp) + band + 1)
    widths = i_hi - i_lo                       # elements per row

    total_nnz = int(np.sum(widths))

    rows = np.empty(total_nnz, dtype=np.int32)
    cols = np.empty(total_nnz, dtype=np.int32)
    vals = np.empty(total_nnz, dtype=np.float64)

    ptr = 0
    for j in range(K):
        w = widths[j]
        if w <= 0:
            continue
        sl = slice(ptr, ptr + w)
        i_range = np.arange(i_lo[j], i_hi[j])
        rows[sl] = j
        cols[sl] = i_range
        diff = z[i_range] - centers[j]
        vals[sl] = coeff * np.exp(-0.5 * (diff / sigma_cond) ** 2) * trap_w[i_range]
        ptr += w

    return csr_matrix((vals[:ptr], (rows[:ptr], cols[:ptr])), shape=(K, K))


# ══════════════════════════════════════════════════════════════════
# Transfer matrix log-likelihood (SCAR-TM-OU)
# ══════════════════════════════════════════════════════════════════

def _tm_loglik(theta, mu, nu, u, copula, K=300, grid_range=5.0, 
               grid_method='auto', adaptive=True, pts_per_sigma = 2):
    """
    Transfer matrix backward pass.  Returns minus log-likelihood.

    Parameters
    ----------
    theta, mu, nu : float
        OU process parameters  (theta > 0, nu > 0).
    u : ndarray (n, 2)
        Pseudo-observations.
    copula : BivariateCopula
        Must expose  copula.pdf_on_grid(u_row, z_grid) -> (K,).
    K : int
        Minimum number of grid points.
    grid_range : float
        Grid spans  [-grid_range*sigma, +grid_range*sigma].
    grid_method : str
        'auto', 'dense', or 'sparse'.

    Returns
    -------
    float : minus log-likelihood  (1e10 on numerical failure).
    """
    if theta <= 0 or nu <= 0:
        return 1e10

    n = len(u)
    if n < 2:
        return 1e10

    sigma = np.sqrt(0.5 * nu ** 2 / theta)
    sigma_cond = sigma * np.sqrt(1.0 - np.exp(-2.0 * theta / (n - 1)))
    if sigma <= 0 or sigma_cond <= 0:
        return 1e10

    try:
        grid = _TMGrid(theta, mu, nu, n, K, grid_range, grid_method, adaptive, pts_per_sigma)
    except Exception:
        return 1e10

    fi_grid = grid.copula_grid(u, copula)

    log_scale, msg = grid.backward_pass(fi_grid)

    if msg is None:
        return 1e10

    # Final convolution with stationary density
    result = np.sum(fi_grid[0] * grid.p0 * msg * grid.trap_w)

    if result <= 0:
        return 1e10

    return -(np.log(result) + log_scale)


# ══════════════════════════════════════════════════════════════════
# Forward-pass functions (all use _TMGrid)
# ══════════════════════════════════════════════════════════════════

def _tm_forward_Jk(theta, mu, nu, u, copula, K=300, grid_range=5.0,
                   grid_method='auto', adaptive=True, pts_per_sigma = 2):
    """
    Forward pass: E[Psi(x_k) | u_{1:k-1}] (predictive smoothed parameter).
    J_k uses data BEFORE time k (not including u_k).

    Returns (n,) array.
    """
    n = len(u)
    grid = _TMGrid(theta, mu, nu, n, K, grid_range, grid_method, adaptive, pts_per_sigma)
    fi_grid = grid.copula_grid(u, copula)
    g_grid = copula.transform(grid.z + grid.mu)

    weights = grid.forward_weights(fi_grid)
    J = np.sum(weights * g_grid[np.newaxis, :], axis=1)
    return J


def _tm_forward_rosenblatt(theta, mu, nu, u, copula, K=300, grid_range=5.0, 
                           grid_method='auto', adaptive=True, pts_per_sigma = 2):
    """
    Forward pass: mixture Rosenblatt transform for GoF test.

    For each k, computes:
        e_{k,1} = u_{k,1}
        e_{k,2} = E[h(u_{k,2}, u_{k,1}, Psi(x_k)) | u_{1:k-1}]

    Returns (n, 2) — Rosenblatt-transformed pseudo-observations.
    """
    n = len(u)
    grid = _TMGrid(theta, mu, nu, n, K, grid_range, grid_method, adaptive, pts_per_sigma)
    fi_grid = grid.copula_grid(u, copula)
    r_grid = copula.transform(grid.z + grid.mu)

    weights = grid.forward_weights(fi_grid)

    e = np.empty((n, 2))
    e[:, 0] = u[:, 0]

    for k in range(n):
        u2_vec = np.full(grid.K, u[k, 1])
        u1_vec = np.full(grid.K, u[k, 0])
        h_vals = copula.h(u2_vec, u1_vec, r_grid)
        e[k, 1] = np.sum(h_vals * weights[k])

    eps = 1e-6
    return np.clip(e, eps, 1.0 - eps)


def _tm_forward_mixture_h(theta, mu, nu, u, copula, K=300, grid_range=5.0, 
                          grid_method='auto', adaptive=True, pts_per_sigma = 2):
    """
    Mixture h-function via TM forward pass.

    Returns (n,) array:
        h_k = E[h(u_{k,2}, u_{k,1}, Psi(x_k)) | u_{1:k-1}]

    Same as second column of _tm_forward_rosenblatt, but standalone
    for use in vine Rosenblatt where we need per-edge mixture h-values.
    """
    n = len(u)
    grid = _TMGrid(theta, mu, nu, n, K, grid_range, grid_method, adaptive, pts_per_sigma)
    fi_grid = grid.copula_grid(u, copula)
    r_grid = copula.transform(grid.z + grid.mu)

    weights = grid.forward_weights(fi_grid)

    h_mix = np.empty(n)
    for k in range(n):
        u2_vec = np.full(grid.K, u[k, 1])
        u1_vec = np.full(grid.K, u[k, 0])
        h_vals = copula.h(u2_vec, u1_vec, r_grid)
        h_mix[k] = np.sum(h_vals * weights[k])

    return np.clip(h_mix, 1e-6, 1.0 - 1e-6)


def _tm_xT_distribution(theta, mu, nu, u, copula, K=300, grid_range=5.0, 
                        grid_method='auto', adaptive=True, pts_per_sigma = 2):
    """
    Forward pass: distribution of x_T on grid.

    Unlike the other forward functions, this uses the *absolute*
    x-grid (z + mu) and accumulates all observations (including the
    last one) into the density before returning.

    Returns (z_grid, prob) where z_grid includes mu offset.
    """
    n = len(u)
    grid = _TMGrid(theta, mu, nu, n, K, grid_range, grid_method, adaptive, pts_per_sigma)
    fi_grid = grid.copula_grid(u, copula)

    alpha = grid.p0.copy()
    log_scale = 0.0

    for t in range(n):
        alpha *= fi_grid[t]

        if t < n - 1:
            alpha = grid.rmatvec(alpha * grid.trap_w)

        mx = np.max(np.abs(alpha))
        if mx > 0:
            log_scale += np.log(mx)
            alpha /= mx

    total = np.sum(alpha * grid.trap_w)
    if total > 0:
        prob = (alpha * grid.trap_w) / total
    else:
        prob = np.ones(grid.K) / grid.K

    z_grid = grid.z + grid.mu
    return z_grid, prob


# ══════════════════════════════════════════════════════════════════
# Analytical gradient of TM log-likelihood (xi-coordinates)
# ══════════════════════════════════════════════════════════════════
#
# Key insight:  working in normalised coordinates  xi = z / sigma
# (fixed grid), several parameter dependencies cancel:
#
#   T_w[i,j] depends on theta ONLY through  rho = exp(-theta*dt).
#             Does NOT depend on nu (or mu).
#
#   p0[j] * trap_w[j] is completely independent of (theta, mu, nu).
#
#   fi[t,j] = c(u_1t, u_2t; Psi(sigma*xi_j + mu))
#             depends on theta and nu through sigma, and on mu directly.

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


def _build_Tw_and_grad_sparse(xi, rho, base_w, K, band):
    """
    Sparse version of T_w and dT_w/drho in xi-coordinates.

    Returns T_w (CSR), dTw_drho (CSR).
    """
    omr2 = 1.0 - rho ** 2
    inv_omr2 = 1.0 / omr2
    coeff = 1.0 / (np.sqrt(omr2) * np.sqrt(2.0 * np.pi))

    d_xi = xi[1] - xi[0]
    xi0 = xi[0]
    inv_dxi = 1.0 / d_xi

    centers_idx = (rho * xi - xi0) * inv_dxi
    i_lo = np.maximum(0, np.floor(centers_idx).astype(np.intp) - band)
    i_hi = np.minimum(K, np.ceil(centers_idx).astype(np.intp) + band + 1)
    widths = i_hi - i_lo
    total_nnz = int(np.sum(widths))

    rows = np.empty(total_nnz, dtype=np.int32)
    cols = np.empty(total_nnz, dtype=np.int32)
    t_vals = np.empty(total_nnz, dtype=np.float64)
    g_vals = np.empty(total_nnz, dtype=np.float64)

    ptr = 0
    for i in range(K):
        w = widths[i]
        if w <= 0:
            continue
        sl = slice(ptr, ptr + w)
        j_range = np.arange(i_lo[i], i_hi[i])
        rows[sl] = i
        cols[sl] = j_range

        q = xi[j_range] - rho * xi[i]
        gauss = coeff * np.exp(-0.5 * q ** 2 * inv_omr2)
        tw = gauss * base_w[j_range]
        t_vals[sl] = tw

        dlog = rho * inv_omr2 + q * xi[i] * inv_omr2 - rho * q ** 2 * inv_omr2 ** 2
        g_vals[sl] = dlog * tw

        ptr += w

    shape = (K, K)
    T_sp = csr_matrix((t_vals[:ptr], (rows[:ptr], cols[:ptr])), shape=shape)
    G_sp = csr_matrix((g_vals[:ptr], (rows[:ptr], cols[:ptr])), shape=shape)
    return T_sp, G_sp


def _tm_loglik_with_grad(theta, mu, nu, u, copula, K=300, grid_range=5.0,
                          grid_method='auto', adaptive=True, pts_per_sigma=2):
    """
    Transfer matrix log-likelihood with analytical gradient.

    Uses normalised coordinates xi = z / sigma so that:
    - T_w depends on theta only through rho = exp(-theta*dt)
    - p0 * w is parameter-independent
    - fi depends on (theta, nu) through sigma and on mu directly

    Parameters
    ----------
    theta, mu, nu : float
        OU process parameters  (theta > 0, nu > 0).
    u : ndarray (n, 2)
        Pseudo-observations.
    copula : BivariateCopula
    K, grid_range, grid_method, adaptive, pts_per_sigma : grid params

    Returns
    -------
    neg_logL : float
        Minus log-likelihood (1e10 on failure).
    neg_grad : ndarray (3,)
        Minus gradient w.r.t. (theta, mu, nu).  Zero on failure.
    """
    FAIL = 1e10, np.zeros(3)

    if theta <= 0 or nu <= 0:
        return FAIL

    n = len(u)
    if n < 2:
        return FAIL

    dt = 1.0 / (n - 1)
    rho = np.exp(-theta * dt)
    sigma2 = 0.5 * nu ** 2 / theta
    sigma = np.sqrt(sigma2)
    sigma_c = sigma * np.sqrt(1.0 - rho ** 2)

    if sigma <= 0 or sigma_c <= 0:
        return FAIL

    # ── fixed normalised grid ────────────────────────────────────
    if adaptive:
        dz_target = sigma_c / pts_per_sigma
        K_min = int(np.ceil(2.0 * grid_range * sigma / dz_target)) + 1
        K_eff = max(K, K_min)
    else:
        K_eff = K

    xi = np.linspace(-grid_range, grid_range, K_eff)
    d_xi = xi[1] - xi[0]

    base_w = np.full(K_eff, d_xi)
    base_w[0] *= 0.5
    base_w[-1] *= 0.5

    # p0 * trap_w is constant (sigma cancels in xi-coordinates)
    pw_const = np.exp(-0.5 * xi ** 2) / np.sqrt(2.0 * np.pi) * base_w

    # ── build T_w and dT_w/drho ──────────────────────────────────
    half_width_xi = 5.0 * np.sqrt(1.0 - rho ** 2)
    band = int(np.ceil(half_width_xi / d_xi))

    if grid_method == 'auto':
        grid_method = 'dense' if band >= K_eff // 4 else 'sparse'

    try:
        if grid_method == 'sparse':
            T_w, dTw_drho = _build_Tw_and_grad_sparse(
                xi, rho, base_w, K_eff, band)
        else:
            T_w, dTw_drho = _build_Tw_and_grad_dense(
                xi, rho, base_w, K_eff)
    except Exception:
        return FAIL

    drho_dtheta = -dt * rho

    # ── copula density and its derivative on grid ─────────────────
    x_grid = sigma * xi + mu

    # Use batch method (fused numba kernel if available, else loop fallback)
    fi, dfi_dx = copula.pdf_and_grad_on_grid_batch(u, x_grid)

    # dx/dalpha:  x = sigma*xi + mu
    d_sigma_dtheta = -0.5 * nu ** 2 / theta ** 2 / (2.0 * sigma)
    d_sigma_dnu = nu / (theta * 2.0 * sigma)

    dx_dalpha = np.zeros((3, K_eff))
    dx_dalpha[0] = d_sigma_dtheta * xi   # dx/dtheta
    dx_dalpha[1] = 1.0                    # dx/dmu
    dx_dalpha[2] = d_sigma_dnu * xi       # dx/dnu

    # ══════════════════════════════════════════════════════════════
    # BACKWARD PASS — compute and store beta[t] and c_vals[t]
    # ══════════════════════════════════════════════════════════════

    def matvec(v):
        return T_w @ v

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
                contrib += (dTw_drho @ target) * drho_dtheta

            new_d_beta[k] = contrib * inv_c

        d_beta = new_d_beta

    # ── Assemble gradient ────────────────────────────────────────
    grad = np.zeros(3)
    for k in range(3):
        dfi_k_0 = dfi_dx[0] * dx_dalpha[k]
        num = np.sum((dfi_k_0 * beta[0] + fi[0] * d_beta[k]) * pw_const)
        grad[k] = num / Z0

    return neg_logL, -grad


# ══════════════════════════════════════════════════════════════════
# Main class
# ══════════════════════════════════════════════════════════════════

METHODS = ('MLE', 'SCAR-P-OU', 'SCAR-M-OU', 'SCAR-TM-OU')


class OULatentProcess:
    """OU latent process for stochastic copula models."""

    def __init__(self, copula):
        self.copula = copula
        self.fit_result = None

    @staticmethod
    def calculate_dwt(T, n_tr, seed=None):
        """Generate Wiener increments (T, n_tr)."""
        rng = np.random.RandomState(seed)
        dt = 1.0 / (T - 1)
        return rng.normal(0, 1, size=(T, n_tr)) * np.sqrt(dt)

    def fit(self, u, method='SCAR-TM-OU', alpha0=None, tol=1e-2,
            n_tr=500, M_iterations=5, seed=None, dwt=None,
            stationary=True, K=300, grid_range=5.0,
            grid_method='auto', adaptive=True, pts_per_sigma=2,
            analytical_grad=True, smart_init=True, verbose=False):
        """
        Fit the stochastic copula model.

        Estimates the OU process parameters alpha = (theta, mu, nu) by
        maximizing the log-likelihood. For SCAR-TM-OU (recommended),
        uses the transfer matrix method with analytical gradient and
        automatic parameter rescaling for L-BFGS-B.

        Parameters
        ----------
        u : array (T, 2)
            Pseudo-observations in [0, 1]^2.
        method : str
            'SCAR-TM-OU' (recommended), 'SCAR-P-OU', 'SCAR-M-OU', or 'MLE'.
        alpha0 : array (3,) or None
            Initial (theta, mu, nu). If None, computed automatically:
            - smart_init=True: analytical heuristic from MLE (zero cost)
            - smart_init=False: [1.0, inv_transform(theta_mle), 1.0]
        tol : float
            L-BFGS-B gradient tolerance. Default 1e-2. Use 5e-2 for
            ~2x speedup with < 1 logL loss (recommended for vine).
        K : int
            Minimum grid size. Auto-increased by adaptive rule.
            Default 300. K=150 is usually sufficient.
        grid_range : float
            Grid spans [-grid_range*sigma, +grid_range*sigma]. Default 5.0.
        grid_method : str
            'auto' (recommended), 'dense', or 'sparse'.
        adaptive : bool
            Adaptive grid refinement (default True).
        pts_per_sigma : int
            Points per conditional sigma for adaptive rule. Default 2.
        analytical_grad : bool
            Use analytical gradient for SCAR-TM-OU (default True).
            Reduces nfev by ~3-4x. Parameters are auto-rescaled to
            help L-BFGS-B estimate the initial Hessian correctly.
        smart_init : bool
            Compute initial point via analytical heuristic (default True).
            Uses MLE to set mu, then estimates theta and nu from target
            autocorrelation (rho=0.95) and volatility (sigma=0.3*|mu|).
            Zero computational cost. Up to 5x speedup on long series.
        n_tr : int
            MC trajectories for P/M samplers (ignored for TM).
        M_iterations : int
            EIS iterations for M-sampler.
        seed : int or None
        dwt : array (T, n_tr) or None
        stationary : bool
        verbose : bool

        Returns
        -------
        scipy.optimize.OptimizeResult with extra fields:
            .alpha : ndarray (3,) — fitted (theta, mu, nu)
            .log_likelihood : float
            .method : str
            .name : str — copula name
            .K : int — grid size used (TM only)
        """
        method = method.upper()
        if method not in METHODS:
            raise ValueError(f"Unknown method {method}. Available: {METHODS}")

        u = np.asarray(u, dtype=np.float64)
        copula = self.copula

        if method == 'MLE':
            return copula._fit_mle(u)

        T_len = len(u)

        # Prepare dwt
        if dwt is None and method in ('SCAR-P-OU', 'SCAR-M-OU'):
            _seed = seed if seed is not None else np.random.randint(1, 1000000)
            dwt = self.calculate_dwt(T_len, n_tr, _seed)

        # Initial guess
        if alpha0 is None:
            if smart_init and method == 'SCAR-TM-OU':
                try:
                    from pyscarcopula.latent.initial_point import \
                        smart_initial_point
                    alpha0, init_info = smart_initial_point(
                        u, copula, verbose=verbose)
                    if verbose:
                        print(f"Smart init: {init_info.get('chosen_method')}, "
                              f"alpha0={alpha0}")
                except Exception:
                    pass  # fall through to default

            if alpha0 is None:
                mle_result = copula._fit_mle(u)
                mu0 = float(np.atleast_1d(copula.inv_transform(np.atleast_1d(mle_result.copula_param)))[0])
                alpha0 = np.array([1.0, mu0, 1.0])

        bounds = Bounds([0.001, -np.inf, 0.001], [np.inf, np.inf, np.inf])

        # ── SCAR-TM-OU with analytical gradient ─────────────────
        if method == 'SCAR-TM-OU' and analytical_grad:

            # Rescale parameters so that all three are O(1) at start.
            # This helps L-BFGS-B estimate the initial Hessian correctly.
            scale = np.array([
                max(abs(alpha0[0]), 1.0),
                max(abs(alpha0[1]), 1.0),
                max(abs(alpha0[2]), 1.0),
            ])
            x0_scaled = alpha0 / scale
            bounds_scaled = Bounds(
                [0.001 / scale[0], -np.inf, 0.001 / scale[2]],
                [np.inf, np.inf, np.inf]
            )

            def objective_and_grad(x_scaled):
                alpha = x_scaled * scale
                if np.isnan(np.sum(alpha)):
                    return 1e10, np.zeros(3)
                th, mu_v, nu_v = alpha
                try:
                    val, grad = _tm_loglik_with_grad(
                        th, mu_v, nu_v, u, copula, K, grid_range,
                        grid_method, adaptive, pts_per_sigma)
                    return val, grad * scale   # chain rule: d/dx = d/dalpha * scale
                except Exception as e:
                    if verbose:
                        print(f"  error at alpha={alpha}: {e}")
                    return 1e10, np.zeros(3)

            if verbose:
                print(f"Fitting {method} (analytical gradient), alpha0={alpha0}")

            result = minimize(
                objective_and_grad, x0_scaled,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds_scaled,
                options={'gtol': tol, 'maxfun': 100, 'maxiter': 100},
            )

            # Unscale result
            result.x = result.x * scale

        # ── All other methods (numerical gradient) ───────────────
        else:
            def objective(alpha):
                if np.isnan(np.sum(alpha)):
                    return 1e10
                th, mu_v, nu_v = alpha
                try:
                    if method == 'SCAR-P-OU':
                        return _p_sampler_loglik(th, mu_v, nu_v, u, dwt,
                                                copula, stationary)
                    elif method == 'SCAR-M-OU':
                        a1t, a2t = _eis_find_auxiliary(alpha, u, M_iterations,
                                                       dwt, copula, stationary)
                        return _m_sampler_loglik(th, mu_v, nu_v, u, dwt,
                                                a1t, a2t, copula, stationary)
                    elif method == 'SCAR-TM-OU':
                        return _tm_loglik(th, mu_v, nu_v, u, copula, K,
                                         grid_range, grid_method,
                                         adaptive, pts_per_sigma)
                except Exception as e:
                    if verbose:
                        print(f"  error at alpha={alpha}: {e}")
                    return 1e10

            if verbose:
                print(f"Fitting {method}, alpha0={alpha0}")

            result = minimize(
                objective, alpha0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'gtol': tol, 'eps': 1e-4, 'maxfun': 100},
            )

        result.alpha = result.x.copy()
        result.log_likelihood = -result.fun
        result.method = method
        result.name = copula.name
        result.n_tr = n_tr if method not in ['MLE', 'SCAR-TM-OU'] else None
        result.K = K if method == 'SCAR-TM-OU' else None
        self.fit_result = result

        if verbose:
            print(f"  => alpha={result.alpha}, logL={result.log_likelihood:.4f}")

        return result

    def smoothed_params(self, u, alpha=None, K=300, grid_range=5.0,
                        grid_method='auto'):
        """E[Psi(x_k) | u_{1:k-1}] via transfer matrix."""
        if alpha is None:
            if self.fit_result is None:
                raise ValueError("Fit first or provide alpha")
            alpha = self.fit_result.alpha
        theta, mu, nu = alpha
        return _tm_forward_Jk(theta, mu, nu, u, self.copula, K, grid_range,
                              grid_method)

    def rosenblatt(self, u, alpha=None, K=300, grid_range=5.0,
                   grid_method='auto'):
        """Mixture Rosenblatt transform for GoF test."""
        if alpha is None:
            if self.fit_result is None:
                raise ValueError("Fit first or provide alpha")
            alpha = self.fit_result.alpha
        theta, mu, nu = alpha
        return _tm_forward_rosenblatt(theta, mu, nu, u, self.copula, K,
                                      grid_range, grid_method)

    def mixture_h(self, u, alpha=None, K=300, grid_range=5.0,
                  grid_method='auto'):
        """Mixture h-function for vine Rosenblatt."""
        if alpha is None:
            if self.fit_result is None:
                raise ValueError("Fit first or provide alpha")
            alpha = self.fit_result.alpha
        theta, mu, nu = alpha
        return _tm_forward_mixture_h(theta, mu, nu, u, self.copula, K,
                                     grid_range, grid_method)

    def xT_distribution(self, u, alpha=None, K=300, grid_range=5.0,
                        grid_method='auto'):
        """Distribution of x_T on grid. Returns (z_grid, prob)."""
        if alpha is None:
            if self.fit_result is None:
                raise ValueError("Fit first or provide alpha")
            alpha = self.fit_result.alpha
        theta, mu, nu = alpha
        return _tm_xT_distribution(theta, mu, nu, u, self.copula, K,
                                   grid_range, grid_method)

    def final_state_sample(self, alpha, n, method='SCAR-TM-OU'):
        """Sample x_T from stationary distribution."""
        theta, mu, nu = alpha
        if method.upper() == 'SCAR-TM-OU':
            sigma2 = nu ** 2 / (2.0 * theta)
        else:
            sigma2 = nu ** 2 / (2.0 * theta) * (1.0 - np.exp(-2.0 * theta))
        return np.random.normal(mu, np.sqrt(sigma2), n)