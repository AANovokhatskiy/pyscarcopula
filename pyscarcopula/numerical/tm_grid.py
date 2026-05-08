"""
pyscarcopula.numerical.tm_grid — Transfer matrix grid infrastructure.

Extracted from latent/ou_process.py (lines 278–554).

Contents:
  - TMGrid: precomputed grid, stationary density, transfer operator
  - _build_dense_T: dense K×K transfer matrix
  - _build_sparse_T_vectorized: sparse banded CSR transfer matrix

The TMGrid encapsulates:
  - Adaptive grid refinement (K_eff from pts_per_sigma)
  - Dense/sparse selection (auto based on bandwidth ratio b/K)
  - Forward and backward passes (unified callback interface)
  - Copula density evaluation on grid

All TM-based functions share this infrastructure.
"""

import numpy as np
from scipy.sparse import csr_matrix


class TMGrid:
    """
    Precomputed transfer-matrix grid and operator for the OU process.

    Encapsulates:
      - adaptive grid z with trapezoidal weights
      - stationary density p0
      - transfer operator (dense or sparse matrix)
      - grid_method selection logic

    Parameters
    ----------
    kappa, mu, nu : float
        OU process parameters.
    n : int
        Number of observations (determines dt = 1/(n-1)).
    K : int
        Minimum number of grid points (default 300).
    grid_range : float
        Grid extent in units of stationary sigma (default 5.0).
        Grid spans [-grid_range*sigma, +grid_range*sigma].
    grid_method : str
        'auto' (recommended), 'dense', or 'sparse'.
        'auto' selects based on bandwidth ratio b/K:
          dense if b >= K/4, sparse (CSR) otherwise.
    adaptive : bool
        Use adaptive grid refinement (default True).
        Guarantees at least pts_per_sigma points per sigma_cond.
    pts_per_sigma : int
        Points per conditional standard deviation for adaptive rule (default 2).
        The paper uses n_pts=4 in formula (20), but the code default is 2
        which provides adequate resolution for most cases.
    """

    __slots__ = (
        'z', 'K', 'dz', 'trap_w', 'p0', 'rho', 'sigma', 'sigma_cond',
        'mu', '_T_op', '_grid_method', '_band',
    )

    def __init__(self, kappa, mu, nu, n, K=300, grid_range=5.0,
                 grid_method='auto', adaptive=True, pts_per_sigma=4):
        dt = 1.0 / (n - 1)
        rho = np.exp(-kappa * dt)
        sigma = np.sqrt(0.5 * nu ** 2 / kappa)
        sigma_cond = sigma * np.sqrt(1.0 - rho ** 2)

        self.rho = rho
        self.sigma = sigma
        self.sigma_cond = sigma_cond
        self.mu = mu

        # ── adaptive grid: at least pts_per_sigma points per sigma_cond ──
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
        """T_trap.T @ v  (transpose of the weighted backward operator)."""
        return self._T_op.T @ v

    def predict_matvec(self, v):
        """
        Forward prediction integral.

        ``_T_op[j, i] = p(z_i | z_j) * w_i`` is weighted for backward
        integration over the next-state index.  Forward prediction integrates
        over the previous-state index:

            phi_next[i] = sum_j p(z_i | z_j) * v[j].

        Therefore we remove the next-state quadrature weights after applying
        the transpose of the stored weighted operator.
        """
        return (self._T_op.T @ v) / self.trap_w

    # ── copula density on grid ───────────────────────────────────

    def copula_grid(self, u, copula):
        """
        Evaluate copula density on grid for all observations.

        Returns (n, K) array: fi_grid[t, j] = c(u1t, u2t; Psi(z_j + mu)).
        """
        x_grid = self.z + self.mu
        return copula.copula_grid_batch(u, x_grid)

    # ── backward pass (used by _tm_loglik) ───────────────────────

    def backward_pass(self, fi_grid):
        """
        Backward message pass with normalization.

        Implements equations (12)-(13) from the paper:
          m_T(x_{T-1}) = integral f_T(x_T) p(x_T|x_{T-1}) dx_T
          m_t(x_{t-1}) = integral f_t(x_t) p(x_t|x_{t-1}) m_{t+1}(x_t) dx_t

        Parameters
        ----------
        fi_grid : (n, K) — copula density on grid

        Returns
        -------
        log_scale : float — accumulated log normalization
        msg : (K,) or None on failure
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

    # ── forward pass (used by all forward functions) ─────────────

    def forward_pass(self, fi_grid, callback):
        """
        Forward message pass with per-step callback.

        Implements equations (22)-(23) from the paper:
          phi_k(z) <- f_k(z) * phi_k(z)             (update)
          phi_{k+1}(z') = integral p(z'|z) phi_k(z) dz

        At each step k the callback receives:
            callback(k, phi, trap_w, is_last)
        where phi is the predictive density BEFORE observing u_k.

        Parameters
        ----------
        fi_grid : (n, K)
        callback : callable(k, phi, trap_w, is_last) -> bool
        """
        n = fi_grid.shape[0]
        phi = self.p0.copy()

        for k in range(n):
            is_last = (k == n - 1)
            cont = callback(k, phi, self.trap_w, is_last)
            if cont is False:
                break

            if not is_last:
                source = fi_grid[k] * phi * self.trap_w
                phi = self.predict_matvec(source)
                mx = np.max(np.abs(phi))
                if mx > 0:
                    phi /= mx
                else:
                    for kk in range(k + 1, n):
                        callback(kk, None, self.trap_w, kk == n - 1)
                    break

    def forward_weights(self, fi_grid):
        """
        Forward pass returning normalized predictive weights.

        weights[k, j] ≈ p(x_k = z_j + mu | u_{1:k-1})

        Parameters
        ----------
        fi_grid : (n, K)

        Returns
        -------
        weights : (n, K) — rows sum to ~1
        """
        n = fi_grid.shape[0]
        K = self.K
        weights = np.zeros((n, K))
        phi = self.p0.copy()

        for k in range(n):
            is_last = (k == n - 1)
            raw_w = phi * self.trap_w
            total = np.sum(raw_w)
            if total > 0:
                weights[k] = raw_w / total
            else:
                weights[k] = 1.0 / K

            if not is_last:
                source = fi_grid[k] * phi * self.trap_w
                phi = self.predict_matvec(source)
                mx = np.max(np.abs(phi))
                if mx > 0:
                    phi /= mx
                else:
                    weights[k + 1:] = 1.0 / K
                    break

        return weights


# ══════════════════════════════════════════════════════════════════
# Transfer operator builders
# ══════════════════════════════════════════════════════════════════

def _build_dense_T(z, rho, sigma_cond, trap_w, K):
    """Dense K×K transfer matrix with trapezoidal weights baked in.

    T_ji = p(z_t=z_i | z_{t-1}=z_j) * w_i
         = N(z_i; rho*z_j, sigma_cond^2) * w_i
    """
    means = rho * z
    diff = z[np.newaxis, :] - means[:, np.newaxis]
    T_mat = (np.exp(-0.5 * (diff / sigma_cond) ** 2)
             / (sigma_cond * np.sqrt(2.0 * np.pi)))
    return T_mat * trap_w[np.newaxis, :]


def _build_sparse_T_vectorized(z, rho, sigma_cond, trap_w, K, band):
    """
    Sparse banded transfer matrix in CSR format.

    For each row j, only columns within [i_lo, i_hi) around the
    kernel center rho*z[j] are stored. When b < K/4, this gives
    O(K*b) matvec cost instead of O(K^2).
    """
    z0 = z[0]
    dz = z[1] - z[0]
    inv_dz = 1.0 / dz
    coeff = 1.0 / (sigma_cond * np.sqrt(2.0 * np.pi))

    centers = rho * z
    i_centers = (centers - z0) * inv_dz

    i_lo = np.maximum(0, np.floor(i_centers).astype(np.intp) - band)
    i_hi = np.minimum(K, np.ceil(i_centers).astype(np.intp) + band + 1)
    widths = i_hi - i_lo

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
