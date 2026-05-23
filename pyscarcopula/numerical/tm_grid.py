"""
pyscarcopula.numerical.tm_grid — Transfer matrix grid infrastructure.

Contents:
  - TMGrid: precomputed grid, stationary density, transfer operator
  - _build_dense_T: dense K×K transfer matrix
  - _build_sparse_T_vectorized: sparse banded CSR transfer matrix
  - _build_local_interpolation_T: Gauss-Hermite local transition

The TMGrid encapsulates:
  - Adaptive grid refinement (K_eff from pts_per_sigma)
  - Dense/sparse selection (auto based on bandwidth ratio b/K)
  - Forward and backward passes (unified callback interface)
  - Copula density evaluation on grid

All TM-based functions share this infrastructure.
"""

import numpy as np
from scipy.sparse import csr_matrix


_GRID_METHODS = frozenset(('auto', 'dense', 'sparse'))
_TRANSITION_METHODS = frozenset(('auto', 'matrix', 'gh', 'spectral'))


def _validate_grid_method(value):
    method = str(value).lower()
    if method not in _GRID_METHODS:
        raise ValueError(
            "grid_method must be one of 'auto', 'dense', or 'sparse', "
            f"got {value!r}"
        )
    return method


def _validate_transition_method(value):
    method = str(value).lower()
    if method not in _TRANSITION_METHODS:
        raise ValueError(
            "transition_method must be one of "
            "'auto', 'matrix', 'gh', or 'spectral', "
            f"got {value!r}"
        )
    if method == 'spectral':
        # Spectral likelihood has no finite grid state; grid-only routines use
        # the automatic matrix/GH fallback for forward distributions.
        return 'auto'
    return method


def _validate_optional_min_int(value, name, minimum):
    if value is None:
        return None
    value = _validate_positive_int(value, name)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _validate_positive_int(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive integer")
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_positive_float(value, name):
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _select_matrix_grid_method(band, K):
    if band >= K // 4:
        return 'dense'
    return 'sparse'


def _select_transition_method(requested, r_kernel_grid, r_gh,
                              adaptive_was_capped):
    if requested != 'auto':
        return requested
    if adaptive_was_capped:
        return 'gh'
    if r_kernel_grid <= r_gh:
        return 'gh'
    return 'matrix'


def _normal_hermite_rule(order):
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    return nodes, weights / np.sqrt(np.pi)


def _local_rule(transition_method, gh_order):
    nodes, weights = _normal_hermite_rule(gh_order)
    return nodes, weights


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
    transition_method : str
        Prototype transition operator selector. 'matrix' preserves the current
        Gaussian matrix path. 'auto' chooses between that path, deterministic
        interpolation, and Gauss-Hermite interpolation. The Gaussian matrix
        path still uses grid_method to choose sparse or dense storage. 'gh'
        forces the local Gauss-Hermite operator.
    max_K : int or None
        Optional cap for the adaptive effective grid size. If the adaptive
        rule requests more than max_K points, K_eff is capped and the
        `_adaptive_was_capped` diagnostic is set.
    r_gh : float
        Locality threshold based on sigma_cond / dz. Auto selection uses GH
        for narrow local kernels.
    gh_order : int
        Reserved Gauss-Hermite order for the local transition operator.
    """

    __slots__ = (
        'z', 'K', 'dz', 'trap_w', 'p0', 'rho', 'sigma', 'sigma_cond',
        'mu', '_T_op', '_grid_method', '_transition_method',
        '_transition_method_requested', '_max_K', '_K_adaptive',
        '_adaptive_was_capped', '_r_gh', '_gh_order',
        '_r_kernel_grid', '_band',
    )

    def __init__(self, kappa, mu, nu, n, K=300, grid_range=5.0,
                 grid_method='auto', adaptive=True, pts_per_sigma=4,
                 transition_method='matrix', max_K=None, r_gh=3.0,
                 gh_order=5):
        grid_method = _validate_grid_method(grid_method)
        transition_method = _validate_transition_method(transition_method)
        max_K = _validate_optional_min_int(max_K, 'max_K', 2)
        gh_order = _validate_positive_int(gh_order, 'gh_order')

        dt = 1.0 / (n - 1)
        rho = np.exp(-kappa * dt)
        sigma = np.sqrt(0.5 * nu ** 2 / kappa)
        sigma_cond = sigma * np.sqrt(1.0 - rho ** 2)

        self.rho = rho
        self.sigma = sigma
        self.sigma_cond = sigma_cond
        self.mu = mu
        self._transition_method_requested = transition_method
        self._max_K = max_K
        self._r_gh = _validate_positive_float(r_gh, 'r_gh')
        self._gh_order = gh_order

        # ── adaptive grid: at least pts_per_sigma points per sigma_cond ──
        if adaptive:
            dz_target = sigma_cond / pts_per_sigma
            K_min = int(np.ceil(2.0 * grid_range * sigma / dz_target)) + 1
            K_adaptive = max(K, K_min)
        else:
            K_adaptive = K

        if max_K is not None:
            K_eff = min(K_adaptive, max_K)
            K_eff = max(K_eff, min(K, max_K))
        else:
            K_eff = K_adaptive
        self._K_adaptive = K_adaptive
        self._adaptive_was_capped = K_eff < K_adaptive

        z = np.linspace(-grid_range * sigma, grid_range * sigma, K_eff)
        dz = z[1] - z[0]
        self._r_kernel_grid = sigma_cond / dz
        transition_method = _select_transition_method(
            transition_method,
            self._r_kernel_grid,
            self._r_gh,
            self._adaptive_was_capped,
        )
        self._transition_method = transition_method

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

        if transition_method == 'gh':
            self._grid_method = 'local'
            self._T_op = _build_local_interpolation_T(
                z, rho, sigma_cond, K_eff, transition_method, gh_order)
        else:
            if grid_method == 'auto':
                grid_method = _select_matrix_grid_method(self._band, K_eff)
            self._grid_method = grid_method

            if grid_method == 'sparse':
                self._T_op = _build_sparse_T_vectorized(
                    z, rho, sigma_cond, trap_w, K_eff, self._band)
            else:
                self._T_op = _build_dense_T(z, rho, sigma_cond, trap_w, K_eff)

    # ── matrix-vector products ───────────────────────────────────

    def diagnostics(self):
        """Return numerical grid and transition-operator diagnostics."""
        return {
            'K': int(self.K),
            'K_adaptive': int(self._K_adaptive),
            'adaptive_was_capped': bool(self._adaptive_was_capped),
            'max_K': None if self._max_K is None else int(self._max_K),
            'grid_method': self._grid_method,
            'transition_method': self._transition_method,
            'transition_method_requested': self._transition_method_requested,
            'r_kernel_grid': float(self._r_kernel_grid),
            'r_gh': float(self._r_gh),
            'gh_order': int(self._gh_order),
            'band': int(self._band),
            'rho': float(self.rho),
            'sigma': float(self.sigma),
            'sigma_cond': float(self.sigma_cond),
            'dz': float(self.dz),
        }

    def matvec(self, v):
        """Transition operator applied to a value vector."""
        return self._T_op @ v

    def rmatvec(self, v):
        """Transpose transition operator applied to a vector."""
        return self._T_op.T @ v

    def predict_matvec(self, v):
        """
        Forward prediction integral.

        For the Gaussian matrix path, ``_T_op[j, i] = p(z_i | z_j) * w_i``
        is weighted for backward integration over the next-state index.  For
        local interpolation paths, rows contain direct interpolation weights.
        Both conventions support the same forward-density update:

            phi_next[i] = sum_j p(z_i | z_j) * v[j].

        Therefore we remove the next-state quadrature weights after applying
        the transpose of the stored operator.
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

    def copula_grid_row(self, u_row, copula):
        """Evaluate copula density on the grid for one observation."""
        x_grid = self.z + self.mu
        return copula.pdf_on_grid(u_row, x_grid)

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
        Helper for forward message passes with a per-step callback.

        This method is retained for custom diagnostics and extension code.
        Built-in predictive routines use ``forward_weights()`` directly.

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

    def predictive_weights_from_phi(self, phi):
        """Normalize a predictive density to grid probability weights."""
        if phi is None:
            return np.full(self.K, 1.0 / self.K, dtype=np.float64)
        raw_w = phi * self.trap_w
        total = np.sum(raw_w)
        if total > 0:
            return raw_w / total
        return np.full(self.K, 1.0 / self.K, dtype=np.float64)

    def advance_forward_phi(self, phi, fi_row):
        """Advance one predictive forward density without storing history."""
        if phi is None:
            return None
        source = fi_row * phi * self.trap_w
        phi_next = self.predict_matvec(source)
        mx = np.max(np.abs(phi_next))
        if mx > 0:
            return phi_next / mx
        return None

    def iter_forward_weights(self, u, copula, need_last_emission=False):
        """
        Stream predictive weights and one-row emissions.

        Yields ``(k, weights, fi_row, phi, posterior_phi)``.  ``weights`` is
        the normalized predictive mass before observing row ``k``.  ``fi_row``
        is evaluated only when needed for the next update, or on the final row
        when ``need_last_emission=True``.  No ``(T, K)`` arrays are allocated.
        """
        n = len(u)
        phi = self.p0.copy()

        for k in range(n):
            is_last = (k == n - 1)
            weights = self.predictive_weights_from_phi(phi)
            fi_row = None
            if (not is_last) or need_last_emission:
                fi_row = self.copula_grid_row(u[k], copula)

            if phi is None or fi_row is None:
                posterior_phi = None
            else:
                posterior_phi = phi * fi_row

            yield k, weights, fi_row, phi, posterior_phi

            if not is_last:
                phi = self.advance_forward_phi(phi, fi_row)

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


def _build_local_interpolation_T(z, rho, sigma_cond, K, transition_method,
                                 gh_order):
    """
    Local transition operator using Gauss-Hermite interpolation.

    Rows are source grid points and columns are target grid points. Entries
    are direct interpolation weights for applying

        (P f)(z_i) ~= E[f(rho * z_i + sigma_cond * eps)]

    on the centered OU grid. Rows sum to one up to floating-point error.
    """
    gh_nodes, gh_weights = _local_rule(transition_method, gh_order)
    offsets = np.sqrt(2.0) * sigma_cond * gh_nodes
    q = len(offsets)

    z0 = z[0]
    z_last = z[-1]
    dz = z[1] - z[0]

    rows = np.empty(K * q * 2, dtype=np.int32)
    cols = np.empty(K * q * 2, dtype=np.int32)
    vals = np.empty(K * q * 2, dtype=np.float64)

    ptr = 0
    for i in range(K):
        center = rho * z[i]
        for offset, weight in zip(offsets, gh_weights):
            y = center + offset

            if y <= z0:
                rows[ptr] = i
                cols[ptr] = 0
                vals[ptr] = weight
                ptr += 1
                continue
            if y >= z_last:
                rows[ptr] = i
                cols[ptr] = K - 1
                vals[ptr] = weight
                ptr += 1
                continue

            left = int(np.floor((y - z0) / dz))
            if left >= K - 1:
                rows[ptr] = i
                cols[ptr] = K - 1
                vals[ptr] = weight
                ptr += 1
                continue

            lam = (y - z[left]) / dz
            rows[ptr] = i
            cols[ptr] = left
            vals[ptr] = weight * (1.0 - lam)
            ptr += 1

            rows[ptr] = i
            cols[ptr] = left + 1
            vals[ptr] = weight * lam
            ptr += 1

    return csr_matrix((vals[:ptr], (rows[:ptr], cols[:ptr])), shape=(K, K))
