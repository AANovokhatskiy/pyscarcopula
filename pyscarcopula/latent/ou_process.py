"""
Ornstein-Uhlenbeck latent process for stochastic copula models.

Supports fitting methods:
  SCAR-P-OU   — MC without importance sampling (p-sampler)
  SCAR-M-OU   — MC with efficient importance sampling (m-sampler)
  SCAR-TM-OU  — transfer matrix (deterministic quadrature)

MLE is handled directly by BivariateCopula._fit_mle.
"""

import numpy as np
from numba import njit
from scipy.optimize import minimize, Bounds
from scipy.signal import savgol_filter


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

    xt = _ou_sample_paths_exact(theta, mu, nu, dwt, x0)

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
# Transfer matrix (SCAR-TM-OU)
# ══════════════════════════════════════════════════════════════════

def _tm_loglik(theta, mu, nu, u, copula, K, grid_range):
    """Transfer matrix backward pass. Returns minus log-likelihood."""
    n = len(u)
    dt = 1.0 / (n - 1)

    rho = np.exp(-theta * dt)
    sigma = np.sqrt(0.5 * nu ** 2 / theta)
    sigma_cond = sigma * np.sqrt(1.0 - rho ** 2)

    z = np.linspace(-grid_range * sigma, grid_range * sigma, K)
    dz = z[1] - z[0]

    trap_w = np.full(K, dz)
    trap_w[0] *= 0.5
    trap_w[-1] *= 0.5

    means = rho * z
    diff = z[np.newaxis, :] - means[:, np.newaxis]
    T_mat = np.exp(-0.5 * (diff / sigma_cond) ** 2) / (sigma_cond * np.sqrt(2.0 * np.pi))
    T_trap = T_mat * trap_w[np.newaxis, :]

    fi_grid = np.empty((n, K))
    for i in range(n):
        fi_grid[i] = copula.pdf_on_grid(u[i], z + mu)

    msg = np.ones(K)
    log_scale = 0.0

    for i in range(n - 1, 0, -1):
        msg = T_trap @ (fi_grid[i] * msg)
        mx = np.max(np.abs(msg))
        if mx > 0:
            log_scale += np.log(mx)
            msg /= mx
        else:
            return 1e10

    p_z1 = np.exp(-0.5 * (z / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    result = np.sum(fi_grid[0] * p_z1 * msg * trap_w)

    if result <= 0:
        return 1e10

    return -(np.log(result) + log_scale)


def _tm_forward_Jk(theta, mu, nu, u, copula, K, grid_range):
    """
    Forward pass: E[Psi(x_k) | u_{1:k-1}] (predictive).
    J_k uses data BEFORE time k (not including u_k).
    Returns (n,).
    """
    n = len(u)
    dt = 1.0 / (n - 1)

    rho = np.exp(-theta * dt)
    sigma = np.sqrt(0.5 * nu ** 2 / theta)
    sigma_cond = sigma * np.sqrt(1.0 - rho ** 2)

    z = np.linspace(-grid_range * sigma, grid_range * sigma, K)
    dz = z[1] - z[0]

    trap_w = np.full(K, dz)
    trap_w[0] *= 0.5
    trap_w[-1] *= 0.5

    means = rho * z
    diff = z[np.newaxis, :] - means[:, np.newaxis]
    T_mat = np.exp(-0.5 * (diff / sigma_cond) ** 2) / (sigma_cond * np.sqrt(2.0 * np.pi))

    fi_grid = np.empty((n, K))
    for i in range(n):
        fi_grid[i] = copula.pdf_on_grid(u[i], z + mu)

    g_grid = copula.transform(z + mu)

    # Prior: p(x_0)
    p_z1 = np.exp(-0.5 * (z / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    alpha = p_z1.copy()

    J = np.empty(n)
    for k in range(n):
        # Predictive: E[Psi(x_k) | u_{1:k-1}] — BEFORE observing u_k
        num = np.sum(g_grid * alpha * trap_w)
        den = np.sum(alpha * trap_w)
        J[k] = num / den if den != 0 else np.nan

        # Update with u_k and propagate: p(x_{k+1} | u_{1:k})
        if k < n - 1:
            source = fi_grid[k] * alpha * trap_w
            alpha = T_mat.T @ source
            mx = np.max(np.abs(alpha))
            if mx > 0:
                alpha /= mx
            else:
                J[k + 1:] = np.nan
                break

    return J


def _tm_forward_rosenblatt(theta, mu, nu, u, copula, K, grid_range):
    """
    Forward pass: mixture Rosenblatt transform for gof test.

    For each k, computes:
        e_{k,1} = u_{k,1}
        e_{k,2} = E[h(u_{k,2}, u_{k,1}, Psi(x_k)) | u_{1:k-1}]
               = integral h(u_{k,2}, u_{k,1}, Psi(z)) * p(z | u_{1:k-1}) dz

    This is the correct predictive conditional CDF, accounting for
    uncertainty in x_k. Using E[Psi(x)] instead introduces Jensen bias.

    Returns (n, 2) — Rosenblatt-transformed pseudo-observations.
    """
    n = len(u)
    dt = 1.0 / (n - 1)

    rho = np.exp(-theta * dt)
    sigma = np.sqrt(0.5 * nu ** 2 / theta)
    sigma_cond = sigma * np.sqrt(1.0 - rho ** 2)

    z = np.linspace(-grid_range * sigma, grid_range * sigma, K)
    dz = z[1] - z[0]

    trap_w = np.full(K, dz)
    trap_w[0] *= 0.5
    trap_w[-1] *= 0.5

    means = rho * z
    diff = z[np.newaxis, :] - means[:, np.newaxis]
    T_mat = np.exp(-0.5 * (diff / sigma_cond) ** 2) / (sigma_cond * np.sqrt(2.0 * np.pi))

    fi_grid = np.empty((n, K))
    for i in range(n):
        fi_grid[i] = copula.pdf_on_grid(u[i], z + mu)

    r_grid = copula.transform(z + mu)  # (K,) — copula param at each grid point

    # Prior
    p_z1 = np.exp(-0.5 * (z / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    alpha = p_z1.copy()

    e = np.empty((n, 2))
    e[:, 0] = u[:, 0]

    for k in range(n):
        # Predictive weights: w_j = alpha_j * trap_w_j / sum(alpha * trap_w)
        raw_w = alpha * trap_w
        total = np.sum(raw_w)
        if total > 0:
            pred_w = raw_w / total
        else:
            pred_w = np.ones(K) / K

        # Mixture h-function: E[h(u2, u1, Psi(z)) | u_{1:k-1}]
        # Evaluate h at each grid point
        u2_vec = np.full(K, u[k, 1])
        u1_vec = np.full(K, u[k, 0])
        h_vals = copula.h(u2_vec, u1_vec, r_grid)  # (K,)

        e[k, 1] = np.sum(h_vals * pred_w)

        # Update and propagate
        if k < n - 1:
            source = fi_grid[k] * alpha * trap_w
            alpha = T_mat.T @ source
            mx = np.max(np.abs(alpha))
            if mx > 0:
                alpha /= mx
            else:
                # Fallback: use mean param
                e[k + 1:, 1] = 0.5
                break

    eps = 1e-6
    e = np.clip(e, eps, 1.0 - eps)
    return e


def _tm_forward_mixture_h(theta, mu, nu, u, copula, K, grid_range):
    """
    Mixture h-function via TM forward pass.

    Returns (n,) array:
        h_k = E[h(u_{k,2}, u_{k,1}, Psi(x_k)) | u_{1:k-1}]

    Same as second column of _tm_forward_rosenblatt, but standalone
    for use in vine Rosenblatt where we need per-edge mixture h-values.
    """
    n = len(u)
    dt = 1.0 / (n - 1)

    rho = np.exp(-theta * dt)
    sigma = np.sqrt(0.5 * nu ** 2 / theta)
    sigma_cond = sigma * np.sqrt(1.0 - rho ** 2)

    z = np.linspace(-grid_range * sigma, grid_range * sigma, K)
    dz = z[1] - z[0]

    trap_w = np.full(K, dz)
    trap_w[0] *= 0.5
    trap_w[-1] *= 0.5

    means = rho * z
    diff = z[np.newaxis, :] - means[:, np.newaxis]
    T_mat = np.exp(-0.5 * (diff / sigma_cond) ** 2) / (sigma_cond * np.sqrt(2.0 * np.pi))

    fi_grid = np.empty((n, K))
    for i in range(n):
        fi_grid[i] = copula.pdf_on_grid(u[i], z + mu)

    r_grid = copula.transform(z + mu)

    p_z1 = np.exp(-0.5 * (z / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    alpha = p_z1.copy()

    h_mix = np.empty(n)

    for k in range(n):
        raw_w = alpha * trap_w
        total = np.sum(raw_w)
        pred_w = raw_w / total if total > 0 else np.ones(K) / K

        u2_vec = np.full(K, u[k, 1])
        u1_vec = np.full(K, u[k, 0])
        h_vals = copula.h(u2_vec, u1_vec, r_grid)
        h_mix[k] = np.sum(h_vals * pred_w)

        if k < n - 1:
            source = fi_grid[k] * alpha * trap_w
            alpha = T_mat.T @ source
            mx = np.max(np.abs(alpha))
            if mx > 0:
                alpha /= mx
            else:
                h_mix[k + 1:] = 0.5
                break

    return np.clip(h_mix, 1e-6, 1.0 - 1e-6)


def _tm_xT_distribution(theta, mu, nu, u, copula, K, grid_range):
    """Forward pass: distribution of x_T on grid. Returns (z, prob)."""
    n = len(u)
    dt = 1.0 / (n - 1)

    rho = np.exp(-theta * dt)
    sigma = np.sqrt(0.5 * nu ** 2 / theta)
    sigma_cond = sigma * np.sqrt(1.0 - rho ** 2)

    z = np.linspace(-grid_range * sigma + mu, grid_range * sigma + mu, K)
    dz = z[1] - z[0]

    trap_w = np.full(K, dz)
    trap_w[0] *= 0.5
    trap_w[-1] *= 0.5

    means_centered = rho * (z - mu) + mu
    diff = z[np.newaxis, :] - means_centered[:, np.newaxis]
    T_mat = np.exp(-0.5 * (diff / sigma_cond) ** 2) / (sigma_cond * np.sqrt(2.0 * np.pi))

    alpha = np.exp(-0.5 * ((z - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    log_scale = 0.0

    for t in range(n):
        fi = copula.pdf_on_grid(u[t], z)
        alpha *= fi

        if t < n - 1:
            alpha = T_mat.T @ (alpha * trap_w)

        mx = np.max(np.abs(alpha))
        if mx > 0:
            log_scale += np.log(mx)
            alpha /= mx

    total = np.sum(alpha * trap_w)
    prob = (alpha * trap_w) / total if total > 0 else np.ones(K) / K

    return z, prob


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
            verbose=False):
        """
        Fit the stochastic copula model.

        Parameters
        ----------
        u : array (T, 2) — pseudo-observations
        method : str — 'SCAR-P-OU', 'SCAR-M-OU', 'SCAR-TM-OU'
        alpha0 : array (3,) or None — initial (theta, mu, nu)
        tol : float — gradient tolerance
        n_tr : int — MC trajectories (P/M samplers)
        M_iterations : int — EIS iterations (M-sampler)
        seed : int or None
        dwt : array (T, n_tr) or None
        stationary : bool
        K : int — grid size (TM)
        grid_range : float
        verbose : bool
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
            mle_result = copula._fit_mle(u)
            mu0 = copula.inv_transform(mle_result.copula_param)
            alpha0 = np.array([1.0, mu0, 1.0])

        bounds = Bounds([0.001, -np.inf, 0.001], [np.inf, np.inf, np.inf])

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
                    return _tm_loglik(th, mu_v, nu_v, u, copula, K, grid_range)
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
        result.n_tr = n_tr if method != 'SCAR-TM-OU' else None
        result.K = K if method == 'SCAR-TM-OU' else None
        self.fit_result = result

        if verbose:
            print(f"  => alpha={result.alpha}, logL={result.log_likelihood:.4f}")

        return result

    def smoothed_params(self, u, alpha=None, K=300, grid_range=5.0):
        """E[Psi(x_k) | u_{1:k-1}] via transfer matrix."""
        if alpha is None:
            if self.fit_result is None:
                raise ValueError("Fit first or provide alpha")
            alpha = self.fit_result.alpha
        theta, mu, nu = alpha
        return _tm_forward_Jk(theta, mu, nu, u, self.copula, K, grid_range)

    def xT_distribution(self, u, alpha=None, K=300, grid_range=5.0):
        """Distribution of x_T on grid. Returns (z_grid, prob)."""
        if alpha is None:
            if self.fit_result is None:
                raise ValueError("Fit first or provide alpha")
            alpha = self.fit_result.alpha
        theta, mu, nu = alpha
        return _tm_xT_distribution(theta, mu, nu, u, self.copula, K, grid_range)

    def final_state_sample(self, alpha, n, method='SCAR-TM-OU'):
        """Sample x_T from stationary distribution."""
        theta, mu, nu = alpha
        if method.upper() == 'SCAR-TM-OU':
            sigma2 = nu ** 2 / (2.0 * theta)
        else:
            sigma2 = nu ** 2 / (2.0 * theta) * (1.0 - np.exp(-2.0 * theta))
        return np.random.normal(mu, np.sqrt(sigma2), n)
