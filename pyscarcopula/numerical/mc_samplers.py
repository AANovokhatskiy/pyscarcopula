"""
pyscarcopula.numerical.mc_samplers — Monte Carlo likelihood estimators.

Extracted from latent/ou_process.py (lines 124–275).

Contents:
  - p_sampler_loglik: MC p-sampler (no importance sampling)
  - m_sampler_loglik: MC m-sampler with EIS auxiliary params
  - eis_find_auxiliary: backward regression for EIS params a1t, a2t
"""

import numpy as np
from scipy.signal import savgol_filter

from pyscarcopula.numerical.ou_kernels import (
    ou_init_state, ou_stationary_state,
    ou_sample_paths, ou_sample_paths_exact,
    log_norm_ou, log_mean_exp,
)
from pyscarcopula._utils import linear_least_squares


# ══════════════════════════════════════════════════════════════════
# P-sampler (SCAR-P-OU) — no importance sampling
# ══════════════════════════════════════════════════════════════════

def p_sampler_loglik(theta, mu, nu, u, dwt, copula, stationary):
    """
    P-sampler log-likelihood (exact OU discretization).

    Returns minus log-likelihood (1e10 on failure).
    """
    T, n_tr = dwt.shape

    if stationary:
        x0 = ou_stationary_state(theta, mu, nu, n_tr)
    else:
        x0 = ou_init_state(mu, n_tr)

    z = np.zeros(T)
    xt = ou_sample_paths(theta, mu, nu, z, z, dwt, x0)

    if np.isnan(np.sum(xt)):
        return 1e10

    copula_log = np.zeros(n_tr)
    for t in range(T):
        r_vals = copula.transform(xt[t])
        u1 = np.full(n_tr, u[t, 0])
        u2 = np.full(n_tr, u[t, 1])
        copula_log += copula.log_pdf(u1, u2, r_vals)

    return -log_mean_exp(copula_log)


# ══════════════════════════════════════════════════════════════════
# M-sampler (SCAR-M-OU) — with EIS
# ══════════════════════════════════════════════════════════════════

def m_sampler_loglik(theta, mu, nu, u, dwt, a1t, a2t, copula, stationary):
    """
    M-sampler log-likelihood with EIS auxiliary params.

    Returns minus log-likelihood (1e10 on failure).
    """
    T, n_tr = dwt.shape
    dt = 1.0 / (T - 1)

    if stationary:
        x0 = ou_stationary_state(theta, mu, nu, n_tr)
    else:
        x0 = ou_init_state(mu, n_tr)

    xt = ou_sample_paths(theta, mu, nu, a1t, a2t, dwt, x0)

    if np.isnan(np.sum(xt)):
        return 1e10

    # Normalizing factors
    norm_log = np.zeros((T, n_tr))
    for i in range(T - 1, 0, -1):
        norm_log[i] = log_norm_ou(theta, mu, nu, a1t[i], a2t[i], dt, xt[i - 1])
    norm_log[0] = log_norm_ou(theta, mu, nu, a1t[0], a2t[0], dt, x0)

    if np.isnan(np.sum(norm_log)):
        return 1e10

    # Copula log-likelihood with IS correction
    log_lik = np.zeros(n_tr)
    for t in range(T):
        r_vals = copula.transform(xt[t])
        u1 = np.full(n_tr, u[t, 0])
        u2 = np.full(n_tr, u[t, 1])
        c_vals = copula.log_pdf(u1, u2, r_vals)
        # Clip xt to avoid overflow in x^2
        xt_clipped = np.clip(xt[t], -500, 500)
        g_vals = a1t[t] * xt_clipped + a2t[t] * xt_clipped ** 2
        log_lik += c_vals + norm_log[t] - g_vals

    if np.isnan(np.sum(log_lik)):
        return 1e10

    return -log_mean_exp(log_lik)


# ══════════════════════════════════════════════════════════════════
# EIS auxiliary parameters
# ══════════════════════════════════════════════════════════════════

def eis_find_auxiliary(alpha, u, M_iterations, dwt, copula, stationary):
    """Find EIS auxiliary params a1t, a2t via backward regression.

    Parameters
    ----------
    alpha : (3,) — [theta, mu, nu]
    u : (T, 2) — pseudo-observations
    M_iterations : int — number of EIS iterations
    dwt : (T, n_tr) — Wiener increments
    copula : CopulaProtocol
    stationary : bool

    Returns
    -------
    a1t, a2t : (T,) arrays
    """
    theta, mu, nu = alpha
    T = len(u)
    n_tr = dwt.shape[1]
    dt = 1.0 / (T - 1)
    t_data = np.linspace(0, 1, T)
    D = nu ** 2 / 2.0

    # Initial variance Dx0 — must match ou_sample_paths
    if stationary:
        Dx0 = D / theta
    else:
        Dx0 = 0.0

    # Precompute upper bounds on a2t for each time step.
    # In ou_sample_paths: p = 1 - 2*a2*sigma2 must be > 0.
    # sigma2(t) = D/theta * (1-exp(-2*theta*t)) + Dx0*exp(-2*theta*t)
    a2_ub = np.zeros(T)
    for k in range(T):
        t = k * dt
        sigma2 = D / theta * (1.0 - np.exp(-2.0 * theta * t)) \
            + Dx0 * np.exp(-2.0 * theta * t)
        if sigma2 > 1e-15:
            # Leave 10% safety margin so p >= 0.1
            a2_ub[k] = 0.9 / (2.0 * sigma2)
        else:
            a2_ub[k] = 1e6

    # EIS damping factor — blend new estimate with old to avoid oscillation
    damping = 0.5

    a1t = np.zeros(T)
    a2t = np.zeros(T)
    best_a1t = np.zeros(T)
    best_a2t = np.zeros(T)
    best_loglik = -np.inf

    for j in range(M_iterations):
        if stationary:
            x0 = ou_stationary_state(theta, mu, nu, n_tr)
        else:
            x0 = ou_init_state(mu, n_tr)

        xt = ou_sample_paths(theta, mu, nu, a1t, a2t, dwt, x0)

        if np.isnan(np.sum(xt)):
            a1t = np.zeros(T)
            a2t = np.zeros(T)
            continue

        a_data = np.zeros((T, 3))
        a_data[-1] = np.array([0.0, np.mean(a1t), min(np.mean(a2t), 0.0)])

        for i in range(T - 1, 0, -1):
            r_vals = copula.transform(xt[i])
            u1 = np.full(n_tr, u[i, 0])
            u2 = np.full(n_tr, u[i, 1])
            copula_log = copula.log_pdf(u1, u2, r_vals)

            norm_log_vals = log_norm_ou(
                theta, mu, nu,
                a_data[i][1], a_data[i][2],
                dt, xt[i - 1])

            A = np.column_stack((np.ones(n_tr), xt[i], xt[i] ** 2))
            b = copula_log + norm_log_vals

            try:
                # Normal equations with small ridge regularization
                # (faster than pinv for 3x3 system, equally stable)
                lr = linear_least_squares(A, b, 1e-6, pseudo_inverse=False)
                if np.isnan(np.sum(lr)):
                    a_data[i - 1] = a_data[i]
                else:
                    # Clamp a2 to upper bound
                    a2_val = min(lr[2], a2_ub[i])
                    a_data[i - 1] = np.array([lr[0], lr[1], a2_val])
            except Exception:
                a_data[i - 1] = a_data[i]

        a1_hat = a_data[:, 1]
        a2_hat = a_data[:, 2]

        # Smooth with Savitzky-Golay filter
        wl = max(T // 10, 5)
        if wl % 2 == 0:
            wl += 1
        d = min(2, wl - 1)
        a1_new = savgol_filter(a1_hat, wl, d)
        a2_new = savgol_filter(a2_hat, wl, d)

        # Enforce a2 upper bounds from the condition 1 - 2*a2*sigma2 > 0
        # (Eq. (condition) in the article). sigma2 here is the unconditional
        # variance from t'=0 used in the SDE solution (m_sampler_solution).
        # a2 can be positive, but must satisfy a2 < 1/(2*sigma2).
        # We keep a safety margin (p >= p_min) because as p -> 0,
        # the volatility B(t) = nu/sqrt(p) -> inf and paths explode.
        for k in range(T):
            a2_new[k] = min(a2_new[k], a2_ub[k])

        # Apply damping: blend new with old to stabilize iterations
        if j > 0:
            a1t = damping * a1_new + (1.0 - damping) * a1t
            a2t = damping * a2_new + (1.0 - damping) * a2t
        else:
            a1t = a1_new
            a2t = a2_new

        # Track best EIS parameters by evaluating current loglik
        try:
            ll_val = -m_sampler_loglik(
                theta, mu, nu, u, dwt, a1t, a2t, copula, stationary)
            if ll_val > best_loglik:
                best_loglik = ll_val
                best_a1t = a1t.copy()
                best_a2t = a2t.copy()
        except Exception:
            pass

    return best_a1t, best_a2t