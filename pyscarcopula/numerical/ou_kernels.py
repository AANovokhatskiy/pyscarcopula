"""
pyscarcopula.numerical.ou_kernels - Numba kernels for the OU process.

Pure numerical functions with no copula dependency.

Contents:
  - ou_init_state: deterministic initial x0 = mu
  - ou_stationary_state_from_dwt: deterministic stationary x0 from dwt
  - ou_sample_paths_exact: exact OU discretization (for p-sampler)
  - ou_sample_paths: EIS-modified OU paths (for m-sampler)
  - log_norm_ou: EIS normalizing factor g2
  - log_mean_exp: numerically stable log(mean(exp(...)))
  - calculate_dwt: generate Wiener increments
"""

import numpy as np
from numba import njit


@njit(cache=True)
def ou_init_state(mu, n_tr):
    """Deterministic initial state x_0 = mu for all trajectories."""
    return np.full(n_tr, mu)


@njit(cache=True)
def ou_stationary_state_from_dwt(kappa, mu, nu, dwt):
    """Deterministic stationary x0 using the reserved final dwt row."""
    T, n_tr = dwt.shape
    dt = 1.0 / (T - 1)
    sigma2 = nu ** 2 / (2.0 * kappa)
    z0 = dwt[T - 1] / np.sqrt(dt)
    return mu + np.sqrt(sigma2) * z0


@njit(cache=True)
def ou_sample_paths_exact(kappa, mu, nu, dwt, x0):
    """
    Exact OU discretization (for p-sampler, no EIS).

    x_{i+1} = mu + rho*(x_i - mu) + sigma_c * eps_i
    where rho = exp(-kappa*dt), sigma_c^2 = nu^2/(2*kappa) * (1 - rho^2).

    Parameters
    ----------
    kappa, mu, nu : float
        OU parameters.
    dwt : (T, n_tr)
        Wiener increments ~ N(0, dt).
    x0 : (n_tr,)
        Initial state.

    Returns
    -------
    xt : (T, n_tr)
    """
    T, n_tr = dwt.shape
    dt = 1.0 / (T - 1)
    rho = np.exp(-kappa * dt)
    sigma2_cond = nu ** 2 / (2.0 * kappa) * (1.0 - rho ** 2)
    sigma_cond = np.sqrt(sigma2_cond)
    scale = sigma_cond / np.sqrt(dt)

    xt = np.empty((T, n_tr))
    xt[0] = x0
    for i in range(1, T):
        xt[i] = mu + rho * (xt[i - 1] - mu) + scale * dwt[i - 1]
    return xt


@njit(cache=True)
def ou_sample_paths(kappa, mu, nu, a1t, a2t, dwt, x0):
    """
    Generate OU trajectories modified by EIS auxiliary params a1t, a2t.

    This is the full path generation used by the m-sampler (SCAR-M-OU).
    When a1t = a2t = 0, it reduces to the standard OU discretization.

    Parameters
    ----------
    kappa, mu, nu : float
        OU parameters.
    a1t, a2t : (T,)
        EIS auxiliary parameters (0 for p-sampler).
    dwt : (T, n_tr)
        Wiener increments.
    x0 : (n_tr,)
        Initial state.

    Returns
    -------
    xt : (T, n_tr)
    """
    T, n_tr = dwt.shape
    dt = 1.0 / (T - 1)

    zero_aux = True
    for i in range(T):
        if a1t[i] != 0.0 or a2t[i] != 0.0:
            zero_aux = False
            break
    if zero_aux:
        return ou_sample_paths_exact(kappa, mu, nu, dwt, x0)

    D = nu ** 2 / 2.0
    xt = np.empty((T, n_tr))
    xt[0] = x0

    Mx0 = np.mean(x0)
    Dx0 = np.var(x0)

    Ito_sum = np.zeros(n_tr)

    for i in range(1, T):
        t = i * dt
        a1, a2 = a1t[i], a2t[i]

        sigma2 = D / kappa * (1.0 - np.exp(-2.0 * kappa * t)) + Dx0 * np.exp(-2.0 * kappa * t)
        p = 1.0 - 2.0 * a2 * sigma2

        if i == 1:
            pm1 = 1.0
        else:
            tm1 = t - dt
            sigma2m1 = D / kappa * (1.0 - np.exp(-2.0 * kappa * tm1)) + Dx0 * np.exp(-2.0 * kappa * tm1)
            pm1 = 1.0 - 2.0 * a2t[i - 1] * sigma2m1

        xs = (Mx0 - mu) * np.exp(-kappa * t) + mu
        xsw = (xs + a1 * sigma2) / p
        st = np.exp(-kappa * t) / np.sqrt(p)
        det_part = xsw + st * x0 - st * (Mx0 + a1t[0] * Dx0) / np.sqrt(1.0 - 2.0 * a2t[0] * Dx0)

        Ito_sum = (Ito_sum * np.sqrt(pm1 / p) + nu / np.sqrt(p) * dwt[i - 1]) * np.exp(-kappa * dt)
        xt[i] = det_part + Ito_sum

    return xt


@njit(cache=True)
def log_norm_ou(kappa, mu, nu, a1, a2, dt, x0):
    """Log normalizing factor g2 for EIS auxiliary distribution."""
    D = nu ** 2 / 2.0
    sigma2 = D / kappa * (1.0 - np.exp(-2.0 * kappa * dt))
    xs = (x0 - mu) * np.exp(-kappa * dt) + mu
    res = (a1 ** 2 * sigma2 + 2.0 * a1 * xs + 2.0 * a2 * xs ** 2) / \
          (2.0 - 4.0 * a2 * sigma2) - 0.5 * np.log(1.0 - 2.0 * a2 * sigma2)
    return res


@njit(cache=True)
def log_mean_exp(log_vals):
    """Numerically stable log(mean(exp(log_vals)))."""
    xc = np.max(log_vals)
    return np.log(np.mean(np.exp(log_vals - xc))) + xc


def calculate_dwt(T, n_tr, seed=None):
    """Generate Wiener increments (T, n_tr).

    Rows ``0..T-2`` drive the OU path increments. Row ``T-1`` is reserved
    as an independent common-random-number source for stationary x0.

    Parameters
    ----------
    T : int
        Number of time steps.
    n_tr : int
        Number of trajectories.
    seed : int or None

    Returns
    -------
    dwt : (T, n_tr), each entry ~ N(0, dt) where dt = 1/(T-1).
    """
    rng = np.random.RandomState(seed)
    dt = 1.0 / (T - 1)
    return rng.normal(0, 1, size=(T, n_tr)) * np.sqrt(dt)
