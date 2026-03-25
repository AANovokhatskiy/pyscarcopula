"""
pyscarcopula.numerical.tm_functions — Transfer matrix computations.

Extracted from latent/ou_process.py (lines 556–734).

All functions share the TMGrid infrastructure. They differ only in
what they compute from the forward/backward messages:

  - tm_loglik: minus log-likelihood via backward pass
  - tm_forward_smoothed: E[Psi(x_k) | u_{1:k-1}]  (smoothed copula param)
  - tm_forward_rosenblatt: mixture Rosenblatt transform for GoF
  - tm_forward_mixture_h: mixture h-function for vine pseudo-obs
  - tm_xT_distribution: distribution of x_T on grid (for VaR/CVaR)
"""

import numpy as np
from pyscarcopula.numerical.tm_grid import TMGrid


def tm_loglik(theta, mu, nu, u, copula, K=300, grid_range=5.0,
              grid_method='auto', adaptive=True, pts_per_sigma=2):
    """
    Transfer matrix backward pass. Returns minus log-likelihood.

    This is the main likelihood function for SCAR-TM-OU.
    Uses the backward message recursion (equations 12-14 in the paper).

    Parameters
    ----------
    theta, mu, nu : float
        OU process parameters (theta > 0, nu > 0).
    u : ndarray (n, 2)
        Pseudo-observations.
    copula : CopulaProtocol
    K, grid_range, grid_method, adaptive, pts_per_sigma
        Grid parameters (all preserved from original).

    Returns
    -------
    float : minus log-likelihood (1e10 on failure).
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
        grid = TMGrid(theta, mu, nu, n, K, grid_range,
                      grid_method, adaptive, pts_per_sigma)
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


def tm_forward_smoothed(theta, mu, nu, u, copula, K=300, grid_range=5.0,
                        grid_method='auto', adaptive=True, pts_per_sigma=2):
    """
    Forward pass: E[Psi(x_k) | u_{1:k-1}] (predictive smoothed parameter).

    Equation (24) in the paper: theta_bar_k = sum_j Psi(z_j + mu) * p_hat_k(z_j) * w_j

    Uses data BEFORE time k (not including u_k).

    Returns (n,) array.
    """
    n = len(u)
    grid = TMGrid(theta, mu, nu, n, K, grid_range,
                  grid_method, adaptive, pts_per_sigma)
    fi_grid = grid.copula_grid(u, copula)
    g_grid = copula.transform(grid.z + grid.mu)

    weights = grid.forward_weights(fi_grid)
    J = np.sum(weights * g_grid[np.newaxis, :], axis=1)
    return J


def tm_forward_rosenblatt(theta, mu, nu, u, copula, K=300, grid_range=5.0,
                          grid_method='auto', adaptive=True, pts_per_sigma=2):
    """
    Forward pass: mixture Rosenblatt transform for GoF test.

    Equation (21)/(25) in the paper:
      e_{k,2} = E[h(u_{k,2}, u_{k,1}, Psi(x_k)) | u_{1:k-1}]
              = sum_j h(u_{k,2}, u_{k,1}; Psi(z_j+mu)) * p_hat_k(z_j) * w_j

    Returns (n, 2) — Rosenblatt-transformed pseudo-observations.
    """
    n = len(u)
    grid = TMGrid(theta, mu, nu, n, K, grid_range,
                  grid_method, adaptive, pts_per_sigma)
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


def tm_forward_mixture_h(theta, mu, nu, u, copula, K=300, grid_range=5.0,
                         grid_method='auto', adaptive=True, pts_per_sigma=2):
    """
    Mixture h-function via TM forward pass.

    Same computation as second column of tm_forward_rosenblatt,
    but standalone for vine pseudo-observation propagation (equation 25).

    Returns (n,) array:
        h_k = E[h(u_{k,2}, u_{k,1}, Psi(x_k)) | u_{1:k-1}]
    """
    n = len(u)
    grid = TMGrid(theta, mu, nu, n, K, grid_range,
                  grid_method, adaptive, pts_per_sigma)
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


def tm_xT_distribution(theta, mu, nu, u, copula, K=300, grid_range=5.0,
                       grid_method='auto', adaptive=True, pts_per_sigma=2):
    """
    Forward pass: distribution of x_T on grid.

    Accumulates all observations (including the last one) into the
    density before returning. Used for VaR/CVaR scenario generation.

    Returns (z_grid, prob) where z_grid includes the mu offset,
    so z_grid[j] = z_j + mu = actual x values.
    """
    n = len(u)
    grid = TMGrid(theta, mu, nu, n, K, grid_range,
                  grid_method, adaptive, pts_per_sigma)
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
