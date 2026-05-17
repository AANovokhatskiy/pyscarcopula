"""
pyscarcopula.numerical.tm_functions — Transfer matrix computations.

All functions share the TMGrid infrastructure. They differ only in
what they compute from the forward/backward messages:

  - tm_loglik: minus log-likelihood via backward pass
  - tm_forward_predictive_mean: E[Psi(x_k) | u_{1:k-1}]
  - tm_forward_rosenblatt: mixture Rosenblatt transform for GoF
  - tm_forward_mixture_h: mixture h-function for vine pseudo-obs
  - tm_xT_distribution: distribution of x_T on grid (for VaR/CVaR)
  - _forward_loglik: test-only forward likelihood oracle
"""

import numpy as np
from pyscarcopula.copula.elliptical import (
    BivariateGaussianCopula,
    _gauss_h_numba,
)
from pyscarcopula.numerical.tm_grid import TMGrid
from pyscarcopula.numerical.predictive_tm import tm_state_distribution


def tm_loglik(kappa, mu, nu, u, copula, K=300, grid_range=5.0,
              grid_method='auto', adaptive=True, pts_per_sigma=4):
    """
    Transfer matrix backward pass. Returns minus log-likelihood.

    This is the main likelihood function for SCAR-TM-OU.
    Uses the backward message recursion (equations 12-14 in the paper).

    Parameters
    ----------
    kappa, mu, nu : float
        OU process parameters (kappa > 0, nu > 0).
    u : ndarray (n, 2)
        Pseudo-observations.
    copula : CopulaProtocol
    K, grid_range, grid_method, adaptive, pts_per_sigma
        Grid parameters (all preserved from original).

    Returns
    -------
    float : minus log-likelihood (1e10 on failure).
    """
    if kappa <= 0 or nu <= 0:
        return 1e10

    n = len(u)
    if n < 2:
        return 1e10

    sigma = np.sqrt(0.5 * nu ** 2 / kappa)
    sigma_cond = sigma * np.sqrt(1.0 - np.exp(-2.0 * kappa / (n - 1)))
    if sigma <= 0 or sigma_cond <= 0:
        return 1e10

    try:
        grid = TMGrid(kappa, mu, nu, n, K, grid_range,
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


def _forward_loglik(kappa, mu, nu, u, copula, K=300, grid_range=5.0,
                    grid_method='auto', adaptive=True, pts_per_sigma=4):
    """
    Test-only forward-pass log-likelihood reference implementation.

    Production SCAR-TM fitting uses :func:`tm_loglik`, which evaluates the
    same likelihood by the backward recursion.  This function intentionally
    lives next to the production TM code so tests that compare forward and
    backward likelihoods stay visible when the main implementation changes.
    """
    grid = TMGrid(kappa, mu, nu, len(u), K, grid_range,
                  grid_method, adaptive, pts_per_sigma)
    fi_grid = grid.copula_grid(u, copula)

    phi = grid.p0.copy()
    log_likelihood = 0.0

    for k in range(len(u)):
        post = fi_grid[k] * phi
        scale = np.sum(post * grid.trap_w)
        if scale <= 0.0:
            return -np.inf

        log_likelihood += np.log(scale)
        post /= scale

        if k < len(u) - 1:
            phi = grid.predict_matvec(post * grid.trap_w)

    return log_likelihood


def tm_forward_predictive_mean(kappa, mu, nu, u, copula, K=300,
                               grid_range=5.0, grid_method='auto',
                               adaptive=True, pts_per_sigma=4):
    """
    Forward pass: E[Psi(x_k) | u_{1:k-1}].

    Equation (24) in the paper:
    r_bar_k = sum_j Psi(z_j + mu) * p_hat_k(z_j) * w_j

    Uses data BEFORE time k (not including u_k).

    Returns (n,) array.
    """
    n = len(u)
    grid = TMGrid(kappa, mu, nu, n, K, grid_range,
                  grid_method, adaptive, pts_per_sigma)
    fi_grid = grid.copula_grid(u, copula)
    g_grid = copula.transform(grid.z + grid.mu)

    weights = grid.forward_weights(fi_grid)
    J = np.sum(weights * g_grid[np.newaxis, :], axis=1)
    return J


def tm_forward_smoothed(kappa, mu, nu, u, copula, K=300, grid_range=5.0,
                        grid_method='auto', adaptive=True, pts_per_sigma=4):
    """Backward-compatible alias for :func:`tm_forward_predictive_mean`."""
    return tm_forward_predictive_mean(
        kappa, mu, nu, u, copula, K, grid_range, grid_method, adaptive,
        pts_per_sigma)


def tm_forward_rosenblatt(kappa, mu, nu, u, copula, K=300, grid_range=5.0,
                          grid_method='auto', adaptive=True, pts_per_sigma=4):
    """
    Forward pass: mixture Rosenblatt transform for GoF test.

    Equation (21)/(25) in the paper:
      e_{k,2} = E[h(u_{k,2}, u_{k,1}, Psi(x_k)) | u_{1:k-1}]
              = sum_j h(u_{k,2}, u_{k,1}; Psi(z_j+mu)) * p_hat_k(z_j) * w_j

    Returns (n, 2) — Rosenblatt-transformed pseudo-observations.
    """
    n = len(u)
    grid = TMGrid(kappa, mu, nu, n, K, grid_range,
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


def tm_forward_mixture_h(kappa, mu, nu, u, copula, K=300, grid_range=5.0,
                         grid_method='auto', adaptive=True, pts_per_sigma=4,
                         state_cache=None, current_cache_key=None,
                         next_cache_key=None):
    """
    Mixture h-function via TM forward pass.

    Same computation as second column of tm_forward_rosenblatt,
    but standalone for vine pseudo-observation propagation (equation 25).

    Returns (n,) array:
        h_k = E[h(u_{k,2}, u_{k,1}, Psi(x_k)) | u_{1:k-1}]
    """
    n = len(u)
    grid = TMGrid(kappa, mu, nu, n, K, grid_range,
                  grid_method, adaptive, pts_per_sigma)
    fi_grid = grid.copula_grid(u, copula)
    r_grid = copula.transform(grid.z + grid.mu)

    if state_cache is None or (
            current_cache_key is None and next_cache_key is None):
        weights = grid.forward_weights(fi_grid)
    else:
        weights = np.zeros((n, grid.K))
        phi = grid.p0.copy()
        for k in range(n):
            raw_w = phi * grid.trap_w
            total = np.sum(raw_w)
            if total > 0:
                weights[k] = raw_w / total
            else:
                weights[k] = 1.0 / grid.K

            phi *= fi_grid[k]
            if k < n - 1:
                phi = grid.predict_matvec(phi * grid.trap_w)
                mx = np.max(np.abs(phi))
                if mx > 0:
                    phi /= mx

        if current_cache_key is not None:
            z_grid = grid.z + grid.mu
            prob = phi * grid.trap_w
            total = np.sum(prob)
            if total > 0:
                prob = prob / total
            else:
                prob = np.full(grid.K, 1.0 / grid.K, dtype=np.float64)
            state_cache[current_cache_key] = (z_grid, prob)

        if next_cache_key is not None:
            phi_next = grid.predict_matvec(phi * grid.trap_w)
            mx = np.max(np.abs(phi_next))
            if mx > 0:
                phi_next /= mx
            z_grid = grid.z + grid.mu
            prob = phi_next * grid.trap_w
            total = np.sum(prob)
            if total > 0:
                prob = prob / total
            else:
                prob = np.full(grid.K, 1.0 / grid.K, dtype=np.float64)
            state_cache[next_cache_key] = (z_grid, prob)

    u2_grid = np.repeat(u[:, 1], grid.K)
    u1_grid = np.repeat(u[:, 0], grid.K)
    r_eval = np.tile(r_grid, n)
    if isinstance(copula, BivariateGaussianCopula):
        h_vals = _gauss_h_numba(u2_grid, u1_grid, r_eval)
    else:
        h_vals = copula.h(u2_grid, u1_grid, r_eval)
    h_vals = h_vals.reshape(n, grid.K)
    h_mix = np.sum(h_vals * weights, axis=1)

    return np.clip(h_mix, 1e-6, 1.0 - 1e-6)


def tm_xT_distribution(kappa, mu, nu, u, copula, K=300, grid_range=5.0,
                       grid_method='auto', adaptive=True, pts_per_sigma=4):
    """
    Forward pass: distribution of x_T on grid.

    Accumulates all observations (including the last one) into the
    density before returning. Used for VaR/CVaR scenario generation.

    Returns (z_grid, prob) where z_grid includes the mu offset,
    so z_grid[j] = z_j + mu = actual x values.
    """
    return tm_state_distribution(
        kappa, mu, nu, u, copula, K, grid_range,
        grid_method, adaptive, pts_per_sigma, horizon='current')
