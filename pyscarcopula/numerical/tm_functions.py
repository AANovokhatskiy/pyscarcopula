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
from pyscarcopula.numerical.gof_blocks import _forward_block_size


def _normalize_transition_method(transition_method):
    method = str(transition_method).lower()
    if method not in {'auto', 'matrix', 'gh', 'spectral'}:
        raise ValueError(
            "transition_method must be one of "
            "'auto', 'matrix', 'gh', or 'spectral'")
    return method


def _grid_transition_method(transition_method):
    method = _normalize_transition_method(transition_method)
    if method == 'spectral':
        return 'auto'
    return method


def _h_on_grid(copula, u2, u1, r_grid):
    if isinstance(copula, BivariateGaussianCopula):
        u2_grid = np.full(len(r_grid), u2, dtype=np.float64)
        u1_grid = np.full(len(r_grid), u1, dtype=np.float64)
        return _gauss_h_numba(u2_grid, u1_grid, r_grid)
    u2_grid = np.full(len(r_grid), u2)
    u1_grid = np.full(len(r_grid), u1)
    return copula.h(u2_grid, u1_grid, r_grid)


def _h_block_on_grid(copula, u_block, r_grid):
    n_block = len(u_block)
    K = len(r_grid)
    if isinstance(copula, BivariateGaussianCopula):
        u2_grid = np.repeat(u_block[:, 1], K).astype(np.float64)
        u1_grid = np.repeat(u_block[:, 0], K).astype(np.float64)
        r_eval = np.tile(r_grid, n_block).astype(np.float64)
        return _gauss_h_numba(u2_grid, u1_grid, r_eval).reshape(n_block, K)
    u2_grid = np.repeat(u_block[:, 1], K)
    u1_grid = np.repeat(u_block[:, 0], K)
    r_eval = np.tile(r_grid, n_block)
    return copula.h(u2_grid, u1_grid, r_eval).reshape(n_block, K)


def _grid_prob_from_phi(grid, phi):
    if phi is None:
        return np.full(grid.K, 1.0 / grid.K, dtype=np.float64)
    prob = phi * grid.trap_w
    total = np.sum(prob)
    if total > 0:
        return prob / total
    return np.full(grid.K, 1.0 / grid.K, dtype=np.float64)


def tm_loglik(kappa, mu, nu, u, copula, K=300, grid_range=5.0,
              grid_method='auto', adaptive=True, pts_per_sigma=4,
              transition_method='matrix', max_K=None, r_gh=3.0,
              gh_order=5):
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
        K, grid_range, grid_method, adaptive, pts_per_sigma,
        transition_method, max_K, r_gh, gh_order
        Grid parameters (all preserved from original).

    Returns
    -------
    float : minus log-likelihood (1e10 on failure).
    """
    if kappa <= 0 or nu <= 0:
        return 1e10

    transition_method = _normalize_transition_method(transition_method)
    if transition_method in {'auto', 'spectral'}:
        from pyscarcopula.numerical.auto_tm import (
            AutoTMConfig,
            auto_neg_loglik,
        )
        return auto_neg_loglik(
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

    n = len(u)
    if n < 2:
        return 1e10

    sigma = np.sqrt(0.5 * nu ** 2 / kappa)
    sigma_cond = sigma * np.sqrt(1.0 - np.exp(-2.0 * kappa / (n - 1)))
    if sigma <= 0 or sigma_cond <= 0:
        return 1e10

    try:
        grid = TMGrid(kappa, mu, nu, n, K, grid_range,
                      grid_method, adaptive, pts_per_sigma,
                      transition_method=transition_method,
                      max_K=max_K, r_gh=r_gh, gh_order=gh_order)
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
                    grid_method='auto', adaptive=True, pts_per_sigma=4,
                    transition_method='matrix', max_K=None, r_gh=3.0,
                    gh_order=5):
    """
    Test-only forward-pass log-likelihood reference implementation.

    Production SCAR-TM fitting uses :func:`tm_loglik`, which evaluates the
    same likelihood by the backward recursion.  This function intentionally
    lives next to the production TM code so tests that compare forward and
    backward likelihoods stay visible when the main implementation changes.
    """
    transition_method = _grid_transition_method(transition_method)
    grid = TMGrid(kappa, mu, nu, len(u), K, grid_range,
                  grid_method, adaptive, pts_per_sigma,
                  transition_method=transition_method, max_K=max_K,
                  r_gh=r_gh, gh_order=gh_order)
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
                               adaptive=True, pts_per_sigma=4,
                               transition_method='matrix', max_K=None,
                               r_gh=3.0, gh_order=5):
    """
    Forward pass: E[Psi(x_k) | u_{1:k-1}].

    Equation (24) in the paper:
    r_bar_k = sum_j Psi(z_j + mu) * p_hat_k(z_j) * w_j

    Uses data BEFORE time k (not including u_k).

    Returns (n,) array.
    """
    n = len(u)
    transition_method = _grid_transition_method(transition_method)
    grid = TMGrid(kappa, mu, nu, n, K, grid_range,
                  grid_method, adaptive, pts_per_sigma,
                  transition_method=transition_method, max_K=max_K,
                  r_gh=r_gh, gh_order=gh_order)
    g_grid = copula.transform(grid.z + grid.mu)

    J = np.empty(n)
    phi = grid.p0.copy()
    block_size = _forward_block_size(grid.K)
    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        fi_block = grid.copula_grid(u[start:stop], copula)
        for local, k in enumerate(range(start, stop)):
            weights = grid.predictive_weights_from_phi(phi)
            J[k] = np.sum(weights * g_grid)
            if k < n - 1:
                phi = grid.advance_forward_phi(phi, fi_block[local])

    return J


def tm_forward_smoothed(kappa, mu, nu, u, copula, K=300, grid_range=5.0,
                        grid_method='auto', adaptive=True, pts_per_sigma=4,
                        transition_method='matrix', max_K=None,
                        r_gh=3.0, gh_order=5):
    """Backward-compatible alias for :func:`tm_forward_predictive_mean`."""
    return tm_forward_predictive_mean(
        kappa, mu, nu, u, copula, K, grid_range, grid_method, adaptive,
        pts_per_sigma, transition_method=transition_method, max_K=max_K,
        r_gh=r_gh, gh_order=gh_order)


def tm_forward_rosenblatt(kappa, mu, nu, u, copula, K=300, grid_range=5.0,
                          grid_method='auto', adaptive=True, pts_per_sigma=4,
                          transition_method='matrix', max_K=None,
                          r_gh=3.0, gh_order=5):
    """
    Forward pass: mixture Rosenblatt transform for GoF test.

    Equation (21)/(25) in the paper:
      e_{k,2} = E[h(u_{k,2}, u_{k,1}, Psi(x_k)) | u_{1:k-1}]
              = sum_j h(u_{k,2}, u_{k,1}; Psi(z_j+mu)) * p_hat_k(z_j) * w_j

    Returns (n, 2) — Rosenblatt-transformed pseudo-observations.
    """
    n = len(u)
    transition_method = _grid_transition_method(transition_method)
    grid = TMGrid(kappa, mu, nu, n, K, grid_range,
                  grid_method, adaptive, pts_per_sigma,
                  transition_method=transition_method, max_K=max_K,
                  r_gh=r_gh, gh_order=gh_order)
    r_grid = copula.transform(grid.z + grid.mu)

    e = np.empty((n, 2))
    e[:, 0] = u[:, 0]

    phi = grid.p0.copy()
    block_size = _forward_block_size(grid.K)
    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        u_block = u[start:stop]
        fi_block = grid.copula_grid(u_block, copula)
        h_block = _h_block_on_grid(copula, u_block, r_grid)
        for local, k in enumerate(range(start, stop)):
            weights = grid.predictive_weights_from_phi(phi)
            e[k, 1] = np.sum(h_block[local] * weights)
            if k < n - 1:
                phi = grid.advance_forward_phi(phi, fi_block[local])

    eps = 1e-6
    return np.clip(e, eps, 1.0 - eps)


def tm_forward_mixture_h(kappa, mu, nu, u, copula, K=300, grid_range=5.0,
                         grid_method='auto', adaptive=True, pts_per_sigma=4,
                         transition_method='matrix', max_K=None,
                         r_gh=3.0, gh_order=5,
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
    transition_method = _grid_transition_method(transition_method)
    grid = TMGrid(kappa, mu, nu, n, K, grid_range,
                  grid_method, adaptive, pts_per_sigma,
                  transition_method=transition_method, max_K=max_K,
                  r_gh=r_gh, gh_order=gh_order)
    r_grid = copula.transform(grid.z + grid.mu)

    h_mix = np.empty(n)
    posterior_phi = None
    phi = grid.p0.copy()
    block_size = _forward_block_size(grid.K)
    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        u_block = u[start:stop]
        fi_block = grid.copula_grid(u_block, copula)
        h_block = _h_block_on_grid(copula, u_block, r_grid)
        for local, k in enumerate(range(start, stop)):
            weights = grid.predictive_weights_from_phi(phi)
            h_mix[k] = np.sum(h_block[local] * weights)
            if phi is None:
                posterior_phi = None
            else:
                posterior_phi = phi * fi_block[local]
            if k < n - 1:
                phi = grid.advance_forward_phi(phi, fi_block[local])

    if state_cache is not None:
        z_grid = grid.z + grid.mu
        if current_cache_key is not None:
            state_cache[current_cache_key] = (
                z_grid, _grid_prob_from_phi(grid, posterior_phi))

        if next_cache_key is not None:
            if posterior_phi is None:
                next_prob = np.full(grid.K, 1.0 / grid.K, dtype=np.float64)
            else:
                phi_next = grid.predict_matvec(posterior_phi * grid.trap_w)
                mx = np.max(np.abs(phi_next))
                if mx > 0:
                    phi_next /= mx
                    next_prob = _grid_prob_from_phi(grid, phi_next)
                else:
                    next_prob = np.full(grid.K, 1.0 / grid.K, dtype=np.float64)
            state_cache[next_cache_key] = (z_grid, next_prob)

    return np.clip(h_mix, 1e-6, 1.0 - 1e-6)


def tm_xT_distribution(kappa, mu, nu, u, copula, K=300, grid_range=5.0,
                       grid_method='auto', adaptive=True, pts_per_sigma=4,
                       transition_method='matrix', max_K=None,
                       r_gh=3.0, gh_order=5):
    """
    Forward pass: distribution of x_T on grid.

    Accumulates all observations (including the last one) into the
    density before returning. Used for VaR/CVaR scenario generation.

    Returns (z_grid, prob) where z_grid includes the mu offset,
    so z_grid[j] = z_j + mu = actual x values.
    """
    transition_method = _grid_transition_method(transition_method)
    return tm_state_distribution(
        kappa, mu, nu, u, copula, K, grid_range,
        grid_method, adaptive, pts_per_sigma, transition_method=transition_method,
        max_K=max_K, r_gh=r_gh, gh_order=gh_order,
        horizon='current')
