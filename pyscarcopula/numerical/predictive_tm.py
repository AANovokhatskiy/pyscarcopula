"""Predictive state distributions for SCAR-TM."""

from __future__ import annotations

import numpy as np

from pyscarcopula.numerical.tm_grid import TMGrid


def _normalize_prob(alpha, trap_w):
    total = np.sum(alpha * trap_w)
    if total > 0:
        return (alpha * trap_w) / total
    return np.full_like(alpha, 1.0 / len(alpha), dtype=np.float64)


def tm_state_distribution(theta, mu, nu, u, copula, K=300, grid_range=5.0,
                          grid_method='auto', adaptive=True, pts_per_sigma=2,
                          horizon='current'):
    """Distribution of x_T or x_{T+1} on the TM grid."""
    horizon = str(horizon).lower()
    if horizon not in ('current', 'next'):
        raise ValueError("horizon must be 'current' or 'next'")

    n = len(u)
    grid = TMGrid(theta, mu, nu, n, K, grid_range,
                  grid_method, adaptive, pts_per_sigma)
    fi_grid = grid.copula_grid(u, copula)

    alpha = grid.p0.copy()

    for t in range(n):
        alpha *= fi_grid[t]

        if t < n - 1:
            alpha = grid.rmatvec(alpha * grid.trap_w)

        mx = np.max(np.abs(alpha))
        if mx > 0:
            alpha /= mx

    if horizon == 'next':
        alpha = grid.rmatvec(alpha * grid.trap_w)
        mx = np.max(np.abs(alpha))
        if mx > 0:
            alpha /= mx

    z_grid = grid.z + grid.mu
    prob = _normalize_prob(alpha, grid.trap_w)
    return z_grid, prob


def sample_grid_distribution(z_grid, prob, n, rng, mode='grid'):
    """Sample states from a discrete grid distribution.

    ``grid`` returns atoms exactly on ``z_grid``. ``histogram`` treats each
    grid point as the center of a local cell and samples uniformly inside
    that cell, preserving cell masses while removing grid atoms.
    """
    mode = 'grid' if mode is None else str(mode).lower()
    z_grid = np.asarray(z_grid, dtype=np.float64)
    prob = np.asarray(prob, dtype=np.float64)
    idx = rng.choice(len(z_grid), size=n, p=prob)

    if mode == 'grid':
        return z_grid[idx]
    if mode != 'histogram':
        raise ValueError("predictive_r_mode must be 'grid' or 'histogram'")
    if len(z_grid) == 1:
        return np.full(n, z_grid[0], dtype=np.float64)

    mid = 0.5 * (z_grid[:-1] + z_grid[1:])
    left = np.empty_like(z_grid)
    right = np.empty_like(z_grid)
    left[0] = z_grid[0]
    left[1:] = mid
    right[:-1] = mid
    right[-1] = z_grid[-1]
    return rng.uniform(left[idx], right[idx])
