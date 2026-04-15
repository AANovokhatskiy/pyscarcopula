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
