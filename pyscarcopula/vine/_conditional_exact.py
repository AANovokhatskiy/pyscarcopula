"""Recursive exact-posterior helpers for R-vine conditional sampling."""

from __future__ import annotations

import numpy as np


def merge_pseudo(pseudo, updates):
    out = pseudo.copy()
    out.update(updates)
    return out


def quadrature_integral(func, a, b, nodes, weights):
    if b <= a:
        return 0.0
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    xs = mid + half * nodes
    vals = np.array([func(float(x)) for x in xs], dtype=np.float64)
    return float(half * np.sum(weights * vals))


def sample_from_tabulated_weight(weight_fn, rng, grid_size):
    """Sample from a 1D density known up to scale via tabulated inverse CDF."""
    edges = np.linspace(0.0, 1.0, int(grid_size) + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    weights = np.array([weight_fn(float(w)) for w in centers], dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)

    cdf = np.empty_like(edges)
    cdf[0] = 0.0
    dx = np.diff(edges)
    cdf[1:] = np.cumsum(dx * weights)

    z_total = float(cdf[-1])
    if not np.isfinite(z_total) or z_total <= 0.0:
        return float(rng.uniform(0.0, 1.0))

    target = float(rng.uniform(0.0, z_total))
    return float(np.interp(target, cdf, edges))


def has_future_given(order_vars, idx, given):
    return any(var in given for var in order_vars[idx:])


def posterior_indices(order_vars, given):
    """Indices whose latent w must account for future fixed variables."""
    given_positions = [idx for idx, var in enumerate(order_vars) if var in given]
    if not given_positions:
        return []
    last_given = max(given_positions)
    return [
        idx for idx in range(last_given + 1)
        if order_vars[idx] not in given
    ]


def future_likelihood(vine, M, edge_map, order_cols, order_vars, idx,
                      pseudo, given, r_all, nodes, weights, sample_idx=0,
                      column_ops=None, column_from_x=None,
                      column_from_w=None):
    d = vine.d
    if idx >= d:
        return 1.0

    if not has_future_given(order_vars, idx, given):
        return 1.0

    s = order_cols[idx]
    var = order_vars[idx]

    if var in given:
        updates = column_from_x(
            vine, pseudo, M, edge_map, s, given[var], r_all, sample_idx,
            column_ops=column_ops)
        density = updates.pop('_density')
        return density * future_likelihood(
            vine, M, edge_map, order_cols, order_vars, idx + 1,
            merge_pseudo(pseudo, updates), given, r_all, nodes, weights,
            sample_idx, column_ops=column_ops,
            column_from_x=column_from_x, column_from_w=column_from_w)

    def _integrand(w):
        updates = column_from_w(
            vine, pseudo, M, edge_map, s, w, r_all, sample_idx,
            column_ops=column_ops)
        updates.pop('_density')
        return future_likelihood(
            vine, M, edge_map, order_cols, order_vars, idx + 1,
            merge_pseudo(pseudo, updates), given, r_all, nodes, weights,
            sample_idx, column_ops=column_ops,
            column_from_x=column_from_x, column_from_w=column_from_w)

    return quadrature_integral(_integrand, 0.0, 1.0, nodes, weights)


def sample_w_posterior(vine, M, edge_map, order_cols, order_vars, idx,
                       pseudo, given, r_all, rng, nodes, weights,
                       sample_idx=0, column_ops=None, column_from_x=None,
                       column_from_w=None):
    s = order_cols[idx]

    def weight_fn(w):
        updates = column_from_w(
            vine, pseudo, M, edge_map, s, w, r_all, sample_idx,
            column_ops=column_ops)
        updates.pop('_density')
        return future_likelihood(
            vine, M, edge_map, order_cols, order_vars, idx + 1,
            merge_pseudo(pseudo, updates), given, r_all, nodes, weights,
            sample_idx, column_ops=column_ops,
            column_from_x=column_from_x, column_from_w=column_from_w)

    grid_size = max(4 * len(nodes) + 1, 33)
    return sample_from_tabulated_weight(weight_fn, rng, grid_size)
