"""Exact conditional predict backend for R-vines."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

from pyscarcopula.vine._helpers import _clip_unit


def validate_rvine_given(given, d):
    """Validate `given` for R-vine conditional predict."""
    if given is None:
        return {}

    if not isinstance(given, dict):
        raise TypeError("given must be a dict[int, float] or None")

    out = {}
    for key, value in given.items():
        try:
            idx = int(key)
        except Exception as exc:
            raise TypeError("given keys must be integers") from exc
        if idx < 0 or idx >= d:
            raise ValueError(f"given key must be in [0, {d - 1}], got {key!r}")
        val = float(value)
        if not (0.0 < val < 1.0):
            raise ValueError(
                f"given[{idx}] must be in pseudo-observation space (0, 1), got {val}"
            )
        out[idx] = val
    return out


def ensure_rvine_conditional_supported(vine):
    """Reject conditional predict for unsupported methods."""
    for edge in vine.edges.values():
        method = edge.method.upper() if edge.method is not None else ''
        if method not in ('MLE', 'GAS', 'SCAR-TM-OU'):
            raise NotImplementedError(
                "Exact conditional predict for RVine is currently "
                "implemented only for MLE, GAS and SCAR-TM-OU edges"
            )


def _edge_r_scalar(r_all, edge_key, sample_idx=0):
    r = np.asarray(r_all[edge_key], dtype=np.float64).ravel()
    if r.size == 0:
        raise ValueError("Empty edge parameter array")
    if sample_idx < 0 or sample_idx >= r.size:
        sample_idx = 0
    return float(r[sample_idx])


def _merge_pseudo(pseudo, updates):
    out = pseudo.copy()
    out.update(updates)
    return out


def _column_from_x(vine, pseudo, M, edge_map, s, x_val, r_all, sample_idx=0):
    """Construct pseudo updates for column s from a fixed base value."""
    d = vine.d
    var = M[s, s]
    updates = {}
    base = float(_clip_unit(np.asarray([x_val], dtype=np.float64))[0])
    updates[(var, frozenset())] = base

    if s == d - 1:
        updates['_density'] = 1.0
        return updates

    n_levels = d - s - 1
    density = 1.0

    for m in range(n_levels):
        edge_key = edge_map[(m, s)]
        edge = vine.edges[edge_key]
        r = _edge_r_scalar(r_all, edge_key, sample_idx)

        partner_var = M[s + m + 1, s]
        cond_set = frozenset(M[s + 1:s + m + 1, s])
        next_cond = frozenset(M[s + 1:s + m + 2, s])

        partner_val = pseudo.get((partner_var, cond_set))
        if partner_val is None:
            raise RuntimeError(
                "Missing partner pseudo-observation during RVine conditional "
                f"construction: var={partner_var}, cond={sorted(cond_set)}"
            )

        var_at_cond = updates.get((var, cond_set), updates[(var, frozenset())])
        density *= max(float(edge.copula.pdf(
            np.array([var_at_cond]),
            np.array([partner_val]),
            np.array([r]))[0]), 1e-300)

        cur = float(_clip_unit(np.atleast_1d(edge.copula.h(
            np.array([var_at_cond]),
            np.array([partner_val]),
            np.array([r])))[0]))
        updates[(var, next_cond)] = cur

        rev = float(_clip_unit(np.atleast_1d(edge.copula.h(
            np.array([partner_val]),
            np.array([var_at_cond]),
            np.array([r])))[0]))
        updates[(partner_var, cond_set | {var})] = rev

    updates['_density'] = density
    return updates


def _column_from_w(vine, pseudo, M, edge_map, s, w_val, r_all, sample_idx=0):
    """Construct pseudo updates for column s from latent Rosenblatt w."""
    d = vine.d
    if s == d - 1:
        return _column_from_x(vine, pseudo, M, edge_map, s, w_val, r_all, sample_idx)

    n_levels = d - s - 1
    val = float(w_val)

    for m in range(n_levels - 1, -1, -1):
        edge_key = edge_map[(m, s)]
        edge = vine.edges[edge_key]
        r = _edge_r_scalar(r_all, edge_key, sample_idx)

        partner_var = M[s + m + 1, s]
        cond_set = frozenset(M[s + 1:s + m + 1, s])

        partner_val = pseudo.get((partner_var, cond_set))
        if partner_val is None:
            raise RuntimeError(
                "Missing partner pseudo-observation during RVine conditional "
                f"inversion: var={partner_var}, cond={sorted(cond_set)}"
            )

        val = float(_clip_unit(np.atleast_1d(edge.copula.h_inverse(
            np.array([val]),
            np.array([partner_val]),
            np.array([r])))[0]))

    return _column_from_x(vine, pseudo, M, edge_map, s, val, r_all, sample_idx)


def _quadrature_integral(func, a, b, nodes, weights):
    if b <= a:
        return 0.0
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    xs = mid + half * nodes
    vals = np.array([func(float(x)) for x in xs], dtype=np.float64)
    return float(half * np.sum(weights * vals))


def _has_future_given(order_vars, idx, given):
    return any(var in given for var in order_vars[idx:])


def _future_likelihood(vine, M, edge_map, order_cols, order_vars, idx,
                       pseudo, given, r_all, nodes, weights, sample_idx=0):
    d = vine.d
    if idx >= d:
        return 1.0

    if not _has_future_given(order_vars, idx, given):
        return 1.0

    s = order_cols[idx]
    var = order_vars[idx]

    if var in given:
        updates = _column_from_x(
            vine, pseudo, M, edge_map, s, given[var], r_all, sample_idx)
        density = updates.pop('_density')
        return density * _future_likelihood(
            vine, M, edge_map, order_cols, order_vars, idx + 1,
            _merge_pseudo(pseudo, updates), given, r_all, nodes, weights, sample_idx)

    def _integrand(w):
        updates = _column_from_w(
            vine, pseudo, M, edge_map, s, w, r_all, sample_idx)
        updates.pop('_density')
        return _future_likelihood(
            vine, M, edge_map, order_cols, order_vars, idx + 1,
            _merge_pseudo(pseudo, updates), given, r_all, nodes, weights, sample_idx)

    return _quadrature_integral(_integrand, 0.0, 1.0, nodes, weights)


def _sample_w_posterior(vine, M, edge_map, order_cols, order_vars, idx,
                        pseudo, given, r_all, rng, nodes, weights,
                        sample_idx=0, tol=1e-6, maxiter=40):
    s = order_cols[idx]

    def weight_fn(w):
        updates = _column_from_w(vine, pseudo, M, edge_map, s, w, r_all, sample_idx)
        updates.pop('_density')
        return _future_likelihood(
            vine, M, edge_map, order_cols, order_vars, idx + 1,
            _merge_pseudo(pseudo, updates), given, r_all, nodes, weights, sample_idx)

    z_total = _quadrature_integral(weight_fn, 0.0, 1.0, nodes, weights)
    if not np.isfinite(z_total) or z_total <= 0.0:
        return float(rng.uniform(0.0, 1.0))

    target = float(rng.uniform(0.0, z_total))
    lo = 0.0
    hi = 1.0

    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        mass = _quadrature_integral(weight_fn, 0.0, mid, nodes, weights)
        if mass < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break

    return 0.5 * (lo + hi)


def sample_rvine_conditional_with_r(vine, n, r_all, given, rng, quad_order=10):
    """Sample from a fitted R-vine conditional on arbitrary given variables."""
    d = vine.d
    M = vine._structure.matrix
    edge_map = vine._edge_map
    order_cols = list(range(d - 1, -1, -1))
    order_vars = [M[s, s] for s in order_cols]
    nodes, weights = leggauss(quad_order)

    x = np.zeros((n, d), dtype=np.float64)

    for t in range(n):
        pseudo = {}

        for idx, s in enumerate(order_cols):
            var = order_vars[idx]

            if var in given:
                updates = _column_from_x(
                    vine, pseudo, M, edge_map, s, given[var], r_all, sample_idx=t)
            elif _has_future_given(order_vars, idx + 1, given):
                w_val = _sample_w_posterior(
                    vine, M, edge_map, order_cols, order_vars, idx,
                    pseudo, given, r_all, rng, nodes, weights, sample_idx=t)
                updates = _column_from_w(
                    vine, pseudo, M, edge_map, s, w_val, r_all, sample_idx=t)
            else:
                w_val = float(rng.uniform(0.0, 1.0))
                updates = _column_from_w(
                    vine, pseudo, M, edge_map, s, w_val, r_all, sample_idx=t)

            updates.pop('_density', None)
            pseudo.update(updates)

        for var in range(d):
            key = (var, frozenset())
            if key not in pseudo:
                raise RuntimeError(f"Missing sampled variable {var}")
            x[t, var] = pseudo[key]

    return x
