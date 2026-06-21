"""Suffix-conditioning helpers for natural-order R-vine sampling."""

import numpy as np

from pyscarcopula.vine._conditional_rvine import (
    find_rvine_peel_order_for_given_suffix,
)
from pyscarcopula.vine._rvine_edges import (
    _edge_h_inverse,
    _edge_h_pair_with_r,
)
from pyscarcopula.vine._rvine_matrix_builder import build_rvine_matrix_with_edge_map
from pyscarcopula.vine._helpers import _clip_unit, _open_unit_uniform


def given_suffix_start_col(d, given, matrix):
    """Return the first fixed suffix column, or ``None`` if not a suffix."""
    if not given:
        return d
    peel_order = [
        int(matrix[d - 1 - col, col])
        for col in range(d)
    ]
    k = len(given)
    suffix = set(peel_order[d - k:])
    if set(given) == suffix:
        return d - k
    return None


def suffix_sampling_state(d, trees, matrix, edge_map, pair_copulas,
                          matrix_key, given):
    """Return matrix/runtime state for exact suffix conditioning if possible."""
    start_col = given_suffix_start_col(d, given, matrix)
    if start_col is not None:
        return start_col, matrix, edge_map, pair_copulas

    given_vars = set(given)
    peel_order = find_rvine_peel_order_for_given_suffix(
        trees, d, given_vars)
    if peel_order is None:
        return None

    to_perm = {var: idx for idx, var in enumerate(peel_order)}
    from_perm = {idx: var for var, idx in to_perm.items()}

    relabeled_trees = []
    for level in trees:
        relabeled_level = []
        for conditioned, conditioning in level:
            relabeled_level.append((
                frozenset(to_perm[v] for v in conditioned),
                frozenset(to_perm[v] for v in conditioning),
            ))
        relabeled_trees.append(relabeled_level)

    try:
        perm_matrix, rebuilt_edge_map = build_rvine_matrix_with_edge_map(
            d, relabeled_trees)
    except RuntimeError:
        return None

    rebuilt_matrix = np.zeros_like(perm_matrix)
    for col in range(d):
        for row in range(d - col):
            rebuilt_matrix[row, col] = from_perm[int(perm_matrix[row, col])]

    start_col = given_suffix_start_col(d, given, rebuilt_matrix)
    if start_col is None:
        return None

    pair_by_orig = {
        (t, orig_idx): pair_copulas[matrix_key(t, orig_idx)]
        for t, level in enumerate(trees)
        for orig_idx in range(len(level))
    }
    rebuilt_pair_copulas = {}
    for key, orig_idx in rebuilt_edge_map.items():
        t = key[0]
        assert 0 <= orig_idx < len(trees[t])
        rebuilt_pair_copulas[key] = pair_by_orig[(t, orig_idx)]
    return start_col, rebuilt_matrix, rebuilt_edge_map, rebuilt_pair_copulas


def sample_suffix_given_with_r(d, n, r_all, rng, given, start_col, matrix,
                               pair_copulas):
    """Sample from an exact suffix-conditioned R-vine with fixed r paths."""
    M = matrix
    w = _open_unit_uniform(rng, size=(n, d))
    pseudo_obs = {}

    last_var = int(M[0, d - 1])
    if d - 1 >= start_col:
        pseudo_obs[(last_var, frozenset())] = np.full(
            n, given[last_var], dtype=np.float64)
    else:
        pseudo_obs[(last_var, frozenset())] = w[:, d - 1].copy()

    for col in range(d - 2, start_col - 1, -1):
        leaf = int(M[d - 1 - col, col])
        top_tree = d - 2 - col
        pseudo_obs[(leaf, frozenset())] = np.full(
            n, given[leaf], dtype=np.float64)
        for t in range(top_tree + 1):
            row = d - 2 - col - t
            partner = int(M[row, col])
            conditioning = frozenset(
                int(M[r, col])
                for r in range(row + 1, d - 1 - col)
            )
            next_leaf_cond = conditioning | {partner}
            next_partner_cond = conditioning | {leaf}
            edge = pair_copulas[(t, col)]
            r = r_all[(t, col)]

            leaf_val = pseudo_obs[(leaf, conditioning)]
            partner_val = pseudo_obs[(partner, conditioning)]
            leaf_next, partner_next = _edge_h_pair_with_r(
                edge, leaf_val, partner_val, r)
            pseudo_obs[(leaf, next_leaf_cond)] = _clip_unit(leaf_next)
            pseudo_obs[(partner, next_partner_cond)] = _clip_unit(partner_next)

    for col in range(start_col - 1, -1, -1):
        leaf = int(M[d - 1 - col, col])
        top_tree = d - 2 - col
        current = w[:, col].copy()

        for t in range(top_tree, -1, -1):
            row = d - 2 - col - t
            partner = int(M[row, col])
            conditioning = frozenset(
                int(M[r, col])
                for r in range(row + 1, d - 1 - col)
            )
            partner_val = pseudo_obs[(partner, conditioning)]
            edge = pair_copulas[(t, col)]
            current = _clip_unit(_edge_h_inverse(
                edge,
                current,
                partner_val,
                config={'r': r_all[(t, col)]},
            ))
            pseudo_obs[(leaf, conditioning)] = current

        for t in range(top_tree + 1):
            row = d - 2 - col - t
            partner = int(M[row, col])
            conditioning = frozenset(
                int(M[r, col])
                for r in range(row + 1, d - 1 - col)
            )
            next_leaf_cond = conditioning | {partner}
            next_partner_cond = conditioning | {leaf}
            edge = pair_copulas[(t, col)]
            r = r_all[(t, col)]

            leaf_val = pseudo_obs[(leaf, conditioning)]
            partner_val = pseudo_obs[(partner, conditioning)]
            leaf_next, partner_next = _edge_h_pair_with_r(
                edge, leaf_val, partner_val, r)
            pseudo_obs[(leaf, next_leaf_cond)] = _clip_unit(leaf_next)
            pseudo_obs[(partner, next_partner_cond)] = _clip_unit(partner_next)

    out = np.empty((n, d), dtype=np.float64)
    for var in range(d):
        out[:, var] = pseudo_obs[(var, frozenset())]
    return out


def given_suffix_edge_observations_with_r(
        d, trees, n, r_all, given, start_col, matrix, pair_copulas, edge_map):
    """Return edge observations fully determined by fixed suffix values."""
    M = matrix
    pseudo_obs = {}
    observed = {}

    last_var = int(M[0, d - 1])
    if d - 1 >= start_col:
        pseudo_obs[(last_var, frozenset())] = np.full(
            n, given[last_var], dtype=np.float64)
    else:
        return observed

    for col in range(d - 2, start_col - 1, -1):
        leaf = int(M[d - 1 - col, col])
        top_tree = d - 2 - col
        pseudo_obs[(leaf, frozenset())] = np.full(
            n, given[leaf], dtype=np.float64)
        for t in range(top_tree + 1):
            row = d - 2 - col - t
            partner = int(M[row, col])
            conditioning = frozenset(
                int(M[r, col])
                for r in range(row + 1, d - 1 - col)
            )
            next_leaf_cond = conditioning | {partner}
            next_partner_cond = conditioning | {leaf}
            edge = pair_copulas[(t, col)]
            r = r_all[(t, col)]

            leaf_val = pseudo_obs[(leaf, conditioning)]
            partner_val = pseudo_obs[(partner, conditioning)]
            observed[(t, col)] = edge_pair_from_pseudo_map(
                trees, (t, col), pseudo_obs, edge_map)
            leaf_next, partner_next = _edge_h_pair_with_r(
                edge, leaf_val, partner_val, r)
            pseudo_obs[(leaf, next_leaf_cond)] = _clip_unit(leaf_next)
            pseudo_obs[(partner, next_partner_cond)] = _clip_unit(partner_next)

    return observed


def edge_pair_from_pseudo_map(trees, key, pseudo_obs, edge_map):
    """Build the observed pair array for an edge from pseudo-observation map."""
    t, col = key
    orig_idx = edge_map[(t, col)]
    conditioned, conditioning = trees[t][orig_idx]
    v1, v2 = sorted(conditioned)
    return np.column_stack((
        _clip_unit(pseudo_obs[(v1, conditioning)]),
        _clip_unit(pseudo_obs[(v2, conditioning)]),
    ))
