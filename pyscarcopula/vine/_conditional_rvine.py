"""
Conditional prediction backends for R-vines.

The graph paths use the vine computational graph view from Cheng et al.
(2025), "Vine Copulas as Differentiable Computational Graphs",
arXiv:2506.13318: pseudo-observation nodes are propagated by h-functions and
graph-compatible unknowns are sampled by inverse h-functions. Patterns that
are not graph-compatible are handled by posterior grid or exact integration.
This module implements the conditional-sampling idea, not the paper's
differentiable PyTorch/autograd layer.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

from pyscarcopula.vine._conditional_scalar import (
    _clip_scalar,
    _copula_family_id,
    _copula_h_inverse_meta,
    _copula_h_inverse_scalar,
    _copula_h_meta,
    _copula_h_scalar,
    _copula_pdf_meta,
    _copula_pdf_scalar,
    _scratch_arrays,
)
from pyscarcopula.copula.independent import IndependentCopula
from pyscarcopula.vine import _conditional_exact as _exact
from pyscarcopula.vine import _conditional_grid as _grid
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


def rvine_conditional_plan(vine, given, quad_order=10):
    """
    Describe the matrix-order conditional sampling workload.

    The plan exposes the computational graph implied by the current R-vine
    matrix order: fixed nodes, sampled nodes, and posterior latent nodes needed
    when fixed variables appear later in the sampling order.
    """
    d = vine.d
    M = vine._structure.matrix
    order_cols = tuple(range(d - 1, -1, -1))
    order_vars = tuple(int(M[s, s]) for s in order_cols)
    posterior_idxs = tuple(_exact.posterior_indices(order_vars, given))
    posterior_vars = tuple(order_vars[idx] for idx in posterior_idxs)
    given_positions = {
        int(var): idx
        for idx, var in enumerate(order_vars)
        if var in given
    }
    last_given_index = (
        max(given_positions.values()) if given_positions else None
    )
    grid_size = max(int(quad_order), 2)
    if posterior_idxs:
        joint_grid_points = int(grid_size ** len(posterior_idxs))
    else:
        joint_grid_points = 0
    posterior_set = set(posterior_idxs)
    graph_steps = tuple(
        {
            'idx': idx,
            'column': int(order_cols[idx]),
            'var': int(var),
            'action': 'given' if var in given else 'sample',
            'posterior': idx in posterior_set,
        }
        for idx, var in enumerate(order_vars)
    )

    optimized = False
    checker = getattr(vine, 'is_conditioning_optimized_for', None)
    if checker is not None:
        optimized = bool(checker(given))

    return {
        'order_cols': order_cols,
        'order_vars': order_vars,
        'given_vars': tuple(sorted(int(var) for var in given)),
        'given_positions': given_positions,
        'last_given_index': last_given_index,
        'posterior_indices': posterior_idxs,
        'posterior_vars': posterior_vars,
        'posterior_dim': len(posterior_idxs),
        'quad_order': int(quad_order),
        'joint_grid_points': joint_grid_points,
        'joint_grid_supported': (
            len(posterior_idxs) >= 2 and joint_grid_points <= 50000
        ),
        'graph_feasible': len(posterior_idxs) == 0,
        'graph_steps': graph_steps,
        'optimized_for_given': optimized,
        'structure_status': getattr(vine, 'conditional_structure_status', None),
    }


def rvine_flexible_graph_plan(vine, given):
    """
    Reachability plan for the flexible graph conditional sampler.

    It tracks pseudo-observation nodes ``(var, cond)`` that are known from
    ``given``, obtainable by deterministic h-propagation, or sampleable from
    a pair-copula edge with one known endpoint.  Higher-tree sampled pseudo
    nodes are marked sampleable only when they can be inverted back to the
    base variable through known h-function inputs. This is the local
    scheduling layer corresponding to the vine computational graph approach in
    arXiv:2506.13318.
    """
    edge_index = _flexible_edge_index(vine)
    d = vine.d
    known = set((int(var), frozenset()) for var in given)
    sampled_base = set(int(var) for var in given)
    sampled_nodes = set()
    steps = []

    def inverse_chain(var, cond_set):
        chain = []
        cur_cond = frozenset(cond_set)
        while cur_cond:
            found = None
            for partner in sorted(cur_cond):
                prev_cond = frozenset(v for v in cur_cond if v != partner)
                if (int(partner), prev_cond) not in known:
                    continue
                edge_key = edge_index.get((
                    frozenset({int(var), int(partner)}), prev_cond))
                if edge_key is None:
                    continue
                found = (int(partner), prev_cond, edge_key)
                break
            if found is None:
                return None
            partner, prev_cond, edge_key = found
            chain.append({
                'edge': edge_key,
                'from': (int(var), cur_cond),
                'known': (partner, prev_cond),
                'to': (int(var), prev_cond),
            })
            cur_cond = prev_cond
        return tuple(chain)

    changed = True

    while changed:
        changed = False

        for tree_level, tree_edges in enumerate(vine.trees):
            for edge_idx, (v1, v2, cond) in enumerate(tree_edges):
                cond_set = frozenset(cond)
                left = (int(v1), cond_set)
                right = (int(v2), cond_set)
                left_next = (int(v1), cond_set | {int(v2)})
                right_next = (int(v2), cond_set | {int(v1)})

                if left in known and right in known:
                    new_nodes = []
                    if left_next not in known:
                        known.add(left_next)
                        new_nodes.append(left_next)
                    if right_next not in known:
                        known.add(right_next)
                        new_nodes.append(right_next)
                    if new_nodes:
                        steps.append({
                            'action': 'h_propagate',
                            'tree': int(tree_level),
                            'edge': int(edge_idx),
                            'inputs': (left, right),
                            'outputs': tuple(new_nodes),
                        })
                        changed = True

        if changed:
            continue

        candidates = []
        for tree_level, tree_edges in enumerate(vine.trees):
            for edge_idx, (v1, v2, cond) in enumerate(tree_edges):
                cond_set = frozenset(cond)
                left = (int(v1), cond_set)
                right = (int(v2), cond_set)
                if left in known and int(v2) not in sampled_base:
                    chain = inverse_chain(int(v2), cond_set)
                    if chain is not None:
                        candidates.append((
                            int(tree_level), int(edge_idx), left, right,
                            chain))
                if right in known and int(v1) not in sampled_base:
                    chain = inverse_chain(int(v1), cond_set)
                    if chain is not None:
                        candidates.append((
                            int(tree_level), int(edge_idx), right, left,
                            chain))

        if candidates:
            tree_level, edge_idx, known_node, sampled, chain = max(
                candidates, key=lambda item: (item[0], -len(item[4])))
            known.add(sampled)
            sampled_nodes.add(sampled)
            inverse_outputs = []
            for inv in chain:
                node = inv['to']
                known.add(node)
                inverse_outputs.append(node)
            sampled_base.add(int(sampled[0]))
            steps.append({
                'action': 'sample_base' if tree_level == 0 else 'sample_pseudo',
                'tree': tree_level,
                'edge': edge_idx,
                'known': known_node,
                'sampled': sampled,
                'inverse_chain': chain,
                'inverse_outputs': tuple(inverse_outputs),
            })
            changed = True

    missing_base = tuple(
        var for var in range(d)
        if (var, frozenset()) not in known
    )
    higher_tree_frontier = []
    for tree_level, tree_edges in enumerate(vine.trees[1:], start=1):
        for edge_idx, (v1, v2, cond) in enumerate(tree_edges):
            cond_set = frozenset(cond)
            left = (int(v1), cond_set)
            right = (int(v2), cond_set)
            if left in known and right not in known:
                higher_tree_frontier.append({
                    'tree': int(tree_level),
                    'edge': int(edge_idx),
                    'known': left,
                    'blocked_sample': right,
                })
            if right in known and left not in known:
                higher_tree_frontier.append({
                    'tree': int(tree_level),
                    'edge': int(edge_idx),
                    'known': right,
                    'blocked_sample': left,
                })

    higher_tree_independent = all(
        isinstance(edge.copula, IndependentCopula)
        for (tree_level, _), edge in vine.edges.items()
        if tree_level > 0
    )

    return {
        'given_vars': tuple(sorted(int(var) for var in given)),
        'known_nodes': tuple(sorted(
            (var, tuple(sorted(cond))) for var, cond in known
        )),
        'sampled_base_vars': tuple(sorted(sampled_base)),
        'sampled_pseudo_nodes': tuple(sorted(
            (var, tuple(sorted(cond))) for var, cond in sampled_nodes
        )),
        'missing_base_vars': missing_base,
        'complete': len(missing_base) == 0,
        'flexible_given_supported': len(given) == 1,
        'higher_tree_independent': higher_tree_independent,
        'sampleable': len(missing_base) == 0 and len(given) == 1,
        'steps': tuple(steps),
        'higher_tree_frontier': tuple(higher_tree_frontier),
        'note': (
            "Experimental reachability plan samples pseudo-observations from "
            "available pair-copula edges and inverts higher-tree nodes back "
            "to base variables when a known inverse h-chain exists. The "
            "flexible executor is currently enabled only for single-given "
            "patterns; multi-given patterns with posterior_dim > 0 require "
            "grid or exact posterior sampling."
        ),
    }


def _flexible_edge_index(vine):
    """Index tree edges by unordered conditioned pair and conditioning set."""
    edge_index = {}
    for tree_level, tree_edges in enumerate(vine.trees):
        for edge_idx, (v1, v2, cond) in enumerate(tree_edges):
            edge_index[(
                frozenset({int(v1), int(v2)}),
                frozenset(int(v) for v in cond),
            )] = (int(tree_level), int(edge_idx))
    return edge_index


def _edge_r_vector(r_all, edge_key, n):
    r = np.asarray(r_all[edge_key], dtype=np.float64).ravel()
    return _grid._r_array_vector(r, n)


def _flexible_h(edge, u, v, r):
    return _clip_unit(_grid._copula_h_vector(
        _copula_family_id(edge.copula), edge.copula.rotate,
        edge.copula, u, v, r))


def _flexible_h_inverse(edge, u, v, r):
    return _clip_unit(_grid._copula_h_inverse_vector(
        _copula_family_id(edge.copula), edge.copula.rotate,
        edge.copula, u, v, r))


def sample_rvine_flexible_graph_with_r(vine, n, r_all, given, rng):
    """
    Sample using the experimental flexible graph executor.

    The executor starts from fixed base variables, repeatedly propagates known
    h-function outputs, and samples an unknown pseudo-observation from the
    highest currently available edge.  If that pseudo-observation is from a
    higher tree, it is inverted through known h-function inputs until the
    corresponding base variable is recovered.
    """
    plan = rvine_flexible_graph_plan(vine, given)
    if not plan['sampleable']:
        raise ValueError(
            "Flexible graph executor requires flexible_graph_plan(...)."
            "sampleable=True")

    d = vine.d
    x = np.empty((n, d), dtype=np.float64)
    filled = np.zeros(d, dtype=bool)
    pseudo = {}

    for var, val in given.items():
        node = (int(var), frozenset())
        arr = np.full(n, float(val), dtype=np.float64)
        pseudo[node] = arr
        x[:, int(var)] = arr
        filled[int(var)] = True

    for step in plan['steps']:
        if step['action'] == 'h_propagate':
            tree_level = int(step['tree'])
            edge_idx = int(step['edge'])
            edge = vine.edges[(tree_level, edge_idx)]
            left, right = step['inputs']
            left = (int(left[0]), frozenset(left[1]))
            right = (int(right[0]), frozenset(right[1]))
            r = _edge_r_vector(r_all, (tree_level, edge_idx), n)
            for out in step['outputs']:
                out = (int(out[0]), frozenset(out[1]))
                if out in pseudo:
                    continue
                if out[0] == left[0]:
                    pseudo[out] = _flexible_h(
                        edge, pseudo[left], pseudo[right], r)
                else:
                    pseudo[out] = _flexible_h(
                        edge, pseudo[right], pseudo[left], r)
            continue

        if step['action'] not in ('sample_base', 'sample_pseudo'):
            continue

        known = (int(step['known'][0]), frozenset(step['known'][1]))
        sampled = (int(step['sampled'][0]), frozenset(step['sampled'][1]))
        sampled_var = int(sampled[0])
        if filled[sampled_var]:
            continue

        tree_level = int(step['tree'])
        edge_idx = int(step['edge'])
        edge = vine.edges[(tree_level, edge_idx)]
        r = _edge_r_vector(r_all, (tree_level, edge_idx), n)
        w = rng.uniform(0.0, 1.0, size=n)
        pseudo[sampled] = _flexible_h_inverse(edge, w, pseudo[known], r)

        current = sampled
        for inv in step.get('inverse_chain', ()):
            inv_edge_key = tuple(inv['edge'])
            inv_edge = vine.edges[inv_edge_key]
            known_node = (int(inv['known'][0]), frozenset(inv['known'][1]))
            out_node = (int(inv['to'][0]), frozenset(inv['to'][1]))
            r_inv = _edge_r_vector(r_all, inv_edge_key, n)
            pseudo[out_node] = _flexible_h_inverse(
                inv_edge, pseudo[current], pseudo[known_node], r_inv)
            current = out_node

        base = (sampled_var, frozenset())
        if base not in pseudo:
            raise RuntimeError(
                "Flexible graph executor did not recover base variable "
                f"{sampled_var}")
        x[:, sampled_var] = pseudo[base]
        filled[sampled_var] = True

    if not np.all(filled):
        missing = tuple(int(idx) for idx in np.where(~filled)[0])
        raise RuntimeError(
            f"Flexible graph executor did not fill variables {missing}")
    return x


def _edge_r_scalar(r_all, edge_key, sample_idx=0):
    r = np.asarray(r_all[edge_key], dtype=np.float64).ravel()
    if r.size == 0:
        raise ValueError("Empty edge parameter array")
    if sample_idx < 0 or sample_idx >= r.size:
        sample_idx = 0
    return float(r[sample_idx])


def _r_array_scalar(r_array, sample_idx=0):
    if r_array.size == 0:
        raise ValueError("Empty edge parameter array")
    if sample_idx < 0 or sample_idx >= r_array.size:
        sample_idx = 0
    return float(r_array[sample_idx])


class _ColumnOps(list):
    """List-like column metadata with dense pseudo-observation keys."""

    def __init__(self, columns, base_keys_by_col, base_keys_by_var):
        super().__init__(columns)
        self.base_keys_by_col = base_keys_by_col
        self.base_keys_by_var = base_keys_by_var


def _build_column_ops(vine, M, edge_map, r_all):
    """Precompute per-column RVine conditional operations."""
    d = vine.d
    columns = [[] for _ in range(d)]
    key_ids = {}

    def _key(var, cond_set):
        key = (int(var), frozenset(cond_set))
        key_id = key_ids.get(key)
        if key_id is None:
            key_id = len(key_ids)
            key_ids[key] = key_id
        return key_id

    base_keys_by_col = [None] * d
    base_keys_by_var = [None] * d
    empty = frozenset()
    for s in range(d):
        var = int(M[s, s])
        base_key = _key(var, empty)
        base_keys_by_col[s] = base_key
        base_keys_by_var[var] = base_key

    for s in range(d - 1):
        n_levels = d - s - 1
        var = int(M[s, s])
        base_key = base_keys_by_col[s]
        for m in range(n_levels):
            edge_key = edge_map[(m, s)]
            edge = vine.edges[edge_key]
            copula = edge.copula
            cond_set = frozenset(M[s + 1:s + m + 1, s])
            next_cond = frozenset(M[s + 1:s + m + 2, s])
            partner_var = int(M[s + m + 1, s])
            columns[s].append((
                np.asarray(r_all[edge_key], dtype=np.float64).ravel(),
                copula,
                _copula_family_id(copula),
                copula.rotate,
                partner_var,
                cond_set,
                next_cond,
                _scratch_arrays(),
                base_key,
                _key(partner_var, cond_set),
                _key(var, cond_set),
                _key(var, next_cond),
                _key(partner_var, cond_set | {var}),
            ))
    return _ColumnOps(columns, base_keys_by_col, base_keys_by_var)


def _column_from_x(vine, pseudo, M, edge_map, s, x_val, r_all,
                   sample_idx=0, column_ops=None):
    """Construct pseudo updates for column s from a fixed base value."""
    d = vine.d
    var = M[s, s]
    updates = {}
    base = _clip_scalar(float(x_val))
    dense_keys = isinstance(column_ops, _ColumnOps)
    empty_key = (var, frozenset())
    base_key = column_ops.base_keys_by_col[s] if dense_keys else empty_key
    updates[base_key] = base

    if s == d - 1:
        updates['_density'] = 1.0
        return updates

    ops = column_ops[s] if column_ops is not None else None
    if ops is None:
        n_levels = d - s - 1
        ops = []
        for m in range(n_levels):
            edge_key = edge_map[(m, s)]
            edge = vine.edges[edge_key]
            copula = edge.copula
            ops.append((
                np.asarray(r_all[edge_key], dtype=np.float64).ravel(),
                copula,
                _copula_family_id(copula),
                copula.rotate,
                M[s + m + 1, s],
                frozenset(M[s + 1:s + m + 1, s]),
                frozenset(M[s + 1:s + m + 2, s]),
                _scratch_arrays(),
            ))
    density = 1.0

    for op in ops:
        (r_array, copula, family_id, rot, partner_var, cond_set, next_cond,
         work) = op[:8]
        if len(op) >= 13:
            _, partner_key, var_cond_key, var_next_key, partner_next_key = op[8:13]
        else:
            partner_key = (partner_var, cond_set)
            var_cond_key = (var, cond_set)
            var_next_key = (var, next_cond)
            partner_next_key = (partner_var, cond_set | {var})
        r = _r_array_scalar(r_array, sample_idx)

        partner_val = pseudo.get(partner_key)
        if partner_val is None:
            raise RuntimeError(
                "Missing partner pseudo-observation during RVine conditional "
                f"construction: var={partner_var}, cond={sorted(cond_set)}"
            )

        var_at_cond = updates.get(var_cond_key, updates[base_key])
        density *= max(
            _copula_pdf_meta(
                family_id, rot, copula, var_at_cond, partner_val, r, work),
            1e-300)

        cur = _clip_scalar(
            _copula_h_meta(
                family_id, rot, copula, var_at_cond, partner_val, r, work))
        updates[var_next_key] = cur

        rev = _clip_scalar(
            _copula_h_meta(
                family_id, rot, copula, partner_val, var_at_cond, r, work))
        updates[partner_next_key] = rev

    updates['_density'] = density
    return updates


def _column_from_w(vine, pseudo, M, edge_map, s, w_val, r_all,
                   sample_idx=0, column_ops=None):
    """Construct pseudo updates for column s from latent Rosenblatt w."""
    d = vine.d
    if s == d - 1:
        return _column_from_x(
            vine, pseudo, M, edge_map, s, w_val, r_all, sample_idx,
            column_ops=column_ops)

    ops = column_ops[s] if column_ops is not None else None
    if ops is None:
        n_levels = d - s - 1
        ops = []
        for m in range(n_levels):
            edge_key = edge_map[(m, s)]
            edge = vine.edges[edge_key]
            copula = edge.copula
            ops.append((
                np.asarray(r_all[edge_key], dtype=np.float64).ravel(),
                copula,
                _copula_family_id(copula),
                copula.rotate,
                M[s + m + 1, s],
                frozenset(M[s + 1:s + m + 1, s]),
                frozenset(M[s + 1:s + m + 2, s]),
                _scratch_arrays(),
            ))
    val = float(w_val)

    for op in reversed(ops):
        (r_array, copula, family_id, rot, partner_var, cond_set, _,
         work) = op[:8]
        if len(op) >= 13:
            partner_key = op[9]
        else:
            partner_key = (partner_var, cond_set)
        r = _r_array_scalar(r_array, sample_idx)

        partner_val = pseudo.get(partner_key)
        if partner_val is None:
            raise RuntimeError(
                "Missing partner pseudo-observation during RVine conditional "
                f"inversion: var={partner_var}, cond={sorted(cond_set)}"
            )

        val = _clip_scalar(
            _copula_h_inverse_meta(
                family_id, rot, copula, val, partner_val, r, work))

    return _column_from_x(
        vine, pseudo, M, edge_map, s, val, r_all, sample_idx,
        column_ops=column_ops)


_sample_from_tabulated_weight = _exact.sample_from_tabulated_weight


def _decode_grid_index(flat_idx, dim, grid_size):
    cells = np.empty(dim, dtype=np.int64)
    val = int(flat_idx)
    for pos in range(dim - 1, -1, -1):
        cells[pos] = val % grid_size
        val //= grid_size
    return cells


def _joint_grid_weight(vine, M, edge_map, order_cols, order_vars,
                       posterior_idxs, w_values, given, r_all, sample_idx=0,
                       column_ops=None, posterior_pos_by_idx=None,
                       last_idx=None):
    pseudo = {}
    density = 1.0
    if posterior_pos_by_idx is None:
        posterior_pos_by_idx = {idx: pos for pos, idx in enumerate(posterior_idxs)}
    if last_idx is None:
        last_idx = max(
            [idx for idx, var in enumerate(order_vars) if var in given]
            + list(posterior_idxs)
        )

    for idx in range(last_idx + 1):
        s = order_cols[idx]
        var = order_vars[idx]

        if var in given:
            updates = _column_from_x(
                vine, pseudo, M, edge_map, s, given[var], r_all, sample_idx,
                column_ops=column_ops)
            density *= updates.pop('_density')
        else:
            pos = posterior_pos_by_idx[idx]
            updates = _column_from_w(
                vine, pseudo, M, edge_map, s, w_values[pos], r_all, sample_idx,
                column_ops=column_ops)
            updates.pop('_density', None)

        pseudo.update(updates)

    return density


def _sample_joint_posterior_grid(vine, M, edge_map, order_cols, order_vars,
                                 posterior_idxs, given, r_all, rng,
                                 grid_size, sample_idx=0, column_ops=None):
    cache = _build_joint_posterior_grid_cache(
        vine, M, edge_map, order_cols, order_vars, posterior_idxs, given,
        r_all, grid_size, sample_idx=sample_idx, column_ops=column_ops)
    if cache is None:
        return {
            idx: float(rng.uniform(0.0, 1.0))
            for idx in posterior_idxs
        }
    return _draw_joint_posterior_grid(cache, rng)


def _build_joint_posterior_grid_cache(vine, M, edge_map, order_cols, order_vars,
                                      posterior_idxs, given, r_all, grid_size,
                                      sample_idx=0, column_ops=None):
    dim = len(posterior_idxs)
    centers = (np.arange(grid_size, dtype=np.float64) + 0.5) / grid_size
    total = int(grid_size ** dim)
    masses = np.empty(total, dtype=np.float64)
    cells = np.zeros(dim, dtype=np.int64)
    w_values = np.empty(dim, dtype=np.float64)
    posterior_pos_by_idx = {
        idx: pos for pos, idx in enumerate(posterior_idxs)
    }
    last_idx = max(
        [idx for idx, var in enumerate(order_vars) if var in given]
        + list(posterior_idxs)
    )

    for flat_idx in range(total):
        for pos in range(dim):
            w_values[pos] = centers[cells[pos]]
        mass = _joint_grid_weight(
            vine, M, edge_map, order_cols, order_vars,
            posterior_idxs, w_values, given, r_all, sample_idx,
            column_ops=column_ops,
            posterior_pos_by_idx=posterior_pos_by_idx,
            last_idx=last_idx)
        if not np.isfinite(mass) or mass <= 0.0:
            mass = 0.0
        masses[flat_idx] = mass
        for pos in range(dim - 1, -1, -1):
            cells[pos] += 1
            if cells[pos] < grid_size:
                break
            cells[pos] = 0

    total_mass = float(np.sum(masses))
    if not np.isfinite(total_mass) or total_mass <= 0.0:
        return None

    cdf = np.cumsum(masses)
    return posterior_idxs, grid_size, dim, total, cdf, total_mass


def _draw_joint_posterior_grid(cache, rng):
    posterior_idxs, grid_size, dim, total, cdf, total_mass = cache
    target = float(rng.uniform(0.0, total_mass))
    flat_idx = int(np.searchsorted(cdf, target, side='right'))
    if flat_idx >= total:
        flat_idx = total - 1

    cells = _decode_grid_index(flat_idx, dim, grid_size)
    out = {}
    for pos, idx in enumerate(posterior_idxs):
        w = (float(cells[pos]) + float(rng.uniform(0.0, 1.0))) / grid_size
        out[idx] = float(_clip_unit(np.asarray([w], dtype=np.float64))[0])
    return out


def sample_rvine_conditional_with_r(vine, n, r_all, given, rng,
                                    quad_order=10,
                                    conditional_method='auto'):
    """
    Sample from a fitted R-vine conditional on arbitrary given variables.

    ``conditional_method='graph'`` uses h-function propagation and inverse
    h-function sampling when the conditioning pattern is graph-compatible,
    following the conditional-sampling idea in arXiv:2506.13318. ``'grid'`` and
    ``'exact'`` handle patterns that require posterior reweighting.
    """
    if conditional_method not in ('auto', 'graph', 'exact', 'grid'):
        raise ValueError(
            "conditional_method must be one of 'auto', 'graph', 'exact', "
            "or 'grid'")

    d = vine.d
    M = vine._structure.matrix
    edge_map = vine._edge_map
    column_ops = _build_column_ops(vine, M, edge_map, r_all)
    order_cols = list(range(d - 1, -1, -1))
    order_vars = [M[s, s] for s in order_cols]
    nodes, weights = leggauss(quad_order)
    posterior_idxs = _exact.posterior_indices(order_vars, given)

    if conditional_method in ('auto', 'graph') and len(posterior_idxs) == 0:
        return _grid.sample_rvine_conditional_grid_many(
            vine, n, order_cols, order_vars, given, rng, column_ops, {})
    if conditional_method == 'graph':
        flexible_plan = rvine_flexible_graph_plan(vine, given)
        if flexible_plan['sampleable']:
            return sample_rvine_flexible_graph_with_r(
                vine, n, r_all, given, rng)
        raise ValueError(
            "conditional_method='graph' requires a graph-compatible "
            "conditioning pattern with graph_feasible=True "
            "(posterior_dim == 0), or flexible_graph_plan(...).sampleable")

    joint_grid_size = max(int(quad_order), 2)
    joint_grid_points = int(joint_grid_size ** max(len(posterior_idxs), 1))
    use_joint_grid = (
        conditional_method in ('auto', 'grid')
        and len(posterior_idxs) >= 2
        and joint_grid_points <= 50000
    )
    if conditional_method == 'grid' and len(posterior_idxs) >= 2 and not use_joint_grid:
        raise ValueError(
            "conditional_method='grid' requires quad_order ** posterior_dim "
            "to be <= 50000")
    joint_grid_cache = None
    joint_grid_many_cache = None
    if use_joint_grid and _grid.column_ops_have_static_r(column_ops):
        joint_grid_cache = _build_joint_posterior_grid_cache(
            vine, M, edge_map, order_cols, order_vars,
            posterior_idxs, given, r_all, joint_grid_size,
            sample_idx=0, column_ops=column_ops)
    elif use_joint_grid:
        joint_grid_many_cache = _grid.build_joint_posterior_grid_cache_many(
            order_cols, order_vars, posterior_idxs, given, joint_grid_size,
            n, column_ops)

    if use_joint_grid:
        cache = joint_grid_cache if joint_grid_cache is not None else joint_grid_many_cache
        posterior_w = _grid.draw_joint_posterior_grid_batch(cache, n, rng)
        return _grid.sample_rvine_conditional_grid_many(
            vine, n, order_cols, order_vars, given, rng, column_ops,
            posterior_w)

    x = np.zeros((n, d), dtype=np.float64)

    for t in range(n):
        pseudo = {}
        posterior_w = {}
        if use_joint_grid:
            if joint_grid_cache is not None:
                posterior_w = _draw_joint_posterior_grid(joint_grid_cache, rng)
            elif joint_grid_many_cache is not None:
                posterior_w = _draw_joint_posterior_grid_many(
                    joint_grid_many_cache, t, rng)
            else:
                posterior_w = _sample_joint_posterior_grid(
                    vine, M, edge_map, order_cols, order_vars,
                    posterior_idxs, given, r_all, rng, joint_grid_size,
                    sample_idx=t, column_ops=column_ops)

        for idx, s in enumerate(order_cols):
            var = order_vars[idx]

            if var in given:
                updates = _column_from_x(
                    vine, pseudo, M, edge_map, s, given[var], r_all,
                    sample_idx=t, column_ops=column_ops)
            elif idx in posterior_w:
                updates = _column_from_w(
                    vine, pseudo, M, edge_map, s, posterior_w[idx],
                    r_all, sample_idx=t, column_ops=column_ops)
            elif _exact.has_future_given(order_vars, idx + 1, given):
                w_val = _exact.sample_w_posterior(
                    vine, M, edge_map, order_cols, order_vars, idx,
                    pseudo, given, r_all, rng, nodes, weights, sample_idx=t,
                    column_ops=column_ops, column_from_x=_column_from_x,
                    column_from_w=_column_from_w)
                updates = _column_from_w(
                    vine, pseudo, M, edge_map, s, w_val, r_all, sample_idx=t,
                    column_ops=column_ops)
            else:
                w_val = float(rng.uniform(0.0, 1.0))
                updates = _column_from_w(
                    vine, pseudo, M, edge_map, s, w_val, r_all, sample_idx=t,
                    column_ops=column_ops)

            updates.pop('_density', None)
            pseudo.update(updates)

        for var in range(d):
            key = column_ops.base_keys_by_var[var]
            if key not in pseudo:
                raise RuntimeError(f"Missing sampled variable {var}")
            x[t, var] = pseudo[key]

    return x
