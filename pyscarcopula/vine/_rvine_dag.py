"""Runtime DAG conditional sampler for natural-order R-vines.

The sampler works on pseudo-observation nodes ``(var, conditioning_set)``.
Known endpoints propagate through h-functions; unknown reachable nodes are
sampled with uniform noise and inverted back to base variables through
available inverse h-chains.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from pyscarcopula.vine._rvine_edges import _edge_h, _edge_h_inverse


_EPS = 1e-10


class ConditionalSamplePlan(list):
    """Execution plan with output dimension and used matrix edges attached."""

    def __init__(self, steps, d):
        super().__init__(steps)
        self.d = int(d)
        self.edges_used = tuple(sorted({
            tuple(step['edge'])
            for step in steps
            if step.get('action') in ('h_prop', 'h_inv') and 'edge' in step
        }))


def _node_key(var, conditioning=()):
    return int(var), frozenset(int(v) for v in conditioning)


def matrix_edge_key(matrix, tree_level, col):
    """Decode one natural-order matrix edge as ``(pair, conditioning)``."""
    d = matrix.shape[0]
    pair = frozenset({
        int(matrix[d - 1 - col, col]),
        int(matrix[d - 2 - col - tree_level, col]),
    })
    conditioning = frozenset(
        int(matrix[row, col])
        for row in range(d - 1 - col - tree_level, d - 1 - col)
    )
    return pair, conditioning


def build_runtime_rvine_dag(matrix, edge_map):
    """Build computational-graph indexes from a natural-order R-vine matrix."""
    matrix = np.asarray(matrix, dtype=int)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"build_runtime_rvine_dag: matrix must be square, got {matrix.shape}"
        )

    edges = {}
    by_endpoint = defaultdict(list)

    for tree_level, col in sorted(edge_map):
        edge_key = matrix_edge_key(matrix, int(tree_level), int(col))
        if edge_key in edges:
            raise ValueError(
                "build_runtime_rvine_dag: duplicate decoded edge "
                f"{edge_key!r}"
            )
        edges[edge_key] = (int(tree_level), int(col))

        pair, conditioning = edge_key
        for var in sorted(pair):
            by_endpoint[(int(var), conditioning)].append(edge_key)

    return {
        'edges': edges,
        'by_endpoint': dict(by_endpoint),
    }


def _inverse_partner_order(candidate):
    """Prefer the deepest/rightmost matrix edge, not the variable id.

    Multiple inverse chains can be feasible under the simplifying assumption.
    This deterministic tie-break follows the fitted R-vine geometry: deeper
    tree levels first, then rightmost natural-order columns. The partner id is
    used only as a final stable tie-breaker.
    """
    partner, _prev_cond, edge = candidate
    return int(edge[0]), int(edge[1]), -int(partner)


def _sample_candidate_order(candidate):
    """Order candidates for the arbitrary-conditioning initializer.

    This is not a probabilistic priority and not the exact R-vine sampling
    order. It only chooses a deterministic DAG initializer before the MCMC
    refinement. Prefer candidates already available at deeper conditional
    levels, then shorter inverse chains to base variables, then rightmost
    natural-order matrix columns. The leaf id is only a stable final tie-break.
    """
    edge = candidate['edge']
    return (
        int(edge[0]),
        -len(candidate['chain']),
        int(edge[1]),
        -int(candidate['leaf']),
    )


def _inverse_chain_to_base(dag, known, var, conditioning):
    chain = []
    cur_cond = frozenset(conditioning)
    var = int(var)

    while cur_cond:
        feasible = []
        for partner in sorted(cur_cond):
            prev_cond = frozenset(v for v in cur_cond if v != partner)
            if _node_key(partner, prev_cond) not in known:
                continue

            edge_key = (frozenset({var, int(partner)}), prev_cond)
            edge = dag['edges'].get(edge_key)
            if edge is None:
                continue

            feasible.append((int(partner), prev_cond, edge))

        if not feasible:
            return None

        partner, prev_cond, edge = max(feasible, key=_inverse_partner_order)
        chain.append({
            'action': 'h_inv',
            'edge': edge,
            'leaf': var,
            'partner': partner,
            'cond': prev_cond,
            'from': _node_key(var, cur_cond),
            'known': _node_key(partner, prev_cond),
            'to': _node_key(var, prev_cond),
        })
        cur_cond = prev_cond

    return chain


def _find_sample_candidate(dag, known):
    candidates = []
    for edge_key, edge in sorted(dag['edges'].items(), key=lambda item: item[1]):
        pair, cond = edge_key
        a, b = sorted(pair)
        for var, partner in ((a, b), (b, a)):
            base_node = _node_key(var)
            if base_node in known:
                continue
            sampled_node = _node_key(var, cond)
            known_partner = _node_key(partner, cond)
            if sampled_node in known or known_partner not in known:
                continue
            chain = _inverse_chain_to_base(dag, known, var, cond)
            if chain is None:
                continue
            candidates.append({
                'edge': edge,
                'leaf': int(var),
                'partner': int(partner),
                'cond': cond,
                'sampled_node': sampled_node,
                'known': known_partner,
                'chain': chain,
            })
    if not candidates:
        return None
    return max(candidates, key=_sample_candidate_order)


def plan_conditional_sample(dag, given, d):
    """Build a runtime conditional sample plan for arbitrary ``given``."""
    known = {
        _node_key(var): 'given'
        for var in given
    }
    steps = []

    while True:
        changed = False

        for edge_key, edge in sorted(dag['edges'].items(), key=lambda item: item[1]):
            pair, cond = edge_key
            a, b = sorted(pair)
            left = _node_key(a, cond)
            right = _node_key(b, cond)
            if left not in known or right not in known:
                continue

            for leaf, partner in ((a, b), (b, a)):
                out_node = _node_key(leaf, cond | {partner})
                if out_node in known:
                    continue
                known[out_node] = 'h_prop'
                steps.append({
                    'action': 'h_prop',
                    'edge': edge,
                    'leaf': int(leaf),
                    'partner': int(partner),
                    'cond': cond,
                    'to': out_node,
                })
                changed = True

        if changed:
            continue

        candidate = _find_sample_candidate(dag, known)
        if candidate is None:
            break

        uniform_node = ('w', int(candidate['leaf']))
        steps.append({
            'action': 'sample_uniform',
            'var': int(candidate['leaf']),
            'node': uniform_node,
        })
        steps.append({
            'action': 'h_inv',
            'edge': candidate['edge'],
            'leaf': int(candidate['leaf']),
            'partner': int(candidate['partner']),
            'cond': candidate['cond'],
            'from': uniform_node,
            'known': candidate['known'],
            'to': candidate['sampled_node'],
        })
        known[candidate['sampled_node']] = 'h_inv'

        for step in candidate['chain']:
            steps.append(step)
            known[step['to']] = 'h_inv'

    missing = [
        var for var in range(int(d))
        if _node_key(var) not in known
    ]
    if missing:
        reachable = sorted(
            (var, tuple(sorted(cond)))
            for var, cond in known
        )
        raise ValueError(
            "plan_conditional_sample: variables are not reachable from "
            f"given={sorted(int(v) for v in given)}: missing={missing}; "
            f"known_nodes={reachable}"
        )

    return ConditionalSamplePlan(steps, d)


def _clip_unit(values):
    return np.clip(np.asarray(values, dtype=np.float64), _EPS, 1.0 - _EPS)


def _edge_payload(step, r_all):
    edge_key = tuple(step['edge'])
    payload = r_all[edge_key]
    if not isinstance(payload, dict):
        raise TypeError(
            "execute_conditional_plan expects r_all edge payloads as "
            "{'edge': pair_copula, 'r': parameter_values} dicts"
        )
    edge = payload['edge']
    r = payload['r']
    return edge, np.asarray(r, dtype=np.float64)


def execute_conditional_plan(plan, r_all, given, n, rng):
    """Execute a conditional DAG plan and return an ``(n, d)`` array."""
    n = int(n)
    pseudo_obs = {}
    for var, value in given.items():
        pseudo_obs[_node_key(var)] = np.full(
            n, float(value), dtype=np.float64)

    for step in plan:
        action = step['action']

        if action == 'sample_uniform':
            pseudo_obs[step['node']] = rng.uniform(_EPS, 1.0 - _EPS, size=n)
            continue

        edge, r = _edge_payload(step, r_all)
        source = step['from'] if action == 'h_inv' else _node_key(
            step['leaf'], step['cond'])
        known = step['known'] if action == 'h_inv' else _node_key(
            step['partner'], step['cond'])

        if action == 'h_prop':
            pseudo_obs[step['to']] = _clip_unit(_edge_h(
                edge,
                pseudo_obs[source],
                pseudo_obs[known],
                config={'r': r},
            ))
        elif action == 'h_inv':
            pseudo_obs[step['to']] = _clip_unit(_edge_h_inverse(
                edge,
                pseudo_obs[source],
                pseudo_obs[known],
                config={'r': r},
            ))
        else:
            raise ValueError(f"execute_conditional_plan: unknown action {action!r}")

    out = np.empty((n, int(plan.d)), dtype=np.float64)
    missing = []
    for var in range(int(plan.d)):
        node = _node_key(var)
        if node not in pseudo_obs:
            missing.append(var)
            continue
        out[:, var] = pseudo_obs[node]
    if missing:
        raise RuntimeError(
            f"execute_conditional_plan: plan did not produce base variables {missing}"
        )
    return out
