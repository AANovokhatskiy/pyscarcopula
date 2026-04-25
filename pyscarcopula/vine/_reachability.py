"""Reachability helpers for R-vine conditional structure scoring."""

from __future__ import annotations

from pyscarcopula.vine._rvine_dag import (
    _find_sample_candidate,
    _node_key,
    build_runtime_rvine_dag,
)


def build_rvine_dag(matrix, edge_map):
    """Build the same runtime DAG used by conditional prediction."""
    return build_runtime_rvine_dag(matrix, edge_map)


def analyze_conditional_reachability(dag, given, d):
    """Return reachability diagnostics for a fixed ``given`` set.

    This mirrors the runtime DAG planner without evaluating h-functions. Keep
    it on the same helper functions as ``_rvine_dag`` so fit-time reachability
    diagnostics cannot drift from predict-time execution.
    """
    known = {
        _node_key(var): 'given'
        for var in given
    }
    steps = 0
    inverse_steps = 0

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
                steps += 1
                changed = True

        if changed:
            continue

        candidate = _find_sample_candidate(dag, known)
        if candidate is None:
            break

        known[candidate['sampled_node']] = 'h_inv'
        steps += 1
        inverse_steps += 1
        for step in candidate['chain']:
            known[step['to']] = 'h_inv'
            steps += 1
            inverse_steps += 1

    missing = tuple(
        var for var in range(int(d))
        if _node_key(var) not in known
    )
    return {
        'complete': len(missing) == 0,
        'missing_base_vars': missing,
        'reachable_base_vars': tuple(
            var for var in range(int(d))
            if _node_key(var) in known
        ),
        'known_nodes': tuple(sorted(
            (var, tuple(sorted(cond)))
            for var, cond in known
        )),
        'n_known_nodes': len(known),
        'n_steps': int(steps),
        'n_inverse_steps': int(inverse_steps),
    }
