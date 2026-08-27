"""Frozen Python traversal references for the native R-vine runtime.

These functions intentionally live outside :mod:`pyscarcopula`.  They are
used only for differential tests and are never available as a production
backend.
"""

from __future__ import annotations

import numpy as np

from pyscarcopula._constants import PSEUDO_OBS_EPS
from pyscarcopula._utils import (
    clip_pseudo_observations,
    clip_rosenblatt_output,
)
from pyscarcopula.vine._helpers import (
    _clip_unit,
    _open_unit_uniform,
    _prepared_open_unit_draws,
)
from pyscarcopula.vine._edge_adapter import (
    edge_copula,
    edge_has_dynamic_params,
    edge_is_independent,
    edge_param,
    edge_result,
    strategy_for_result,
)
from pyscarcopula.vine._rvine_edges import (
    _edge_h,
    _edge_h_inverse,
    _edge_h_inverse_for_variables,
    _edge_h_pair,
    _edge_h_pair_for_variables,
)
from pyscarcopula.vine._rvine_conditional_plan import _node_key
from pyscarcopula.vine._rvine_sampling_plan import build_rvine_sampling_plan


def _sample_with_r_python(
        vine, n, r_all, rng, return_pseudo=False, max_active_tree=None,
        traversal_plan=None, *, uniforms=None):
    """Reference implementation of canonical R-vine sampling."""
    d = vine.d
    if uniforms is None:
        w = _open_unit_uniform(rng, size=(n, d))
    else:
        w = _prepared_open_unit_draws(
            uniforms, (n, d), name="R-vine sampling uniforms")
    if max_active_tree is None:
        max_active_tree = vine._max_non_independent_tree_level()
    if max_active_tree < 0:
        if return_pseudo:
            pseudo_obs = {
                (var, frozenset()): w[:, var].copy()
                for var in range(d)
            }
            return w, pseudo_obs
        return w

    if traversal_plan is None:
        active_keys = vine._sample_active_edge_keys(max_active_tree)
        traversal_plan = build_rvine_sampling_plan(
            d,
            vine._natural_order_matrix,
            vine._trees,
            vine._edge_map,
            active_keys,
            max_active_tree,
        )
    nodes = [None] * len(traversal_plan.node_keys)
    nodes[traversal_plan.last_output_node] = w[
        :, traversal_plan.last_uniform_column].copy()

    for column, uniform_column in enumerate(traversal_plan.column_uniforms):
        current = w[:, uniform_column].copy()
        for index in range(
                traversal_plan.inverse_offsets[column],
                traversal_plan.inverse_offsets[column + 1]):
            edge_key = traversal_plan.active_keys[
                traversal_plan.inverse_edges[index]]
            partner_node = traversal_plan.inverse_partner_nodes[index]
            output_node = traversal_plan.inverse_output_nodes[index]
            target_variable = traversal_plan.node_keys[output_node][0]
            partner_variable = traversal_plan.node_keys[partner_node][0]
            current = _clip_unit(_edge_h_inverse_for_variables(
                vine.pair_copulas[edge_key],
                target_variable,
                current,
                partner_variable,
                nodes[partner_node],
                config={"r": r_all[edge_key]},
            ))
            nodes[output_node] = current

        for index in range(
                traversal_plan.forward_offsets[column],
                traversal_plan.forward_offsets[column + 1]):
            edge_key = traversal_plan.active_keys[
                traversal_plan.forward_edges[index]]
            leaf_node = traversal_plan.forward_leaf_nodes[index]
            partner_node = traversal_plan.forward_partner_nodes[index]
            leaf_output = traversal_plan.forward_leaf_output_nodes[index]
            partner_output = (
                traversal_plan.forward_partner_output_nodes[index])
            leaf_variable = traversal_plan.node_keys[leaf_node][0]
            partner_variable = traversal_plan.node_keys[partner_node][0]
            leaf_next, partner_next = _edge_h_pair_for_variables(
                vine.pair_copulas[edge_key],
                leaf_variable,
                nodes[leaf_node],
                partner_variable,
                nodes[partner_node],
                config={"r": r_all[edge_key]},
            )
            nodes[leaf_output] = _clip_unit(leaf_next)
            nodes[partner_output] = _clip_unit(partner_next)

    out = np.empty((n, d), dtype=np.float64)
    for variable, node in enumerate(traversal_plan.output_nodes):
        out[:, variable] = nodes[node]
    if return_pseudo:
        return out, {
            key: nodes[index]
            for index, key in enumerate(traversal_plan.node_keys)
        }
    return out


def _sample_suffix_given_with_r_python(
        d, n, r_all, rng, given, start_col, matrix, pair_copulas, *,
        uniforms=None):
    """Reference implementation of exact suffix-conditioned sampling."""
    w = (
        _open_unit_uniform(rng, size=(n, d))
        if uniforms is None else
        _prepared_open_unit_draws(
            uniforms, (n, d), name="suffix sampling uniforms")
    )
    pseudo_obs = {}
    last_var = int(matrix[0, d - 1])
    pseudo_obs[(last_var, frozenset())] = (
        np.full(n, given[last_var], dtype=np.float64)
        if d - 1 >= start_col else w[:, d - 1].copy()
    )

    for col in range(d - 2, start_col - 1, -1):
        leaf = int(matrix[d - 1 - col, col])
        top_tree = d - 2 - col
        pseudo_obs[(leaf, frozenset())] = np.full(
            n, given[leaf], dtype=np.float64)
        for tree in range(top_tree + 1):
            row = d - 2 - col - tree
            partner = int(matrix[row, col])
            conditioning = frozenset(
                int(matrix[source_row, col])
                for source_row in range(row + 1, d - 1 - col)
            )
            edge = pair_copulas[(tree, col)]
            leaf_next, partner_next = _edge_h_pair_for_variables(
                edge,
                leaf,
                pseudo_obs[(leaf, conditioning)],
                partner,
                pseudo_obs[(partner, conditioning)],
                config={"r": r_all[(tree, col)]},
            )
            pseudo_obs[(leaf, conditioning | {partner})] = _clip_unit(
                leaf_next)
            pseudo_obs[(partner, conditioning | {leaf})] = _clip_unit(
                partner_next)

    for col in range(start_col - 1, -1, -1):
        leaf = int(matrix[d - 1 - col, col])
        top_tree = d - 2 - col
        current = w[:, col].copy()
        for tree in range(top_tree, -1, -1):
            row = d - 2 - col - tree
            partner = int(matrix[row, col])
            conditioning = frozenset(
                int(matrix[source_row, col])
                for source_row in range(row + 1, d - 1 - col)
            )
            current = _clip_unit(_edge_h_inverse_for_variables(
                pair_copulas[(tree, col)],
                leaf,
                current,
                partner,
                pseudo_obs[(partner, conditioning)],
                config={"r": r_all[(tree, col)]},
            ))
            pseudo_obs[(leaf, conditioning)] = current

        for tree in range(top_tree + 1):
            row = d - 2 - col - tree
            partner = int(matrix[row, col])
            conditioning = frozenset(
                int(matrix[source_row, col])
                for source_row in range(row + 1, d - 1 - col)
            )
            leaf_next, partner_next = _edge_h_pair_for_variables(
                pair_copulas[(tree, col)],
                leaf,
                pseudo_obs[(leaf, conditioning)],
                partner,
                pseudo_obs[(partner, conditioning)],
                config={"r": r_all[(tree, col)]},
            )
            pseudo_obs[(leaf, conditioning | {partner})] = _clip_unit(
                leaf_next)
            pseudo_obs[(partner, conditioning | {leaf})] = _clip_unit(
                partner_next)

    out = np.empty((n, d), dtype=np.float64)
    for var in range(d):
        out[:, var] = pseudo_obs[(var, frozenset())]
    return out


def _vine_sample_suffix_given_with_r_python(
        vine, n, r_all, rng, given, start_col, matrix=None,
        pair_copulas=None, *, uniforms=None):
    matrix = vine._natural_order_matrix if matrix is None else matrix
    pair_copulas = (
        vine.pair_copulas if pair_copulas is None else pair_copulas)
    return _sample_suffix_given_with_r_python(
        vine.d,
        n,
        r_all,
        rng,
        given,
        start_col,
        matrix,
        pair_copulas,
        uniforms=uniforms,
    )


def _edge_payload(step, r_all):
    edge_key = tuple(step["edge"])
    payload = r_all[edge_key]
    if not isinstance(payload, dict):
        raise TypeError(
            "reference conditional plan expects edge payloads as "
            "{'edge': pair_copula, 'r': parameter_values} dictionaries"
        )
    return payload["edge"], np.asarray(payload["r"], dtype=np.float64)


def _execute_conditional_plan_python(
        plan, r_all, given, n, rng, *, uniforms=None):
    """Reference executor for a conditional DAG program."""
    n = int(n)
    replay_uniforms = None
    replay_by_variable = False
    if uniforms is not None:
        uniform_count = sum(
            step.get("action") == "sample_uniform" for step in plan)
        supplied = np.asarray(uniforms)
        if supplied.shape == (n, uniform_count):
            replay_uniforms = _prepared_open_unit_draws(
                supplied, (n, uniform_count),
                name="conditional DAG uniforms")
        elif supplied.shape == (n, int(plan.d)):
            replay_uniforms = _prepared_open_unit_draws(
                supplied, (n, int(plan.d)),
                name="conditional DAG uniforms")
            replay_by_variable = True
        else:
            raise ValueError(
                "conditional DAG uniforms must have shape "
                f"{(n, uniform_count)} (draw order) or "
                f"{(n, int(plan.d))} (variable columns), got "
                f"{supplied.shape}"
            )
    replay_index = 0
    pseudo_obs = {
        _node_key(var): np.full(n, float(value), dtype=np.float64)
        for var, value in given.items()
    }

    for step in plan:
        action = step["action"]
        if action == "sample_uniform":
            if replay_uniforms is None:
                values = _open_unit_uniform(rng, size=n)
            elif replay_by_variable:
                values = replay_uniforms[:, int(step["var"])].copy()
            else:
                values = replay_uniforms[:, replay_index].copy()
                replay_index += 1
            pseudo_obs[step["node"]] = values
            continue

        edge, r = _edge_payload(step, r_all)
        source = (
            step["from"] if action == "h_inv" else
            _node_key(step["leaf"], step["cond"])
        )
        known = (
            step["known"] if action == "h_inv" else
            _node_key(step["partner"], step["cond"])
        )
        if action == "h_prop":
            pseudo_obs[step["to"]] = _clip_unit(_edge_h(
                edge,
                pseudo_obs[source],
                pseudo_obs[known],
                config={"r": r},
            ))
        elif action == "h_inv":
            pseudo_obs[step["to"]] = _clip_unit(_edge_h_inverse(
                edge,
                pseudo_obs[source],
                pseudo_obs[known],
                config={"r": r},
            ))
        else:
            raise ValueError(f"unknown reference action {action!r}")

    out = np.empty((n, int(plan.d)), dtype=np.float64)
    missing = []
    for var in range(int(plan.d)):
        node = _node_key(var)
        if node not in pseudo_obs:
            missing.append(var)
        else:
            out[:, var] = pseudo_obs[node]
    if missing:
        raise RuntimeError(
            f"reference plan did not produce base variables {missing}")
    return out


def _sample_dag_given_with_r_python(
        vine, n, r_all, rng, given, plan, pair_copulas, *, uniforms=None):
    missing = sorted(set(plan.edges_used) - set(r_all))
    if missing:
        raise KeyError(
            "RVineCopula._sample_dag_given_with_r: missing predicted "
            f"parameters for DAG edges {missing}"
        )
    payload = {
        key: {"edge": pair_copulas[key], "r": r_all[key]}
        for key in plan.edges_used
    }
    return _execute_conditional_plan_python(
        plan, payload, given, n, rng, uniforms=uniforms)


def _log_pdf_rows_with_r_python(
        vine, observations, r_all, pair_copulas=None, edge_map=None):
    """Reference row-wise R-vine log-density."""
    pair_copulas = (
        vine.pair_copulas if pair_copulas is None else pair_copulas)
    edge_map = vine._edge_map if edge_map is None else edge_map
    pseudo_obs = {
        (var, frozenset()): observations[:, var].copy()
        for var in range(vine.d)
    }
    logp = np.zeros(len(observations), dtype=np.float64)
    for tree, level in enumerate(vine._trees):
        for original_index, (conditioned, conditioning) in enumerate(level):
            key = vine._matrix_key_from_map(tree, original_index, edge_map)
            edge = pair_copulas[key]
            first, second = sorted(conditioned)
            u_first = _clip_unit(pseudo_obs[(first, conditioning)])
            u_second = _clip_unit(pseudo_obs[(second, conditioning)])
            r = np.asarray(r_all[key], dtype=np.float64)
            if len(r) == 1 and len(observations) != 1:
                r = np.full(
                    len(observations), float(r[0]), dtype=np.float64)
            elif len(r) != len(observations):
                raise ValueError(
                    "VineCopula._log_pdf_rows_with_r: parameter path "
                    f"for edge {key} has length {len(r)}, expected 1 "
                    f"or {len(observations)}"
                )
            if not edge_is_independent(edge):
                logp += edge_copula(edge).log_pdf(
                    u_first, u_second, r)
            if tree < vine.d - 2:
                first_next, second_next = _edge_h_pair(
                    edge, u_first, u_second, config={"r": r})
                pseudo_obs[(
                    second, conditioning | {first})] = _clip_unit(second_next)
                pseudo_obs[(
                    first, conditioning | {second})] = _clip_unit(first_next)
    return logp


def _log_likelihood_python(vine, observations):
    """Candidate-composed traversal used to check native likelihood order."""
    u = np.asarray(observations, dtype=np.float64)
    max_active_tree = vine._max_non_independent_tree_level()
    if max_active_tree < 0:
        return 0.0

    pseudo_obs = {
        (variable, frozenset()): u[:, variable].copy()
        for variable in range(vine.d)
    }
    total = 0.0
    for tree, level in enumerate(vine._trees[:max_active_tree + 1]):
        for original, (conditioned, conditioning) in enumerate(level):
            edge = vine.pair_copulas[vine._matrix_key(tree, original)]
            first, second = sorted(conditioned)
            first_values = _clip_unit(pseudo_obs[(first, conditioning)])
            second_values = _clip_unit(pseudo_obs[(second, conditioning)])
            copula = edge_copula(edge)
            result = edge_result(edge)
            parameter = None
            if result is None or not edge_has_dynamic_params(edge):
                parameter = edge_param(edge)

            pair = np.column_stack((first_values, second_values))
            if not edge_is_independent(edge):
                if result is not None and parameter is None:
                    total += strategy_for_result(result).log_likelihood(
                        copula, pair, result)
                else:
                    total += copula.log_likelihood(pair, parameter)

            if tree >= max_active_tree:
                continue
            if parameter is None:
                first_next, second_next = _edge_h_pair(
                    edge, first_values, second_values)
            else:
                parameters = np.full(
                    len(first_values), parameter, dtype=np.float64)
                first_next, second_next = copula.h_pair(
                    first_values, second_values, parameters)
            pseudo_obs[(second, conditioning | {first})] = _clip_unit(
                second_next)
            pseudo_obs[(first, conditioning | {second})] = _clip_unit(
                first_next)
    return float(total)


def _empty_mcmc_diagnostics():
    return {
        "accepted": {}, "proposed": {}, "acceptance_rate": {},
        "accepted_per_chain": {}, "proposals_per_chain": {},
        "acceptance_min": None, "acceptance_mean": None,
        "acceptance_max": None, "low_acceptance_warning": False,
        "insufficient_moves_warning": False,
        "minimum_accepted_moves_per_chain": None,
        "convergence_warning": False, "warning_codes": (),
        "step_unit": "single_coordinate_update", "n_free": 0,
        "n_steps": 0, "burnin_steps": 0, "total_steps": 0,
        "completed_sweeps": 0, "partial_sweep_steps": 0,
    }


def _mcmc_diagnostics(
        free_vars, accepted, proposed, n, n_steps, burnin_steps):
    free_vars = [int(var) for var in free_vars]
    accepted = {int(var): int(accepted[var]) for var in free_vars}
    proposed = {int(var): int(proposed[var]) for var in free_vars}
    rates = {
        var: accepted[var] / proposed[var] if proposed[var] else 0.0
        for var in free_vars
    }
    accepted_per_chain = {
        var: accepted[var] / n if n else 0.0 for var in free_vars}
    proposals_per_chain = {
        var: proposed[var] / n if n else 0.0 for var in free_vars}
    values = np.asarray(tuple(rates.values()), dtype=np.float64)
    has_proposals = any(proposed[var] > 0 for var in free_vars)
    acceptance_min = float(np.min(values)) if has_proposals else None
    acceptance_mean = float(np.mean(values)) if has_proposals else None
    acceptance_max = float(np.max(values)) if has_proposals else None
    low_acceptance = bool(
        has_proposals and acceptance_min is not None
        and acceptance_min < 0.02)
    minimum_moves = (
        float(min(accepted_per_chain.values())) if has_proposals else None)
    insufficient_moves = minimum_moves is not None and minimum_moves < 5.0
    warning_codes = []
    if low_acceptance:
        warning_codes.append("low_acceptance")
    if insufficient_moves:
        warning_codes.append("insufficient_accepted_moves")
    total_steps = int(burnin_steps) + int(n_steps)
    return {
        "accepted": accepted,
        "proposed": proposed,
        "acceptance_rate": rates,
        "accepted_per_chain": accepted_per_chain,
        "proposals_per_chain": proposals_per_chain,
        "acceptance_min": acceptance_min,
        "acceptance_mean": acceptance_mean,
        "acceptance_max": acceptance_max,
        "low_acceptance_warning": low_acceptance,
        "insufficient_moves_warning": insufficient_moves,
        "minimum_accepted_moves_per_chain": minimum_moves,
        "convergence_warning": bool(warning_codes),
        "warning_codes": tuple(warning_codes),
        "step_unit": "single_coordinate_update",
        "n_free": len(free_vars),
        "n_steps": int(n_steps),
        "burnin_steps": int(burnin_steps),
        "total_steps": total_steps,
        "completed_sweeps": total_steps // len(free_vars),
        "partial_sweep_steps": total_steps % len(free_vars),
    }


def _sample_arbitrary_given_mcmc_python(
        vine, n, r_all, rng, given, initial=None, n_steps=None,
        burnin_steps=None, *, initial_uniforms=None, random_draws=None,
        step_offset=0):
    """Reference Metropolis-within-Gibbs conditional sampler."""
    free_vars = [var for var in range(vine.d) if var not in given]
    if not free_vars:
        out = np.empty((n, vine.d), dtype=np.float64)
        for var in range(vine.d):
            out[:, var] = given[var]
        return out, _empty_mcmc_diagnostics()

    if initial is None:
        current = (
            _open_unit_uniform(rng, size=(n, vine.d))
            if initial_uniforms is None else
            _prepared_open_unit_draws(
                initial_uniforms,
                (n, vine.d),
                name="MCMC initial uniforms",
            ).copy()
        )
        for var, value in given.items():
            current[:, var] = value
    else:
        if initial_uniforms is not None:
            raise ValueError(
                "MCMC initial_uniforms cannot be supplied with initial")
        current = np.asarray(initial, dtype=np.float64).copy()
        for var, value in given.items():
            current[:, var] = value

    current_logp = _log_pdf_rows_with_r_python(vine, current, r_all)
    n_steps = max(80, 30 * len(free_vars)) if n_steps is None else int(n_steps)
    burnin_steps = (
        max(40, 10 * len(free_vars))
        if burnin_steps is None else int(burnin_steps)
    )
    total_steps = burnin_steps + n_steps
    step_offset = int(step_offset)
    if step_offset < 0:
        raise ValueError("MCMC step_offset must be non-negative")
    replay_draws = None
    if random_draws is not None:
        replay_draws = _prepared_open_unit_draws(
            random_draws,
            (total_steps, n, 2),
            name="MCMC interleaved random draws",
        )
    accepted = {int(var): 0 for var in free_vars}
    proposed = {int(var): 0 for var in free_vars}

    for step_index in range(total_steps):
        var = free_vars[(step_offset + step_index) % len(free_vars)]
        proposal = current.copy()
        proposal[:, var] = (
            _open_unit_uniform(rng, size=n)
            if replay_draws is None else replay_draws[step_index, :, 0]
        )
        proposal_logp = _log_pdf_rows_with_r_python(vine, proposal, r_all)
        acceptance_uniforms = (
            rng.uniform(PSEUDO_OBS_EPS, 1.0, size=n)
            if replay_draws is None else replay_draws[step_index, :, 1]
        )
        accept = np.log(acceptance_uniforms) < proposal_logp - current_logp
        if np.any(accept):
            current[accept, var] = proposal[accept, var]
            current_logp[accept] = proposal_logp[accept]
        accepted[int(var)] += int(np.sum(accept))
        proposed[int(var)] += int(n)

    return clip_pseudo_observations(current), _mcmc_diagnostics(
        free_vars, accepted, proposed, n, n_steps, burnin_steps)


def _rvine_rosenblatt_transform_python(
        vine, observations, K=300, grid_range=5.0, *, vine_type=None):
    """Reference Rosenblatt traversal for a fitted R-vine."""
    if vine_type is None:
        vine_type = getattr(vine, "vine_type", "rvine")
    if vine_type not in {"cvine", "dvine", "rvine"}:
        raise ValueError(
            "vine_type must be 'cvine', 'dvine' or 'rvine', "
            f"got {vine_type!r}")
    if getattr(vine, "matrix", None) is None:
        raise ValueError("Fit the vine first")
    u = np.asarray(observations, dtype=np.float64)
    rows, dimension = u.shape
    if dimension != vine.d:
        raise ValueError(
            f"u has d={dimension}, but fitted vine has d={vine.d}")

    if dimension == 2:
        edge = vine.pair_copulas[(0, 0)]
        first = clip_pseudo_observations(u[:, 0])
        second = clip_pseudo_observations(u[:, 1])
        pair = np.column_stack((first, second))
        result = np.empty((rows, dimension), dtype=np.float64)
        result[:, 0] = first
        result[:, 1] = clip_pseudo_observations(_edge_h(
            edge,
            second,
            first,
            u_pair=pair,
            K=K,
            grid_range=grid_range,
        ))
        return clip_rosenblatt_output(result)

    matrix = vine.matrix
    pseudo = {
        (var, frozenset()): clip_pseudo_observations(u[:, var].copy())
        for var in range(dimension)
    }
    result = np.empty((rows, dimension), dtype=np.float64)
    last_var = int(matrix[0, dimension - 1])
    result[:, dimension - 1] = pseudo[(last_var, frozenset())]

    for col in range(dimension - 2, -1, -1):
        leaf = int(matrix[dimension - 1 - col, col])
        top_tree = dimension - 2 - col
        current = pseudo[(leaf, frozenset())]
        for tree in range(top_tree + 1):
            row = dimension - 2 - col - tree
            partner = int(matrix[row, col])
            conditioning = frozenset(
                int(matrix[source_row, col])
                for source_row in range(row + 1, dimension - 1 - col)
            )
            leaf_value = pseudo.get((leaf, conditioning))
            partner_value = pseudo.get((partner, conditioning))
            if leaf_value is None or partner_value is None:
                missing = "leaf" if leaf_value is None else "partner"
                raise RuntimeError(
                    f"Missing {missing} pseudo-observation during "
                    f"Rosenblatt: column={col}, tree={tree}")
            leaf_next, partner_next = _edge_h_pair_for_variables(
                vine.pair_copulas[(tree, col)],
                leaf,
                leaf_value,
                partner,
                partner_value,
                K=K,
                grid_range=grid_range,
            )
            current = clip_pseudo_observations(leaf_next)
            pseudo[(leaf, conditioning | {partner})] = current
            pseudo[(partner, conditioning | {leaf})] = (
                clip_pseudo_observations(partner_next))
        result[:, col] = current
    return clip_rosenblatt_output(result)
