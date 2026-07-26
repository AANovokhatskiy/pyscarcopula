"""Native sequential R-vine sampler for fitted GAS edge states."""

from __future__ import annotations

import numpy as np

from pyscarcopula._types import GASResult
from pyscarcopula.numerical import _cpp_copula, _cpp_extension, _cpp_gas
from pyscarcopula.numerical._cpp_extension import (
    CppError,
    CppUnsupported,
    cpp_status_name,
)
from pyscarcopula.vine._edge_adapter import edge_copula, edge_result
from pyscarcopula.vine._helpers import _open_unit_uniform
from pyscarcopula.vine._rvine_edges import (
    _edge_r_for_sample,
    _edge_requires_stepwise_sample,
)


def _node_id(nodes, variable, conditioning):
    key = (int(variable), frozenset(int(item) for item in conditioning))
    try:
        return nodes[key]
    except KeyError:
        value = len(nodes)
        nodes[key] = value
        return value


def _build_plan(module, vine, active_keys, max_active_tree):
    edge_indices = {key: index for index, key in enumerate(active_keys)}
    nodes = {}
    plan = module.GasRvinePlan()
    plan.dimension = int(vine.d)
    plan.last_uniform_column = int(vine.d - 1)
    last_var = int(vine.matrix[0, vine.d - 1])
    plan.last_output_node = _node_id(nodes, last_var, ())

    column_uniforms = []
    inverse_offsets = [0]
    inverse_edges = []
    inverse_partner_nodes = []
    inverse_output_nodes = []
    inverse_transposed = []
    forward_offsets = [0]
    forward_edges = []
    forward_leaf_nodes = []
    forward_partner_nodes = []
    forward_leaf_output_nodes = []
    forward_partner_output_nodes = []
    forward_transposed = []

    d = vine.d
    M = vine.matrix
    for col in range(d - 2, -1, -1):
        leaf = int(M[d - 1 - col, col])
        active_top = min(d - 2 - col, max_active_tree)
        column_uniforms.append(int(col))

        for t in range(active_top, -1, -1):
            row = d - 2 - col - t
            partner = int(M[row, col])
            conditioning = frozenset(
                int(M[r, col])
                for r in range(row + 1, d - 1 - col)
            )
            inverse_edges.append(edge_indices[(t, col)])
            inverse_partner_nodes.append(
                _node_id(nodes, partner, conditioning))
            inverse_output_nodes.append(
                _node_id(nodes, leaf, conditioning))
            inverse_transposed.append(int(leaf > partner))
        inverse_offsets.append(len(inverse_edges))

        for t in range(active_top + 1):
            row = d - 2 - col - t
            partner = int(M[row, col])
            conditioning = frozenset(
                int(M[r, col])
                for r in range(row + 1, d - 1 - col)
            )
            forward_edges.append(edge_indices[(t, col)])
            forward_leaf_nodes.append(_node_id(nodes, leaf, conditioning))
            forward_partner_nodes.append(
                _node_id(nodes, partner, conditioning))
            forward_leaf_output_nodes.append(
                _node_id(nodes, leaf, conditioning | {partner}))
            forward_partner_output_nodes.append(
                _node_id(nodes, partner, conditioning | {leaf}))
            forward_transposed.append(int(leaf > partner))
        forward_offsets.append(len(forward_edges))

    update_u1_nodes = []
    update_u2_nodes = []
    for t, col in active_keys:
        orig_idx = vine._edge_map[(t, col)]
        conditioned, conditioning = vine.trees[t][orig_idx]
        v1, v2 = sorted(conditioned)
        update_u1_nodes.append(_node_id(nodes, v1, conditioning))
        update_u2_nodes.append(_node_id(nodes, v2, conditioning))

    plan.output_nodes = [_node_id(nodes, var, ()) for var in range(d)]
    plan.column_uniforms = column_uniforms
    plan.inverse_offsets = inverse_offsets
    plan.inverse_edges = inverse_edges
    plan.inverse_partner_nodes = inverse_partner_nodes
    plan.inverse_output_nodes = inverse_output_nodes
    plan.inverse_transposed = inverse_transposed
    plan.forward_offsets = forward_offsets
    plan.forward_edges = forward_edges
    plan.forward_leaf_nodes = forward_leaf_nodes
    plan.forward_partner_nodes = forward_partner_nodes
    plan.forward_leaf_output_nodes = forward_leaf_output_nodes
    plan.forward_partner_output_nodes = forward_partner_output_nodes
    plan.forward_transposed = forward_transposed
    plan.update_u1_nodes = update_u1_nodes
    plan.update_u2_nodes = update_u2_nodes
    # output_nodes can introduce base nodes only if the matrix plan is invalid.
    plan.node_count = len(nodes)
    return plan


def _native_edges(module, vine, active_keys):
    native_edges = []
    dynamic = []
    for key in active_keys:
        edge = vine.pair_copulas[key]
        result = edge_result(edge)
        copula = edge_copula(edge)
        native = module.GasRvineEdge()
        native.copula = _cpp_copula.make_copula_ops_spec(module, copula)
        is_dynamic = isinstance(result, GASResult)
        native.dynamic = is_dynamic
        if is_dynamic:
            params = result.params
            native.gas_params = _cpp_gas._params(
                module, params.omega, params.gamma, params.beta)
            native.gas_config = _cpp_gas._config(
                module,
                result.scaling,
                result.score_eps,
            )
        native_edges.append(native)
        dynamic.append(is_dynamic)
    return native_edges, dynamic


def sample(vine, n, rng, active_keys, max_active_tree):
    """Return a native GAS R-vine sample, or ``None`` when unsupported."""
    module = _cpp_extension.load()
    if not hasattr(module, "gas_rvine_sample"):
        return None
    for key in active_keys:
        edge = vine.pair_copulas[key]
        if (
            not isinstance(edge_result(edge), GASResult)
            and _edge_requires_stepwise_sample(edge)
        ):
            return None
    try:
        native_edges, dynamic = _native_edges(module, vine, active_keys)
    except CppUnsupported:
        return None
    if not any(dynamic):
        return None

    parameter_paths = np.zeros((n, len(active_keys)), dtype=np.float64)
    for edge_index, key in enumerate(active_keys):
        if not dynamic[edge_index]:
            parameter_paths[:, edge_index] = _edge_r_for_sample(
                vine.pair_copulas[key], n, rng)

    uniforms = np.ascontiguousarray(
        _open_unit_uniform(rng, size=(n, vine.d)),
        dtype=np.float64,
    )
    plan = _build_plan(module, vine, active_keys, max_active_tree)
    result = module.gas_rvine_sample(
        native_edges,
        plan,
        uniforms,
        parameter_paths,
    )
    status = int(result["status"])
    if status != 0:
        row = int(result["failure_row"])
        edge = int(result["failure_edge"])
        message = (
            "C++ GAS R-vine sample failed: "
            f"status={status} ({cpp_status_name(status)})"
        )
        if row >= 0:
            message += f", row={row}"
        if edge >= 0:
            message += f", edge={edge}"
        if status in (2, 6):
            raise ValueError(message)
        if status in (3, 4, 5):
            raise CppUnsupported(message)
        if status == 7:
            raise FloatingPointError(message)
        raise CppError(message)
    values = np.asarray(result["values"], dtype=np.float64)
    return values.reshape(n, vine.d)


__all__ = ["sample"]
