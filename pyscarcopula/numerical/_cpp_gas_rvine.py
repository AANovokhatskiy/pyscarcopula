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
from pyscarcopula.vine._rvine_sampling_plan import (
    build_rvine_sampling_plan,
)


def _native_plan(module, traversal_plan):
    """Serialize the canonical Python traversal plan for the C++ executor."""
    plan = module.RvineTraversalPlan()
    plan.dimension = traversal_plan.dimension
    plan.node_count = len(traversal_plan.node_keys)
    plan.last_uniform_column = traversal_plan.last_uniform_column
    plan.last_output_node = traversal_plan.last_output_node
    for name in (
        "output_nodes",
        "column_uniforms",
        "inverse_offsets",
        "inverse_edges",
        "inverse_partner_nodes",
        "inverse_output_nodes",
        "inverse_transposed",
        "forward_offsets",
        "forward_edges",
        "forward_leaf_nodes",
        "forward_partner_nodes",
        "forward_leaf_output_nodes",
        "forward_partner_output_nodes",
        "forward_transposed",
        "update_u1_nodes",
        "update_u2_nodes",
    ):
        setattr(plan, name, list(getattr(traversal_plan, name)))
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


def sample(
        vine,
        n,
        rng,
        active_keys,
        max_active_tree,
        traversal_plan=None):
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
    if traversal_plan is None:
        traversal_plan = build_rvine_sampling_plan(
            vine.d,
            vine.matrix,
            vine.trees,
            vine._edge_map,
            active_keys,
            max_active_tree,
        )
    elif traversal_plan.active_keys != tuple(active_keys):
        raise ValueError(
            "R-vine traversal plan does not match active edge keys")
    plan = _native_plan(module, traversal_plan)
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
