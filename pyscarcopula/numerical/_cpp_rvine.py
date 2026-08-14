"""Shared serializers and capability gates for native R-vine runtimes.

The topology builders remain in :mod:`pyscarcopula.vine`; this module owns
only the flat pybind11 boundary.  It deliberately does not dispatch a generic
runtime until the corresponding migration phase adds a native entry point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyscarcopula.numerical import _cpp_copula
from pyscarcopula.numerical._cpp_extension import (
    CppError,
    CppUnsupported,
    cpp_status_name,
)
from pyscarcopula.vine._edge_adapter import edge_copula


_NATIVE_EQUIVALENT_ATTR = "__pyscarcopula_native_rvine__"


def _builtin_copula_types():
    from pyscarcopula.copula.clayton import ClaytonCopula
    from pyscarcopula.copula.elliptical import BivariateGaussianCopula
    from pyscarcopula.copula.frank import FrankCopula
    from pyscarcopula.copula.gumbel import GumbelCopula
    from pyscarcopula.copula.independent import IndependentCopula
    from pyscarcopula.copula.joe import JoeCopula

    return (
        IndependentCopula,
        ClaytonCopula,
        GumbelCopula,
        JoeCopula,
        FrankCopula,
        BivariateGaussianCopula,
    )


def native_copula_supported(copula) -> bool:
    """Return whether a pair copula has native-equivalent point semantics.

    Built-ins are accepted by exact type.  A subclass must opt in on its own
    class body so an inherited marker cannot accidentally bypass overrides.
    """
    if type(copula) in _builtin_copula_types():
        return True
    return vars(type(copula)).get(_NATIVE_EQUIVALENT_ATTR) is True


def native_edges_supported(pair_copulas, active_keys) -> bool:
    """Check an edge collection without constructing native objects."""
    return all(
        native_copula_supported(edge_copula(pair_copulas[key]))
        for key in active_keys
    )


def compile_copula_spec(module, copula):
    """Serialize one exact built-in or explicitly opted-in pair copula."""
    if not native_copula_supported(copula):
        raise CppUnsupported(
            "native R-vine execution requires an exact built-in copula type "
            "or an explicit native-equivalent opt-in; got "
            f"{type(copula).__name__}"
        )
    return _cpp_copula.make_copula_ops_spec(module, copula)


def compile_traversal_plan(module, traversal_plan):
    """Serialize a validated canonical Python traversal plan."""
    plan = module.RVineTraversalPlan()
    plan.dimension = int(traversal_plan.dimension)
    plan.node_count = len(traversal_plan.node_keys)
    plan.last_uniform_column = int(traversal_plan.last_uniform_column)
    plan.last_output_node = int(traversal_plan.last_output_node)
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
        setattr(plan, name, [int(value) for value in getattr(
            traversal_plan, name)])
    return plan


def _node_key(variable, conditioning=()):
    return int(variable), frozenset(int(value) for value in conditioning)


def compile_conditional_plan(module, plan, active_keys, given):
    """Compile a suffix or DAG action program into one flat native format."""
    active_keys = tuple(tuple(int(value) for value in key) for key in active_keys)
    edge_indices = {key: index for index, key in enumerate(active_keys)}
    given_variables = sorted(int(variable) for variable in given)
    nodes = {}
    node_sources = []
    node_source_indices = []

    def node_id(key):
        if key not in nodes:
            nodes[key] = len(nodes)
            node_sources.append(0)  # RVineNodeSource.COMPUTED
            node_source_indices.append(-1)
        return nodes[key]

    def mark_source(key, source, source_index):
        node = node_id(key)
        if node_sources[node] != 0 or node_source_indices[node] != -1:
            raise ValueError(
                f"compile_conditional_plan: duplicate source for node {key!r}")
        node_sources[node] = source
        node_source_indices[node] = int(source_index)
        return node

    given_nodes = [
        mark_source(_node_key(variable), 1, variable)
        for variable in given_variables
    ]
    uniform_nodes = []
    opcodes = []
    operation_edges = []
    input1_nodes = []
    input2_nodes = []
    output1_nodes = []
    output2_nodes = []
    transposed = []

    for step in plan:
        action = step.get("action")
        if action == "sample_uniform":
            uniform_column = int(step.get("column", step["var"]))
            uniform_nodes.append(mark_source(
                step["node"], 2, uniform_column))
            continue
        if action == "copy":
            opcodes.append(4)
            operation_edges.append(-1)
            input1_nodes.append(node_id(step["from"]))
            input2_nodes.append(-1)
            output1_nodes.append(node_id(step["to"]))
            output2_nodes.append(-1)
            transposed.append(0)
            continue
        if action not in {"h_prop", "h_pair", "h_inv"}:
            raise ValueError(
                "compile_conditional_plan: unsupported action "
                f"{action!r}"
            )
        edge_key = tuple(int(value) for value in step["edge"])
        try:
            edge_index = edge_indices[edge_key]
        except KeyError as exc:
            raise ValueError(
                "compile_conditional_plan: plan references edge "
                f"{edge_key} outside active_keys"
            ) from exc
        leaf = int(step["leaf"])
        partner = int(step["partner"])
        if action == "h_prop":
            source = _node_key(leaf, step["cond"])
            known = _node_key(partner, step["cond"])
            opcode = 1
            output1 = step["to"]
            output2 = None
        elif action == "h_pair":
            source = step["first"]
            known = step["second"]
            opcode = 2
            output1 = step["first_to"]
            output2 = step["second_to"]
        else:
            source = step["from"]
            known = step["known"]
            opcode = 3
            output1 = step["to"]
            output2 = None
        opcodes.append(opcode)
        operation_edges.append(edge_index)
        input1_nodes.append(node_id(source))
        input2_nodes.append(node_id(known))
        output1_nodes.append(node_id(output1))
        output2_nodes.append(
            -1 if output2 is None else node_id(output2))
        transposed.append(int(leaf > partner))

    output_nodes = [node_id(_node_key(variable)) for variable in range(
        int(plan.d))]
    used_edges = sorted(set(
        edge for edge in operation_edges if edge >= 0))

    native = module.RVineConditionalPlan()
    native.dimension = int(plan.d)
    native.node_count = len(nodes)
    native.given_variables = given_variables
    native.given_nodes = given_nodes
    native.uniform_nodes = uniform_nodes
    native.node_sources = node_sources
    native.node_source_indices = node_source_indices
    native.opcodes = opcodes
    native.edge_indices = operation_edges
    native.input1_nodes = input1_nodes
    native.input2_nodes = input2_nodes
    native.output1_nodes = output1_nodes
    native.output2_nodes = output2_nodes
    native.transposed = transposed
    native.output_nodes = output_nodes
    native.used_edges = used_edges
    if not module.validate_rvine_conditional_plan(native, len(active_keys)):
        raise ValueError(
            "compile_conditional_plan: native plan validation failed")
    return native


def compile_density_plan(
        module,
        dimension,
        trees,
        edge_map,
        active_keys,
        *,
        residual_node_keys=()):
    """Compile the canonical tree-order density traversal into flat arrays."""
    dimension = int(dimension)
    active_keys = tuple(tuple(int(value) for value in key) for key in active_keys)
    edge_indices = {key: index for index, key in enumerate(active_keys)}
    reverse_edge_map = {
        (int(tree), int(original)): (int(tree), int(column))
        for (tree, column), original in edge_map.items()
    }
    nodes = {}

    def node_id(key):
        normalized = _node_key(key[0], key[1])
        if normalized not in nodes:
            nodes[normalized] = len(nodes)
        return nodes[normalized]

    input_nodes = [node_id(_node_key(variable)) for variable in range(dimension)]
    operation_edges = []
    input1_nodes = []
    input2_nodes = []
    output1_nodes = []
    output2_nodes = []
    transposed = []

    for tree, level in enumerate(trees):
        for original, (conditioned, conditioning) in enumerate(level):
            try:
                edge_key = reverse_edge_map[(tree, original)]
                edge_index = edge_indices[edge_key]
            except KeyError as exc:
                raise ValueError(
                    "compile_density_plan: cannot resolve tree edge "
                    f"{(tree, original)}"
                ) from exc
            first, second = sorted(int(value) for value in conditioned)
            conditioning = frozenset(int(value) for value in conditioning)
            operation_edges.append(edge_index)
            input1_nodes.append(node_id(_node_key(first, conditioning)))
            input2_nodes.append(node_id(_node_key(second, conditioning)))
            transposed.append(0)
            if tree < dimension - 2:
                output1_nodes.append(node_id(
                    _node_key(first, conditioning | {second})))
                output2_nodes.append(node_id(
                    _node_key(second, conditioning | {first})))
            else:
                output1_nodes.append(-1)
                output2_nodes.append(-1)

    residual_nodes = [node_id(key) for key in residual_node_keys]
    native = module.RVineDensityPlan()
    native.dimension = dimension
    native.node_count = len(nodes)
    native.input_nodes = input_nodes
    native.edge_indices = operation_edges
    native.input1_nodes = input1_nodes
    native.input2_nodes = input2_nodes
    native.output1_nodes = output1_nodes
    native.output2_nodes = output2_nodes
    native.transposed = transposed
    native.residual_nodes = residual_nodes
    native.used_edges = sorted(set(operation_edges))
    if not module.validate_rvine_density_plan(native, len(active_keys)):
        raise ValueError("compile_density_plan: native plan validation failed")
    return native


@dataclass(frozen=True)
class RVineParameterPack:
    """Owning Python buffers backing one non-owning native parameter pack."""

    scalar_parameters: np.ndarray
    row_parameters: np.ndarray
    n_rows: int

    @property
    def row_parameter_columns(self):
        return int(self.row_parameters.shape[1])


def _parameter_path(value, n_rows, edge_key):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise TypeError(
            f"R-vine parameter path for edge {edge_key} must contain real values"
        )
    path = np.asarray(raw, dtype=np.float64)
    if path.ndim == 0:
        path = path.reshape(1)
    if path.ndim != 1 or len(path) not in {1, n_rows}:
        raise ValueError(
            f"R-vine parameter path for edge {edge_key} must have length 1 "
            f"or {n_rows}, got shape {path.shape}"
        )
    if not np.all(np.isfinite(path)):
        raise ValueError(
            f"R-vine parameter path for edge {edge_key} must be finite"
        )
    return path


def compile_edge_specs(
        module,
        pair_copulas,
        active_keys,
        parameter_paths,
        n_rows,
        *,
        parameter_sources=None):
    """Pack edge metadata plus separate scalar and per-row parameter buffers."""
    n_rows = int(n_rows)
    if n_rows < 0:
        raise ValueError("R-vine parameter row count must be non-negative")
    declared_sources = {
        tuple(int(value) for value in key): source
        for key, source in (parameter_sources or {}).items()
    }
    scalar_parameters = []
    row_parameters = []
    native_edges = []
    independent_type = _builtin_copula_types()[0]

    for key in active_keys:
        key = tuple(int(value) for value in key)
        edge = pair_copulas[key]
        copula = edge_copula(edge)
        native = module.RVineEdgeSpec()
        native.copula = compile_copula_spec(module, copula)
        parameter_free = isinstance(copula, independent_type)
        native.parameter_free = parameter_free
        if parameter_free:
            native.parameter_source = module.RVineParameterSource.NONE
            native.parameter_index = -1
        else:
            if key not in parameter_paths:
                raise KeyError(f"missing R-vine parameter path for edge {key}")
            raw_parameter = parameter_paths[key]
            raw_is_scalar = np.asarray(raw_parameter).ndim == 0
            path = _parameter_path(raw_parameter, n_rows, key)
            declared = declared_sources.get(key)
            if declared is None:
                if n_rows == 1 and len(path) == 1 and not raw_is_scalar:
                    raise ValueError(
                        "R-vine parameter source is ambiguous for one row; "
                        f"declare parameter_sources[{key!r}] as 'scalar' "
                        "or 'row_path'"
                    )
                is_row_path = len(path) != 1
            elif declared in {
                    "scalar", "SCALAR",
                    module.RVineParameterSource.SCALAR}:
                if len(path) != 1:
                    raise ValueError(
                        f"scalar R-vine parameter for edge {key} must have "
                        f"length 1, got {len(path)}")
                is_row_path = False
            elif declared in {
                    "row_path", "ROW_PATH",
                    module.RVineParameterSource.ROW_PATH}:
                if len(path) != n_rows:
                    raise ValueError(
                        f"row-path R-vine parameter for edge {key} must have "
                        f"length {n_rows}, got {len(path)}")
                is_row_path = True
            else:
                raise ValueError(
                    f"invalid R-vine parameter source for edge {key}: "
                    f"{declared!r}")
            if not is_row_path:
                native.parameter_source = module.RVineParameterSource.SCALAR
                native.parameter_index = len(scalar_parameters)
                scalar_parameters.append(float(path[0]))
            else:
                native.parameter_source = module.RVineParameterSource.ROW_PATH
                native.parameter_index = len(row_parameters)
                row_parameters.append(path)
        native_edges.append(native)

    scalar_buffer = np.ascontiguousarray(
        scalar_parameters, dtype=np.float64)
    if row_parameters:
        row_buffer = np.ascontiguousarray(
            np.column_stack(row_parameters), dtype=np.float64)
    else:
        row_buffer = np.empty((n_rows, 0), dtype=np.float64)
    return native_edges, RVineParameterPack(
        scalar_parameters=scalar_buffer,
        row_parameters=row_buffer,
        n_rows=n_rows,
    )


def raise_for_status(result, operation):
    """Translate a structured native R-vine status without fallback."""
    status = int(result["status"])
    if status == 0:
        return
    row = int(result.get("failure_row", -1))
    edge = int(result.get("failure_edge", -1))
    message = (
        f"C++ {operation} failed: status={status} "
        f"({cpp_status_name(status)})"
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


__all__: list[str] = []
