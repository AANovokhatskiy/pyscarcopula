"""Shared serializers and capability gates for native R-vine runtimes.

The topology builders remain in :mod:`pyscarcopula.vine`; this module owns
only the flat pybind11 boundary.  Random numbers and parameter trajectories
remain Python-owned; native entry points receive validated contiguous buffers.
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
from pyscarcopula.vine._edge_adapter import (
    edge_copula,
    edge_has_dynamic_params,
    edge_is_independent,
    edge_param,
    edge_result,
)
from pyscarcopula.vine._helpers import (
    _open_unit_uniform,
    _prepared_open_unit_draws,
)
from pyscarcopula._constants import PSEUDO_OBS_EPS
from pyscarcopula._utils import clip_pseudo_observations


_NATIVE_EQUIVALENT_ATTR = "__pyscarcopula_native_rvine__"
_DEFAULT_MCMC_DRAW_MEMORY_BUDGET_BYTES = 64 * 1024 * 1024


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
        # The preserved arbitrary-DAG oracle calls the fitted edge directly
        # in plan argument order.  Suffix actions instead use the explicit
        # variable-aware helpers and therefore transpose when leaf > partner.
        is_dag_action = action == "h_prop" or (
            action == "h_inv" and "cond" in step
        )
        transposed.append(0 if is_dag_action else int(leaf > partner))

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


def conditional_active_keys(plan):
    """Return the sorted matrix-edge keys actually referenced by a plan."""
    return tuple(sorted({
        tuple(int(value) for value in step["edge"])
        for step in plan
        if step.get("action") in {"h_prop", "h_pair", "h_inv"}
    }))


def density_active_keys(trees, edge_map):
    """Return matrix-edge keys in the Python density oracle's tree order."""
    reverse_edge_map = {
        (int(tree), int(original)): (int(tree), int(column))
        for (tree, column), original in edge_map.items()
    }
    keys = []
    for tree, level in enumerate(trees):
        for original, _edge in enumerate(level):
            try:
                keys.append(reverse_edge_map[(tree, original)])
            except KeyError as exc:
                raise ValueError(
                    "density_active_keys: cannot resolve tree edge "
                    f"{(tree, original)}"
                ) from exc
    return tuple(keys)


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
    residual_node_keys = tuple(
        _node_key(key[0], key[1]) for key in residual_node_keys)
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
            # The public d=2 oracle calls the original rotated copula as
            # h(u1 | u0), rather than requesting the variable-aware second
            # direction from h_pair. Swapping h_pair orientation preserves
            # that historical 90/270-degree rotation behavior.
            transposed.append(int(dimension == 2 and bool(
                residual_node_keys)))
            if tree < dimension - 2 or residual_node_keys:
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


def rosenblatt_residual_node_keys(matrix):
    """Return residual nodes in the preserved R-vine output-column order."""
    normalized = np.asarray(matrix, dtype=int)
    if (
            normalized.ndim != 2
            or normalized.shape[0] != normalized.shape[1]
            or normalized.shape[0] < 2):
        raise ValueError(
            "R-vine Rosenblatt matrix must be square with dimension >= 2")
    dimension = int(normalized.shape[0])
    if dimension == 2:
        # The preserved public oracle intentionally follows the legacy
        # C-vine order in two dimensions, independently of matrix peel order.
        return (
            _node_key(0),
            _node_key(1, (0,)),
        )
    keys = []
    for column in range(dimension - 1):
        leaf_row = dimension - 1 - column
        leaf = int(normalized[leaf_row, column])
        conditioning = frozenset(
            int(normalized[row, column]) for row in range(leaf_row))
        keys.append(_node_key(leaf, conditioning))
    keys.append(_node_key(int(normalized[0, dimension - 1])))
    return tuple(keys)


@dataclass(frozen=True)
class RVineParameterPack:
    """Owning Python buffers backing one non-owning native parameter pack."""

    scalar_parameters: np.ndarray
    row_parameters: np.ndarray
    n_rows: int

    @property
    def row_parameter_columns(self):
        return int(self.row_parameters.shape[1])


@dataclass(frozen=True)
class _RVineScalarRequestPack:
    """Request-owned scalar buffer reusable across bounded row batches."""

    key: tuple
    scalar_parameters: np.ndarray


def _scalar_request_key(active_keys, parameter_sources, native_edges):
    sources = tuple(sorted(
        (tuple(int(value) for value in key), str(source))
        for key, source in (parameter_sources or {}).items()
    ))
    return (
        tuple(active_keys),
        sources,
        tuple(id(edge) for edge in (native_edges or ())),
    )


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


def _validate_parameter_domain(copula, path, edge_key):
    """Validate the mathematical domain of a native pair-copula parameter."""
    if not native_copula_supported(copula):
        # Custom Python semantics remain owned by the fallback implementation.
        return

    from pyscarcopula.copula.clayton import ClaytonCopula
    from pyscarcopula.copula.elliptical import BivariateGaussianCopula
    from pyscarcopula.copula.frank import FrankCopula
    from pyscarcopula.copula.gumbel import GumbelCopula
    from pyscarcopula.copula.independent import IndependentCopula
    from pyscarcopula.copula.joe import JoeCopula

    values = np.asarray(path, dtype=np.float64)
    if isinstance(copula, IndependentCopula):
        return
    if isinstance(copula, (ClaytonCopula, FrankCopula)):
        valid = values > 0.0
        domain = "(0, +inf)"
    elif isinstance(copula, (GumbelCopula, JoeCopula)):
        valid = values >= 1.0
        domain = "[1, +inf)"
    elif isinstance(copula, BivariateGaussianCopula):
        valid = (values > -1.0) & (values < 1.0)
        domain = "(-1, 1)"
    else:  # pragma: no cover - guarded by native_copula_supported
        return
    if not np.all(valid):
        invalid = float(values[np.flatnonzero(~valid)[0]])
        raise ValueError(
            f"R-vine parameter path for edge {edge_key} must lie in "
            f"{domain}; got {invalid!r}"
        )


def compile_edge_specs(
        module,
        pair_copulas,
        active_keys,
        parameter_paths,
        n_rows,
        *,
        parameter_sources=None,
        native_edges=None):
    """Pack edge metadata plus separate scalar and per-row parameter buffers."""
    n_rows = int(n_rows)
    if n_rows < 0:
        raise ValueError("R-vine parameter row count must be non-negative")
    active_keys = tuple(
        tuple(int(value) for value in key) for key in active_keys)
    cached_edges = None if native_edges is None else tuple(native_edges)
    if cached_edges is not None:
        if len(cached_edges) != len(active_keys):
            raise ValueError(
                "cached native R-vine edge count does not match active_keys")
    declared_sources = {
        tuple(int(value) for value in key): source
        for key, source in (parameter_sources or {}).items()
    }
    scalar_parameters = []
    row_parameters = []
    compiled_edges = []
    for edge_position, key in enumerate(active_keys):
        edge = pair_copulas[key]
        copula = edge_copula(edge)
        cached_native = (
            None if cached_edges is None else cached_edges[edge_position])
        native = (
            module.RVineEdgeSpec()
            if cached_native is None
            else cached_native
        )
        if cached_native is None:
            native.copula = compile_copula_spec(module, copula)
        parameter_free = edge_is_independent(edge)
        if cached_native is None:
            native.parameter_free = parameter_free
        elif bool(native.parameter_free) != parameter_free:
            raise ValueError(
                f"cached native R-vine edge metadata is stale for edge {key}")
        if parameter_free:
            source = module.RVineParameterSource.NONE
            parameter_index = -1
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
                source = module.RVineParameterSource.SCALAR
                parameter_index = len(scalar_parameters)
                scalar_parameters.append(float(path[0]))
            else:
                source = module.RVineParameterSource.ROW_PATH
                parameter_index = len(row_parameters)
                row_parameters.append(path)
        if cached_native is None:
            native.parameter_source = source
            native.parameter_index = parameter_index
        elif (
                native.parameter_source != source
                or int(native.parameter_index) != parameter_index):
            raise ValueError(
                "cached native R-vine parameter layout does not match edge "
                f"{key}")
        compiled_edges.append(native)

    scalar_buffer = np.ascontiguousarray(
        scalar_parameters, dtype=np.float64)
    if row_parameters:
        row_buffer = np.ascontiguousarray(
            np.column_stack(row_parameters), dtype=np.float64)
    else:
        row_buffer = np.empty((n_rows, 0), dtype=np.float64)
    return compiled_edges, RVineParameterPack(
        scalar_parameters=scalar_buffer,
        row_parameters=row_buffer,
        n_rows=n_rows,
    )


def conditional_parameter_layout(
        pair_copulas, active_keys, parameter_paths, n_rows):
    """Normalize conditional edge paths and classify scalar/path storage.

    Static prediction currently materializes constant length-``n`` arrays.
    They are collapsed back to one scalar at the native boundary, while
    genuinely dynamic or varying paths retain one value per row.
    """
    n_rows = int(n_rows)
    normalized = dict(parameter_paths)
    sources = {}
    for raw_key in active_keys:
        key = tuple(int(value) for value in raw_key)
        edge = pair_copulas[key]
        if edge_is_independent(edge):
            continue
        if key not in parameter_paths:
            raise KeyError(f"missing R-vine parameter path for edge {key}")
        raw = parameter_paths[key]
        path = _parameter_path(raw, n_rows, key)
        _validate_parameter_domain(edge_copula(edge), path, key)
        dynamic = edge_has_dynamic_params(edge)
        if dynamic:
            if len(path) != n_rows:
                raise ValueError(
                    f"dynamic R-vine parameter path for edge {key} must "
                    f"have length {n_rows}, got {len(path)}")
            sources[key] = "row_path"
            normalized[key] = path
        elif len(path) == 1 or (
                len(path) > 1 and np.all(path == path[0])):
            sources[key] = "scalar"
            normalized[key] = float(path[0])
        else:
            sources[key] = "row_path"
            normalized[key] = path
    return normalized, sources


def density_parameter_layout(
        pair_copulas, active_keys, parameter_paths, n_rows):
    """Validate every oracle edge and classify native parameter storage."""
    missing = [key for key in active_keys if key not in parameter_paths]
    if missing:
        raise KeyError(f"missing R-vine parameter path for edge {missing[0]}")
    return conditional_parameter_layout(
        pair_copulas, active_keys, parameter_paths, n_rows)


def static_rosenblatt_parameter_layout(pair_copulas, active_keys):
    """Return scalar parameters or ``None`` for a Stage 5 fallback model."""
    from pyscarcopula._types import IndependentResult, MLEResult

    parameters = {}
    sources = {}
    for raw_key in active_keys:
        key = tuple(int(value) for value in raw_key)
        edge = pair_copulas[key]
        if edge_is_independent(edge):
            continue
        result = edge_result(edge)
        if result is None:
            parameter = edge_param(edge)
        elif type(result) is MLEResult:
            parameter = result.copula_param
        elif type(result) is IndependentResult:
            continue
        else:
            return None
        if parameter is None:
            return None
        path = np.asarray([parameter], dtype=np.float64)
        _validate_parameter_domain(edge_copula(edge), path, key)
        parameters[key] = float(path[0])
        sources[key] = "scalar"
    return parameters, sources


def _conditional_given_values(dimension, given, expected_variables):
    expected = tuple(sorted(int(value) for value in expected_variables))
    actual = tuple(sorted(int(value) for value in given))
    if actual != expected:
        raise ValueError(
            "conditional R-vine plan given variables do not match values: "
            f"expected {expected}, got {actual}")
    values = np.full(int(dimension), 0.5, dtype=np.float64)
    for variable, raw_value in given.items():
        if isinstance(raw_value, complex) or np.iscomplexobj(raw_value):
            raise TypeError(
                "conditional R-vine given values must contain real values")
        value = float(raw_value)
        if not np.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(
                "conditional R-vine given values must lie in the open unit "
                f"interval; variable {int(variable)} has {value!r}")
        values[int(variable)] = value
    return np.ascontiguousarray(values)


def _conditional_uniform_buffer(plan, n_rows, rng, uniforms):
    """Prepare full-width uniforms while preserving each Python RNG contract."""
    n_rows = int(n_rows)
    dimension = int(plan.d)
    draw_steps = [
        step for step in plan if step.get("action") == "sample_uniform"
    ]
    is_suffix = any("column" in step for step in draw_steps)
    if is_suffix:
        if uniforms is None:
            return np.ascontiguousarray(
                _open_unit_uniform(rng, size=(n_rows, dimension)),
                dtype=np.float64,
            )
        return _prepared_open_unit_draws(
            uniforms,
            (n_rows, dimension),
            name="suffix sampling uniforms",
        )

    draw_count = len(draw_steps)
    prepared = np.full((n_rows, dimension), 0.5, dtype=np.float64)
    if uniforms is None:
        for step in draw_steps:
            prepared[:, int(step["var"])] = _open_unit_uniform(
                rng, size=n_rows)
        return prepared

    supplied = np.asarray(uniforms)
    if supplied.shape == (n_rows, draw_count):
        compact = _prepared_open_unit_draws(
            supplied,
            (n_rows, draw_count),
            name="conditional DAG uniforms",
        )
        for draw_index, step in enumerate(draw_steps):
            prepared[:, int(step["var"])] = compact[:, draw_index]
        return prepared
    if supplied.shape == (n_rows, dimension):
        return _prepared_open_unit_draws(
            supplied,
            (n_rows, dimension),
            name="conditional DAG uniforms",
        )
    raise ValueError(
        "conditional DAG uniforms must have shape "
        f"{(n_rows, draw_count)} (draw order) or "
        f"{(n_rows, dimension)} (variable columns), got {supplied.shape}"
    )


def conditional_sample(
        module,
        pair_copulas,
        plan,
        n,
        rng,
        given,
        parameter_paths,
        *,
        uniforms=None,
        active_keys=None,
        parameter_sources=None,
        normalized_parameter_paths=None,
        native_plan=None,
        native_edges=None,
        parameter_pack=None,
        n_threads=1):
    """Execute one suffix or DAG conditional program in the native runtime.

    ``None`` is returned only when the exact-type capability gate rejects an
    edge before parameter validation or RNG consumption.
    """
    n = int(n)
    if n < 0:
        raise ValueError(
            "R-vine conditional sample row count must be non-negative")
    if not isinstance(n_threads, (int, np.integer)) or int(n_threads) <= 0:
        raise ValueError(
            "R-vine conditional sample n_threads must be a positive integer")
    n_threads = int(n_threads)
    active_keys = (
        conditional_active_keys(plan)
        if active_keys is None else
        tuple(tuple(int(value) for value in key) for key in active_keys)
    )
    if not native_edges_supported(pair_copulas, active_keys):
        return None

    if normalized_parameter_paths is None or parameter_sources is None:
        normalized_parameter_paths, parameter_sources = (
            conditional_parameter_layout(
                pair_copulas, active_keys, parameter_paths, n)
        )
    if parameter_pack is None:
        try:
            edges, parameters = compile_edge_specs(
                module,
                pair_copulas,
                active_keys,
                normalized_parameter_paths,
                n,
                parameter_sources=parameter_sources,
                native_edges=native_edges,
            )
        except CppUnsupported:
            return None
    else:
        if native_edges is None:
            raise ValueError(
                "precompiled conditional R-vine parameter pack requires "
                "native edges")
        if not isinstance(parameter_pack, RVineParameterPack):
            raise TypeError("parameter_pack must be an RVineParameterPack")
        if int(parameter_pack.n_rows) != n:
            raise ValueError(
                "precompiled conditional R-vine parameter pack row count "
                "mismatch")
        edges = list(native_edges)
        parameters = parameter_pack
    native = (
        compile_conditional_plan(module, plan, active_keys, given)
        if native_plan is None else native_plan
    )
    given_values = _conditional_given_values(
        int(plan.d), given, native.given_variables)
    prepared_uniforms = _conditional_uniform_buffer(
        plan, n, rng, uniforms)
    result = module.rvine_conditional_sample(
        native,
        edges,
        parameters.scalar_parameters,
        parameters.row_parameters,
        given_values,
        prepared_uniforms,
        n_threads,
    )
    raise_for_status(result, "R-vine conditional sample")
    if int(result["n_rows"]) != n or int(result["dimension"]) != int(plan.d):
        raise CppError(
            "C++ R-vine conditional sample returned inconsistent dimensions")
    values = np.asarray(result["values"], dtype=np.float64)
    if values.size != n * int(plan.d):
        raise CppError(
            "C++ R-vine conditional sample returned an invalid value count")
    return np.ascontiguousarray(values.reshape(n, int(plan.d)))


def sample(
        module,
        vine,
        n,
        rng,
        active_keys,
        traversal_plan,
        parameter_paths,
        *,
        uniforms=None,
        parameter_sources=None,
        native_plan=None,
        native_edges=None,
        n_threads=1,
        parameter_pack=None,
        request_state=None):
    """Execute one generic unconditional R-vine batch.

    ``None`` is returned only when the exact-type capability gate rejects an
    edge before entering C++.  Validation and numerical failures after the
    native call are always raised and must never trigger Python fallback.
    """
    n = int(n)
    if n < 0:
        raise ValueError("R-vine sample row count must be non-negative")
    if not isinstance(n_threads, (int, np.integer)) or int(n_threads) <= 0:
        raise ValueError("R-vine sample n_threads must be a positive integer")
    n_threads = int(n_threads)
    active_keys = tuple(
        tuple(int(value) for value in key) for key in active_keys)
    if traversal_plan is None or tuple(traversal_plan.active_keys) != active_keys:
        raise ValueError(
            "R-vine traversal plan does not match active edge keys")
    if not native_edges_supported(vine.pair_copulas, active_keys):
        return None

    request_key = _scalar_request_key(
        active_keys, parameter_sources, native_edges)
    cached_pack = (
        None if request_state is None
        else request_state.get("scalar_parameter_pack")
    )
    if (
            isinstance(cached_pack, _RVineScalarRequestPack)
            and cached_pack.key == request_key):
        if native_edges is None:
            raise ValueError(
                "cached scalar R-vine pack requires cached native edges")
        edges = list(native_edges)
        parameters = RVineParameterPack(
            scalar_parameters=cached_pack.scalar_parameters,
            row_parameters=np.empty((n, 0), dtype=np.float64),
            n_rows=n,
        )
    else:
        if parameter_pack is None:
            try:
                edges, parameters = compile_edge_specs(
                    module,
                    vine.pair_copulas,
                    active_keys,
                    parameter_paths,
                    n,
                    parameter_sources=parameter_sources,
                    native_edges=native_edges,
                )
            except CppUnsupported:
                return None
        else:
            if native_edges is None:
                raise ValueError(
                    "precompiled R-vine parameter pack requires native edges")
            if not isinstance(parameter_pack, RVineParameterPack):
                raise TypeError(
                    "parameter_pack must be an RVineParameterPack")
            if int(parameter_pack.n_rows) != n:
                raise ValueError(
                    "precompiled R-vine parameter pack row count mismatch")
            edges = list(native_edges)
            parameters = parameter_pack
        if request_state is not None and parameters.row_parameter_columns == 0:
            request_state["scalar_parameter_pack"] = _RVineScalarRequestPack(
                key=request_key,
                scalar_parameters=parameters.scalar_parameters,
            )
    plan = (
        compile_traversal_plan(module, traversal_plan)
        if native_plan is None
        else native_plan
    )
    if uniforms is None:
        prepared_uniforms = np.ascontiguousarray(
            _open_unit_uniform(rng, size=(n, int(vine.d))),
            dtype=np.float64,
        )
    else:
        prepared_uniforms = _prepared_open_unit_draws(
            uniforms,
            (n, int(vine.d)),
            name="R-vine sampling uniforms",
        )
    result = module.rvine_sample(
        plan,
        edges,
        parameters.scalar_parameters,
        parameters.row_parameters,
        prepared_uniforms,
        n_threads,
    )
    raise_for_status(result, "R-vine sample")
    if int(result["n_rows"]) != n or int(result["dimension"]) != int(vine.d):
        raise CppError("C++ R-vine sample returned inconsistent dimensions")
    values = np.asarray(result["values"], dtype=np.float64)
    if values.size != n * int(vine.d):
        raise CppError("C++ R-vine sample returned an invalid value count")
    return np.ascontiguousarray(values.reshape(n, int(vine.d)))


def _density_observations(values, dimension):
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise TypeError("R-vine density observations must contain real values")
    observations = np.asarray(raw, dtype=np.float64)
    if observations.ndim != 2 or observations.shape[1] != int(dimension):
        raise ValueError(
            "R-vine density observations must have shape "
            f"(n, {int(dimension)}), got {observations.shape}"
        )
    if not np.all(np.isfinite(observations)):
        raise ValueError("R-vine density observations must be finite")
    return np.ascontiguousarray(observations)


def _rosenblatt_observations(values, dimension):
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise TypeError(
            "R-vine Rosenblatt observations must contain real values")
    observations = np.asarray(raw, dtype=np.float64)
    if observations.ndim != 2 or observations.shape[1] != int(dimension):
        raise ValueError(
            "R-vine Rosenblatt observations must have shape "
            f"(n, {int(dimension)}), got {observations.shape}"
        )
    if not np.all(np.isfinite(observations)):
        raise ValueError("R-vine Rosenblatt observations must be finite")
    return np.ascontiguousarray(observations)


def _execute_rosenblatt(
        module, native_plan, native_edges, parameters, observations,
        n_threads):
    result = module.rvine_rosenblatt_transform(
        native_plan,
        list(native_edges),
        parameters.scalar_parameters,
        parameters.row_parameters,
        observations,
        int(n_threads),
    )
    raise_for_status(result, "R-vine Rosenblatt transform")
    n_rows, dimension = observations.shape
    if (
            int(result["n_rows"]) != n_rows
            or int(result["dimension"]) != dimension):
        raise CppError(
            "C++ R-vine Rosenblatt transform returned inconsistent "
            "dimensions")
    residuals = np.asarray(result["residuals"], dtype=np.float64)
    if residuals.size != n_rows * dimension:
        raise CppError(
            "C++ R-vine Rosenblatt transform returned an invalid buffer")
    residuals = residuals.reshape(n_rows, dimension)
    if not np.all(np.isfinite(residuals)):
        raise CppError(
            "C++ R-vine Rosenblatt transform returned invalid values")
    return np.ascontiguousarray(residuals)


def rosenblatt(
        module,
        pair_copulas,
        dimension,
        trees,
        edge_map,
        matrix,
        observations,
        *,
        active_keys=None,
        parameter_paths=None,
        parameter_sources=None,
        residual_node_keys=None,
        native_plan=None,
        native_edges=None,
        parameter_pack=None,
        n_threads=1):
    """Execute the scalar-only static R-vine Rosenblatt traversal."""
    if not isinstance(n_threads, (int, np.integer)) or int(n_threads) <= 0:
        raise ValueError(
            "R-vine Rosenblatt n_threads must be a positive integer")
    dimension = int(dimension)
    active_keys = (
        density_active_keys(trees, edge_map)
        if active_keys is None else
        tuple(tuple(int(value) for value in key) for key in active_keys)
    )
    if not native_edges_supported(pair_copulas, active_keys):
        return None
    if parameter_paths is None or parameter_sources is None:
        layout = static_rosenblatt_parameter_layout(
            pair_copulas, active_keys)
        if layout is None:
            return None
        parameter_paths, parameter_sources = layout
    scalar_sources = {
        "scalar",
        "SCALAR",
        module.RVineParameterSource.SCALAR,
    }
    if any(
            source not in scalar_sources
            for source in parameter_sources.values()):
        return None

    prepared_observations = _rosenblatt_observations(
        observations, dimension)
    n_rows = len(prepared_observations)
    if parameter_pack is None:
        try:
            edges, parameters = compile_edge_specs(
                module,
                pair_copulas,
                active_keys,
                parameter_paths,
                n_rows,
                parameter_sources=parameter_sources,
                native_edges=native_edges,
            )
        except CppUnsupported:
            return None
    else:
        if native_edges is None:
            raise ValueError(
                "precompiled Rosenblatt parameter pack requires native "
                "edges")
        if not isinstance(parameter_pack, RVineParameterPack):
            raise TypeError("parameter_pack must be an RVineParameterPack")
        if int(parameter_pack.n_rows) != n_rows:
            raise ValueError(
                "precompiled Rosenblatt parameter pack row count mismatch")
        edges = list(native_edges)
        parameters = parameter_pack
    residual_node_keys = (
        rosenblatt_residual_node_keys(matrix)
        if residual_node_keys is None else tuple(residual_node_keys)
    )
    plan = (
        compile_density_plan(
            module,
            dimension,
            trees,
            edge_map,
            active_keys,
            residual_node_keys=residual_node_keys,
        )
        if native_plan is None else native_plan
    )
    return _execute_rosenblatt(
        module,
        plan,
        edges,
        parameters,
        prepared_observations,
        int(n_threads),
    )


def _mcmc_draw_chunk_capacity(n, chunk_steps, memory_budget_bytes):
    """Bound generated proposal/acceptance buffers by a byte budget."""
    if (
            isinstance(memory_budget_bytes, (bool, np.bool_))
            or not isinstance(memory_budget_bytes, (int, np.integer))):
        raise TypeError("R-vine MCMC memory_budget_bytes must be an integer")
    budget = int(memory_budget_bytes)
    if budget < 0:
        raise ValueError(
            "R-vine MCMC memory_budget_bytes must be non-negative")
    rows = int(n)
    if rows == 0:
        return int(chunk_steps)
    bytes_per_step = 2 * rows * np.dtype(np.float64).itemsize
    capacity = budget // bytes_per_step
    if capacity < 1:
        raise MemoryError(
            "R-vine MCMC requires at least "
            f"{bytes_per_step} bytes for one proposal/acceptance step, "
            f"exceeding memory_budget_bytes={budget}"
        )
    return min(int(chunk_steps), int(capacity))


def _execute_log_pdf_rows(
        module, native_plan, native_edges, parameters, observations,
        n_threads):
    result = module.rvine_log_pdf_rows(
        native_plan,
        list(native_edges),
        parameters.scalar_parameters,
        parameters.row_parameters,
        observations,
        int(n_threads),
    )
    raise_for_status(result, "R-vine row log-density")
    n_rows, dimension = observations.shape
    if (
            int(result["n_rows"]) != n_rows
            or int(result["dimension"]) != dimension):
        raise CppError(
            "C++ R-vine row log-density returned inconsistent dimensions")
    values = np.asarray(result["log_pdf"], dtype=np.float64)
    if values.shape != (n_rows,) or not np.all(np.isfinite(values)):
        raise CppError(
            "C++ R-vine row log-density returned invalid values")
    return np.ascontiguousarray(values)


def log_pdf_rows(
        module,
        pair_copulas,
        dimension,
        trees,
        edge_map,
        parameter_paths,
        observations,
        *,
        active_keys=None,
        normalized_parameter_paths=None,
        parameter_sources=None,
        native_plan=None,
        native_edges=None,
        parameter_pack=None,
        n_threads=1):
    """Execute one fused row-wise R-vine density traversal."""
    if not isinstance(n_threads, (int, np.integer)) or int(n_threads) <= 0:
        raise ValueError(
            "R-vine row log-density n_threads must be a positive integer")
    dimension = int(dimension)
    prepared_observations = _density_observations(observations, dimension)
    n_rows = len(prepared_observations)
    active_keys = (
        density_active_keys(trees, edge_map)
        if active_keys is None else
        tuple(tuple(int(value) for value in key) for key in active_keys)
    )
    if not native_edges_supported(pair_copulas, active_keys):
        return None
    if normalized_parameter_paths is None or parameter_sources is None:
        normalized_parameter_paths, parameter_sources = (
            density_parameter_layout(
                pair_copulas, active_keys, parameter_paths, n_rows)
        )
    if parameter_pack is None:
        try:
            edges, parameters = compile_edge_specs(
                module,
                pair_copulas,
                active_keys,
                normalized_parameter_paths,
                n_rows,
                parameter_sources=parameter_sources,
                native_edges=native_edges,
            )
        except CppUnsupported:
            return None
    else:
        if native_edges is None:
            raise ValueError(
                "precompiled density parameter pack requires native edges")
        if not isinstance(parameter_pack, RVineParameterPack):
            raise TypeError("parameter_pack must be an RVineParameterPack")
        if int(parameter_pack.n_rows) != n_rows:
            raise ValueError(
                "precompiled density parameter pack row count mismatch")
        edges = list(native_edges)
        parameters = parameter_pack
    plan = (
        compile_density_plan(
            module, dimension, trees, edge_map, active_keys)
        if native_plan is None else native_plan
    )
    return _execute_log_pdf_rows(
        module,
        plan,
        edges,
        parameters,
        prepared_observations,
        int(n_threads),
    )


def mcmc(
        module,
        pair_copulas,
        dimension,
        trees,
        edge_map,
        parameter_paths,
        n,
        rng,
        given,
        *,
        initial=None,
        n_steps=None,
        burnin_steps=None,
        initial_uniforms=None,
        random_draws=None,
        step_offset=0,
        active_keys=None,
        normalized_parameter_paths=None,
        parameter_sources=None,
        native_plan=None,
        native_edges=None,
        parameter_pack=None,
        n_threads=1,
        chunk_steps=256,
        memory_budget_bytes=_DEFAULT_MCMC_DRAW_MEMORY_BUDGET_BYTES):
    """Run coordinate-wise MCMC in bounded native chunks with Python RNG."""
    from pyscarcopula.vine._rvine_conditional_runtime import (
        _empty_mcmc_diagnostics,
        _mcmc_diagnostics,
    )

    dimension = int(dimension)
    n = int(n)
    if n < 0:
        raise ValueError("R-vine MCMC row count must be non-negative")
    if not isinstance(n_threads, (int, np.integer)) or int(n_threads) <= 0:
        raise ValueError("R-vine MCMC n_threads must be a positive integer")
    if not isinstance(chunk_steps, (int, np.integer)) or int(chunk_steps) <= 0:
        raise ValueError("R-vine MCMC chunk_steps must be a positive integer")
    effective_chunk_steps = _mcmc_draw_chunk_capacity(
        n, chunk_steps, memory_budget_bytes)
    active_keys = (
        density_active_keys(trees, edge_map)
        if active_keys is None else
        tuple(tuple(int(value) for value in key) for key in active_keys)
    )
    if not native_edges_supported(pair_copulas, active_keys):
        return None
    if normalized_parameter_paths is None or parameter_sources is None:
        normalized_parameter_paths, parameter_sources = (
            density_parameter_layout(
                pair_copulas, active_keys, parameter_paths, n)
        )
    if parameter_pack is None:
        try:
            edges, parameters = compile_edge_specs(
                module,
                pair_copulas,
                active_keys,
                normalized_parameter_paths,
                n,
                parameter_sources=parameter_sources,
                native_edges=native_edges,
            )
        except CppUnsupported:
            return None
    else:
        if native_edges is None:
            raise ValueError(
                "precompiled MCMC parameter pack requires native edges")
        if not isinstance(parameter_pack, RVineParameterPack):
            raise TypeError("parameter_pack must be an RVineParameterPack")
        if int(parameter_pack.n_rows) != n:
            raise ValueError("precompiled MCMC parameter pack row mismatch")
        edges = list(native_edges)
        parameters = parameter_pack
    plan = (
        compile_density_plan(
            module, dimension, trees, edge_map, active_keys)
        if native_plan is None else native_plan
    )

    free_vars = [var for var in range(dimension) if var not in given]
    if not free_vars:
        out = np.empty((n, dimension), dtype=np.float64)
        for variable in range(dimension):
            out[:, variable] = given[variable]
        return out, _empty_mcmc_diagnostics()

    if initial is None:
        current = (
            _open_unit_uniform(rng, size=(n, dimension))
            if initial_uniforms is None else
            _prepared_open_unit_draws(
                initial_uniforms,
                (n, dimension),
                name="MCMC initial uniforms",
            ).copy()
        )
        for variable, value in given.items():
            current[:, variable] = value
    else:
        if initial_uniforms is not None:
            raise ValueError(
                "MCMC initial_uniforms cannot be supplied with initial")
        current = np.asarray(initial, dtype=np.float64).copy()
        if current.shape != (n, dimension):
            raise ValueError(
                f"MCMC initial state must have shape {(n, dimension)}, "
                f"got {current.shape}")
        if not np.all(np.isfinite(current)):
            raise ValueError("MCMC initial state must be finite")
        for variable, value in given.items():
            current[:, variable] = value
    current = np.ascontiguousarray(current, dtype=np.float64)
    current_log_pdf = _execute_log_pdf_rows(
        module, plan, edges, parameters, current, int(n_threads))

    n_steps = (
        max(80, 30 * len(free_vars))
        if n_steps is None else int(n_steps)
    )
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

    given_indices = sorted(int(variable) for variable in given)
    full_given_values = _conditional_given_values(
        dimension, given, given_indices)
    given_values = np.ascontiguousarray(
        full_given_values[given_indices], dtype=np.float64)
    accepted = {int(variable): 0 for variable in free_vars}
    proposed = {int(variable): 0 for variable in free_vars}
    offset = 0
    chunk_steps = effective_chunk_steps
    while offset < total_steps:
        count = min(chunk_steps, total_steps - offset)
        if replay_draws is None:
            proposal_uniforms = np.empty((count, n), dtype=np.float64)
            acceptance_uniforms = np.empty((count, n), dtype=np.float64)
            for local_step in range(count):
                proposal_uniforms[local_step] = rng.uniform(
                    PSEUDO_OBS_EPS,
                    1.0 - PSEUDO_OBS_EPS,
                    size=n,
                )
                acceptance_uniforms[local_step] = rng.uniform(
                    PSEUDO_OBS_EPS, 1.0, size=n)
        else:
            chunk = replay_draws[offset:offset + count]
            proposal_uniforms = np.ascontiguousarray(chunk[:, :, 0])
            acceptance_uniforms = np.ascontiguousarray(chunk[:, :, 1])
        result = module.rvine_mcmc(
            plan,
            edges,
            parameters.scalar_parameters,
            parameters.row_parameters,
            given_indices,
            given_values,
            free_vars,
            current,
            current_log_pdf,
            step_offset + offset,
            proposal_uniforms,
            acceptance_uniforms,
            int(n_threads),
        )
        raise_for_status(result, "R-vine conditional MCMC chunk")
        if (
                int(result["n_rows"]) != n
                or int(result["dimension"]) != dimension
                or int(result["coordinate_steps"]) != count):
            raise CppError(
                "C++ R-vine MCMC returned inconsistent dimensions")
        state = np.asarray(result["state"], dtype=np.float64)
        log_pdf = np.asarray(result["log_pdf"], dtype=np.float64)
        if state.size != n * dimension or log_pdf.shape != (n,):
            raise CppError("C++ R-vine MCMC returned invalid buffers")
        current = np.ascontiguousarray(state.reshape(n, dimension))
        current_log_pdf = np.ascontiguousarray(log_pdf)
        raw = dict(result["diagnostics"])
        chunk_proposed = list(raw["proposed"])
        chunk_accepted = list(raw["accepted"])
        if (
                len(chunk_proposed) != len(free_vars)
                or len(chunk_accepted) != len(free_vars)):
            raise CppError("C++ R-vine MCMC returned invalid counters")
        for index, variable in enumerate(free_vars):
            proposed[variable] += int(chunk_proposed[index])
            accepted[variable] += int(chunk_accepted[index])
        offset += count

    return clip_pseudo_observations(current), _mcmc_diagnostics(
        free_vars,
        accepted,
        proposed,
        n,
        n_steps,
        burnin_steps,
    )


def raise_for_status(result, operation):
    """Translate a structured native R-vine status without fallback."""
    status = int(result["status"])
    if status == 0:
        return
    row = int(result.get("failure_row", -1))
    edge = int(result.get("failure_edge", -1))
    operation_index = int(result.get("failure_operation", -1))
    message = (
        f"C++ {operation} failed: status={status} "
        f"({cpp_status_name(status)})"
    )
    if row >= 0:
        message += f", row={row}"
    if edge >= 0:
        message += f", edge={edge}"
    if operation_index >= 0:
        message += f", operation={operation_index}"
    if status in (2, 6):
        raise ValueError(message)
    if status in (3, 4, 5):
        raise CppUnsupported(message)
    if status == 7:
        raise FloatingPointError(message)
    raise CppError(message)


__all__: list[str] = []
