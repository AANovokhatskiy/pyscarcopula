"""Contracts for the shared native R-vine layer."""

from __future__ import annotations

import numpy as np
import pytest

from pyscarcopula import ClaytonCopula
from pyscarcopula.copula._rotation import transposed_bivariate_copula
from pyscarcopula._native import _extension as _cpp_extension, vine as _cpp_rvine
from pyscarcopula._native.errors import NativeError, NativeUnsupported
from pyscarcopula.vine._edge_adapter import edge_copula
from pyscarcopula.vine._helpers import _clip_unit
from pyscarcopula.vine._rvine_dag import (
    build_runtime_rvine_dag,
    plan_conditional_sample,
)
from pyscarcopula.vine._rvine_sampling_plan import (
    build_rvine_sampling_plan,
)
from pyscarcopula.vine._rvine_suffix import (
    build_suffix_conditional_plan,
    given_suffix_start_col,
)
from rvine_runtime_cases import (
    configured_mixed_family_vine,
    scalar_parameters,
)


pytestmark = pytest.mark.rvine_native


def _traversal_plan(vine):
    max_active_tree = vine._max_non_independent_tree_level()
    active_keys = vine._sample_active_edge_keys(max_active_tree)
    return build_rvine_sampling_plan(
        vine.d,
        vine.matrix,
        vine.trees,
        vine._edge_map,
        active_keys,
        max_active_tree,
    )


def _execute_flat_conditional(
        native, vine, active_keys, parameters, given, uniforms):
    """Small independent oracle for the packed conditional-plan contract."""
    uniforms = np.asarray(uniforms, dtype=np.float64)
    n = len(uniforms)
    nodes = [None] * native.node_count
    for node, (source, source_index) in enumerate(zip(
            native.node_sources, native.node_source_indices)):
        if source == 1:  # RVineNodeSource.GIVEN
            nodes[node] = np.full(n, given[source_index], dtype=np.float64)
        elif source == 2:  # RVineNodeSource.UNIFORM
            nodes[node] = uniforms[:, source_index].copy()

    for values in zip(
            native.opcodes,
            native.edge_indices,
            native.input1_nodes,
            native.input2_nodes,
            native.output1_nodes,
            native.output2_nodes,
            native.transposed):
        opcode, edge_index, input1, input2, output1, output2, transposed = values
        if opcode == 4:  # RVineOpcode.COPY
            nodes[output1] = nodes[input1].copy()
            continue
        key = active_keys[edge_index]
        copula = edge_copula(vine.pair_copulas[key])
        transposed_copula = transposed_bivariate_copula(copula)
        first_copula = transposed_copula if transposed else copula
        parameter = parameters[key]
        if opcode == 1:  # RVineOpcode.H
            nodes[output1] = _clip_unit(first_copula.h(
                nodes[input1], nodes[input2], parameter))
        elif opcode == 2:  # RVineOpcode.H_PAIR
            second_copula = copula if transposed else transposed_copula
            nodes[output1] = _clip_unit(first_copula.h(
                nodes[input1], nodes[input2], parameter))
            nodes[output2] = _clip_unit(second_copula.h(
                nodes[input2], nodes[input1], parameter))
        elif opcode == 3:  # RVineOpcode.H_INV
            nodes[output1] = _clip_unit(first_copula.h_inverse(
                nodes[input1], nodes[input2], parameter))
        else:  # pragma: no cover - native validation rejects this first
            raise AssertionError(f"unexpected opcode {opcode}")
    return np.column_stack([nodes[node] for node in native.output_nodes])


def test_common_binding_owns_plan_and_preserves_gas_alias():
    module = _cpp_extension.load()
    assert module.GasRvinePlan is module.RVineTraversalPlan
    assert type(module.GasRvinePlan()) is module.RVineTraversalPlan
    assert hasattr(module, "RVineEdgeSpec")
    assert hasattr(module, "RVineConditionalPlan")
    assert hasattr(module, "RVineDensityPlan")


def test_common_binding_advertises_all_rosenblatt_entry_points():
    module = _cpp_extension.load()
    assert hasattr(module, "rvine_sample")
    assert hasattr(module, "rvine_conditional_sample")
    assert hasattr(module, "rvine_log_pdf_rows")
    assert hasattr(module, "rvine_mcmc_chunk")
    assert hasattr(module, "rvine_mcmc_policy")
    assert hasattr(module, "rvine_mcmc_default_steps")
    assert not hasattr(module, "rvine_mcmc")
    assert hasattr(module, "rvine_rosenblatt_transform")
    assert hasattr(module, "dense_student_rosenblatt_transform")


def test_native_mcmc_policy_owns_defaults_and_85_percent_threshold():
    module = _cpp_extension.load()
    assert _cpp_rvine.mcmc_default_steps(1) == (80, 40)
    assert _cpp_rvine.mcmc_default_steps(4) == (120, 40)

    plan = module.RVineDensityPlan()
    plan.dimension = 1
    plan.node_count = 1
    plan.edge_indices = [0] * 100
    plan.affected_operation_offsets = [0, 85]
    policy = module.rvine_mcmc_policy(
        plan, [0], 2, True, "auto", 1024**3)
    assert policy["status"] == 0
    assert policy["density_algorithm"] == "incremental"

    plan.affected_operation_offsets = [0, 86]
    policy = module.rvine_mcmc_policy(
        plan, [0], 2, True, "auto", 1024**3)
    assert policy["status"] == 0
    assert policy["density_algorithm"] == "full_recompute"


def test_common_traversal_serializer_preserves_every_flat_array():
    module = _cpp_extension.load()
    python_plan = _traversal_plan(configured_mixed_family_vine())
    native = _cpp_rvine.compile_traversal_plan(module, python_plan)
    assert native.dimension == python_plan.dimension
    assert native.node_count == len(python_plan.node_keys)
    assert native.last_uniform_column == python_plan.last_uniform_column
    assert native.last_output_node == python_plan.last_output_node
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
            "update_u2_nodes"):
        assert tuple(getattr(native, name)) == tuple(getattr(python_plan, name))


def test_edge_pack_separates_none_scalar_and_row_path_storage():
    module = _cpp_extension.load()
    vine = configured_mixed_family_vine()
    active_keys = tuple(sorted(vine.pair_copulas))
    paths = scalar_parameters(vine)
    dynamic_key = (0, 0)
    paths[dynamic_key] = np.linspace(0.7, 1.0, 4)

    edges, parameters = _cpp_rvine.compile_edge_specs(
        module,
        vine.pair_copulas,
        active_keys,
        paths,
        4,
    )

    assert parameters.scalar_parameters.shape == (1,)
    assert parameters.row_parameters.shape == (4, 1)
    assert parameters.scalar_parameters.flags.c_contiguous
    assert parameters.row_parameters.flags.c_contiguous
    by_key = dict(zip(active_keys, edges))
    assert by_key[(0, 0)].parameter_source == (
        module.RVineParameterSource.ROW_PATH)
    assert by_key[(0, 1)].parameter_source == (
        module.RVineParameterSource.SCALAR)
    assert by_key[(1, 0)].parameter_source == module.RVineParameterSource.NONE
    assert by_key[(1, 0)].parameter_free is True
    assert by_key[(1, 0)].parameter_index == -1


def test_edge_pack_requires_source_for_ambiguous_one_row_array():
    module = _cpp_extension.load()
    vine = configured_mixed_family_vine()
    key = (0, 0)
    path = np.array([0.8], dtype=np.float64)

    with pytest.raises(ValueError, match="ambiguous for one row"):
        _cpp_rvine.compile_edge_specs(
            module, vine.pair_copulas, (key,), {key: path}, 1)

    edges, parameters = _cpp_rvine.compile_edge_specs(
        module,
        vine.pair_copulas,
        (key,),
        {key: path},
        1,
        parameter_sources={key: "row_path"},
    )
    assert edges[0].parameter_source == module.RVineParameterSource.ROW_PATH
    assert parameters.scalar_parameters.shape == (0,)
    assert parameters.row_parameters.shape == (1, 1)

    edges, parameters = _cpp_rvine.compile_edge_specs(
        module, vine.pair_copulas, (key,), {key: 0.8}, 1)
    assert edges[0].parameter_source == module.RVineParameterSource.SCALAR
    assert parameters.scalar_parameters.shape == (1,)
    assert parameters.row_parameters.shape == (1, 0)


@pytest.mark.parametrize(
    ("bad_path", "error", "message"),
    [
        (np.array([0.2, 0.3]), ValueError, "length 1 or 4"),
        (np.array([0.2, np.nan, 0.4, 0.5]), ValueError, "finite"),
        (np.full(4, 0.2 + 0.1j), TypeError, "real values"),
    ],
)
def test_edge_pack_rejects_invalid_paths_before_native_call(
        bad_path, error, message):
    module = _cpp_extension.load()
    vine = configured_mixed_family_vine()
    active_keys = tuple(sorted(vine.pair_copulas))
    paths = scalar_parameters(vine)
    paths[(0, 0)] = bad_path
    with pytest.raises(error, match=message):
        _cpp_rvine.compile_edge_specs(
            module,
            vine.pair_copulas,
            active_keys,
            paths,
            4,
        )


def test_exact_type_gate_rejects_all_subclasses():
    module = _cpp_extension.load()

    class CustomClayton(ClaytonCopula):
        pass

    class NativeEquivalentClayton(ClaytonCopula):
        __pyscarcopula_native_rvine__ = True

    assert not _cpp_rvine.native_copula_supported(CustomClayton())
    with pytest.raises(NativeUnsupported, match="exact registered built-in"):
        _cpp_rvine.compile_copula_spec(module, CustomClayton())
    opted_in = NativeEquivalentClayton()
    assert not _cpp_rvine.native_copula_supported(opted_in)
    with pytest.raises(NativeUnsupported, match="exact registered built-in"):
        _cpp_rvine.compile_copula_spec(module, opted_in)


def test_conditional_and_density_compilers_emit_flat_indexed_programs():
    module = _cpp_extension.load()
    vine = configured_mixed_family_vine()
    active_keys = tuple(sorted(vine.pair_copulas))
    given = {2: 0.4}
    conditional = plan_conditional_sample(
        build_runtime_rvine_dag(vine.matrix, vine._edge_map),
        given,
        vine.d,
    )
    native_conditional = _cpp_rvine.compile_conditional_plan(
        module, conditional, active_keys, given)

    operation_count = sum(
        step["action"] in {"h_prop", "h_inv"}
        for step in conditional
    )
    uniform_count = sum(
        step["action"] == "sample_uniform"
        for step in conditional
    )
    assert native_conditional.dimension == vine.d
    assert len(native_conditional.edge_indices) == operation_count
    assert len(native_conditional.opcodes) == operation_count
    assert len(native_conditional.uniform_nodes) == uniform_count
    assert len(native_conditional.node_sources) == native_conditional.node_count
    assert len(native_conditional.node_source_indices) == (
        native_conditional.node_count)
    assert len(native_conditional.output_nodes) == vine.d
    assert all(
        0 <= value < native_conditional.node_count
        for value in native_conditional.output_nodes
    )

    native_density = _cpp_rvine.compile_density_plan(
        module,
        vine.d,
        vine.trees,
        vine._edge_map,
        active_keys,
    )
    edge_count = sum(len(level) for level in vine.trees)
    assert native_density.dimension == vine.d
    assert len(native_density.edge_indices) == edge_count
    assert len(native_density.input1_nodes) == edge_count
    assert len(native_density.input2_nodes) == edge_count
    assert len(native_density.output1_nodes) == edge_count
    assert native_density.used_edges == sorted(
        set(native_density.edge_indices))

    assert module.validate_rvine_conditional_plan(
        native_conditional, len(active_keys))
    assert module.validate_rvine_density_plan(native_density, len(active_keys))
    native_conditional.output_nodes = [native_conditional.node_count] * vine.d
    native_density.transposed = [2] * edge_count
    assert not module.validate_rvine_conditional_plan(
        native_conditional, len(active_keys))
    assert not module.validate_rvine_density_plan(
        native_density, len(active_keys))


def test_conditional_trace_captures_canonical_fixed_suffix_inputs():
    module = _cpp_extension.load()
    vine = configured_mixed_family_vine()
    peel_order = [
        int(vine.matrix[vine.d - 1 - column, column])
        for column in range(vine.d)
    ]
    fixed = peel_order[-2:]
    given = {fixed[0]: 0.37, fixed[1]: 0.71}
    start_col = given_suffix_start_col(vine.d, given, vine.matrix)
    plan = build_suffix_conditional_plan(
        vine.d, start_col, vine.matrix, given)
    trace = _cpp_rvine.conditional_trace(
        module,
        vine.pair_copulas,
        plan,
        5,
        given,
        scalar_parameters(vine),
    )

    first_step = next(step for step in plan if step["action"] == "h_pair")
    variables = sorted((int(first_step["leaf"]), int(first_step["partner"])))
    expected = np.tile([given[variables[0]], given[variables[1]]], (5, 1))
    np.testing.assert_array_equal(trace[0], expected)


def test_pseudo_observation_trace_contains_native_rosenblatt_nodes():
    module = _cpp_extension.load()
    vine = configured_mixed_family_vine()
    observations = np.random.default_rng(2026082901).uniform(
        0.02, 0.98, size=(9, vine.d))
    pseudo = _cpp_rvine.pseudo_observations(
        module,
        vine.pair_copulas,
        vine.d,
        vine.trees,
        vine._edge_map,
        vine.matrix,
        observations,
    )
    residual_keys = _cpp_rvine.rosenblatt_residual_node_keys(vine.matrix)
    residuals = _cpp_rvine.rosenblatt(
        module,
        vine.pair_copulas,
        vine.d,
        vine.trees,
        vine._edge_map,
        vine.matrix,
        observations,
    )

    for variable in range(vine.d):
        np.testing.assert_array_equal(
            pseudo[(variable, frozenset())], observations[:, variable])
    for column, key in enumerate(residual_keys):
        np.testing.assert_array_equal(pseudo[key], residuals[:, column])


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (2, ValueError),
        (3, NativeUnsupported),
        (7, FloatingPointError),
        (99, NativeError),
    ],
)
def test_common_status_mapper_preserves_failure_context(status, error):
    result = {
        "status": status,
        "failure_row": 5,
        "failure_edge": 2,
    }
    with pytest.raises(error, match=r"row=5, edge=2"):
        _cpp_rvine.raise_for_status(result, "test operation")
