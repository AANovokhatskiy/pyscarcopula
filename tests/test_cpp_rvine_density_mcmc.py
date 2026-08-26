"""Differential and native-boundary tests for R-vine stage 4."""

from __future__ import annotations

from copy import deepcopy
import threading

import numpy as np
import pytest

from pyscarcopula import (
    BivariateGaussianCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    IndependentCopula,
    JoeCopula,
)
from pyscarcopula.numerical import _cpp_extension, _cpp_rvine

from rvine_runtime_cases import (
    configured_mixed_family_vine,
    configured_static_dvine,
    fitted_pair,
    scalar_parameters,
)


def _density_request(vine, parameters, n):
    module = _cpp_extension.load()
    active_keys = _cpp_rvine.density_active_keys(
        vine._trees, vine._edge_map)
    normalized, sources = _cpp_rvine.density_parameter_layout(
        vine.pair_copulas, active_keys, parameters, n)
    context, pack = vine._native_density_context(
        module,
        vine.pair_copulas,
        vine._edge_map,
        parameters,
        n,
        active_keys=active_keys,
        normalized_paths=normalized,
        parameter_sources=sources,
    )
    assert context is not None
    return module, context, pack


def test_density_plan_precompiles_and_validates_coordinate_closures():
    vine = configured_static_dvine(6)
    parameters = scalar_parameters(vine)
    module, context, _pack = _density_request(vine, parameters, 3)
    plan = context["plan"]

    dependencies = [set() for _ in range(int(plan.node_count))]
    for variable, node in enumerate(plan.input_nodes):
        dependencies[int(node)] = {variable}
    expected_operations = [[] for _ in range(vine.d)]
    expected_nodes = [[int(plan.input_nodes[variable])]
                      for variable in range(vine.d)]
    for operation, (input1, input2, output1, output2) in enumerate(zip(
            plan.input1_nodes,
            plan.input2_nodes,
            plan.output1_nodes,
            plan.output2_nodes,
    )):
        affected = dependencies[int(input1)] | dependencies[int(input2)]
        for variable in sorted(affected):
            expected_operations[variable].append(operation)
        if int(output1) >= 0:
            dependencies[int(output1)] = set(affected)
            dependencies[int(output2)] = set(affected)
            for variable in sorted(affected):
                expected_nodes[variable].extend(
                    (int(output1), int(output2)))

    for variable in range(vine.d):
        operation_begin = plan.affected_operation_offsets[variable]
        operation_end = plan.affected_operation_offsets[variable + 1]
        node_begin = plan.affected_node_offsets[variable]
        node_end = plan.affected_node_offsets[variable + 1]
        assert list(plan.affected_operations[
            operation_begin:operation_end]) == expected_operations[variable]
        assert list(plan.affected_nodes[
            node_begin:node_end]) == expected_nodes[variable]
    assert module.validate_rvine_density_plan(
        plan, len(context["active_keys"]))

    corrupted = list(plan.affected_operations)
    corrupted[0] = corrupted[-1]
    plan.affected_operations = corrupted
    assert not module.validate_rvine_density_plan(
        plan, len(context["active_keys"]))


@pytest.mark.parametrize(
    ("family", "rotation", "parameter"),
    [
        *[(ClaytonCopula, rotation, 0.8)
          for rotation in (0, 90, 180, 270)],
        *[(GumbelCopula, rotation, 1.6)
          for rotation in (0, 90, 180, 270)],
        *[(JoeCopula, rotation, 1.7)
          for rotation in (0, 90, 180, 270)],
        (FrankCopula, 0, 2.5),
        (BivariateGaussianCopula, 0, -0.4),
        (IndependentCopula, 0, 0.0),
    ],
)
def test_native_density_matches_python_family_rotation_matrix_exactly(
        monkeypatch, family, rotation, parameter):
    vine = configured_static_dvine(2)
    copula = family() if family is IndependentCopula else family(
        rotate=rotation)
    vine.pair_copulas[(0, 0)] = fitted_pair(copula, parameter)
    observations = np.asfortranarray(np.array([
        [0.0, 1.0],
        [1e-14, 1.0 - 1e-14],
        [0.27, 0.63],
        [0.91, 0.08],
    ], dtype=np.float64))
    parameters = scalar_parameters(vine)

    expected = vine._log_pdf_rows_with_r_python(observations, parameters)
    actual = vine._log_pdf_rows_with_r(observations, parameters)

    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.c_contiguous


def test_native_density_transposed_plan_matches_transposed_rotation_exactly():
    vine = configured_static_dvine(2)
    parameter = 0.8
    vine.pair_copulas[(0, 0)] = fitted_pair(
        ClaytonCopula(rotate=90), parameter)
    parameters = scalar_parameters(vine)
    observations = np.array([
        [0.13, 0.82],
        [0.37, 0.44],
        [0.79, 0.21],
    ])
    module, context, pack = _density_request(vine, parameters, len(observations))
    context["plan"].transposed = [1]

    result = module.rvine_log_pdf_rows(
        context["plan"],
        context["edges"],
        pack.scalar_parameters,
        pack.row_parameters,
        observations,
    )
    _cpp_rvine.raise_for_status(result, "test transposed density")
    expected = ClaytonCopula(rotate=270).log_pdf(
        observations[:, 0], observations[:, 1], np.array([parameter]))
    np.testing.assert_array_equal(result["log_pdf"], expected)


def test_native_density_supports_empty_singleton_and_mixed_parameter_paths(
        monkeypatch):
    vine = configured_mixed_family_vine()
    parameters = scalar_parameters(vine)
    parameters[(0, 0)] = np.linspace(0.55, 1.05, 9)
    observations = np.asfortranarray(
        np.random.default_rng(2026082240).uniform(
            1e-12, 1.0 - 1e-12, size=(9, vine.d)))

    expected = vine._log_pdf_rows_with_r_python(observations, parameters)
    actual = vine._log_pdf_rows_with_r(observations, parameters)
    np.testing.assert_array_equal(actual, expected)

    scalar = scalar_parameters(vine)
    singleton = observations[:1]
    expected_one = vine._log_pdf_rows_with_r_python(singleton, scalar)
    actual_one = vine._log_pdf_rows_with_r(singleton, scalar)
    np.testing.assert_array_equal(actual_one, expected_one)

    empty = vine._log_pdf_rows_with_r(
        np.empty((0, vine.d), dtype=np.float64), scalar)
    assert empty.shape == (0,)
    assert empty.dtype == np.float64


def test_density_oracles_reject_invalid_parameter_and_native_precedes_rng(
        monkeypatch):
    vine = configured_static_dvine(2)
    parameters = scalar_parameters(vine)
    parameters[(0, 0)] = np.array([2.0])
    observations = np.array([[0.3, 0.7]])

    with pytest.raises(Exception):
        vine._log_pdf_rows_with_r_python(observations, parameters)
    with pytest.raises(ValueError, match="must lie in"):
        vine._log_pdf_rows_with_r(observations, parameters)

    rng = np.random.default_rng(2026082239)
    state_before = deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="must lie in"):
        vine._sample_arbitrary_given_mcmc(
            3,
            parameters,
            rng,
            {0: 0.4},
            n_steps=1,
            burnin_steps=0,
        )
    assert rng.bit_generator.state == state_before


def test_density_direct_validation_diagnostics_and_thread_parity():
    vine = configured_mixed_family_vine()
    n = 11
    parameters = scalar_parameters(vine)
    module, context, pack = _density_request(vine, parameters, n)
    observations = np.random.default_rng(2026082241).uniform(
        0.01, 0.99, size=(n, vine.d))

    sequential = module.rvine_log_pdf_rows(
        context["plan"],
        context["edges"],
        pack.scalar_parameters,
        pack.row_parameters,
        observations,
        1,
    )
    requested_many = module.rvine_log_pdf_rows(
        context["plan"],
        context["edges"],
        pack.scalar_parameters,
        pack.row_parameters,
        observations,
        4,
    )
    _cpp_rvine.raise_for_status(sequential, "test density")
    _cpp_rvine.raise_for_status(requested_many, "test density")
    np.testing.assert_array_equal(
        sequential["log_pdf"], requested_many["log_pdf"])
    diagnostics = dict(requested_many["diagnostics"])
    assert diagnostics["n_threads_requested"] == 4
    assert diagnostics["n_threads_used"] == 1
    assert diagnostics["density_operations"] == n * 3
    assert diagnostics["h_pair_operations"] == n * 2
    assert diagnostics["independence_fast_paths"] == n

    malformed = _cpp_rvine.compile_density_plan(
        module,
        vine.d,
        vine._trees,
        vine._edge_map,
        context["active_keys"],
    )
    malformed.input1_nodes = [malformed.node_count] * len(
        malformed.input1_nodes)
    invalid_plan = module.rvine_log_pdf_rows(
        malformed,
        context["edges"],
        pack.scalar_parameters,
        pack.row_parameters,
        observations,
    )
    assert invalid_plan["status"] == 2

    invalid_observations = observations.copy()
    invalid_observations[3, 1] = np.nan
    invalid_values = module.rvine_log_pdf_rows(
        context["plan"],
        context["edges"],
        pack.scalar_parameters,
        pack.row_parameters,
        invalid_observations,
    )
    assert invalid_values["status"] == 6
    assert invalid_values["failure_row"] == 3

    invalid_parameters = pack.scalar_parameters.copy()
    invalid_parameters[:] = 2.0
    invalid_parameter_result = module.rvine_log_pdf_rows(
        context["plan"],
        context["edges"],
        invalid_parameters,
        pack.row_parameters,
        observations,
    )
    assert invalid_parameter_result["status"] == 6


@pytest.mark.parametrize("chunk_steps", [1, 2, 3, 8])
def test_native_mcmc_matches_python_across_chunk_boundaries_exactly(
        chunk_steps):
    vine = configured_mixed_family_vine()
    n = 7
    parameters = scalar_parameters(vine)
    parameters[(0, 0)] = np.linspace(0.65, 0.95, n)
    given = {0: 0.57, 2: 0.31}
    initial = np.random.default_rng(2026082242).uniform(
        0.05, 0.95, size=(n, vine.d))
    for variable, value in given.items():
        initial[:, variable] = value
    draws = np.random.default_rng(2026082243).uniform(
        0.01, 0.99, size=(11, n, 2))

    expected, expected_diagnostics = vine._sample_arbitrary_given_mcmc_python(
        n,
        parameters,
        np.random.default_rng(1),
        given,
        initial=initial,
        n_steps=7,
        burnin_steps=4,
        random_draws=draws,
        step_offset=1,
    )
    module, context, pack = _density_request(vine, parameters, n)
    actual, actual_diagnostics = _cpp_rvine.mcmc(
        module,
        vine.pair_copulas,
        vine.d,
        vine._trees,
        vine._edge_map,
        parameters,
        n,
        np.random.default_rng(2),
        given,
        initial=initial,
        n_steps=7,
        burnin_steps=4,
        random_draws=draws,
        step_offset=1,
        active_keys=context["active_keys"],
        normalized_parameter_paths=parameters,
        parameter_sources=context["parameter_sources"],
        native_plan=context["plan"],
        native_edges=context["edges"],
        parameter_pack=pack,
        chunk_steps=chunk_steps,
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual_diagnostics == expected_diagnostics


def test_native_and_python_mcmc_support_zero_chains(monkeypatch):
    vine = configured_mixed_family_vine()
    parameters = scalar_parameters(vine)
    given = {0: 0.57, 2: 0.31}
    initial = np.empty((0, vine.d), dtype=np.float64)
    draws = np.empty((5, 0, 2), dtype=np.float64)
    results = [
        executor(
            0,
            parameters,
            np.random.default_rng(2026082280),
            given,
            initial=initial,
            n_steps=3,
            burnin_steps=2,
            random_draws=draws,
        )
        for executor in (
            vine._sample_arbitrary_given_mcmc_python,
            vine._sample_arbitrary_given_mcmc,
        )
    ]

    for samples, diagnostics in results:
        assert samples.shape == (0, vine.d)
        assert diagnostics["proposed"] == {1: 0}
        assert diagnostics["accepted"] == {1: 0}
        assert diagnostics["proposals_per_chain"] == {1: 0.0}
        assert diagnostics["accepted_per_chain"] == {1: 0.0}
        assert diagnostics["acceptance_mean"] is None
    assert results[0][1] == results[1][1]


def test_incremental_mcmc_matches_full_recompute_bitwise_with_row_paths():
    vine = configured_mixed_family_vine()
    n = 13
    parameters = scalar_parameters(vine)
    parameters[(0, 0)] = np.linspace(0.65, 1.05, n)
    module, context, pack = _density_request(vine, parameters, n)
    given_indices = [0]
    given_values = np.array([0.57])
    free_indices = [1, 2]
    current = np.random.default_rng(2026082271).uniform(
        0.03, 0.97, size=(n, vine.d))
    current[:, given_indices] = given_values
    density = module.rvine_log_pdf_rows(
        context["plan"],
        context["edges"],
        pack.scalar_parameters,
        pack.row_parameters,
        current,
    )
    _cpp_rvine.raise_for_status(density, "test initial density")
    steps = 17
    draws = np.random.default_rng(2026082272).uniform(
        0.01, 0.99, size=(steps, n, 2))
    proposal = np.ascontiguousarray(draws[:, :, 0])
    acceptance = np.ascontiguousarray(draws[:, :, 1])

    def execute(algorithm, budget=64 * 1024 * 1024):
        return module.rvine_mcmc_chunk(
            context["plan"],
            context["edges"],
            pack.scalar_parameters,
            pack.row_parameters,
            given_indices,
            given_values,
            free_indices,
            current,
            density["log_pdf"],
            1,
            proposal,
            acceptance,
            3,
            algorithm,
            budget,
        )

    expected = execute("full_recompute")
    plan = context["plan"]
    item_size = np.dtype(np.float64).itemsize
    fixed_bytes = (
        n * (vine.d + 1) * item_size
        + 2 * steps * n * item_size)
    per_row_workspace = 2 * (
        int(plan.node_count) + len(plan.edge_indices)) * item_size + (
            int(plan.node_count) * np.dtype(np.int32).itemsize)
    budget = fixed_bytes + 5 * per_row_workspace
    actual = execute("incremental", budget)
    _cpp_rvine.raise_for_status(expected, "test full MCMC")
    _cpp_rvine.raise_for_status(actual, "test incremental MCMC")

    expected_draws_used = steps * n
    for result in (expected, actual):
        assert result["coordinate_steps"] == steps
        assert result["proposal_draws_used"] == expected_draws_used
        assert result["acceptance_draws_used"] == expected_draws_used
        diagnostics = dict(result["diagnostics"])
        assert diagnostics["proposal_draws_used"] == expected_draws_used
        assert diagnostics["acceptance_draws_used"] == expected_draws_used

    np.testing.assert_array_equal(actual["state"], expected["state"])
    np.testing.assert_array_equal(actual["log_pdf"], expected["log_pdf"])
    expected_diag = dict(expected["diagnostics"])
    actual_diag = dict(actual["diagnostics"])
    for key in ("proposed", "accepted", "non_finite_proposals"):
        assert actual_diag[key] == expected_diag[key]
    assert expected_diag["mcmc_density_algorithm"] == "full_recompute"
    assert actual_diag["mcmc_density_algorithm"] == "incremental"
    assert actual_diag["affected_operations"] == [3, 2]
    assert actual_diag["row_chunks"] == 3
    assert actual_diag["max_chunk_rows"] == 5
    assert actual_diag["peak_workspace_bytes"] <= budget
    assert actual_diag["cache_bytes"] == (
        5 * (int(plan.node_count) + len(plan.edge_indices)) * item_size)


def test_incremental_mcmc_preserves_cycle_across_non_sweep_chunks(
        monkeypatch):
    vine = configured_static_dvine(5)
    n = 7
    parameters = scalar_parameters(vine)
    given = {0: 0.43, 2: 0.67}
    initial = np.random.default_rng(2026082273).uniform(
        0.04, 0.96, size=(n, vine.d))
    for variable, value in given.items():
        initial[:, variable] = value
    draws = np.random.default_rng(2026082274).uniform(
        0.01, 0.99, size=(11, n, 2))
    expected, expected_diagnostics = vine._sample_arbitrary_given_mcmc_python(
        n,
        parameters,
        np.random.default_rng(1),
        given,
        initial=initial,
        n_steps=8,
        burnin_steps=3,
        random_draws=draws,
        step_offset=2,
    )

    module, context, pack = _density_request(vine, parameters, n)
    native_mcmc = module.rvine_mcmc_chunk
    offsets = []
    algorithms = []

    def recording_mcmc(*args, **kwargs):
        offsets.append(args[9])
        algorithms.append(args[13])
        return native_mcmc(*args, **kwargs)

    monkeypatch.setattr(module, "rvine_mcmc_chunk", recording_mcmc)
    actual, actual_diagnostics = _cpp_rvine.mcmc(
        module,
        vine.pair_copulas,
        vine.d,
        vine._trees,
        vine._edge_map,
        parameters,
        n,
        np.random.default_rng(2),
        given,
        initial=initial,
        n_steps=8,
        burnin_steps=3,
        random_draws=draws,
        step_offset=2,
        active_keys=context["active_keys"],
        normalized_parameter_paths=parameters,
        parameter_sources=context["parameter_sources"],
        native_plan=context["plan"],
        native_edges=context["edges"],
        parameter_pack=pack,
        chunk_steps=4,
        density_algorithm="incremental",
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual_diagnostics == expected_diagnostics
    assert offsets == [2, 6, 10]
    assert algorithms == ["incremental"] * 3


def test_incremental_mcmc_matches_every_full_recompute_acceptance_decision():
    vine = configured_static_dvine(5)
    n = 7
    parameters = scalar_parameters(vine)
    parameters[(0, 0)] = np.linspace(-0.30, -0.20, n)
    module, context, pack = _density_request(vine, parameters, n)
    given_indices = [0, 2]
    given_values = np.array([0.43, 0.67])
    free_indices = [1, 3, 4]
    initial = np.random.default_rng(2026082276).uniform(
        0.04, 0.96, size=(n, vine.d))
    initial[:, given_indices] = given_values
    density = module.rvine_log_pdf_rows(
        context["plan"],
        context["edges"],
        pack.scalar_parameters,
        pack.row_parameters,
        initial,
    )
    _cpp_rvine.raise_for_status(density, "test decision initial density")
    draws = np.random.default_rng(2026082277).uniform(
        0.01, 0.99, size=(11, n, 2))

    def execute(algorithm):
        state = initial.copy()
        log_pdf = np.asarray(density["log_pdf"]).copy()
        trajectory = []
        for step in range(len(draws)):
            result = module.rvine_mcmc_chunk(
                context["plan"],
                context["edges"],
                pack.scalar_parameters,
                pack.row_parameters,
                given_indices,
                given_values,
                free_indices,
                state,
                log_pdf,
                2 + step,
                np.ascontiguousarray(draws[step:step + 1, :, 0]),
                np.ascontiguousarray(draws[step:step + 1, :, 1]),
                1,
                algorithm,
            )
            _cpp_rvine.raise_for_status(
                result, f"test {algorithm} decision step {step}")
            state = np.asarray(result["state"]).reshape(n, vine.d).copy()
            log_pdf = np.asarray(result["log_pdf"]).copy()
            diagnostics = dict(result["diagnostics"])
            trajectory.append((
                state.copy(),
                log_pdf.copy(),
                list(diagnostics["proposed"]),
                list(diagnostics["accepted"]),
                int(diagnostics["non_finite_proposals"]),
            ))
        return trajectory

    expected = execute("full_recompute")
    actual = execute("incremental")
    assert len(actual) == len(expected)
    for actual_step, expected_step in zip(actual, expected):
        np.testing.assert_array_equal(actual_step[0], expected_step[0])
        np.testing.assert_array_equal(actual_step[1], expected_step[1])
        assert actual_step[2:] == expected_step[2:]


def test_native_mcmc_preserves_interleaved_rng_state_across_internal_chunk(
        monkeypatch):
    vine = configured_static_dvine(4)
    parameters = scalar_parameters(vine)
    n = 5
    given = {0: 0.43, 2: 0.67}
    initial = np.random.default_rng(2026082244).uniform(
        0.05, 0.95, size=(n, vine.d))
    python_rng = np.random.default_rng(2026082245)
    native_rng = np.random.default_rng(2026082245)
    module = _cpp_extension.load()
    native_mcmc = module.rvine_mcmc_chunk
    selected_algorithms = []

    def recording_mcmc(*args, **kwargs):
        selected_algorithms.append(args[13])
        return native_mcmc(*args, **kwargs)

    monkeypatch.setattr(module, "rvine_mcmc_chunk", recording_mcmc)

    expected, expected_diagnostics = vine._sample_arbitrary_given_mcmc_python(
        n,
        parameters,
        python_rng,
        given,
        initial=initial,
        n_steps=257,
        burnin_steps=3,
        step_offset=1,
    )
    actual, actual_diagnostics = vine._sample_arbitrary_given_mcmc(
        n,
        parameters,
        native_rng,
        given,
        initial=initial,
        n_steps=257,
        burnin_steps=3,
        step_offset=1,
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual_diagnostics == expected_diagnostics
    np.testing.assert_array_equal(native_rng.random(32), python_rng.random(32))
    assert selected_algorithms
    assert set(selected_algorithms) == {"incremental"}


def test_native_mcmc_auto_keeps_single_chain_on_full_recompute(monkeypatch):
    vine = configured_static_dvine(4)
    parameters = scalar_parameters(vine)
    given = {0: 0.43, 2: 0.67}
    initial = np.array([[0.43, 0.25, 0.67, 0.75]])
    module = _cpp_extension.load()
    native_mcmc = module.rvine_mcmc_chunk
    selected_algorithms = []

    def recording_mcmc(*args, **kwargs):
        selected_algorithms.append(args[13])
        return native_mcmc(*args, **kwargs)

    monkeypatch.setattr(module, "rvine_mcmc_chunk", recording_mcmc)
    vine._sample_arbitrary_given_mcmc(
        1,
        parameters,
        np.random.default_rng(2026082275),
        given,
        initial=initial,
        n_steps=3,
        burnin_steps=0,
    )

    assert selected_algorithms == ["full_recompute"]


def test_native_mcmc_chunk_size_obeys_memory_budget_before_rng(monkeypatch):
    vine = configured_mixed_family_vine()
    n = 4
    parameters = scalar_parameters(vine)
    given = {0: 0.43, 2: 0.67}
    initial = np.random.default_rng(2026082248).uniform(
        0.05, 0.95, size=(n, vine.d))
    for variable, value in given.items():
        initial[:, variable] = value
    draws = np.random.default_rng(2026082249).uniform(
        0.01, 0.99, size=(5, n, 2))
    expected, expected_diagnostics = vine._sample_arbitrary_given_mcmc_python(
        n,
        parameters,
        np.random.default_rng(1),
        given,
        initial=initial,
        n_steps=5,
        burnin_steps=0,
        random_draws=draws,
    )
    module, context, pack = _density_request(vine, parameters, n)
    native_mcmc = module.rvine_mcmc_chunk
    chunk_sizes = []
    selected_algorithms = []
    native_budgets = []

    def recording_mcmc(*args, **kwargs):
        chunk_sizes.append(np.asarray(args[10]).shape[0])
        selected_algorithms.append(args[13])
        native_budgets.append(args[14])
        return native_mcmc(*args, **kwargs)

    monkeypatch.setattr(module, "rvine_mcmc_chunk", recording_mcmc)
    bytes_per_step = 2 * n * np.dtype(np.float64).itemsize
    full_reserved = _cpp_rvine._mcmc_full_reserved_bytes(
        context["plan"], n)
    incremental_reserved = _cpp_rvine._mcmc_incremental_reserved_bytes(
        context["plan"], n)
    fallback_budget = full_reserved + bytes_per_step
    assert fallback_budget < incremental_reserved + bytes_per_step
    actual, actual_diagnostics = _cpp_rvine.mcmc(
        module,
        vine.pair_copulas,
        vine.d,
        vine._trees,
        vine._edge_map,
        parameters,
        n,
        np.random.default_rng(2),
        given,
        initial=initial,
        n_steps=5,
        burnin_steps=0,
        random_draws=draws,
        active_keys=context["active_keys"],
        normalized_parameter_paths=parameters,
        parameter_sources=context["parameter_sources"],
        native_plan=context["plan"],
        native_edges=context["edges"],
        parameter_pack=pack,
        memory_budget_bytes=fallback_budget,
    )
    np.testing.assert_array_equal(actual, expected)
    assert actual_diagnostics == expected_diagnostics
    assert chunk_sizes == [1] * 5
    assert selected_algorithms == ["full_recompute"] * 5
    adapter_state_bytes = _cpp_rvine._mcmc_adapter_state_bytes(
        context["plan"], n)
    assert native_budgets == [
        fallback_budget - adapter_state_bytes
    ] * 5

    density = module.rvine_log_pdf_rows(
        context["plan"],
        context["edges"],
        pack.scalar_parameters,
        pack.row_parameters,
        initial,
    )
    _cpp_rvine.raise_for_status(density, "test budget initial density")
    direct_full = native_mcmc(
        context["plan"],
        context["edges"],
        pack.scalar_parameters,
        pack.row_parameters,
        [0, 2],
        np.array([0.43, 0.67]),
        [1],
        initial,
        density["log_pdf"],
        0,
        np.ascontiguousarray(draws[:1, :, 0]),
        np.ascontiguousarray(draws[:1, :, 1]),
        1,
        "full_recompute",
        fallback_budget - adapter_state_bytes,
    )
    _cpp_rvine.raise_for_status(direct_full, "test bounded full MCMC")
    assert dict(direct_full["diagnostics"])[
        "peak_workspace_bytes"] + adapter_state_bytes <= fallback_budget
    rejected_full = native_mcmc(
        context["plan"],
        context["edges"],
        pack.scalar_parameters,
        pack.row_parameters,
        [0, 2],
        np.array([0.43, 0.67]),
        [1],
        initial,
        density["log_pdf"],
        0,
        np.ascontiguousarray(draws[:1, :, 0]),
        np.ascontiguousarray(draws[:1, :, 1]),
        1,
        "full_recompute",
        fallback_budget - adapter_state_bytes - 1,
    )
    assert rejected_full["status"] == 2

    rng = np.random.default_rng(2026082253)
    state_before = deepcopy(rng.bit_generator.state)
    with pytest.raises(MemoryError, match="full-recompute fallback"):
        _cpp_rvine.mcmc(
            module,
            vine.pair_copulas,
            vine.d,
            vine._trees,
            vine._edge_map,
            parameters,
            n,
            rng,
            given,
            n_steps=1,
            burnin_steps=0,
            memory_budget_bytes=fallback_budget - 1,
        )
    assert rng.bit_generator.state == state_before

    incremental_rng = np.random.default_rng(2026082254)
    incremental_state_before = deepcopy(incremental_rng.bit_generator.state)
    with pytest.raises(MemoryError, match="incremental MCMC fixed workspace"):
        _cpp_rvine.mcmc(
            module,
            vine.pair_copulas,
            vine.d,
            vine._trees,
            vine._edge_map,
            parameters,
            n,
            incremental_rng,
            given,
            n_steps=1,
            burnin_steps=0,
            memory_budget_bytes=(
                incremental_reserved + bytes_per_step - 1),
            density_algorithm="incremental",
        )
    assert incremental_rng.bit_generator.state == incremental_state_before


def test_mcmc_direct_counters_offset_and_validation():
    vine = configured_static_dvine(4)
    n = 6
    parameters = scalar_parameters(vine)
    module, context, pack = _density_request(vine, parameters, n)
    given_indices = [0, 2]
    given_values = np.array([0.43, 0.67])
    free_indices = [1, 3]
    current = np.random.default_rng(2026082246).uniform(
        0.05, 0.95, size=(n, vine.d))
    current[:, given_indices] = given_values
    density = module.rvine_log_pdf_rows(
        context["plan"],
        context["edges"],
        pack.scalar_parameters,
        pack.row_parameters,
        current,
    )
    _cpp_rvine.raise_for_status(density, "test initial density")
    proposal = np.full((5, n), 0.5)
    acceptance = np.full((5, n), 0.9)
    result = module.rvine_mcmc_chunk(
        context["plan"],
        context["edges"],
        pack.scalar_parameters,
        pack.row_parameters,
        given_indices,
        given_values,
        free_indices,
        current,
        density["log_pdf"],
        1,
        proposal,
        acceptance,
        4,
    )
    _cpp_rvine.raise_for_status(result, "test MCMC")
    diagnostics = dict(result["diagnostics"])
    assert diagnostics["n_threads_requested"] == 4
    assert diagnostics["n_threads_used"] == 1
    assert diagnostics["proposed"] == [2 * n, 3 * n]
    assert sum(diagnostics["accepted"]) <= 5 * n
    assert diagnostics["non_finite_proposals"] == 0

    invalid = module.rvine_mcmc_chunk(
        context["plan"],
        context["edges"],
        pack.scalar_parameters,
        pack.row_parameters,
        given_indices,
        given_values,
        free_indices,
        current,
        density["log_pdf"],
        -1,
        proposal,
        acceptance,
    )
    assert invalid["status"] == 2


def test_mcmc_algorithms_report_the_same_invalid_row_location():
    vine = configured_static_dvine(4)
    n = 4
    parameters = scalar_parameters(vine)
    module, context, pack = _density_request(vine, parameters, n)
    given_indices = [0, 2]
    given_values = np.array([0.43, 0.67])
    current = np.full((n, vine.d), 0.5)
    current[:, given_indices] = given_values
    current[2, 1] = np.nan

    failures = []
    for algorithm in ("full_recompute", "incremental"):
        result = module.rvine_mcmc_chunk(
            context["plan"],
            context["edges"],
            pack.scalar_parameters,
            pack.row_parameters,
            given_indices,
            given_values,
            [1, 3],
            current,
            np.zeros(n),
            0,
            np.full((1, n), 0.5),
            np.full((1, n), 0.5),
            1,
            algorithm,
        )
        failures.append((
            int(result["status"]),
            int(result["failure_row"]),
            int(result["failure_edge"]),
            int(result["failure_operation"]),
        ))

    assert failures == [(6, 2, -1, -1)] * 2


def test_mcmc_direct_counts_non_finite_proposals_without_failing_chunk():
    vine = configured_mixed_family_vine()
    n = 3
    parameters = scalar_parameters(vine)
    parameters[(0, 0)] = np.array([1e308])
    module, context, pack = _density_request(vine, parameters, n)
    given_indices = [0, 2]
    given_values = np.array([0.4, 0.6])
    current = np.full((n, vine.d), 0.5)
    current[:, given_indices] = given_values

    results = {}
    for algorithm in ("full_recompute", "incremental"):
        result = module.rvine_mcmc_chunk(
            context["plan"],
            context["edges"],
            pack.scalar_parameters,
            pack.row_parameters,
            given_indices,
            given_values,
            [1],
            current,
            np.zeros(n),
            0,
            np.full((1, n), 1e-300),
            np.full((1, n), 0.5),
            1,
            algorithm,
        )
        _cpp_rvine.raise_for_status(
            result, f"test non-finite {algorithm} MCMC proposal")
        results[algorithm] = result

    expected = results["full_recompute"]
    actual = results["incremental"]
    np.testing.assert_array_equal(actual["state"], expected["state"])
    np.testing.assert_array_equal(actual["log_pdf"], expected["log_pdf"])
    for result in results.values():
        diagnostics = dict(result["diagnostics"])
        assert diagnostics["proposed"] == [n]
        assert diagnostics["accepted"] == [0]
        assert diagnostics["non_finite_proposals"] == n
        np.testing.assert_array_equal(
            np.asarray(result["state"]).reshape(n, vine.d), current)


def test_custom_builtin_subclass_uses_density_and_mcmc_python_fallback(
        monkeypatch):
    calls = []

    class CustomClayton(ClaytonCopula):
        def log_pdf(self, u1, u2, r):
            calls.append(len(np.asarray(u1)))
            return np.full_like(np.asarray(u1, dtype=np.float64), 0.125)

    vine = configured_mixed_family_vine()
    vine.pair_copulas[(0, 0)] = fitted_pair(
        CustomClayton(rotate=90), 0.8)
    parameters = scalar_parameters(vine)
    observations = np.full((4, vine.d), 0.5)
    density = vine._log_pdf_rows_with_r_python(observations, parameters)
    assert density.shape == (4,)
    assert calls

    initial = observations.copy()
    initial[:, 0] = 0.57
    vine._sample_arbitrary_given_mcmc_python(
        4,
        parameters,
        np.random.default_rng(1),
        {0: 0.57, 2: 0.31},
        initial=initial,
        n_steps=1,
        burnin_steps=0,
    )
    assert len(calls) >= 3

    actual_density = vine._log_pdf_rows_with_r(observations, parameters)
    np.testing.assert_array_equal(actual_density, density)
    expected_mcmc = vine._sample_arbitrary_given_mcmc_python(
        4,
        parameters,
        np.random.default_rng(1),
        {0: 0.57, 2: 0.31},
        initial=initial,
        n_steps=1,
        burnin_steps=0,
    )
    actual_mcmc = vine._sample_arbitrary_given_mcmc(
        4,
        parameters,
        np.random.default_rng(1),
        {0: 0.57, 2: 0.31},
        initial=initial,
        n_steps=1,
        burnin_steps=0,
    )
    np.testing.assert_array_equal(actual_mcmc[0], expected_mcmc[0])
    assert actual_mcmc[1] == expected_mcmc[1]


def test_density_context_reuses_plan_and_refreshes_mutable_edge(monkeypatch):
    vine = configured_mixed_family_vine()
    parameters = scalar_parameters(vine)
    observations = np.full((5, vine.d), 0.5)

    vine._log_pdf_rows_with_r(observations, parameters)
    first = vine._native_rvine_cache["density"]
    first_plan = first["plan"]
    first_edges = first["edges"]
    vine._log_pdf_rows_with_r(observations[:3], parameters)
    reused = vine._native_rvine_cache["density"]
    assert reused["plan"] is first_plan
    assert all(
        current is previous
        for current, previous in zip(reused["edges"], first_edges)
    )

    vine.pair_copulas[(0, 0)].copula._rotate = 180
    vine._log_pdf_rows_with_r(observations, parameters)
    refreshed = vine._native_rvine_cache["density"]
    assert refreshed["plan"] is not first_plan
    assert refreshed["edges"][0] is not first_edges[0]


def test_native_density_releases_the_gil():
    vine = configured_static_dvine(25)
    vine.pair_copulas = {
        key: fitted_pair(BivariateGaussianCopula(), 0.05)
        for key in vine.pair_copulas
    }
    n = 3000
    parameters = scalar_parameters(vine)
    module, context, pack = _density_request(vine, parameters, n)
    observations = np.random.default_rng(2026082247).uniform(
        0.02, 0.98, size=(n, vine.d))
    started = threading.Event()
    stop = threading.Event()
    counter = [0]

    def worker():
        started.set()
        while not stop.is_set():
            counter[0] += 1

    thread = threading.Thread(target=worker)
    thread.start()
    assert started.wait(timeout=2.0)
    before = counter[0]
    try:
        result = module.rvine_log_pdf_rows(
            context["plan"],
            context["edges"],
            pack.scalar_parameters,
            pack.row_parameters,
            observations,
        )
    finally:
        stop.set()
        thread.join()
    _cpp_rvine.raise_for_status(result, "test density")
    assert counter[0] > before


def test_native_mcmc_releases_the_gil():
    vine = configured_static_dvine(20)
    vine.pair_copulas = {
        key: fitted_pair(BivariateGaussianCopula(), 0.05)
        for key in vine.pair_copulas
    }
    n = 1500
    parameters = scalar_parameters(vine)
    module, context, pack = _density_request(vine, parameters, n)
    given_indices = [0, 2]
    given_values = np.array([0.4, 0.6])
    current = np.random.default_rng(2026082254).uniform(
        0.02, 0.98, size=(n, vine.d))
    current[:, given_indices] = given_values
    density = module.rvine_log_pdf_rows(
        context["plan"],
        context["edges"],
        pack.scalar_parameters,
        pack.row_parameters,
        current,
    )
    _cpp_rvine.raise_for_status(density, "test MCMC initial density")
    free_indices = [
        variable for variable in range(vine.d)
        if variable not in given_indices
    ]
    proposal = np.full((6, n), 0.5)
    acceptance = np.full((6, n), 0.9)
    started = threading.Event()
    stop = threading.Event()
    counter = [0]

    def worker():
        started.set()
        while not stop.is_set():
            counter[0] += 1

    thread = threading.Thread(target=worker)
    thread.start()
    assert started.wait(timeout=2.0)
    before = counter[0]
    try:
        result = module.rvine_mcmc_chunk(
            context["plan"],
            context["edges"],
            pack.scalar_parameters,
            pack.row_parameters,
            given_indices,
            given_values,
            free_indices,
            current,
            density["log_pdf"],
            0,
            proposal,
            acceptance,
        )
    finally:
        stop.set()
        thread.join()
    _cpp_rvine.raise_for_status(result, "test MCMC GIL release")
    assert counter[0] > before
