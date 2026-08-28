"""Native-boundary tests for R-vine density and MCMC execution."""

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
from pyscarcopula._native import _extension as _cpp_extension, vine as _cpp_rvine
from pyscarcopula._native.errors import NativeUnsupported
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


def test_custom_builtin_subclass_is_rejected_by_density_and_mcmc():
    class CustomClayton(ClaytonCopula):
        def log_pdf(self, u1, u2, r):
            return np.full_like(np.asarray(u1, dtype=np.float64), 0.125)

    vine = configured_mixed_family_vine()
    vine.pair_copulas[(0, 0)] = fitted_pair(
        CustomClayton(rotate=90), 0.8)
    parameters = scalar_parameters(vine)
    observations = np.full((4, vine.d), 0.5)
    initial = observations.copy()
    initial[:, 0] = 0.57
    with pytest.raises(NativeUnsupported, match="exact registered"):
        vine._log_pdf_rows_with_r(observations, parameters)
    with pytest.raises(NativeUnsupported, match="exact registered"):
        vine._sample_arbitrary_given_mcmc(
            4,
            parameters,
            np.random.default_rng(1),
            {0: 0.57, 2: 0.31},
            initial=initial,
            n_steps=1,
            burnin_steps=0,
        )


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
