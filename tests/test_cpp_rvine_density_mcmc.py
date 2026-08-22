"""Differential and native-boundary tests for R-vine stage 4."""

from __future__ import annotations

from copy import deepcopy
import threading
import time

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
from pyscarcopula.numerical._cpp_extension import CppUnsupported
from pyscarcopula.numerical._rvine_backend import _RVINE_BACKEND_ENV

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

    monkeypatch.setenv(_RVINE_BACKEND_ENV, "python_executor")
    expected = vine._log_pdf_rows_with_r(observations, parameters)
    monkeypatch.setenv(_RVINE_BACKEND_ENV, "native_strict")
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

    monkeypatch.setenv(_RVINE_BACKEND_ENV, "python_executor")
    expected = vine._log_pdf_rows_with_r(observations, parameters)
    monkeypatch.setenv(_RVINE_BACKEND_ENV, "native_strict")
    actual = vine._log_pdf_rows_with_r(observations, parameters)
    np.testing.assert_array_equal(actual, expected)

    scalar = scalar_parameters(vine)
    singleton = observations[:1]
    monkeypatch.setenv(_RVINE_BACKEND_ENV, "python_executor")
    expected_one = vine._log_pdf_rows_with_r(singleton, scalar)
    monkeypatch.setenv(_RVINE_BACKEND_ENV, "native_strict")
    actual_one = vine._log_pdf_rows_with_r(singleton, scalar)
    np.testing.assert_array_equal(actual_one, expected_one)

    empty = vine._log_pdf_rows_with_r(
        np.empty((0, vine.d), dtype=np.float64), scalar)
    assert empty.shape == (0,)
    assert empty.dtype == np.float64


def test_density_and_mcmc_reject_parameter_outside_family_domain_before_rng(
        monkeypatch):
    vine = configured_static_dvine(2)
    parameters = scalar_parameters(vine)
    parameters[(0, 0)] = np.array([2.0])
    observations = np.array([[0.3, 0.7]])

    messages = []
    for mode in ("python_executor", "native_strict"):
        monkeypatch.setenv(_RVINE_BACKEND_ENV, mode)
        with pytest.raises(ValueError, match="must lie in") as exc_info:
            vine._log_pdf_rows_with_r(observations, parameters)
        messages.append(str(exc_info.value))
    assert messages[0] == messages[1]

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

    monkeypatch.setenv(_RVINE_BACKEND_ENV, "python_executor")
    expected, expected_diagnostics = vine._sample_arbitrary_given_mcmc(
        n,
        parameters,
        python_rng,
        given,
        initial=initial,
        n_steps=257,
        burnin_steps=3,
        step_offset=1,
    )
    monkeypatch.setenv(_RVINE_BACKEND_ENV, "native_strict")
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
    native_mcmc = module.rvine_mcmc
    chunk_sizes = []

    def recording_mcmc(*args, **kwargs):
        chunk_sizes.append(np.asarray(args[10]).shape[0])
        return native_mcmc(*args, **kwargs)

    monkeypatch.setattr(module, "rvine_mcmc", recording_mcmc)
    bytes_per_step = 2 * n * np.dtype(np.float64).itemsize
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
        memory_budget_bytes=2 * bytes_per_step,
    )
    np.testing.assert_array_equal(actual, expected)
    assert actual_diagnostics == expected_diagnostics
    assert chunk_sizes == [2, 2, 1]

    rng = np.random.default_rng(2026082253)
    state_before = deepcopy(rng.bit_generator.state)
    with pytest.raises(MemoryError, match="one proposal/acceptance step"):
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
            memory_budget_bytes=bytes_per_step - 1,
        )
    assert rng.bit_generator.state == state_before


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
    result = module.rvine_mcmc(
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

    invalid = module.rvine_mcmc(
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

    result = module.rvine_mcmc(
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
    )
    _cpp_rvine.raise_for_status(result, "test non-finite MCMC proposal")
    diagnostics = dict(result["diagnostics"])
    assert diagnostics["proposed"] == [n]
    assert diagnostics["accepted"] == [0]
    assert diagnostics["non_finite_proposals"] == n
    np.testing.assert_array_equal(
        np.asarray(result["state"]).reshape(n, vine.d), current)


def test_custom_builtin_subclass_falls_back_for_density_and_mcmc(monkeypatch):
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
    monkeypatch.setenv(_RVINE_BACKEND_ENV, "auto")
    density = vine._log_pdf_rows_with_r(observations, parameters)
    assert density.shape == (4,)
    assert calls

    initial = observations.copy()
    initial[:, 0] = 0.57
    vine._sample_arbitrary_given_mcmc(
        4,
        parameters,
        np.random.default_rng(1),
        {0: 0.57, 2: 0.31},
        initial=initial,
        n_steps=1,
        burnin_steps=0,
    )
    assert len(calls) >= 3

    monkeypatch.setenv(_RVINE_BACKEND_ENV, "native_strict")
    with pytest.raises(CppUnsupported, match="does not support"):
        vine._log_pdf_rows_with_r(observations, parameters)
    with pytest.raises(CppUnsupported, match="does not support"):
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
    monkeypatch.setenv(_RVINE_BACKEND_ENV, "native_strict")

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
        time.sleep(0.01)
        while not stop.is_set():
            counter[0] += 1

    thread = threading.Thread(target=worker)
    thread.start()
    started.wait()
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
        time.sleep(0.01)
        while not stop.is_set():
            counter[0] += 1

    thread = threading.Thread(target=worker)
    thread.start()
    started.wait()
    before = counter[0]
    try:
        result = module.rvine_mcmc(
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
