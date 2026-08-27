"""Contracts for native unconditional R-vine sampling."""

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
    RVineCopula,
)
from pyscarcopula._constants import PSEUDO_OBS_EPS
from pyscarcopula._native import _extension as _cpp_extension, vine as _cpp_rvine
from pyscarcopula._native.errors import NativeUnsupported
from pyscarcopula.vine._edge_adapter import edge_is_independent
from pyscarcopula.vine._rvine_sampling_plan import build_rvine_sampling_plan
from rvine_candidate_harness import _sample_with_r_python

from rvine_runtime_cases import (
    configured_mixed_family_vine,
    configured_static_dvine,
    fitted_pair,
    scalar_parameters,
)


pytestmark = pytest.mark.rvine_native


def _plan(vine):
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


def _packed_request(vine, n, parameters, *, parameter_sources=None):
    module = _cpp_extension.load()
    plan = _plan(vine)
    edges, pack = _cpp_rvine.compile_edge_specs(
        module,
        vine.pair_copulas,
        plan.active_keys,
        parameters,
        n,
        parameter_sources=parameter_sources,
    )
    return module, plan, _cpp_rvine.compile_traversal_plan(
        module, plan), edges, pack


_FAMILY_ROTATION_CELLS = [
    pytest.param(IndependentCopula, 0, 0.0, id="independent-r0"),
    *[
        pytest.param(family, rotation, parameter,
                     id=f"{family.__name__}-r{rotation}")
        for family, parameter in (
            (ClaytonCopula, 0.8),
            (GumbelCopula, 1.4),
            (JoeCopula, 1.5),
        )
        for rotation in (0, 90, 180, 270)
    ],
    pytest.param(FrankCopula, 0, 2.0, id="frank-r0"),
    pytest.param(BivariateGaussianCopula, 0, -0.35, id="gaussian-r0"),
]


@pytest.mark.parametrize(
    ("family", "rotation", "parameter"), _FAMILY_ROTATION_CELLS)
def test_native_matches_python_for_every_family_rotation_and_orientation(
        monkeypatch, family, rotation, parameter):
    vine = configured_mixed_family_vine()
    vine.pair_copulas[(0, 0)] = fitted_pair(
        family(rotate=rotation), parameter)
    plan = _plan(vine)
    parameters = scalar_parameters(vine)
    uniforms = np.random.default_rng(2026081501).uniform(
        PSEUDO_OBS_EPS,
        1.0 - PSEUDO_OBS_EPS,
        size=(19, vine.d),
    )

    expected = _sample_with_r_python(vine,
        len(uniforms),
        parameters,
        np.random.default_rng(1),
        traversal_plan=plan,
        uniforms=uniforms,
    )
    actual = vine._sample_with_r(
        len(uniforms),
        parameters,
        np.random.default_rng(2),
        traversal_plan=plan,
        uniforms=uniforms,
    )

    np.testing.assert_array_equal(actual, expected)


def test_native_matches_python_for_transposed_edges(monkeypatch):
    vine = configured_static_dvine(3, order=(1, 0, 2))
    plan = _plan(vine)
    assert 1 in plan.inverse_transposed
    assert 1 in plan.forward_transposed
    parameters = scalar_parameters(vine)
    uniforms = np.random.default_rng(2026081511).uniform(
        0.01, 0.99, size=(23, vine.d))

    expected = _sample_with_r_python(vine,
        len(uniforms), parameters, np.random.default_rng(1),
        traversal_plan=plan, uniforms=uniforms)
    actual = vine._sample_with_r(
        len(uniforms), parameters, np.random.default_rng(2),
        traversal_plan=plan, uniforms=uniforms)
    np.testing.assert_array_equal(actual, expected)


def test_native_mixed_scalar_row_path_and_noncontiguous_replay_are_exact(
        monkeypatch):
    vine = configured_mixed_family_vine()
    plan = _plan(vine)
    parameters = scalar_parameters(vine)
    parameters[(0, 0)] = np.linspace(0.55, 1.05, 11)
    base = np.random.default_rng(2026081502).uniform(
        0.01, 0.99, size=(11, vine.d))
    uniforms = np.asfortranarray(base)
    before = uniforms.copy()

    expected = _sample_with_r_python(vine,
        len(uniforms), parameters, np.random.default_rng(3),
        traversal_plan=plan, uniforms=uniforms)
    actual_rng = np.random.default_rng(41)
    expected_rng = np.random.default_rng(41)
    actual = vine._sample_with_r(
        len(uniforms), parameters, actual_rng,
        traversal_plan=plan, uniforms=uniforms)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(uniforms, before)
    np.testing.assert_array_equal(actual_rng.random(16), expected_rng.random(16))
    assert actual.flags.c_contiguous


def test_native_adapter_supports_empty_and_single_row_batches(monkeypatch):
    vine = configured_mixed_family_vine()
    plan = _plan(vine)
    parameters = scalar_parameters(vine)
    module = _cpp_extension.load()
    scalar_sources = {
        key: "scalar"
        for key in plan.active_keys
        if not edge_is_independent(vine.pair_copulas[key])
    }

    empty = _cpp_rvine.sample(
        module,
        vine,
        0,
        np.random.default_rng(1),
        plan.active_keys,
        plan,
        parameters,
        uniforms=np.empty((0, vine.d), dtype=np.float64),
        parameter_sources=scalar_sources,
    )
    assert empty.shape == (0, vine.d)
    assert empty.dtype == np.float64

    uniforms = np.array([[PSEUDO_OBS_EPS, 0.5, 1.0 - PSEUDO_OBS_EPS]])
    expected = _sample_with_r_python(vine,
        1, parameters, np.random.default_rng(2), uniforms=uniforms)
    actual = vine._sample_with_r(
        1, parameters, np.random.default_rng(3), uniforms=uniforms)
    np.testing.assert_array_equal(actual, expected)


def test_native_supports_a_fully_independent_vine(monkeypatch):
    vine = configured_static_dvine(4, independent=True)
    plan = _plan(vine)
    assert plan.active_keys == ()
    uniforms = np.array([
        [1e-15, 0.25, 0.75, 1.0 - 1e-15],
        [0.9, 0.7, 0.3, 0.1],
    ], dtype=np.float64)

    expected = _sample_with_r_python(vine,
        len(uniforms), {}, np.random.default_rng(1),
        traversal_plan=plan, uniforms=uniforms)
    actual = vine._sample_with_r(
        len(uniforms), {}, np.random.default_rng(2),
        traversal_plan=plan, uniforms=uniforms)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, uniforms)


def test_native_result_reports_sequential_threads_and_independence_fast_path():
    vine = configured_mixed_family_vine()
    vine.pair_copulas[(0, 0)] = fitted_pair(IndependentCopula(), 0.0)
    n = 13
    parameters = scalar_parameters(vine)
    sources = {
        key: "scalar"
        for key, edge in vine.pair_copulas.items()
        if not edge_is_independent(edge)
    }
    module, python_plan, plan, edges, pack = _packed_request(
        vine, n, parameters, parameter_sources=sources)
    uniforms = np.ascontiguousarray(
        np.random.default_rng(4).uniform(0.02, 0.98, size=(n, vine.d)))

    sequential = module.rvine_sample(
        plan, edges, pack.scalar_parameters, pack.row_parameters, uniforms, 1)
    requested_many = module.rvine_sample(
        plan, edges, pack.scalar_parameters, pack.row_parameters, uniforms, 4)
    _cpp_rvine.raise_for_status(sequential, "test sample")
    _cpp_rvine.raise_for_status(requested_many, "test sample")

    np.testing.assert_array_equal(
        sequential["values"], requested_many["values"])
    diagnostics = dict(requested_many["diagnostics"])
    assert diagnostics["n_threads_requested"] == 4
    assert diagnostics["n_threads_used"] == 1
    assert diagnostics["inverse_operations"] == (
        n * len(python_plan.inverse_edges))
    assert diagnostics["forward_operations"] == (
        n * len(python_plan.forward_edges))
    assert diagnostics["independence_fast_paths"] > 0


def test_native_sample_releases_the_gil_for_the_row_loop():
    vine = configured_static_dvine(50)
    vine.pair_copulas = {
        key: fitted_pair(BivariateGaussianCopula(), 0.05)
        for key in vine.pair_copulas
    }
    n = 1000
    module, _, plan, edges, pack = _packed_request(
        vine, n, scalar_parameters(vine))
    uniforms = np.random.default_rng(2026081513).uniform(
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
        result = module.rvine_sample(
            plan,
            edges,
            pack.scalar_parameters,
            pack.row_parameters,
            uniforms,
        )
    finally:
        stop.set()
        thread.join()

    _cpp_rvine.raise_for_status(result, "test sample")
    assert counter[0] > before


def test_native_uses_fitted_result_as_the_independence_source_of_truth(
        monkeypatch):
    vine = configured_mixed_family_vine()
    logically_independent = fitted_pair(ClaytonCopula(), 0.8)
    logically_independent.fit_result = fitted_pair(
        IndependentCopula(), 0.0).fit_result
    vine.pair_copulas[(0, 0)] = logically_independent
    assert edge_is_independent(logically_independent)

    plan = _plan(vine)
    parameters = scalar_parameters(vine)
    uniforms = np.random.default_rng(2026081512).uniform(
        0.02, 0.98, size=(17, vine.d))

    expected = _sample_with_r_python(vine,
        len(uniforms), parameters, np.random.default_rng(1),
        traversal_plan=plan, uniforms=uniforms)
    actual = vine._sample_with_r(
        len(uniforms), parameters, np.random.default_rng(2),
        traversal_plan=plan, uniforms=uniforms)

    np.testing.assert_array_equal(actual, expected)


def test_native_revalidates_plan_and_parameter_row_shape():
    vine = configured_mixed_family_vine()
    n = 3
    parameters = scalar_parameters(vine)
    sources = {key: "scalar" for key in _plan(vine).active_keys}
    module, _, plan, edges, pack = _packed_request(
        vine, n, parameters, parameter_sources=sources)
    uniforms = np.full((n, vine.d), 0.5, dtype=np.float64)

    plan.output_nodes = [plan.node_count] * vine.d
    invalid_plan = module.rvine_sample(
        plan, edges, pack.scalar_parameters, pack.row_parameters, uniforms)
    assert invalid_plan["status"] == 2
    with pytest.raises(ValueError, match="invalid_size"):
        _cpp_rvine.raise_for_status(invalid_plan, "test sample")

    plan = _cpp_rvine.compile_traversal_plan(module, _plan(vine))
    plan.node_count = 2_000_000_000
    oversized_plan = module.rvine_sample(
        plan, edges, pack.scalar_parameters, pack.row_parameters, uniforms)
    assert oversized_plan["status"] == 2
    with pytest.raises(ValueError, match="invalid_size"):
        _cpp_rvine.raise_for_status(oversized_plan, "test sample")

    plan = _cpp_rvine.compile_traversal_plan(module, _plan(vine))
    bad_rows = np.empty((n + 1, 0), dtype=np.float64)
    invalid_pack = module.rvine_sample(
        plan, edges, pack.scalar_parameters, bad_rows, uniforms)
    assert invalid_pack["status"] == 2
    with pytest.raises(ValueError, match="invalid_size"):
        _cpp_rvine.raise_for_status(invalid_pack, "test sample")


def test_native_validation_happens_before_rng_consumption():
    vine = configured_mixed_family_vine()
    plan = _plan(vine)
    parameters = scalar_parameters(vine)
    parameters[(0, 0)] = np.array([0.7, np.nan, 0.9])
    actual_rng = np.random.default_rng(51)
    expected_rng = np.random.default_rng(51)

    with pytest.raises(ValueError, match="finite"):
        _cpp_rvine.sample(
            _cpp_extension.load(),
            vine,
            3,
            actual_rng,
            plan.active_keys,
            plan,
            parameters,
        )
    np.testing.assert_array_equal(actual_rng.random(16), expected_rng.random(16))


def test_public_native_sampling_preserves_batch_and_rng_contract(monkeypatch):
    vine = configured_mixed_family_vine()
    one_rng = np.random.default_rng(2026081503)
    many_rng = np.random.default_rng(2026081503)

    one_batch = vine.sample(97, rng=one_rng, batch_rows=97)
    many_batches = vine.sample(97, rng=many_rng, batch_rows=7)

    np.testing.assert_array_equal(one_batch, many_batches)
    np.testing.assert_array_equal(one_rng.random(16), many_rng.random(16))


def test_public_static_scalar_pack_is_compiled_once_per_request(monkeypatch):
    vine = configured_mixed_family_vine()
    calls = []
    original = _cpp_rvine.compile_edge_specs

    def counted(*args, **kwargs):
        calls.append((int(args[4]), kwargs.get("native_edges") is not None))
        return original(*args, **kwargs)

    monkeypatch.setattr(_cpp_rvine, "compile_edge_specs", counted)
    vine.sample(17, rng=np.random.default_rng(71), batch_rows=5)
    assert calls == [(5, False)]

    vine.sample(11, rng=np.random.default_rng(72), batch_rows=3)
    assert calls == [(5, False), (3, True)]


def test_public_native_memory_rejection_precedes_rng_consumption(monkeypatch):
    vine = configured_mixed_family_vine()
    actual_rng = np.random.default_rng(52)
    expected_rng = np.random.default_rng(52)

    with pytest.raises(MemoryError, match="memory_budget_bytes"):
        vine.sample(
            100,
            rng=actual_rng,
            batch_rows=10,
            memory_budget_bytes=1,
        )
    np.testing.assert_array_equal(actual_rng.random(16), expected_rng.random(16))


def test_compiled_context_is_reused_then_invalidated_and_not_persisted(
        monkeypatch):
    vine = configured_mixed_family_vine()
    vine.sample(8, rng=np.random.default_rng(61))
    first = vine._native_rvine_cache["unconditional"]
    first_plan = first["plan"]
    first_edges = first["edges"]

    vine.sample(5, rng=np.random.default_rng(62))
    reused = vine._native_rvine_cache["unconditional"]
    assert reused["plan"] is first_plan
    assert all(
        current is previous
        for current, previous in zip(reused["edges"], first_edges))

    edge_position = tuple(_plan(vine).active_keys).index((0, 0))
    vine.pair_copulas[(0, 0)].copula._rotate = 180
    vine.sample(5, rng=np.random.default_rng(63))
    rotated = vine._native_rvine_cache["unconditional"]
    assert rotated["edges"][edge_position] is not first_edges[edge_position]

    prior_result_edges = rotated["edges"]
    vine.pair_copulas[(0, 0)].fit_result = deepcopy(
        vine.pair_copulas[(0, 0)].fit_result)
    vine.sample(5, rng=np.random.default_rng(64))
    result_refreshed = vine._native_rvine_cache["unconditional"]
    assert result_refreshed["edges"][edge_position] is not (
        prior_result_edges[edge_position])

    vine.pair_copulas[(0, 0)] = fitted_pair(
        BivariateGaussianCopula(), 0.2)
    vine.sample(5, rng=np.random.default_rng(65))
    refreshed = vine._native_rvine_cache["unconditional"]
    assert refreshed["plan"] is not first_plan
    assert refreshed["edges"][edge_position] is not first_edges[edge_position]

    state = vine.__getstate__()
    assert "_native_rvine_cache" not in state
    assert "_native_rvine_generation" not in state
    restored = deepcopy(vine)
    assert restored._native_rvine_cache == {}
    assert restored._native_rvine_generation == 0

    generation = vine._native_rvine_generation
    vine.matrix = vine.matrix
    assert vine._native_rvine_cache == {}
    assert vine._native_rvine_generation == generation + 1


def test_refit_invalidates_native_cache(monkeypatch):
    data = np.random.default_rng(2026081514).uniform(
        0.01, 0.99, size=(40, 3))
    vine = RVineCopula(candidates=[IndependentCopula]).fit(data)
    vine.sample(4, rng=np.random.default_rng(1))
    assert vine._native_rvine_cache
    generation = vine._native_rvine_generation

    vine.fit(data)

    assert vine._native_rvine_cache == {}
    assert vine._native_rvine_generation > generation


def test_custom_builtin_subclass_is_rejected():
    class CustomClayton(ClaytonCopula):
        def h_inverse(self, v, u_given, r):
            return np.full_like(np.asarray(v, dtype=np.float64), 0.271)

    vine = configured_mixed_family_vine()
    vine.pair_copulas[(0, 0)] = fitted_pair(CustomClayton(), 0.8)
    uniforms = np.full((4, vine.d), 0.5, dtype=np.float64)
    parameters = scalar_parameters(vine)

    with pytest.raises(NativeUnsupported, match="exact registered"):
        vine._sample_with_r(
            4, parameters, np.random.default_rng(1), uniforms=uniforms)
