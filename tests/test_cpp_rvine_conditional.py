"""Contracts for native suffix and DAG R-vine sampling."""

from __future__ import annotations

from collections import Counter
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
from pyscarcopula._constants import PSEUDO_OBS_EPS
from pyscarcopula._native import _extension as _cpp_extension, vine as _cpp_rvine
from pyscarcopula._native.errors import NativeUnsupported
from pyscarcopula.vine._rvine_dag import (
    build_runtime_rvine_dag,
    plan_conditional_sample,
)
from pyscarcopula.vine._rvine_suffix import (
    SuffixConditionalPlan,
    build_suffix_conditional_plan,
)
from rvine_candidate_harness import (
    _sample_dag_given_with_r_python,
    _vine_sample_suffix_given_with_r_python,
)

from rvine_runtime_cases import (
    configured_mixed_family_vine,
    configured_mixed_gas_vine,
    configured_static_dvine,
    fitted_pair,
    scalar_parameters,
)


pytestmark = pytest.mark.rvine_native


_FAMILY_ROTATION_CELLS = [
    pytest.param(IndependentCopula, 0, 0.0, id="independent-r0"),
    *[
        pytest.param(
            family,
            rotation,
            parameter,
            id=f"{family.__name__}-r{rotation}",
        )
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


def _suffix_plan(vine, given):
    start_col = vine._given_suffix_start_col(given)
    assert start_col is not None
    return start_col, build_suffix_conditional_plan(
        vine.d, start_col, vine.matrix, given)


def _dag_plan(vine, given):
    return plan_conditional_sample(
        build_runtime_rvine_dag(vine.matrix, vine._edge_map),
        given,
        vine.d,
    )


def _native_request(vine, plan, given, parameters, n):
    module = _cpp_extension.load()
    active_keys = _cpp_rvine.conditional_active_keys(plan)
    normalized, sources = _cpp_rvine.conditional_parameter_layout(
        vine.pair_copulas, active_keys, parameters, n)
    edges, pack = _cpp_rvine.compile_edge_specs(
        module,
        vine.pair_copulas,
        active_keys,
        normalized,
        n,
        parameter_sources=sources,
    )
    native = _cpp_rvine.compile_conditional_plan(
        module, plan, active_keys, given)
    given_values = _cpp_rvine._conditional_given_values(
        vine.d, given, native.given_variables)
    return module, active_keys, native, edges, pack, given_values


@pytest.mark.parametrize(
    ("family", "rotation", "parameter"), _FAMILY_ROTATION_CELLS)
def test_suffix_native_matches_every_family_and_rotation_exactly(
        monkeypatch, family, rotation, parameter):
    vine = configured_mixed_family_vine()
    vine.pair_copulas[(0, 0)] = fitted_pair(
        family(rotate=rotation), parameter)
    given = {2: 0.42}
    start_col, _ = _suffix_plan(vine, given)
    parameters = scalar_parameters(vine)
    uniforms = np.asfortranarray(
        np.random.default_rng(2026082201).uniform(
            PSEUDO_OBS_EPS,
            1.0 - PSEUDO_OBS_EPS,
            size=(19, vine.d),
        )
    )
    before = uniforms.copy()

    expected = _vine_sample_suffix_given_with_r_python(vine,
        len(uniforms),
        parameters,
        np.random.default_rng(1),
        given,
        start_col,
        uniforms=uniforms,
    )
    actual = vine._sample_suffix_given_with_r(
        len(uniforms),
        parameters,
        np.random.default_rng(2),
        given,
        start_col,
        uniforms=uniforms,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(uniforms, before)
    assert actual.flags.c_contiguous


def test_dag_native_matches_python_for_non_suffix_given_exactly(
        monkeypatch):
    vine = configured_static_dvine(4)
    given = {0: 0.57, 2: 0.31}
    plan = _dag_plan(vine, given)
    assert vine._suffix_sampling_state(given) is None
    edge_visits = Counter(
        tuple(step["edge"])
        for step in plan
        if step["action"] in {"h_prop", "h_inv"}
    )
    assert max(edge_visits.values()) > 1
    parameters = scalar_parameters(vine)
    draw_count = sum(
        step["action"] == "sample_uniform" for step in plan)
    uniforms = np.asfortranarray(
        np.random.default_rng(2026082202).uniform(
            0.01, 0.99, size=(23, draw_count)))

    expected = _sample_dag_given_with_r_python(vine,
        len(uniforms),
        parameters,
        np.random.default_rng(3),
        given,
        plan,
        vine.pair_copulas,
        uniforms=uniforms,
    )
    actual = vine._sample_dag_given_with_r(
        len(uniforms),
        parameters,
        np.random.default_rng(4),
        given,
        plan,
        vine.pair_copulas,
        uniforms=uniforms,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[:, 0], np.full(len(actual), 0.57))
    np.testing.assert_array_equal(actual[:, 2], np.full(len(actual), 0.31))


def test_native_suffix_one_free_column_and_single_row_are_exact(monkeypatch):
    vine = configured_static_dvine(4)
    given = {
        int(vine.matrix[vine.d - 1 - column, column]):
            0.19 + 0.13 * column
        for column in range(1, vine.d)
    }
    start_col, _ = _suffix_plan(vine, given)
    assert start_col == 1
    uniforms = np.array([[0.17, 0.37, 0.57, 0.77]], dtype=np.float64)
    parameters = scalar_parameters(vine)

    expected = _vine_sample_suffix_given_with_r_python(vine,
        1, parameters, np.random.default_rng(1), given, start_col,
        uniforms=uniforms)
    actual = vine._sample_suffix_given_with_r(
        1, parameters, np.random.default_rng(2), given, start_col,
        uniforms=uniforms)

    np.testing.assert_array_equal(actual, expected)
    for variable, value in given.items():
        np.testing.assert_array_equal(actual[:, variable], [value])


def test_suffix_native_matches_transposed_edge_orientation(monkeypatch):
    vine = configured_static_dvine(3, order=(1, 0, 2))
    given = {2: 0.4}
    start_col, plan = _suffix_plan(vine, given)
    native = _cpp_rvine.compile_conditional_plan(
        _cpp_extension.load(),
        plan,
        _cpp_rvine.conditional_active_keys(plan),
        given,
    )
    assert 1 in native.transposed
    parameters = scalar_parameters(vine)
    uniforms = np.random.default_rng(2026082208).uniform(
        0.01, 0.99, size=(21, vine.d))

    expected = _vine_sample_suffix_given_with_r_python(vine,
        21, parameters, np.random.default_rng(1), given, start_col,
        uniforms=uniforms)
    actual = vine._sample_suffix_given_with_r(
        21, parameters, np.random.default_rng(2), given, start_col,
        uniforms=uniforms)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("kind", ["suffix", "dag"])
def test_native_conditional_rng_state_matches_python(kind, monkeypatch):
    vine = configured_mixed_family_vine()
    parameters = scalar_parameters(vine)
    if kind == "suffix":
        given = {2: 0.42}
        start_col, _ = _suffix_plan(vine, given)

        def execute_python(rng):
            return _vine_sample_suffix_given_with_r_python(vine,
                17, parameters, rng, given, start_col)

        def execute_native(rng):
            return vine._sample_suffix_given_with_r(
                17, parameters, rng, given, start_col)
    else:
        vine = configured_static_dvine(4)
        parameters = scalar_parameters(vine)
        given = {0: 0.57, 2: 0.31}
        plan = _dag_plan(vine, given)

        def execute_python(rng):
            return _sample_dag_given_with_r_python(vine,
                17, parameters, rng, given, plan, vine.pair_copulas)

        def execute_native(rng):
            return vine._sample_dag_given_with_r(
                17, parameters, rng, given, plan, vine.pair_copulas)

    python_rng = np.random.default_rng(2026082203)
    native_rng = np.random.default_rng(2026082203)
    expected = execute_python(python_rng)
    actual = execute_native(native_rng)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(native_rng.random(32), python_rng.random(32))


def test_native_suffix_supports_mixed_scalar_and_row_parameter_paths(
        monkeypatch):
    vine = configured_mixed_family_vine()
    given = {2: 0.42}
    start_col, _ = _suffix_plan(vine, given)
    parameters = scalar_parameters(vine)
    parameters[(0, 0)] = np.linspace(0.55, 1.05, 13)
    uniforms = np.random.default_rng(2026082204).uniform(
        0.01, 0.99, size=(13, vine.d))

    expected = _vine_sample_suffix_given_with_r_python(vine,
        13, parameters, np.random.default_rng(1), given, start_col,
        uniforms=uniforms)
    actual = vine._sample_suffix_given_with_r(
        13, parameters, np.random.default_rng(2), given, start_col,
        uniforms=uniforms)

    np.testing.assert_array_equal(actual, expected)


def test_native_suffix_supports_predictive_dynamic_row_paths(monkeypatch):
    vine = configured_mixed_gas_vine()
    given = {2: 0.42}
    start_col, _ = _suffix_plan(vine, given)
    parameters = scalar_parameters(vine)
    parameters[(0, 1)] = np.linspace(-0.4, 0.4, 11)
    uniforms = np.random.default_rng(2026082205).uniform(
        0.01, 0.99, size=(11, vine.d))

    expected = _vine_sample_suffix_given_with_r_python(vine,
        11, parameters, np.random.default_rng(1), given, start_col,
        uniforms=uniforms)
    actual = vine._sample_suffix_given_with_r(
        11, parameters, np.random.default_rng(2), given, start_col,
        uniforms=uniforms)

    np.testing.assert_array_equal(actual, expected)


def test_conditional_context_is_reused_and_semantic_mutation_refreshes_it(
        monkeypatch):
    vine = configured_mixed_family_vine()
    given = {2: 0.42}
    start_col, _ = _suffix_plan(vine, given)
    parameters = scalar_parameters(vine)

    vine._sample_suffix_given_with_r(
        8, parameters, np.random.default_rng(1), given, start_col)
    conditional = vine._native_rvine_cache["conditional"]
    first = next(iter(conditional.values()))
    first_plan = first["plan"]
    first_edges = first["edges"]

    vine._sample_suffix_given_with_r(
        5, parameters, np.random.default_rng(2), {2: 0.61}, start_col)
    reused = next(iter(vine._native_rvine_cache["conditional"].values()))
    assert reused["plan"] is first_plan
    assert all(
        current is previous
        for current, previous in zip(reused["edges"], first_edges)
    )

    vine.pair_copulas[(0, 0)].copula._rotate = 180
    vine._sample_suffix_given_with_r(
        5, parameters, np.random.default_rng(3), given, start_col)
    refreshed = next(iter(vine._native_rvine_cache["conditional"].values()))
    assert refreshed["plan"] is not first_plan
    assert refreshed["edges"][0] is not first_edges[0]


def test_conditional_cache_distinguishes_full_immutable_plan_signature(
        monkeypatch):
    vine = configured_mixed_family_vine()
    given = {2: 0.42}
    _, plan = _suffix_plan(vine, given)
    parameters = scalar_parameters(vine)
    module = _cpp_extension.load()

    first, _ = vine._native_conditional_context(
        module, plan, vine.pair_copulas, parameters, given, 4)
    with pytest.raises(AttributeError):
        plan.append({"action": "copy"})

    changed = SuffixConditionalPlan(
        [
            *plan,
            {
                "action": "copy",
                "from": (0, frozenset()),
                "to": ("unused", frozenset()),
            },
        ],
        plan.d,
    )
    second, _ = vine._native_conditional_context(
        module, changed, vine.pair_copulas, parameters, given, 4)

    assert second is not first
    assert second["plan"] is not first["plan"]
    assert len(second["plan"].opcodes) == len(first["plan"].opcodes) + 1
    assert changed.native_signature_digest != plan.native_signature_digest


def test_conditional_cache_reuses_only_unchanged_scalar_parameter_pack():
    vine = configured_mixed_family_vine()
    given = {2: 0.42}
    _, plan = _suffix_plan(vine, given)
    parameters = scalar_parameters(vine)
    module = _cpp_extension.load()

    context, first = vine._native_conditional_context(
        module, plan, vine.pair_copulas, parameters, given, 4)
    reused_context, reused = vine._native_conditional_context(
        module, plan, vine.pair_copulas, parameters, given, 7)
    assert reused_context is context
    assert reused.scalar_parameters is first.scalar_parameters
    assert reused.row_parameters.shape == (7, 0)

    key = next(
        key for key in _cpp_rvine.conditional_active_keys(plan)
        if not isinstance(vine.pair_copulas[key].copula, IndependentCopula)
    )
    parameters[key] = np.array([float(parameters[key][0]) + 0.05])
    changed_context, changed = vine._native_conditional_context(
        module, plan, vine.pair_copulas, parameters, given, 7)
    assert changed_context is context
    assert changed.scalar_parameters is not first.scalar_parameters

    _, changed_reused = vine._native_conditional_context(
        module, plan, vine.pair_copulas, parameters, given, 3)
    assert changed_reused.scalar_parameters is changed.scalar_parameters
    assert changed_reused.row_parameters.shape == (3, 0)


def test_missing_used_parameter_path_is_rejected_before_rng(monkeypatch):
    vine = configured_mixed_family_vine()
    given = {2: 0.42}
    start_col, plan = _suffix_plan(vine, given)
    parameters = scalar_parameters(vine)
    missing = next(
        key for key in _cpp_rvine.conditional_active_keys(plan)
        if not isinstance(vine.pair_copulas[key].copula, IndependentCopula)
    )
    del parameters[missing]
    actual_rng = np.random.default_rng(2026082206)
    expected_rng = np.random.default_rng(2026082206)

    with pytest.raises(KeyError, match="missing R-vine parameter path"):
        vine._sample_suffix_given_with_r(
            7, parameters, actual_rng, given, start_col)
    np.testing.assert_array_equal(
        actual_rng.random(24), expected_rng.random(24))


@pytest.mark.parametrize("case", ["missing", "wrong_length", "uniform_shape"])
def test_conditional_python_and_native_reject_the_same_invalid_requests(
        monkeypatch, case):
    vine = configured_mixed_family_vine()
    given = {2: 0.42}
    start_col, plan = _suffix_plan(vine, given)
    parameters = scalar_parameters(vine)
    n = 7
    uniforms = np.full((n, vine.d), 0.5, dtype=np.float64)
    used = next(
        key for key in _cpp_rvine.conditional_active_keys(plan)
        if not isinstance(vine.pair_copulas[key].copula, IndependentCopula)
    )
    if case == "missing":
        del parameters[used]
    elif case == "wrong_length":
        parameters[used] = np.full(n - 1, 0.7, dtype=np.float64)
    else:
        uniforms = uniforms[:, :-1]

    errors = []
    for executor in (
            lambda *args, **kwargs: _vine_sample_suffix_given_with_r_python(
                vine, *args, **kwargs),
            vine._sample_suffix_given_with_r):
        with pytest.raises(Exception) as captured:
            executor(
                n,
                parameters,
                np.random.default_rng(1),
                given,
                start_col,
                uniforms=uniforms,
            )
        errors.append(type(captured.value))

    if case == "wrong_length":
        assert issubclass(errors[0], RuntimeError)
        assert errors[1] is ValueError
    else:
        assert errors[1] is errors[0]
        assert errors[1] is (
            KeyError if case == "missing" else ValueError)


def test_custom_builtin_subclass_is_rejected_by_conditional_runtime():
    class CustomClayton(ClaytonCopula):
        def h_inverse(self, v, u_given, r):
            return np.full_like(np.asarray(v, dtype=np.float64), 0.271)

    vine = configured_mixed_family_vine()
    vine.pair_copulas[(0, 0)] = fitted_pair(CustomClayton(), 0.8)
    given = {2: 0.42}
    start_col, _ = _suffix_plan(vine, given)
    parameters = scalar_parameters(vine)
    uniforms = np.full((4, vine.d), 0.5, dtype=np.float64)

    with pytest.raises(NativeUnsupported, match="exact registered"):
        vine._sample_suffix_given_with_r(
            4, parameters, np.random.default_rng(1), given, start_col,
            uniforms=uniforms)


def test_native_conditional_direct_validation_and_diagnostics():
    vine = configured_mixed_family_vine()
    given = {2: 0.42}
    _, plan = _suffix_plan(vine, given)
    n = 7
    parameters = scalar_parameters(vine)
    module, _, native, edges, pack, given_values = _native_request(
        vine, plan, given, parameters, n)
    uniforms = np.full((n, vine.d), 0.5, dtype=np.float64)

    result = module.rvine_conditional_sample(
        native,
        edges,
        pack.scalar_parameters,
        pack.row_parameters,
        given_values,
        uniforms,
        4,
    )
    _cpp_rvine.raise_for_status(result, "test conditional sample")
    diagnostics = dict(result["diagnostics"])
    assert diagnostics["n_threads_requested"] == 4
    assert diagnostics["n_threads_used"] == 1
    assert diagnostics["h_pair_operations"] > 0
    assert diagnostics["inverse_operations"] > 0
    assert diagnostics["independence_fast_paths"] > 0
    assert diagnostics["row_blocks"] == 1
    assert diagnostics["max_block_rows"] == n
    assert diagnostics["peak_workspace_bytes"] > 0

    malformed = _cpp_rvine.compile_conditional_plan(
        module,
        plan,
        _cpp_rvine.conditional_active_keys(plan),
        given,
    )
    malformed.output_nodes = [malformed.node_count] * vine.d
    invalid_plan = module.rvine_conditional_sample(
        malformed,
        edges,
        pack.scalar_parameters,
        pack.row_parameters,
        given_values,
        uniforms,
    )
    assert invalid_plan["status"] == 2

    invalid_rows = module.rvine_conditional_sample(
        native,
        edges,
        pack.scalar_parameters,
        np.empty((n + 1, 0), dtype=np.float64),
        given_values,
        uniforms,
    )
    assert invalid_rows["status"] == 2

    bad_uniforms = uniforms.copy()
    bad_uniforms[3, 1] = 1.0
    invalid_uniform = module.rvine_conditional_sample(
        native,
        edges,
        pack.scalar_parameters,
        pack.row_parameters,
        given_values,
        bad_uniforms,
    )
    assert invalid_uniform["status"] == 6
    assert invalid_uniform["failure_row"] == 3


def test_conditional_native_bounds_workspace_and_preserves_thread_parity(
        monkeypatch):
    vine = configured_mixed_family_vine()
    given = {2: 0.42}
    start_col, plan = _suffix_plan(vine, given)
    n = 2053
    parameters = scalar_parameters(vine)
    module, _, native, edges, pack, given_values = _native_request(
        vine, plan, given, parameters, n)
    uniforms = np.random.default_rng(2026082213).uniform(
        0.01, 0.99, size=(n, vine.d))

    sequential = module.rvine_conditional_sample(
        native,
        edges,
        pack.scalar_parameters,
        pack.row_parameters,
        given_values,
        uniforms,
        1,
    )
    requested_many = module.rvine_conditional_sample(
        native,
        edges,
        pack.scalar_parameters,
        pack.row_parameters,
        given_values,
        uniforms,
        4,
    )
    _cpp_rvine.raise_for_status(sequential, "test conditional sample")
    _cpp_rvine.raise_for_status(requested_many, "test conditional sample")
    np.testing.assert_array_equal(
        sequential["values"], requested_many["values"])

    diagnostics = dict(requested_many["diagnostics"])
    assert diagnostics["n_threads_requested"] == 4
    assert diagnostics["n_threads_used"] == 1
    assert diagnostics["row_blocks"] == 9
    assert diagnostics["max_block_rows"] == 256
    assert diagnostics["peak_workspace_bytes"] <= 64 * 1024 * 1024

    expected = _vine_sample_suffix_given_with_r_python(vine,
        n, parameters, np.random.default_rng(1), given, start_col,
        uniforms=uniforms)
    np.testing.assert_array_equal(
        np.asarray(sequential["values"]).reshape(n, vine.d), expected)


def test_native_conditional_sample_releases_the_gil_for_block_execution():
    vine = configured_static_dvine(30)
    vine.pair_copulas = {
        key: fitted_pair(BivariateGaussianCopula(), 0.05)
        for key in vine.pair_copulas
    }
    given = {int(vine.matrix[0, vine.d - 1]): 0.4}
    _, plan = _suffix_plan(vine, given)
    n = 2000
    module, _, native, edges, pack, given_values = _native_request(
        vine, plan, given, scalar_parameters(vine), n)
    uniforms = np.random.default_rng(2026082214).uniform(
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
        result = module.rvine_conditional_sample(
            native,
            edges,
            pack.scalar_parameters,
            pack.row_parameters,
            given_values,
            uniforms,
        )
    finally:
        stop.set()
        thread.join()

    _cpp_rvine.raise_for_status(result, "test conditional sample")
    assert counter[0] > before


def test_conditional_adapter_supports_empty_rows_and_independence(monkeypatch):
    vine = configured_static_dvine(4, independent=True)
    given = {3: 0.4}
    start_col, plan = _suffix_plan(vine, given)
    assert start_col == 3
    result = _cpp_rvine.conditional_sample(
        _cpp_extension.load(),
        vine.pair_copulas,
        plan,
        0,
        np.random.default_rng(1),
        given,
        {},
        uniforms=np.empty((0, vine.d), dtype=np.float64),
    )
    assert result.shape == (0, vine.d)
    assert result.dtype == np.float64

    uniforms = np.random.default_rng(2026082209).uniform(
        0.01, 0.99, size=(9, vine.d))
    parameters = scalar_parameters(vine)
    expected = _vine_sample_suffix_given_with_r_python(vine,
        9, parameters, np.random.default_rng(1), given, start_col,
        uniforms=uniforms)
    actual = vine._sample_suffix_given_with_r(
        9, parameters, np.random.default_rng(2), given, start_col,
        uniforms=uniforms)
    np.testing.assert_array_equal(actual, expected)


def test_all_given_predict_remains_early_and_rng_free(monkeypatch):
    vine = configured_mixed_family_vine()
    given = {0: 0.2, 1: 0.4, 2: 0.6}
    actual_rng = np.random.default_rng(2026082207)
    expected_rng = np.random.default_rng(2026082207)

    result, diagnostics = vine.predict(
        6, given=given, rng=actual_rng, return_diagnostics=True)

    np.testing.assert_array_equal(
        result,
        np.tile(np.array([0.2, 0.4, 0.6]), (6, 1)),
    )
    assert diagnostics["all_variables_given"] is True
    assert "conditional" not in vine._native_rvine_cache
    np.testing.assert_array_equal(
        actual_rng.random(24), expected_rng.random(24))


def test_none_given_predict_stays_on_unconditional_runtime(monkeypatch):
    vine = configured_mixed_family_vine()

    result = vine.predict(7, given={}, rng=np.random.default_rng(2026082210))

    assert result.shape == (7, vine.d)
    assert "unconditional" in vine._native_rvine_cache
    assert "conditional" not in vine._native_rvine_cache


def test_conditional_context_is_excluded_from_persistence(monkeypatch):
    vine = configured_mixed_family_vine()
    given = {2: 0.42}
    start_col, _ = _suffix_plan(vine, given)
    vine._sample_suffix_given_with_r(
        4,
        scalar_parameters(vine),
        np.random.default_rng(1),
        given,
        start_col,
    )
    assert "conditional" in vine._native_rvine_cache
    assert "_native_rvine_cache" not in vine.__getstate__()
    restored = deepcopy(vine)
    assert restored._native_rvine_cache == {}
