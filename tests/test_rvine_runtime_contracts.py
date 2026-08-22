"""Oracle and dispatch contracts for the R-vine runtime."""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pyscarcopula
from pyscarcopula import (
    BivariateGaussianCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    IndependentCopula,
    JoeCopula,
)
from pyscarcopula._constants import PSEUDO_OBS_EPS
from pyscarcopula.numerical import _cpp_extension
from pyscarcopula.numerical._cpp_extension import (
    CppUnavailable,
    CppUnsupported,
)
from pyscarcopula.numerical._rvine_backend import (
    _RVINE_BACKEND_ENV,
    dispatch_rvine_backend,
    native_rvine_symbol_available,
    rvine_backend_mode,
)
from pyscarcopula.stattests import (
    _rvine_rosenblatt_transform_python,
    _student_rosenblatt_transform_python,
    rvine_rosenblatt_transform,
    student_rosenblatt_transform,
)
from pyscarcopula.vine._rvine_dag import (
    _execute_conditional_plan_python,
    build_runtime_rvine_dag,
    plan_conditional_sample,
)
from pyscarcopula.vine.vine import VineCopula

from rvine_runtime_cases import (
    configured_mixed_gas_vine,
    configured_mixed_family_vine,
    configured_static_dvine,
    fitted_pair,
    scalar_parameters,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "rvine_runtime_oracle_v1.json"


@pytest.fixture
def golden():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _fixture_parameters(payload):
    return {
        tuple(int(part) for part in key.split(",")): np.asarray(
            value, dtype=np.float64)
        for key, value in payload["inputs"]["parameter_paths"].items()
    }


def _assert_rng_equal(left, right):
    np.testing.assert_array_equal(left.random(16), right.random(16))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("auto", "auto"),
        ("NATIVE_STRICT", "native_strict"),
        (" python_executor ", "python_executor"),
    ],
)
def test_private_backend_selector_accepts_three_modes(
        monkeypatch, value, expected):
    monkeypatch.setenv(_RVINE_BACKEND_ENV, value)
    assert rvine_backend_mode() == expected


def test_private_backend_selector_rejects_unknown_mode(monkeypatch):
    monkeypatch.setenv(_RVINE_BACKEND_ENV, "silent-fallback")
    with pytest.raises(ValueError, match=_RVINE_BACKEND_ENV):
        rvine_backend_mode()


def test_python_executor_ignores_advertised_native_symbol(monkeypatch):
    monkeypatch.setenv(_RVINE_BACKEND_ENV, "python_executor")
    monkeypatch.setattr(
        _cpp_extension,
        "load",
        lambda: SimpleNamespace(rvine_sample=object()),
    )
    calls = []
    result = dispatch_rvine_backend(
        capability="unconditional_sampling",
        native_symbol="rvine_sample",
        python_executor=lambda: calls.append("python") or "oracle",
        native_executor=lambda _module: calls.append("native") or "native",
    )
    assert result == "oracle"
    assert calls == ["python"]


def test_auto_falls_back_only_when_new_symbol_is_missing(monkeypatch):
    monkeypatch.setenv(_RVINE_BACKEND_ENV, "auto")
    monkeypatch.setattr(_cpp_extension, "load", lambda: SimpleNamespace())
    assert dispatch_rvine_backend(
        capability="log_pdf_rows",
        native_symbol="rvine_log_pdf_rows",
        python_executor=lambda: "oracle",
    ) == "oracle"


@pytest.mark.parametrize("mode", ["auto", "native_strict"])
def test_auto_and_native_strict_select_advertised_native_symbol(
        monkeypatch, mode):
    monkeypatch.setenv(_RVINE_BACKEND_ENV, mode)
    module = SimpleNamespace(rvine_sample=object())
    monkeypatch.setattr(_cpp_extension, "load", lambda: module)
    calls = []
    result = dispatch_rvine_backend(
        capability="unconditional_sampling",
        native_symbol="rvine_sample",
        python_executor=lambda: calls.append("python") or "oracle",
        native_executor=lambda received: (
            calls.append("native") or (
                "native" if received is module else "wrong module")
        ),
    )
    assert result == "native"
    assert calls == ["native"]


def test_native_strict_rejects_missing_new_symbol(monkeypatch):
    monkeypatch.setenv(_RVINE_BACKEND_ENV, "native_strict")
    monkeypatch.setattr(_cpp_extension, "load", lambda: SimpleNamespace())
    with pytest.raises(CppUnsupported, match="rvine_log_pdf_rows"):
        dispatch_rvine_backend(
            capability="log_pdf_rows",
            native_symbol="rvine_log_pdf_rows",
            python_executor=lambda: pytest.fail("unexpected fallback"),
        )


def test_auto_does_not_hide_native_validation_failure(monkeypatch):
    monkeypatch.setenv(_RVINE_BACKEND_ENV, "auto")
    monkeypatch.setattr(
        _cpp_extension,
        "load",
        lambda: SimpleNamespace(rvine_sample=object()),
    )

    def invalid(_module):
        raise ValueError("invalid packed plan")

    with pytest.raises(ValueError, match="invalid packed plan"):
        dispatch_rvine_backend(
            capability="unconditional_sampling",
            native_symbol="rvine_sample",
            python_executor=lambda: pytest.fail("unexpected fallback"),
            native_executor=invalid,
        )


def test_auto_falls_back_on_declared_unsupported_capability(monkeypatch):
    monkeypatch.setenv(_RVINE_BACKEND_ENV, "auto")
    monkeypatch.setattr(
        _cpp_extension,
        "load",
        lambda: SimpleNamespace(rvine_sample=object()),
    )
    calls = []

    def unsupported(_module):
        calls.append("native")
        return None

    result = dispatch_rvine_backend(
        capability="unconditional_sampling",
        native_symbol="rvine_sample",
        python_executor=lambda: calls.append("python") or "oracle",
        native_executor=unsupported,
    )
    assert result == "oracle"
    assert calls == ["native", "python"]


def test_auto_propagates_unsupported_raised_after_native_entry(monkeypatch):
    monkeypatch.setenv(_RVINE_BACKEND_ENV, "auto")
    monkeypatch.setattr(
        _cpp_extension,
        "load",
        lambda: SimpleNamespace(rvine_sample=object()),
    )

    def unsupported(_module):
        raise CppUnsupported("custom edge")

    with pytest.raises(CppUnsupported, match="custom edge"):
        dispatch_rvine_backend(
            capability="unconditional_sampling",
            native_symbol="rvine_sample",
            python_executor=lambda: pytest.fail("unexpected fallback"),
            native_executor=unsupported,
        )


@pytest.mark.parametrize("mode", ["auto", "native_strict", "python_executor"])
def test_missing_base_extension_remains_cpp_unavailable(monkeypatch, mode):
    monkeypatch.setenv(_RVINE_BACKEND_ENV, mode)

    def unavailable():
        raise CppUnavailable("compiled extension missing")

    monkeypatch.setattr(_cpp_extension, "load", unavailable)
    with pytest.raises(CppUnavailable, match="compiled extension missing"):
        dispatch_rvine_backend(
            capability="unconditional_sampling",
            native_symbol="rvine_sample",
            python_executor=lambda: pytest.fail("unexpected pure Python path"),
        )


def test_extension_loader_reports_real_import_failure(monkeypatch):
    monkeypatch.setattr(_cpp_extension, "_MODULE", None)
    monkeypatch.setattr(_cpp_extension, "_MODULE_ERROR", None)
    original_import = _cpp_extension.importlib.import_module

    def fail_extension_import(name):
        if name == "pyscarcopula._scar_cpp":
            raise ImportError("synthetic missing _scar_cpp")
        return original_import(name)

    monkeypatch.setattr(
        _cpp_extension.importlib, "import_module", fail_extension_import)
    with pytest.raises(CppUnavailable, match="synthetic missing _scar_cpp"):
        _cpp_extension.load()


def test_selector_is_not_part_of_public_signatures_or_exports():
    for callable_object in (
            VineCopula.sample,
            VineCopula.predict,
            rvine_rosenblatt_transform,
            student_rosenblatt_transform):
        parameters = inspect.signature(callable_object).parameters
        assert "backend" not in parameters
        assert _RVINE_BACKEND_ENV not in parameters
    assert not hasattr(pyscarcopula, "rvine_backend_mode")
    assert not hasattr(pyscarcopula, "_sample_with_r_python")


def test_oracle_fixture_has_immutable_provenance(golden):
    assert golden["schema_version"] == 1
    assert golden["fixture_id"] == "rvine-runtime-oracle-v1"
    assert golden["provenance"]["automatic_regeneration"] is False
    assert len(golden["provenance"]["source_commit"]) == 40
    assert "Python traversal" in golden["provenance"]["oracle"]
    assert len(golden["provenance"]["extension_source_commit"]) == 40
    assert "_student_rosenblatt_transform_python" in (
        golden["provenance"]["extension_oracle"])


def test_python_oracles_match_runtime_golden_exactly(monkeypatch, golden):
    monkeypatch.setenv(_RVINE_BACKEND_ENV, "python_executor")
    vine = configured_mixed_family_vine()
    inputs = golden["inputs"]
    expected = golden["expected"]
    uniforms = np.asarray(inputs["uniforms"], dtype=np.float64)
    r_all = _fixture_parameters(golden)
    observations = np.asarray(inputs["observations"], dtype=np.float64)

    unconditional = vine._sample_with_r(
        len(uniforms),
        r_all,
        np.random.default_rng(1),
        uniforms=uniforms,
    )
    np.testing.assert_array_equal(
        unconditional, expected["unconditional_sample"])

    suffix_given = {
        int(key): value for key, value in inputs["suffix_given"].items()
    }
    suffix = vine._sample_suffix_given_with_r(
        len(uniforms),
        r_all,
        np.random.default_rng(2),
        suffix_given,
        vine._given_suffix_start_col(suffix_given),
        uniforms=uniforms,
    )
    np.testing.assert_array_equal(suffix, expected["suffix_sample"])

    dag_given = {
        int(key): value for key, value in inputs["dag_given"].items()
    }
    dag = build_runtime_rvine_dag(vine.matrix, vine._edge_map)
    plan = plan_conditional_sample(dag, dag_given, vine.d)
    dag_sample = vine._sample_dag_given_with_r(
        len(uniforms),
        r_all,
        np.random.default_rng(3),
        dag_given,
        plan,
        vine.pair_copulas,
        uniforms=np.asarray(inputs["dag_uniforms"], dtype=np.float64),
    )
    np.testing.assert_array_equal(dag_sample, expected["dag_sample"])

    log_pdf = vine._log_pdf_rows_with_r(observations, r_all)
    np.testing.assert_array_equal(log_pdf, expected["log_pdf_rows"])

    transformed = rvine_rosenblatt_transform(vine, observations)
    np.testing.assert_array_equal(
        transformed, expected["rvine_rosenblatt"])

    student = student_rosenblatt_transform(
        np.asarray(inputs["student_correlation"], dtype=np.float64),
        float(inputs["student_df"]),
        observations,
    )
    np.testing.assert_allclose(
        student, expected["student_rosenblatt"], rtol=2e-15, atol=2e-15)

    student_df_path = student_rosenblatt_transform(
        np.asarray(inputs["student_correlation"], dtype=np.float64),
        np.asarray(inputs["student_df_path"], dtype=np.float64),
        observations,
    )
    np.testing.assert_array_equal(
        student_df_path, expected["student_rosenblatt_df_path"])

    student_low_df = student_rosenblatt_transform(
        np.asarray(inputs["student_correlation"], dtype=np.float64),
        float(inputs["student_low_df"]),
        np.asarray(inputs["student_low_df_observations"], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        student_low_df, expected["student_rosenblatt_low_df"])

    mcmc, diagnostics = vine._sample_arbitrary_given_mcmc(
        len(uniforms),
        r_all,
        np.random.default_rng(4),
        dag_given,
        initial=dag_sample,
        n_steps=3,
        burnin_steps=2,
        random_draws=np.asarray(
            inputs["mcmc_random_draws"], dtype=np.float64),
    )
    np.testing.assert_array_equal(mcmc, expected["mcmc_sample"])
    assert json.loads(json.dumps(diagnostics)) == expected["mcmc_diagnostics"]


@pytest.mark.rvine_native
def test_auto_dense_student_capability_fallback_matches_gate_zero_golden(
        monkeypatch, golden):
    monkeypatch.setenv(_RVINE_BACKEND_ENV, "auto")
    inputs = golden["inputs"]
    expected = golden["expected"]
    correlation = np.asarray(
        inputs["student_correlation"], dtype=np.float64)

    path_result = student_rosenblatt_transform(
        correlation,
        np.asarray(inputs["student_df_path"], dtype=np.float64),
        np.asarray(inputs["observations"], dtype=np.float64),
    )
    low_df_result = student_rosenblatt_transform(
        correlation,
        float(inputs["student_low_df"]),
        np.asarray(inputs["student_low_df_observations"], dtype=np.float64),
    )

    np.testing.assert_array_equal(
        path_result, expected["student_rosenblatt_df_path"])
    np.testing.assert_array_equal(
        low_df_result, expected["student_rosenblatt_low_df"])


def test_replay_inputs_are_owned_contiguous_and_do_not_mutate_or_consume_rng(
        golden):
    vine = configured_mixed_family_vine()
    r_all = _fixture_parameters(golden)
    base = np.asarray(golden["inputs"]["uniforms"], dtype=np.float64)
    non_contiguous = np.asfortranarray(base)
    before = non_contiguous.copy()
    actual_rng = np.random.default_rng(9123)
    expected_rng = np.random.default_rng(9123)

    result = vine._sample_with_r_python(
        len(base), r_all, actual_rng, uniforms=non_contiguous)

    assert result.flags.c_contiguous
    np.testing.assert_array_equal(non_contiguous, before)
    _assert_rng_equal(actual_rng, expected_rng)


def test_sampling_rng_state_and_output_match_across_batch_sizes(monkeypatch):
    monkeypatch.setenv(_RVINE_BACKEND_ENV, "python_executor")
    vine = configured_mixed_family_vine()
    one_rng = np.random.default_rng(20260814)
    many_rng = np.random.default_rng(20260814)
    one_batch = vine.sample(97, rng=one_rng, batch_rows=97)
    many_batches = vine.sample(97, rng=many_rng, batch_rows=7)
    np.testing.assert_array_equal(one_batch, many_batches)
    _assert_rng_equal(one_rng, many_rng)


@pytest.mark.rvine_native
def test_existing_gas_native_and_python_selector_consume_identical_rng(
        monkeypatch):
    if not native_rvine_symbol_available("gas_rvine_sample"):
        pytest.skip("existing _scar_cpp.gas_rvine_sample is unavailable")
    vine = configured_mixed_gas_vine()
    python_rng = np.random.default_rng(202608141)
    native_rng = np.random.default_rng(202608141)
    monkeypatch.setenv(_RVINE_BACKEND_ENV, "python_executor")
    expected = vine.sample(64, rng=python_rng)
    monkeypatch.setenv(_RVINE_BACKEND_ENV, "native_strict")
    actual = vine.sample(64, rng=native_rng)
    np.testing.assert_array_equal(actual, expected)
    _assert_rng_equal(native_rng, python_rng)


def test_memory_budget_rejection_precedes_rng_consumption(monkeypatch):
    monkeypatch.setenv(_RVINE_BACKEND_ENV, "python_executor")
    vine = configured_mixed_family_vine()
    actual_rng = np.random.default_rng(118)
    expected_rng = np.random.default_rng(118)
    with pytest.raises(MemoryError, match="memory_budget_bytes"):
        vine.sample(
            100,
            rng=actual_rng,
            batch_rows=10,
            memory_budget_bytes=1,
        )
    _assert_rng_equal(actual_rng, expected_rng)


def test_mcmc_interleaved_draw_replay_preserves_output_and_rng_state(golden):
    vine = configured_mixed_family_vine()
    r_all = _fixture_parameters(golden)
    inputs = golden["inputs"]
    initial = np.asarray(golden["expected"]["dag_sample"], dtype=np.float64)
    given = {int(key): value for key, value in inputs["dag_given"].items()}
    seed = 771
    actual_rng = np.random.default_rng(seed)
    replay_rng = np.random.default_rng(seed)

    actual, actual_diag = vine._sample_arbitrary_given_mcmc_python(
        len(initial),
        r_all,
        actual_rng,
        given,
        initial=initial,
        n_steps=3,
        burnin_steps=2,
    )
    draws = np.empty((5, len(initial), 2), dtype=np.float64)
    for step in range(5):
        draws[step, :, 0] = replay_rng.uniform(
            PSEUDO_OBS_EPS, 1.0 - PSEUDO_OBS_EPS, size=len(initial))
        draws[step, :, 1] = replay_rng.uniform(
            PSEUDO_OBS_EPS, 1.0, size=len(initial))
    replay, replay_diag = vine._sample_arbitrary_given_mcmc_python(
        len(initial),
        r_all,
        np.random.default_rng(999),
        given,
        initial=initial,
        n_steps=3,
        burnin_steps=2,
        random_draws=draws,
    )
    np.testing.assert_array_equal(actual, replay)
    assert actual_diag == replay_diag
    _assert_rng_equal(actual_rng, replay_rng)


def test_dag_draw_replay_preserves_plan_order_and_rng_state(golden):
    vine = configured_mixed_family_vine()
    r_all = _fixture_parameters(golden)
    given = {
        int(key): value
        for key, value in golden["inputs"]["dag_given"].items()
    }
    plan = plan_conditional_sample(
        build_runtime_rvine_dag(vine.matrix, vine._edge_map),
        given,
        vine.d,
    )
    seed = 881
    actual_rng = np.random.default_rng(seed)
    replay_rng = np.random.default_rng(seed)
    actual = vine._sample_dag_given_with_r_python(
        4,
        r_all,
        actual_rng,
        given,
        plan,
        vine.pair_copulas,
    )
    draw_count = sum(step["action"] == "sample_uniform" for step in plan)
    draws = np.column_stack([
        replay_rng.uniform(
            PSEUDO_OBS_EPS, 1.0 - PSEUDO_OBS_EPS, size=4)
        for _ in range(draw_count)
    ])
    replay = vine._sample_dag_given_with_r_python(
        4,
        r_all,
        np.random.default_rng(999),
        given,
        plan,
        vine.pair_copulas,
        uniforms=draws,
    )
    np.testing.assert_array_equal(actual, replay)
    _assert_rng_equal(actual_rng, replay_rng)


@pytest.mark.parametrize("chunks", [(1, 1, 1, 1, 1), (2, 3), (3, 2)])
def test_mcmc_chunk_boundaries_preserve_cyclic_coordinate(chunks, golden):
    vine = configured_mixed_family_vine()
    r_all = _fixture_parameters(golden)
    initial = np.asarray(golden["expected"]["dag_sample"], dtype=np.float64)
    draws = np.asarray(
        golden["inputs"]["mcmc_random_draws"], dtype=np.float64)
    given = {
        int(key): value
        for key, value in golden["inputs"]["dag_given"].items()
    }
    expected, _ = vine._sample_arbitrary_given_mcmc_python(
        len(initial),
        r_all,
        np.random.default_rng(1),
        given,
        initial=initial,
        n_steps=5,
        burnin_steps=0,
        random_draws=draws,
    )

    offset = 0
    chunked = initial
    for chunk_size in chunks:
        chunked, _ = vine._sample_arbitrary_given_mcmc_python(
            len(initial),
            r_all,
            np.random.default_rng(2),
            given,
            initial=chunked,
            n_steps=chunk_size,
            burnin_steps=0,
            random_draws=draws[offset:offset + chunk_size],
            step_offset=offset,
        )
        offset += chunk_size
    assert offset == len(draws)
    np.testing.assert_array_equal(chunked, expected)


@pytest.mark.parametrize(
    ("family", "rotation", "parameter"),
    [
        *[(ClaytonCopula, rotation, 0.8) for rotation in (0, 90, 180, 270)],
        *[(GumbelCopula, rotation, 1.6) for rotation in (0, 90, 180, 270)],
        *[(JoeCopula, rotation, 1.7) for rotation in (0, 90, 180, 270)],
        (FrankCopula, 0, 2.5),
        (BivariateGaussianCopula, 0, -0.4),
        (IndependentCopula, 0, 0.0),
    ],
)
def test_python_sampling_oracle_characterizes_public_pair_matrix(
        family, rotation, parameter):
    vine = configured_static_dvine(2)
    copula = family() if family is IndependentCopula else family(rotate=rotation)
    vine.pair_copulas[(0, 0)] = fitted_pair(copula, parameter)
    uniforms = np.array([
        [PSEUDO_OBS_EPS * 2.0, 0.2],
        [0.35, 0.65],
        [0.8, 1.0 - PSEUDO_OBS_EPS * 2.0],
    ])
    r_all = scalar_parameters(vine)
    result = vine._sample_with_r_python(
        len(uniforms),
        r_all,
        np.random.default_rng(1),
        uniforms=uniforms,
    )
    assert result.shape == uniforms.shape
    assert result.dtype == np.float64
    assert result.flags.c_contiguous
    assert np.all(np.isfinite(result))
    assert np.all((result > 0.0) & (result < 1.0))


def test_internal_oracles_cover_empty_singleton_and_invalid_replay_shapes():
    vine = configured_mixed_family_vine()
    scalar = scalar_parameters(vine)
    empty = vine._sample_with_r_python(
        0,
        scalar,
        np.random.default_rng(1),
        uniforms=np.empty((0, vine.d)),
    )
    assert empty.shape == (0, vine.d)
    singleton = vine._sample_with_r_python(
        1,
        scalar,
        np.random.default_rng(1),
        uniforms=np.full((1, vine.d), 0.5),
    )
    assert singleton.shape == (1, vine.d)
    with pytest.raises(ValueError, match="shape"):
        vine._sample_with_r_python(
            2,
            scalar,
            np.random.default_rng(1),
            uniforms=np.full((2, vine.d + 1), 0.5),
        )
    with pytest.raises(ValueError, match="open unit interval"):
        vine._sample_with_r_python(
            1,
            scalar,
            np.random.default_rng(1),
            uniforms=np.array([[0.0, 0.5, 1.0]]),
        )
    with pytest.raises(TypeError, match="real values"):
        vine._sample_with_r_python(
            1,
            scalar,
            np.random.default_rng(1),
            uniforms=np.full((1, vine.d), 0.5 + 0.1j),
        )
    with pytest.raises(ValueError, match="finite values"):
        vine._sample_with_r_python(
            1,
            scalar,
            np.random.default_rng(1),
            uniforms=np.array([[0.5, np.nan, 0.5]]),
        )


def test_mixed_scalar_and_row_paths_are_not_expanded_by_oracle(golden):
    vine = configured_mixed_family_vine()
    r_all = _fixture_parameters(golden)
    snapshots = {key: value.copy() for key, value in r_all.items()}
    vine._sample_with_r_python(
        4,
        r_all,
        np.random.default_rng(1),
        uniforms=np.asarray(golden["inputs"]["uniforms"]),
    )
    assert r_all[(0, 0)].shape == (4,)
    assert r_all[(0, 1)].shape == (1,)
    assert r_all[(1, 0)].shape == (1,)
    for key in r_all:
        np.testing.assert_array_equal(r_all[key], snapshots[key])


def test_custom_builtin_subclass_keeps_python_override(monkeypatch, golden):
    class CustomClayton(ClaytonCopula):
        def h_inverse(self, v, u_given, r):
            return np.full_like(np.asarray(v, dtype=np.float64), 0.314)

    monkeypatch.setenv(_RVINE_BACKEND_ENV, "auto")
    vine = configured_mixed_family_vine()
    vine.pair_copulas[(0, 0)] = fitted_pair(CustomClayton(rotate=0), 0.8)
    r_all = _fixture_parameters(golden)
    result = vine._sample_with_r(
        4,
        r_all,
        np.random.default_rng(1),
        uniforms=np.asarray(golden["inputs"]["uniforms"]),
    )
    np.testing.assert_array_equal(result[:, 0], np.full(4, 0.314))


def test_public_predict_diagnostics_contract(monkeypatch):
    monkeypatch.setenv(_RVINE_BACKEND_ENV, "python_executor")
    vine = configured_mixed_family_vine()
    given = {2: 0.42}
    samples, diagnostics = vine.predict(
        5,
        given=given,
        rng=np.random.default_rng(44),
        return_diagnostics=True,
    )
    assert samples.shape == (5, 3)
    np.testing.assert_array_equal(samples[:, 2], np.full(5, 0.42))
    required = {
        "given",
        "dynamic_conditioning",
        "suffix_start_col",
        "matrix_rebuilt",
        "conditional_method",
        "updated_edges",
        "skipped_edges",
        "timings_ms",
    }
    assert required <= diagnostics.keys()
    assert diagnostics["conditional_method"] == "suffix"
    assert all(
        isinstance(value, float) and value >= 0.0
        for value in diagnostics["timings_ms"].values()
    )


def test_oracle_names_are_stable_and_independent():
    vine = configured_mixed_family_vine()
    assert callable(vine._sample_with_r_python)
    assert callable(vine._sample_suffix_given_with_r_python)
    assert callable(vine._sample_dag_given_with_r_python)
    assert callable(vine._log_pdf_rows_with_r_python)
    assert callable(vine._sample_arbitrary_given_mcmc_python)
    assert callable(_execute_conditional_plan_python)
    assert callable(_rvine_rosenblatt_transform_python)
    assert callable(_student_rosenblatt_transform_python)


@pytest.mark.parametrize(
    "native_symbol",
    [
        "rvine_sample",
        "rvine_conditional_sample",
        "rvine_log_pdf_rows",
        "rvine_mcmc_chunk",
        "rvine_rosenblatt_transform",
        "dense_student_rosenblatt_transform",
    ],
)
def test_native_strict_differential_harness_reserved_for_future_entry_points(
        monkeypatch, native_symbol, golden):
    if not native_rvine_symbol_available(native_symbol):
        pytest.skip(f"_scar_cpp.{native_symbol} is not implemented yet")

    vine = configured_mixed_family_vine()
    r_all = _fixture_parameters(golden)
    observations = np.asarray(golden["inputs"]["observations"])
    uniforms = np.asarray(golden["inputs"]["uniforms"])

    def run():
        if native_symbol == "rvine_sample":
            return vine._sample_with_r(
                4, r_all, np.random.default_rng(1), uniforms=uniforms)
        if native_symbol == "rvine_conditional_sample":
            given = {
                int(key): value
                for key, value in golden["inputs"]["suffix_given"].items()
            }
            return vine._sample_suffix_given_with_r(
                4,
                r_all,
                np.random.default_rng(2),
                given,
                vine._given_suffix_start_col(given),
                uniforms=uniforms,
            )
        if native_symbol == "rvine_log_pdf_rows":
            return vine._log_pdf_rows_with_r(observations, r_all)
        if native_symbol == "rvine_mcmc_chunk":
            given = {
                int(key): value
                for key, value in golden["inputs"]["dag_given"].items()
            }
            return vine._sample_arbitrary_given_mcmc(
                4,
                r_all,
                np.random.default_rng(4),
                given,
                initial=np.asarray(golden["expected"]["dag_sample"]),
                n_steps=3,
                burnin_steps=2,
                random_draws=np.asarray(
                    golden["inputs"]["mcmc_random_draws"]),
            )[0]
        if native_symbol == "rvine_rosenblatt_transform":
            return rvine_rosenblatt_transform(vine, observations)
        if native_symbol == "dense_student_rosenblatt_transform":
            return student_rosenblatt_transform(
                np.asarray(golden["inputs"]["student_correlation"]),
                golden["inputs"]["student_df"],
                observations,
            )
        raise AssertionError(f"unhandled native symbol {native_symbol}")

    monkeypatch.setenv(_RVINE_BACKEND_ENV, "python_executor")
    expected = deepcopy(run())
    monkeypatch.setenv(_RVINE_BACKEND_ENV, "native_strict")
    actual = run()
    if native_symbol == "dense_student_rosenblatt_transform":
        np.testing.assert_allclose(
            actual, expected, rtol=5e-12, atol=5e-13)
    else:
        np.testing.assert_array_equal(actual, expected)
