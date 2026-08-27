"""Oracle and dispatch contracts for the R-vine runtime."""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path

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
from pyscarcopula._native import _extension as _cpp_extension
from pyscarcopula._native.errors import (
    NativeUnavailable,
    NativeUnsupported,
)
from pyscarcopula.stattests import (
    rvine_rosenblatt_transform,
    student_rosenblatt_transform,
)
from pyscarcopula.vine._rvine_dag import (
    build_runtime_rvine_dag,
    plan_conditional_sample,
)
from pyscarcopula.vine.vine import VineCopula

from rvine_candidate_harness import (
    _execute_conditional_plan_python,
    _log_pdf_rows_with_r_python,
    _rvine_rosenblatt_transform_python,
    _sample_arbitrary_given_mcmc_python,
    _sample_dag_given_with_r_python,
    _sample_with_r_python,
    _vine_sample_suffix_given_with_r_python,
)

from rvine_runtime_cases import (
    configured_mixed_gas_vine,
    configured_mixed_family_vine,
    configured_static_dvine,
    fitted_pair,
    scalar_parameters,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "rvine_runtime_oracle_v1.json"
_GOLDEN_RTOL = 2e-15
_GOLDEN_ATOL = 2e-15
_SCIPY_GOLDEN_RTOL = 1e-10
_SCIPY_GOLDEN_ATOL = 1e-11


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


def _assert_golden_close(actual, expected):
    """Compare floating-point oracles across supported platform toolchains."""
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=_GOLDEN_RTOL,
        atol=_GOLDEN_ATOL,
    )


def _assert_scipy_golden_close(actual, expected):
    """Allow supported SciPy releases to differ in Student-t tails."""
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=_SCIPY_GOLDEN_RTOL,
        atol=_SCIPY_GOLDEN_ATOL,
    )


def _native_symbol_available(name):
    try:
        module = _cpp_extension.load()
    except NativeUnavailable:
        return False
    return callable(getattr(module, name, None))


def test_missing_base_extension_is_native_unavailable(monkeypatch):
    def unavailable():
        raise NativeUnavailable("compiled extension missing")

    monkeypatch.setattr(_cpp_extension, "load", unavailable)
    with pytest.raises(NativeUnavailable, match="compiled extension missing"):
        configured_static_dvine(3).sample(2, rng=np.random.default_rng(7))


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
    with pytest.raises(NativeUnavailable, match="synthetic missing _scar_cpp"):
        _cpp_extension.load()


def test_selector_is_not_part_of_public_signatures_or_exports():
    for callable_object in (
            VineCopula.sample,
            VineCopula.predict,
            rvine_rosenblatt_transform,
            student_rosenblatt_transform):
        parameters = inspect.signature(callable_object).parameters
        assert "backend" not in parameters
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


def test_preserved_python_oracles_match_runtime_golden(golden):
    vine = configured_mixed_family_vine()
    inputs = golden["inputs"]
    expected = golden["expected"]
    uniforms = np.asarray(inputs["uniforms"], dtype=np.float64)
    r_all = _fixture_parameters(golden)
    observations = np.asarray(inputs["observations"], dtype=np.float64)

    unconditional = _sample_with_r_python(vine,
        len(uniforms),
        r_all,
        np.random.default_rng(1),
        uniforms=uniforms,
    )
    _assert_golden_close(unconditional, expected["unconditional_sample"])

    suffix_given = {
        int(key): value for key, value in inputs["suffix_given"].items()
    }
    suffix = _vine_sample_suffix_given_with_r_python(vine,
        len(uniforms),
        r_all,
        np.random.default_rng(2),
        suffix_given,
        vine._given_suffix_start_col(suffix_given),
        uniforms=uniforms,
    )
    _assert_golden_close(suffix, expected["suffix_sample"])

    dag_given = {
        int(key): value for key, value in inputs["dag_given"].items()
    }
    dag = build_runtime_rvine_dag(vine.matrix, vine._edge_map)
    plan = plan_conditional_sample(dag, dag_given, vine.d)
    dag_sample = _sample_dag_given_with_r_python(vine,
        len(uniforms),
        r_all,
        np.random.default_rng(3),
        dag_given,
        plan,
        vine.pair_copulas,
        uniforms=np.asarray(inputs["dag_uniforms"], dtype=np.float64),
    )
    _assert_golden_close(dag_sample, expected["dag_sample"])

    log_pdf = _log_pdf_rows_with_r_python(vine, observations, r_all)
    _assert_golden_close(log_pdf, expected["log_pdf_rows"])

    transformed = _rvine_rosenblatt_transform_python(vine, observations)
    _assert_golden_close(transformed, expected["rvine_rosenblatt"])

    student = student_rosenblatt_transform(
        np.asarray(inputs["student_correlation"], dtype=np.float64),
        float(inputs["student_df"]),
        observations,
    )
    _assert_scipy_golden_close(student, expected["student_rosenblatt"])

    student_df_path = student_rosenblatt_transform(
        np.asarray(inputs["student_correlation"], dtype=np.float64),
        np.asarray(inputs["student_df_path"], dtype=np.float64),
        observations,
    )
    _assert_scipy_golden_close(
        student_df_path, expected["student_rosenblatt_df_path"])

    student_low_df = student_rosenblatt_transform(
        np.asarray(inputs["student_correlation"], dtype=np.float64),
        float(inputs["student_low_df"]),
        np.asarray(inputs["student_low_df_observations"], dtype=np.float64),
    )
    _assert_scipy_golden_close(
        student_low_df, expected["student_rosenblatt_low_df"])

    mcmc, diagnostics = _sample_arbitrary_given_mcmc_python(vine,
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
    _assert_golden_close(mcmc, expected["mcmc_sample"])
    assert json.loads(json.dumps(diagnostics)) == expected["mcmc_diagnostics"]


@pytest.mark.rvine_native
def test_dense_student_uses_mandatory_native_path(golden):
    inputs = golden["inputs"]
    correlation = np.asarray(
        inputs["student_correlation"], dtype=np.float64)
    observations = np.asarray(inputs["observations"], dtype=np.float64)
    low_df_observations = np.asarray(
        inputs["student_low_df_observations"], dtype=np.float64)
    df_path = np.asarray(inputs["student_df_path"], dtype=np.float64)
    low_df = float(inputs["student_low_df"])

    path_result = student_rosenblatt_transform(
        correlation,
        df_path,
        observations,
    )
    low_df_result = student_rosenblatt_transform(
        correlation,
        low_df,
        low_df_observations,
    )

    _assert_scipy_golden_close(
        path_result, golden["expected"]["student_rosenblatt_df_path"])
    _assert_scipy_golden_close(
        low_df_result, golden["expected"]["student_rosenblatt_low_df"])


def test_replay_inputs_are_owned_contiguous_and_do_not_mutate_or_consume_rng(
        golden):
    vine = configured_mixed_family_vine()
    r_all = _fixture_parameters(golden)
    base = np.asarray(golden["inputs"]["uniforms"], dtype=np.float64)
    non_contiguous = np.asfortranarray(base)
    before = non_contiguous.copy()
    actual_rng = np.random.default_rng(9123)
    expected_rng = np.random.default_rng(9123)

    result = _sample_with_r_python(vine,
        len(base), r_all, actual_rng, uniforms=non_contiguous)

    assert result.flags.c_contiguous
    np.testing.assert_array_equal(non_contiguous, before)
    _assert_rng_equal(actual_rng, expected_rng)


def test_sampling_rng_state_and_output_match_across_batch_sizes(monkeypatch):
    vine = configured_mixed_family_vine()
    one_rng = np.random.default_rng(20260814)
    many_rng = np.random.default_rng(20260814)
    one_batch = vine.sample(97, rng=one_rng, batch_rows=97)
    many_batches = vine.sample(97, rng=many_rng, batch_rows=7)
    np.testing.assert_array_equal(one_batch, many_batches)
    _assert_rng_equal(one_rng, many_rng)


@pytest.mark.rvine_native
def test_existing_gas_sampling_is_reproducible_on_native_path():
    if not _native_symbol_available("gas_rvine_sample"):
        pytest.skip("existing _scar_cpp.gas_rvine_sample is unavailable")
    vine = configured_mixed_gas_vine()
    expected_rng = np.random.default_rng(202608141)
    actual_rng = np.random.default_rng(202608141)
    expected = vine.sample(64, rng=expected_rng)
    actual = vine.sample(64, rng=actual_rng)
    np.testing.assert_array_equal(actual, expected)
    _assert_rng_equal(actual_rng, expected_rng)


def test_memory_budget_rejection_precedes_rng_consumption(monkeypatch):
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

    actual, actual_diag = _sample_arbitrary_given_mcmc_python(vine,
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
    replay, replay_diag = _sample_arbitrary_given_mcmc_python(vine,
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
    actual = _sample_dag_given_with_r_python(vine,
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
    replay = _sample_dag_given_with_r_python(vine,
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
    expected, _ = _sample_arbitrary_given_mcmc_python(vine,
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
        chunked, _ = _sample_arbitrary_given_mcmc_python(vine,
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
    result = _sample_with_r_python(vine,
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
    empty = _sample_with_r_python(vine,
        0,
        scalar,
        np.random.default_rng(1),
        uniforms=np.empty((0, vine.d)),
    )
    assert empty.shape == (0, vine.d)
    singleton = _sample_with_r_python(vine,
        1,
        scalar,
        np.random.default_rng(1),
        uniforms=np.full((1, vine.d), 0.5),
    )
    assert singleton.shape == (1, vine.d)
    with pytest.raises(ValueError, match="shape"):
        _sample_with_r_python(vine,
            2,
            scalar,
            np.random.default_rng(1),
            uniforms=np.full((2, vine.d + 1), 0.5),
        )
    with pytest.raises(ValueError, match="open unit interval"):
        _sample_with_r_python(vine,
            1,
            scalar,
            np.random.default_rng(1),
            uniforms=np.array([[0.0, 0.5, 1.0]]),
        )
    with pytest.raises(TypeError, match="real values"):
        _sample_with_r_python(vine,
            1,
            scalar,
            np.random.default_rng(1),
            uniforms=np.full((1, vine.d), 0.5 + 0.1j),
        )
    with pytest.raises(ValueError, match="finite values"):
        _sample_with_r_python(vine,
            1,
            scalar,
            np.random.default_rng(1),
            uniforms=np.array([[0.5, np.nan, 0.5]]),
        )


def test_mixed_scalar_and_row_paths_are_not_expanded_by_oracle(golden):
    vine = configured_mixed_family_vine()
    r_all = _fixture_parameters(golden)
    snapshots = {key: value.copy() for key, value in r_all.items()}
    _sample_with_r_python(vine,
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


def test_custom_builtin_subclass_is_rejected_by_runtime(golden):
    class CustomClayton(ClaytonCopula):
        def h_inverse(self, v, u_given, r):
            return np.full_like(np.asarray(v, dtype=np.float64), 0.314)

    vine = configured_mixed_family_vine()
    vine.pair_copulas[(0, 0)] = fitted_pair(CustomClayton(rotate=0), 0.8)
    r_all = _fixture_parameters(golden)
    with pytest.raises(NativeUnsupported, match="exact registered"):
        vine._sample_with_r(
            4,
            r_all,
            np.random.default_rng(1),
            uniforms=np.asarray(golden["inputs"]["uniforms"]),
        )


def test_public_predict_diagnostics_contract(monkeypatch):
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
    assert not hasattr(VineCopula, "_sample_with_r_python")
    assert not hasattr(VineCopula, "_sample_suffix_given_with_r_python")
    assert not hasattr(VineCopula, "_sample_dag_given_with_r_python")
    assert not hasattr(VineCopula, "_log_pdf_rows_with_r_python")
    assert not hasattr(VineCopula, "_sample_arbitrary_given_mcmc_python")
    assert callable(_execute_conditional_plan_python)
    assert callable(_rvine_rosenblatt_transform_python)


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
def test_native_differential_harness_uses_explicit_python_oracles(
        monkeypatch, native_symbol, golden):
    if not _native_symbol_available(native_symbol):
        pytest.skip(f"_scar_cpp.{native_symbol} is not implemented yet")

    vine = configured_mixed_family_vine()
    r_all = _fixture_parameters(golden)
    observations = np.asarray(golden["inputs"]["observations"])
    uniforms = np.asarray(golden["inputs"]["uniforms"])

    def run(native):
        if native_symbol == "rvine_sample":
            method = (
                vine._sample_with_r
                if native else lambda *args, **kwargs:
                _sample_with_r_python(vine, *args, **kwargs))
            return method(
                4, r_all, np.random.default_rng(1), uniforms=uniforms)
        if native_symbol == "rvine_conditional_sample":
            given = {
                int(key): value
                for key, value in golden["inputs"]["suffix_given"].items()
            }
            method = (
                vine._sample_suffix_given_with_r
                if native else lambda *args, **kwargs:
                _vine_sample_suffix_given_with_r_python(
                    vine, *args, **kwargs))
            return method(
                4,
                r_all,
                np.random.default_rng(2),
                given,
                vine._given_suffix_start_col(given),
                uniforms=uniforms,
            )
        if native_symbol == "rvine_log_pdf_rows":
            method = (
                vine._log_pdf_rows_with_r
                if native else lambda *args, **kwargs:
                _log_pdf_rows_with_r_python(vine, *args, **kwargs))
            return method(observations, r_all)
        if native_symbol == "rvine_mcmc_chunk":
            given = {
                int(key): value
                for key, value in golden["inputs"]["dag_given"].items()
            }
            method = (
                vine._sample_arbitrary_given_mcmc
                if native else lambda *args, **kwargs:
                _sample_arbitrary_given_mcmc_python(
                    vine, *args, **kwargs))
            return method(
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
            method = (
                rvine_rosenblatt_transform
                if native else _rvine_rosenblatt_transform_python)
            return method(vine, observations)
        if native_symbol == "dense_student_rosenblatt_transform":
            if not native:
                return np.asarray(golden["expected"]["student_rosenblatt"])
            return student_rosenblatt_transform(
                np.asarray(golden["inputs"]["student_correlation"]),
                golden["inputs"]["student_df"],
                observations,
            )
        raise AssertionError(f"unhandled native symbol {native_symbol}")

    expected = deepcopy(run(False))
    actual = run(True)
    if native_symbol == "dense_student_rosenblatt_transform":
        np.testing.assert_allclose(
            actual, expected, rtol=5e-12, atol=5e-13)
    else:
        np.testing.assert_array_equal(actual, expected)
