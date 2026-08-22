"""Contracts for the native dense Student Rosenblatt transform."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

import pyscarcopula.stattests as statt
from pyscarcopula.numerical import _cpp_extension
from pyscarcopula.numerical._cpp_extension import CppUnsupported
from pyscarcopula.numerical._rvine_backend import _RVINE_BACKEND_ENV
from pyscarcopula.numerical.multivariate_native import (
    dense_student_rosenblatt,
)
from pyscarcopula.stattests import (
    _student_rosenblatt_transform_python,
    student_rosenblatt_transform,
)


pytestmark = pytest.mark.rvine_native


def _correlation():
    return np.array(
        [
            [1.0, 0.45, -0.20],
            [0.45, 1.0, 0.30],
            [-0.20, 0.30, 1.0],
        ],
        dtype=np.float64,
    )


def _observations():
    return np.array(
        [
            [1e-10, 0.20, 0.80],
            [0.10, 0.50, 0.90],
            [0.90, 0.70, 1.0 - 1e-10],
            [0.35, 0.65, 0.45],
        ],
        dtype=np.float64,
    )


@pytest.mark.parametrize("df", [0.1, 0.5, 1.0, 2.0, 5.0, 1000.0])
def test_native_scalar_df_matches_independent_scipy_oracle(
        monkeypatch, df):
    correlation = _correlation()
    observations = _observations()
    expected = _student_rosenblatt_transform_python(
        correlation, df, observations)

    monkeypatch.setenv(_RVINE_BACKEND_ENV, "native_strict")
    actual = student_rosenblatt_transform(correlation, df, observations)

    np.testing.assert_allclose(actual, expected, rtol=5e-12, atol=5e-13)
    assert actual.flags.c_contiguous
    assert np.all((actual > 0.0) & (actual < 1.0))


def test_df_path_is_dispatched_once_and_matches_rowwise_oracle(monkeypatch):
    correlation = _correlation()
    observations = _observations()
    df_path = np.array([0.5, 2.0, 5.0, 17.0])
    expected = np.vstack([
        _student_rosenblatt_transform_python(
            correlation, df, observations[row:row + 1])
        for row, df in enumerate(df_path)
    ])

    monkeypatch.setenv(_RVINE_BACKEND_ENV, "native_strict")
    actual = student_rosenblatt_transform(
        correlation, df_path, observations)

    np.testing.assert_allclose(actual, expected, rtol=5e-12, atol=5e-13)


@pytest.mark.parametrize("df", [1e-4, 1e-3, 1e-2, 2e-2, 5e-2])
def test_low_positive_df_auto_preserves_scipy_oracle_exactly(
        monkeypatch, df):
    correlation = _correlation()
    observations = _observations()
    expected = _student_rosenblatt_transform_python(
        correlation, df, observations)

    monkeypatch.setenv(_RVINE_BACKEND_ENV, "auto")
    actual = student_rosenblatt_transform(correlation, df, observations)

    np.testing.assert_array_equal(actual, expected)


def test_low_df_tail_coordinate_does_not_silently_use_native_approximation(
        monkeypatch):
    correlation = _correlation()
    observations = np.array(
        [[1e-10, 0.2, 0.8], [0.1, 0.5, 0.9],
         [0.9, 0.7, 1.0 - 1e-10]],
        dtype=np.float64,
    )
    expected = _student_rosenblatt_transform_python(
        correlation, 0.02, observations)
    assert expected[0, 0] > 1e-4

    monkeypatch.setenv(_RVINE_BACKEND_ENV, "auto")
    actual = student_rosenblatt_transform(
        correlation, 0.02, observations)

    np.testing.assert_array_equal(actual, expected)


def test_df_path_with_low_value_falls_back_as_one_legacy_operation(
        monkeypatch):
    correlation = _correlation()
    observations = _observations()
    df_path = np.array([0.02, 0.5, 2.0, 10.0])
    expected = _student_rosenblatt_transform_python(
        correlation, df_path, observations)

    monkeypatch.setenv(_RVINE_BACKEND_ENV, "auto")
    actual = student_rosenblatt_transform(
        correlation, df_path, observations)
    np.testing.assert_array_equal(actual, expected)

    monkeypatch.setenv(_RVINE_BACKEND_ENV, "native_strict")
    with pytest.raises(CppUnsupported, match="does not support"):
        student_rosenblatt_transform(correlation, df_path, observations)


@pytest.mark.parametrize(
    ("observations", "df"),
    [
        (np.empty((0, 3), dtype=np.float64), 0.5),
        (np.array([[0.2, 0.5, 0.8]]), 2.0),
    ],
)
def test_empty_and_single_row_contracts(monkeypatch, observations, df):
    correlation = _correlation()
    monkeypatch.setenv(_RVINE_BACKEND_ENV, "native_strict")

    actual = student_rosenblatt_transform(correlation, df, observations)
    expected = _student_rosenblatt_transform_python(
        correlation, df, observations)

    assert actual.shape == observations.shape
    np.testing.assert_allclose(actual, expected, rtol=5e-12, atol=5e-13)


def test_one_dimensional_contract(monkeypatch):
    correlation = np.ones((1, 1), dtype=np.float64)
    observations = np.array([[1e-10], [0.2], [0.8], [1.0 - 1e-10]])
    df_path = np.array([0.1, 0.5, 2.0, 1000.0])
    monkeypatch.setenv(_RVINE_BACKEND_ENV, "native_strict")

    actual = student_rosenblatt_transform(
        correlation, df_path, observations)

    np.testing.assert_allclose(actual, observations, rtol=5e-12, atol=5e-13)


def test_noncontiguous_inputs_and_four_threads_match_serial():
    rng = np.random.default_rng(20260822)
    dimension = 6
    correlation = np.full((dimension, dimension), 0.15)
    np.fill_diagonal(correlation, 1.0)
    backing = rng.uniform(0.01, 0.99, size=(1024, dimension * 2))
    observations = backing[:, ::2]
    df_backing = np.column_stack((
        rng.uniform(0.25, 30.0, size=len(observations)),
        np.zeros(len(observations)),
    ))
    df_path = df_backing[:, 0]
    assert not observations.flags.c_contiguous
    assert not df_path.flags.c_contiguous

    serial = dense_student_rosenblatt(
        correlation, df_path, observations, n_threads=1)
    parallel = dense_student_rosenblatt(
        correlation, df_path, observations, n_threads=4)

    np.testing.assert_array_equal(parallel, serial)
    assert parallel.flags.c_contiguous


def test_native_diagnostics_report_one_factorization_and_parallel_rows():
    module = _cpp_extension.load()
    rng = np.random.default_rng(812)
    dimension = 6
    correlation = np.full((dimension, dimension), 0.1)
    np.fill_diagonal(correlation, 1.0)
    observations = rng.uniform(0.01, 0.99, size=(1024, dimension))
    result = module.dense_student_rosenblatt_transform(
        correlation,
        observations,
        np.array([0.5]),
        4,
    )

    assert result["status"] == module.SCAR_OK
    assert result["n_rows"] == len(observations)
    assert result["dimension"] == dimension
    assert result["diagnostics"] == {
        "n_threads_requested": 4,
        "parallel_blocks": 4,
        "correlation_factorizations": 1,
    }


def test_native_transform_releases_the_gil():
    module = _cpp_extension.load()
    rng = np.random.default_rng(2026082256)
    dimension = 15
    correlation = np.fromfunction(
        lambda i, j: 0.25 ** np.abs(i - j),
        (dimension, dimension),
    )
    observations = rng.uniform(0.01, 0.99, size=(3_000, dimension))
    df_path = np.linspace(0.5, 20.0, len(observations))
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
        result = module.dense_student_rosenblatt_transform(
            correlation, observations, df_path, 1)
    finally:
        stop.set()
        thread.join()

    assert result["status"] == module.SCAR_OK
    assert counter[0] > before


@pytest.mark.parametrize(
    ("df", "match"),
    [
        (0.0, "finite positive"),
        (-0.5, "finite positive"),
        (np.nan, "finite positive"),
        ([1.0, 2.0], "scalar or have one value per row"),
    ],
)
def test_invalid_df_is_rejected_before_native_call(df, match):
    with pytest.raises(ValueError, match=match):
        dense_student_rosenblatt(
            _correlation(), df, _observations())


@pytest.mark.parametrize("df", [0.0, -1.0, np.nan, np.inf])
def test_public_auto_preserves_legacy_df_boundary_results(monkeypatch, df):
    correlation = _correlation()
    observations = _observations()
    expected = _student_rosenblatt_transform_python(
        correlation, df, observations)

    monkeypatch.setenv(_RVINE_BACKEND_ENV, "auto")
    actual = student_rosenblatt_transform(correlation, df, observations)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("correlation", "observations", "df", "error", "match"),
    [
        (np.eye(2), np.ones(2), 2.0, ValueError, "2D"),
        (np.eye(2), np.ones((2, 3)), 2.0, ValueError, "shape"),
        (np.eye(3, dtype=np.complex128), np.ones((2, 3)), 2.0,
         TypeError, "real"),
        (np.eye(3), np.ones((2, 3), dtype=np.complex128), 2.0,
         TypeError, "real"),
        (np.eye(3), np.ones((2, 3)), 2.0 + 1.0j,
         TypeError, "real"),
        (np.eye(3), np.array([[0.2, np.nan, 0.8]]), 2.0,
         ValueError, "NaN"),
    ],
)
def test_adapter_rejects_invalid_array_contracts(
        correlation, observations, df, error, match):
    with pytest.raises(error, match=match):
        dense_student_rosenblatt(correlation, df, observations)


def test_correlation_validation_and_near_singular_spd_contract():
    module = _cpp_extension.load()
    observations = np.array([[0.2, 0.5, 0.8]])
    df = np.array([0.5])

    nonsymmetric = _correlation()
    nonsymmetric[0, 1] += 0.01
    invalid = module.dense_student_rosenblatt_transform(
        nonsymmetric, observations, df, 1)
    assert invalid["status"] == module.SCAR_INVALID_PARAMETER

    non_spd = np.array(
        [[1.0, 1.1, 0.0], [1.1, 1.0, 0.0], [0.0, 0.0, 1.0]])
    failed = module.dense_student_rosenblatt_transform(
        non_spd, observations, df, 1)
    assert failed["status"] == module.SCAR_NUMERICAL_FAILURE
    with pytest.raises(CppUnsupported, match="positive-definite"):
        dense_student_rosenblatt(non_spd, df, observations)

    near_singular = np.full((3, 3), 0.999)
    np.fill_diagonal(near_singular, 1.0)
    actual = dense_student_rosenblatt(
        near_singular, df, observations)
    expected = _student_rosenblatt_transform_python(
        near_singular, float(df[0]), observations)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize(
    "correlation",
    [
        np.array([[1.0, 0.46, -0.2],
                  [0.45, 1.0, 0.3],
                  [-0.2, 0.3, 1.0]]),
        np.array([[1.2, 0.45, -0.2],
                  [0.45, 1.0, 0.3],
                  [-0.2, 0.3, 1.0]]),
        np.array([[1.0, 1.1, 0.0],
                  [1.1, 1.0, 0.0],
                  [0.0, 0.0, 1.0]]),
    ],
)
def test_public_auto_preserves_legacy_correlation_results(
        monkeypatch, correlation):
    observations = np.array([[0.2, 0.5, 0.8]])
    expected = _student_rosenblatt_transform_python(
        correlation, 0.5, observations)

    monkeypatch.setenv(_RVINE_BACKEND_ENV, "auto")
    actual = student_rosenblatt_transform(
        correlation, 0.5, observations)

    np.testing.assert_array_equal(actual, expected)


def test_capability_symmetry_tolerance_matches_native_validation(monkeypatch):
    correlation = _correlation()
    correlation[0, 1] += 1.5e-12
    observations = np.array([[0.2, 0.5, 0.8]])
    expected = _student_rosenblatt_transform_python(
        correlation, 0.5, observations)

    monkeypatch.setenv(_RVINE_BACKEND_ENV, "auto")
    actual = student_rosenblatt_transform(
        correlation, 0.5, observations)

    np.testing.assert_array_equal(actual, expected)


def test_public_auto_preserves_legacy_singular_failure(monkeypatch):
    correlation = np.ones((3, 3), dtype=np.float64)
    observations = np.array([[0.2, 0.5, 0.8]])
    monkeypatch.setenv(_RVINE_BACKEND_ENV, "auto")

    with pytest.raises(np.linalg.LinAlgError):
        student_rosenblatt_transform(correlation, 0.5, observations)


def test_public_auto_preserves_nan_observation_result(monkeypatch):
    observations = np.array([[0.2, np.nan, 0.8]])
    expected = _student_rosenblatt_transform_python(
        _correlation(), 0.5, observations)
    monkeypatch.setenv(_RVINE_BACKEND_ENV, "auto")

    actual = student_rosenblatt_transform(
        _correlation(), 0.5, observations)

    np.testing.assert_array_equal(actual, expected)


def test_ill_conditioned_spd_uses_legacy_oracle_in_auto(monkeypatch):
    correlation = np.full((3, 3), 0.9999999)
    np.fill_diagonal(correlation, 1.0)
    observations = np.array([[0.2, 0.5, 0.8]])
    expected = _student_rosenblatt_transform_python(
        correlation, 0.5, observations)

    monkeypatch.setenv(_RVINE_BACKEND_ENV, "auto")
    actual = student_rosenblatt_transform(
        correlation, 0.5, observations)
    np.testing.assert_array_equal(actual, expected)

    with pytest.raises(CppUnsupported, match="condition number"):
        dense_student_rosenblatt(correlation, 0.5, observations)

    monkeypatch.setenv(_RVINE_BACKEND_ENV, "native_strict")
    with pytest.raises(CppUnsupported, match="does not support"):
        student_rosenblatt_transform(correlation, 0.5, observations)


def test_infinite_and_out_of_range_uniforms_are_clipped():
    observations = np.array(
        [[-np.inf, -1.0, 0.5], [0.5, 2.0, np.inf]],
        dtype=np.float64,
    )
    actual = dense_student_rosenblatt(
        _correlation(), 0.5, observations)

    assert np.all(np.isfinite(actual))
    assert np.all((actual > 0.0) & (actual < 1.0))


def test_dense_gas_path_passes_entire_df_trajectory_once(monkeypatch):
    observations = _observations()
    correlation = _correlation()
    df_path = np.array([0.5, 1.0, 2.0, 5.0])
    calls = []

    monkeypatch.setattr(
        statt, "_gas_parameter_path",
        lambda copula, u, fit_result: df_path,
    )

    def record_transform(received_correlation, received_df, received_u):
        calls.append((received_correlation, received_df, received_u))
        return np.full_like(received_u, 0.5)

    monkeypatch.setattr(
        statt, "student_rosenblatt_transform", record_transform)
    copula = SimpleNamespace(R=correlation, corr_mode="dense")
    fit_result = SimpleNamespace(method="GAS")

    actual = statt.stochastic_student_rosenblatt_transform(
        copula, observations, fit_result)

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], correlation)
    assert calls[0][1] is df_path
    assert calls[0][2] is observations
    np.testing.assert_array_equal(actual, np.full_like(observations, 0.5))
