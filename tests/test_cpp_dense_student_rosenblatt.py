"""Contracts for the native dense Student Rosenblatt transform."""

from __future__ import annotations

import ctypes
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import t as t_dist

import pyscarcopula.stattests as statt
from pyscarcopula._native import _extension as _cpp_extension
from pyscarcopula._native.multivariate import (
    dense_student_rosenblatt,
)
from pyscarcopula.stattests import student_rosenblatt_transform


pytestmark = pytest.mark.rvine_native
_NATIVE_SCIPY_RTOL = 1e-10
_NATIVE_SCIPY_ATOL = 1e-11


def _assert_native_matches_scipy(actual, expected):
    """Compare implementations across supported SciPy and C++ toolchains."""
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=_NATIVE_SCIPY_RTOL,
        atol=_NATIVE_SCIPY_ATOL,
    )


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


def _student_scipy_oracle(correlation, df, observations):
    """Frozen pre-8.2 dense Student implementation used only as an oracle."""
    values = np.asarray(observations)
    df_values = np.asarray(df)
    if df_values.ndim != 0:
        df_path = np.asarray(df_values, dtype=np.float64).ravel()
        if df_path.size == 1:
            df = float(df_path[0])
        elif len(values) == 0:
            return np.empty(values.shape, dtype=np.float64)
        else:
            return np.vstack([
                _student_scipy_oracle(
                    correlation, float(row_df), values[row:row + 1])
                for row, row_df in enumerate(df_path)
            ])

    clipped = np.clip(values, 1e-10, 1.0 - 1e-10)
    quantiles = t_dist.ppf(clipped, df=df)
    rows, dimension = quantiles.shape
    transformed = np.empty((rows, dimension))
    transformed[:, 0] = t_dist.cdf(quantiles[:, 0], df=df)
    for index in range(1, dimension):
        leading = correlation[:index, :index]
        cross = correlation[index, :index]
        inverse = np.linalg.inv(leading)
        beta = cross @ inverse
        variance = correlation[index, index] - cross @ inverse @ cross
        previous = quantiles[:, :index]
        mean = previous @ beta
        quadratic = np.sum(previous @ inverse * previous, axis=1)
        conditional_df = df + index
        scale = (df + quadratic) / conditional_df
        standardized = (
            (quantiles[:, index] - mean)
            / (np.sqrt(max(variance, 1e-12)) * np.sqrt(scale))
        )
        transformed[:, index] = t_dist.cdf(
            standardized, df=conditional_df)
    return np.clip(transformed, 1e-10, 1.0 - 1e-10)


@pytest.mark.parametrize("df", [0.1, 0.5, 1.0, 2.0, 5.0, 1000.0])
def test_native_scalar_df_matches_independent_scipy_oracle(
        monkeypatch, df):
    correlation = _correlation()
    observations = _observations()
    expected = _student_scipy_oracle(
        correlation, df, observations)

    actual = student_rosenblatt_transform(correlation, df, observations)

    _assert_native_matches_scipy(actual, expected)
    assert actual.flags.c_contiguous
    assert np.all((actual > 0.0) & (actual < 1.0))


def test_df_path_is_dispatched_once_and_matches_rowwise_oracle(monkeypatch):
    correlation = _correlation()
    observations = _observations()
    df_path = np.array([0.5, 2.0, 5.0, 17.0])
    expected = np.vstack([
        _student_scipy_oracle(
            correlation, df, observations[row:row + 1])
        for row, df in enumerate(df_path)
    ])

    actual = student_rosenblatt_transform(
        correlation, df_path, observations)

    _assert_native_matches_scipy(actual, expected)


@pytest.mark.parametrize("df", [1e-4, 1e-3, 1e-2, 2e-2, 5e-2])
def test_low_positive_df_mandatory_native_matches_scipy_oracle(
        monkeypatch, df):
    correlation = _correlation()
    observations = _observations()
    expected = _student_scipy_oracle(
        correlation, df, observations)

    actual = student_rosenblatt_transform(correlation, df, observations)

    _assert_native_matches_scipy(actual, expected)


def test_low_df_tail_coordinate_preserves_scipy_finite_endpoint(
        monkeypatch):
    correlation = _correlation()
    observations = np.array(
        [[1e-10, 0.2, 0.8], [0.1, 0.5, 0.9],
         [0.9, 0.7, 1.0 - 1e-10]],
        dtype=np.float64,
    )
    expected = _student_scipy_oracle(
        correlation, 0.02, observations)
    assert expected[0, 0] > 1e-4

    actual = student_rosenblatt_transform(
        correlation, 0.02, observations)

    _assert_native_matches_scipy(actual, expected)


def test_df_path_with_low_value_uses_one_mandatory_native_operation(
        monkeypatch):
    correlation = _correlation()
    observations = _observations()
    df_path = np.array([0.02, 0.5, 2.0, 10.0])
    expected = _student_scipy_oracle(
        correlation, df_path, observations)

    actual = student_rosenblatt_transform(
        correlation, df_path, observations)
    _assert_native_matches_scipy(actual, expected)

    strict = student_rosenblatt_transform(
        correlation, df_path, observations)
    np.testing.assert_array_equal(strict, actual)


@pytest.mark.parametrize(
    ("observations", "df"),
    [
        (np.empty((0, 3), dtype=np.float64), 0.5),
        (np.array([[0.2, 0.5, 0.8]]), 2.0),
    ],
)
def test_empty_and_single_row_contracts(monkeypatch, observations, df):
    correlation = _correlation()

    actual = student_rosenblatt_transform(correlation, df, observations)
    expected = _student_scipy_oracle(
        correlation, df, observations)

    assert actual.shape == observations.shape
    _assert_native_matches_scipy(actual, expected)


def test_one_dimensional_contract(monkeypatch):
    correlation = np.ones((1, 1), dtype=np.float64)
    observations = np.array([[1e-10], [0.2], [0.8], [1.0 - 1e-10]])
    df_path = np.array([0.1, 0.5, 2.0, 1000.0])

    actual = student_rosenblatt_transform(
        correlation, df_path, observations)

    _assert_native_matches_scipy(actual, observations)


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


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="UCRT floating-point environment regression contract",
)
def test_parallel_workers_inherit_calling_floating_point_environment():
    rng = np.random.default_rng(20260824)
    dimension = 6
    correlation = np.full((dimension, dimension), 0.15)
    np.fill_diagonal(correlation, 1.0)
    observations = rng.uniform(0.01, 0.99, size=(256, dimension))
    df_path = rng.uniform(0.25, 30.0, size=len(observations))

    module = _cpp_extension.load()
    module._parallel_runtime_shutdown()
    module._parallel_for_blocks_probe(16, 1, 4)

    ucrt = ctypes.CDLL("ucrtbase")
    ucrt.fegetround.restype = ctypes.c_int
    ucrt.fesetround.argtypes = [ctypes.c_int]
    ucrt.fesetround.restype = ctypes.c_int
    original_rounding = ucrt.fegetround()
    ucrt_round_downward = 0x100
    try:
        assert ucrt.fesetround(ucrt_round_downward) == 0
        serial = dense_student_rosenblatt(
            correlation, df_path, observations, n_threads=1)
        parallel = dense_student_rosenblatt(
            correlation, df_path, observations, n_threads=4)
    finally:
        assert ucrt.fesetround(original_rounding) == 0

    np.testing.assert_array_equal(parallel, serial)


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
def test_public_entry_rejects_invalid_df_without_fallback(monkeypatch, df):
    with pytest.raises(ValueError, match="finite positive"):
        student_rosenblatt_transform(_correlation(), df, _observations())


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
    with pytest.raises(np.linalg.LinAlgError, match="numerical_failure"):
        dense_student_rosenblatt(non_spd, df, observations)

    near_singular = np.full((3, 3), 0.999)
    np.fill_diagonal(near_singular, 1.0)
    actual = dense_student_rosenblatt(
        near_singular, df, observations)
    expected = _student_scipy_oracle(
        near_singular, float(df[0]), observations)
    _assert_native_matches_scipy(actual, expected)


@pytest.mark.parametrize(
    ("correlation", "error"),
    [
        (np.array([[1.0, 0.46, -0.2],
                   [0.45, 1.0, 0.3],
                   [-0.2, 0.3, 1.0]]), ValueError),
        (np.array([[1.2, 0.45, -0.2],
                   [0.45, 1.0, 0.3],
                   [-0.2, 0.3, 1.0]]), ValueError),
        (np.array([[1.0, 1.1, 0.0],
                   [1.1, 1.0, 0.0],
                   [0.0, 0.0, 1.0]]), np.linalg.LinAlgError),
    ],
)
def test_public_entry_rejects_invalid_correlation_without_fallback(
        monkeypatch, correlation, error):
    with pytest.raises(error):
        student_rosenblatt_transform(
            correlation, 0.5, np.array([[0.2, 0.5, 0.8]]))


def test_capability_symmetry_tolerance_matches_native_validation(monkeypatch):
    correlation = _correlation()
    correlation[0, 1] += 1.5e-12
    observations = np.array([[0.2, 0.5, 0.8]])
    with pytest.raises(ValueError, match="invalid_parameter"):
        student_rosenblatt_transform(correlation, 0.5, observations)


def test_public_entry_preserves_singular_failure_type(monkeypatch):
    correlation = np.ones((3, 3), dtype=np.float64)
    observations = np.array([[0.2, 0.5, 0.8]])

    with pytest.raises(np.linalg.LinAlgError):
        student_rosenblatt_transform(correlation, 0.5, observations)


def test_public_entry_rejects_nan_observation_without_fallback(monkeypatch):
    observations = np.array([[0.2, np.nan, 0.8]])

    with pytest.raises(ValueError, match="NaN"):
        student_rosenblatt_transform(_correlation(), 0.5, observations)


def test_ill_conditioned_spd_uses_mandatory_native_path(monkeypatch):
    correlation = np.full((3, 3), 0.9999999)
    np.fill_diagonal(correlation, 1.0)
    observations = np.array([[0.2, 0.5, 0.8]])
    actual = student_rosenblatt_transform(
        correlation, 0.5, observations)
    assert np.all(np.isfinite(actual))
    assert np.all((actual > 0.0) & (actual < 1.0))
    direct = dense_student_rosenblatt(correlation, 0.5, observations)
    np.testing.assert_array_equal(direct, actual)

    strict = student_rosenblatt_transform(
        correlation, 0.5, observations)
    np.testing.assert_array_equal(strict, actual)


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
