"""Contracts for conditional sampling parallelism."""

import json
import os

import numpy as np
import pytest

from tools.benchmark_timing import interleaved_timings
from pyscarcopula import (
    EquicorrGaussianCopula,
    StochasticStudentCopula,
    StudentCopula,
)
from pyscarcopula._native import _extension as _cpp_extension, multivariate as multivariate_native


def equicorr_matrix(d, rho):
    """Test-only dense fixture for generic conditional kernels."""
    matrix = np.full((int(d), int(d)), float(rho), dtype=np.float64)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _student_model(d):
    correlation = np.full((d, d), 0.1, dtype=np.float64)
    np.fill_diagonal(correlation, 1.0)
    return StochasticStudentCopula(d=d, R=correlation), correlation


def _conditional_inputs(n=256, d=20, seed=801):
    rng = np.random.default_rng(seed)
    given = np.arange(5, dtype=np.int32)
    n_free = d - len(given)
    given_latent = rng.normal(size=(n, len(given)))
    normal_draws = rng.normal(size=(n, n_free))
    return given, given_latent, normal_draws


def test_gaussian_conditional_fixed_draws_are_bitwise_equivalent():
    n, d = 256, 20
    given, given_latent, normal_draws = _conditional_inputs(n=n, d=d)
    correlation = equicorr_matrix(d, 0.2)

    expected, expected_info = (
        multivariate_native.gaussian_conditional_latent_info(
            correlation,
            given,
            given_latent,
            normal_draws,
            n_threads=1,
        )
    )
    actual, actual_info = (
        multivariate_native.gaussian_conditional_latent_info(
            correlation,
            given,
            given_latent,
            normal_draws,
            n_threads=4,
        )
    )

    np.testing.assert_array_equal(actual, expected)
    assert expected_info["correlation_factorizations"] == 1
    assert actual_info["correlation_factorizations"] == 1
    assert actual_info["parallel_blocks"] == 4


def test_student_conditional_fixed_draws_are_bitwise_equivalent():
    n, d = 256, 20
    given, given_latent, normal_draws = _conditional_inputs(
        n=n, d=d, seed=802)
    _, correlation = _student_model(d)
    df = np.linspace(4.5, 8.5, n)
    chi_square = np.random.default_rng(803).chisquare(
        df + len(given))

    expected, _ = multivariate_native.student_conditional_latent_info(
        correlation,
        given,
        given_latent,
        df,
        normal_draws,
        chi_square,
        n_threads=1,
    )
    actual, diagnostics = (
        multivariate_native.student_conditional_latent_info(
            correlation,
            given,
            given_latent,
            df,
            normal_draws,
            chi_square,
            n_threads=4,
        )
    )

    np.testing.assert_array_equal(actual, expected)
    assert diagnostics["correlation_factorizations"] == 1
    assert diagnostics["parallel_blocks"] == 4


def test_conditional_row_correlation_factorizes_once_per_row():
    n, d = 128, 24
    given, given_latent, normal_draws = _conditional_inputs(
        n=n, d=d, seed=804)
    correlations = np.stack([
        equicorr_matrix(d, rho)
        for rho in np.linspace(0.05, 0.25, n)
    ])

    expected, _ = multivariate_native.gaussian_conditional_latent_info(
        correlations,
        given,
        given_latent,
        normal_draws,
        n_threads=1,
    )
    actual, diagnostics = (
        multivariate_native.gaussian_conditional_latent_info(
            correlations,
            given,
            given_latent,
            normal_draws,
            n_threads=4,
        )
    )

    np.testing.assert_array_equal(actual, expected)
    assert diagnostics["correlation_factorizations"] == n
    assert diagnostics["parallel_blocks"] == 4


def test_conditional_parallel_failure_index_matches_sequential():
    module = _cpp_extension.load()
    n, d = 256, 20
    given, given_latent, normal_draws = _conditional_inputs(
        n=n, d=d, seed=805)
    correlation = equicorr_matrix(d, 0.2)
    df = np.full(n, 6.0)
    df[5] = 2.0
    chi_square = np.ones(n)

    sequential = dict(module.multivariate_student_conditional(
        correlation, given, given_latent, df, normal_draws,
        chi_square, 1))
    parallel = dict(module.multivariate_student_conditional(
        correlation, given, given_latent, df, normal_draws,
        chi_square, 4))

    assert sequential["status"] == module.SCAR_INVALID_PARAMETER
    assert parallel["status"] == module.SCAR_INVALID_PARAMETER
    assert sequential["failure_index"] == parallel["failure_index"] == 5
    np.testing.assert_array_equal(
        parallel["values"][:5], sequential["values"][:5])
    np.testing.assert_array_equal(parallel["values"][6:], 0.0)


@pytest.mark.parametrize("family", ["gaussian", "student"])
def test_public_conditional_seed_reproducibility_across_threads(family):
    n, d = 256, 20
    given = {index: 0.25 + 0.05 * index for index in range(5)}
    if family == "gaussian":
        copula = EquicorrGaussianCopula(d=d)
        kwargs = {"r": 0.2}
    else:
        copula, _ = _student_model(d)
        kwargs = {"r": 6.0}

    expected = copula.sample_conditional(
        n, given=given, rng=np.random.default_rng(806),
        n_threads=1, **kwargs)
    actual = copula.sample_conditional(
        n, given=given, rng=np.random.default_rng(806),
        n_threads=4, **kwargs)

    np.testing.assert_array_equal(actual, expected)


def test_static_student_conditional_defaults_to_one_thread(monkeypatch):
    d = 8
    correlation = equicorr_matrix(d, 0.2)
    copula = StudentCopula()
    copula.shape = correlation
    copula.df = 6.0
    given = {0: 0.3, 2: 0.7}
    monkeypatch.setenv("PYSCARCOPULA_NUM_THREADS", "8")

    implicit = copula.sample_conditional(
        64, given=given, rng=np.random.default_rng(812))
    explicit = copula.sample_conditional(
        64, given=given, rng=np.random.default_rng(812), n_threads=1)

    np.testing.assert_array_equal(implicit, explicit)


def test_conditional_small_workload_stays_sequential():
    n, d = 16, 10
    given, given_latent, normal_draws = _conditional_inputs(
        n=n, d=d, seed=807)
    module = _cpp_extension.load()
    batches_before = dict(module._parallel_runtime_info())[
        "batches_submitted"]

    _, diagnostics = multivariate_native.gaussian_conditional_latent_info(
        equicorr_matrix(d, 0.2),
        given,
        given_latent,
        normal_draws,
        n_threads=8,
    )

    batches_after = dict(module._parallel_runtime_info())[
        "batches_submitted"]
    assert diagnostics["parallel_blocks"] == 1
    assert batches_after == batches_before


@pytest.mark.parametrize("n_threads", [0, 257, True, 1.5])
def test_sampling_kernels_reject_invalid_thread_count(n_threads):
    given, given_latent, normal_draws = _conditional_inputs(n=8, d=10)
    with pytest.raises(ValueError, match="n_threads"):
        multivariate_native.gaussian_conditional_latent(
            equicorr_matrix(10, 0.2),
            given,
            given_latent,
            normal_draws,
            n_threads=n_threads,
        )


def _benchmark_enabled():
    if os.environ.get("PYSCA_RUN_BENCHMARKS") != "1":
        pytest.skip("set PYSCA_RUN_BENCHMARKS=1 to run benchmark baselines")


@pytest.mark.benchmark
def test_conditional_internal_thread_scaling_benchmark():
    _benchmark_enabled()
    n, d = 2_000, 40
    given, given_latent, normal_draws = _conditional_inputs(
        n=n, d=d, seed=811)
    correlation = equicorr_matrix(d, 0.2)
    workers = (1, 2, 4, 8)
    calls = {
        count: lambda count=count: (
                multivariate_native.gaussian_conditional_latent(
                    correlation,
                    given,
                    given_latent,
                    normal_draws,
                    n_threads=count,
                )
            )
        for count in workers
    }
    for call in calls.values():
        call()
    measured = interleaved_timings(calls, repeats=5)
    timings = measured.medians
    reference = measured.results[1]
    for count in workers[1:]:
        np.testing.assert_array_equal(measured.results[count], reference)

    payload = {
        "name": "conditional_gaussian_internal_thread_scaling",
        "workload": {"n": n, "d": d, "n_given": len(given)},
        "seconds": {str(key): value for key, value in timings.items()},
        "speedup": {
            str(key): measured.median_ratio(1, key) for key in workers
        },
    }
    print("PYSCA_BENCHMARK " + json.dumps(
        payload, sort_keys=True), flush=True)
