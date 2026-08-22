"""Contracts for multivariate rows and static likelihood."""

from concurrent.futures import ThreadPoolExecutor
import json
import os

import numpy as np
import pytest

from benchmark_timing import interleaved_timings
from pyscarcopula import (
    EquicorrGaussianCopula,
    NumericalConfig,
    StochasticStudentCopula,
)
from pyscarcopula.numerical import (
    _cpp_copula,
    _cpp_extension,
    multivariate_native,
    static_likelihood,
)


def _student_case(T=128, d=20, seed=701):
    rng = np.random.default_rng(seed)
    correlation = np.full((d, d), 0.1, dtype=np.float64)
    np.fill_diagonal(correlation, 1.0)
    copula = StochasticStudentCopula(d=d, R=correlation)
    u = rng.uniform(0.02, 0.98, (T, d))
    return copula, u


@pytest.mark.parametrize("family", ["student", "equicorr"])
def test_multivariate_rows_are_bitwise_equivalent(family):
    if family == "student":
        copula, u = _student_case(T=256, d=20)
        cache = copula.prepare_emission_cache(u)
    else:
        rng = np.random.default_rng(702)
        copula = EquicorrGaussianCopula(d=40)
        u = rng.uniform(0.02, 0.98, (7_000, 40))
        cache = None

    expected = multivariate_native.log_pdf_and_dlog_rows_info(
        copula, u, 6.0 if family == "student" else 0.2,
        cache=cache, n_threads=1)
    actual = multivariate_native.log_pdf_and_dlog_rows_info(
        copula, u, 6.0 if family == "student" else 0.2,
        cache=cache, n_threads=4)

    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2]["n_threads_requested"] == 4
    assert 1 < actual[2]["row_parallel_blocks"] <= 4


def test_multivariate_rows_small_workload_stays_sequential():
    copula, u = _student_case(T=8, d=20, seed=703)
    module = _cpp_extension.load()
    batches_before = dict(module._parallel_runtime_info())[
        "batches_submitted"]

    _, _, diagnostics = multivariate_native.log_pdf_and_dlog_rows_info(
        copula, u, 6.0, n_threads=8)

    batches_after = dict(module._parallel_runtime_info())[
        "batches_submitted"]
    assert diagnostics["row_parallel_blocks"] == 1
    assert batches_after == batches_before


def test_multivariate_rows_parallel_failure_index_matches_sequential():
    module = _cpp_extension.load()
    copula, u = _student_case(T=256, d=20, seed=704)
    u[:5] = 0.5
    spec = _cpp_copula.make_multivariate_spec(module, copula, cache=None)
    spec.l_inv = (np.eye(copula.d) * 1e200).reshape(-1).tolist()

    sequential = dict(module.multivariate_log_pdf_and_grad(
        spec, u, np.array([6.0]), 0, 1))
    parallel = dict(module.multivariate_log_pdf_and_grad(
        spec, u, np.array([6.0]), 0, 4))

    assert sequential["status"] == module.SCAR_NUMERICAL_FAILURE
    assert parallel["status"] == module.SCAR_NUMERICAL_FAILURE
    assert sequential["failure_index"] == parallel["failure_index"] == 5
    np.testing.assert_array_equal(
        parallel["log_pdf"][:5], sequential["log_pdf"][:5])
    assert np.all(np.isneginf(parallel["log_pdf"][6:]))
    assert np.all(np.isnan(parallel["dlog_dr"][6:]))


def test_static_student_joint_objective_parallel_tolerance():
    copula, u = _student_case(T=160, d=8, seed=705)
    sequential = static_likelihood.prepare(
        copula, u, n_threads=1).joint_result(6.0)
    parallel = static_likelihood.prepare(
        copula, u, n_threads=4).joint_result(6.0)

    assert sequential["status"] == parallel["status"] == 0
    assert parallel["parallel_blocks"] == 4
    assert parallel["failure_index"] == sequential["failure_index"] == -1
    assert parallel["negative_log_likelihood"] == pytest.approx(
        sequential["negative_log_likelihood"], rel=0.0, abs=1e-11)
    assert parallel["negative_gradient"] == pytest.approx(
        sequential["negative_gradient"], rel=0.0, abs=1e-11)
    np.testing.assert_allclose(
        parallel["negative_correlation_gradient"],
        sequential["negative_correlation_gradient"],
        rtol=0.0,
        atol=1e-11,
    )


def test_static_parallel_evaluator_is_thread_safe():
    copula, u = _student_case(T=160, d=8, seed=706)
    evaluator = static_likelihood.prepare(copula, u, n_threads=4)
    expected = evaluator.joint_result(6.0)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(
            lambda _: evaluator.joint_result(6.0), range(8)))

    for result in results:
        assert result["negative_log_likelihood"] == (
            expected["negative_log_likelihood"])
        assert result["negative_gradient"] == expected["negative_gradient"]
        np.testing.assert_array_equal(
            result["negative_correlation_gradient"],
            expected["negative_correlation_gradient"],
        )


def test_static_equicorr_medium_workload_stays_sequential():
    rng = np.random.default_rng(707)
    copula = EquicorrGaussianCopula(d=4)
    u = rng.uniform(0.02, 0.98, (20_000, 4))
    evaluator = static_likelihood.prepare(copula, u, n_threads=8)
    module = _cpp_extension.load()
    batches_before = dict(module._parallel_runtime_info())[
        "batches_submitted"]

    result = evaluator.result(0.2)

    batches_after = dict(module._parallel_runtime_info())[
        "batches_submitted"]
    assert result["parallel_blocks"] == 1
    assert batches_after == batches_before


@pytest.mark.parametrize("n_threads", [0, 257, True, 1.5])
def test_static_evaluator_rejects_invalid_thread_count(n_threads):
    copula, u = _student_case(T=10, d=4, seed=708)
    with pytest.raises(ValueError, match="n_threads"):
        static_likelihood.prepare(copula, u, n_threads=n_threads)


def test_static_student_fit_is_equivalent_across_thread_counts():
    copula1, u = _student_case(T=120, d=8, seed=709)
    copula4 = StochasticStudentCopula(d=8, R=copula1.R.copy())
    result1 = copula1._fit_mle(
        u, config=NumericalConfig(n_threads=1), maxiter=12)
    result4 = copula4._fit_mle(
        u, config=NumericalConfig(n_threads=4), maxiter=12)

    assert result4.success == result1.success
    assert result4.copula_param == pytest.approx(
        result1.copula_param, rel=0.0, abs=1e-10)
    assert result4.log_likelihood == pytest.approx(
        result1.log_likelihood, rel=0.0, abs=1e-9)


def _benchmark_enabled():
    if os.environ.get("PYSCA_RUN_BENCHMARKS") != "1":
        pytest.skip("set PYSCA_RUN_BENCHMARKS=1 to run benchmark baselines")


@pytest.mark.benchmark
def test_static_student_internal_thread_scaling_benchmark():
    _benchmark_enabled()
    copula, u = _student_case(T=700, d=80, seed=710)
    evaluators = {
        count: static_likelihood.prepare(copula, u, n_threads=count)
        for count in (1, 2, 4, 8)
    }
    calls = {
        count: lambda evaluator=evaluator: evaluator.result(6.0)
        for count, evaluator in evaluators.items()
    }
    for call in calls.values():
        call()
    measured = interleaved_timings(calls, repeats=5)
    timings = measured.medians
    reference = measured.results[1]
    for count in (2, 4, 8):
        result = measured.results[count]
        assert result["negative_log_likelihood"] == pytest.approx(
            reference["negative_log_likelihood"],
            rel=0.0,
            abs=1e-9,
        )
        assert result["negative_gradient"] == pytest.approx(
            reference["negative_gradient"], rel=0.0, abs=1e-9)

    payload = {
        "name": "static_student_internal_thread_scaling",
        "workload": {"T": 700, "d": 80},
        "seconds": {str(key): value for key, value in timings.items()},
        "speedup": {
            str(key): measured.median_ratio(1, key) for key in evaluators
        },
    }
    print("PYSCA_BENCHMARK " + json.dumps(
        payload, sort_keys=True), flush=True)


@pytest.mark.benchmark
def test_static_equicorr_internal_thread_scaling_benchmark():
    _benchmark_enabled()
    rng = np.random.default_rng(711)
    copula = EquicorrGaussianCopula(d=2)
    u = rng.uniform(0.02, 0.98, (400_000, 2))
    evaluators = {
        count: static_likelihood.prepare(copula, u, n_threads=count)
        for count in (1, 2, 4, 8)
    }
    calls = {
        count: lambda evaluator=evaluator: evaluator.result(0.2)
        for count, evaluator in evaluators.items()
    }
    for call in calls.values():
        call()
    measured = interleaved_timings(calls, repeats=5)
    timings = measured.medians
    reference = measured.results[1]
    for count in (2, 4, 8):
        result = measured.results[count]
        assert result["negative_log_likelihood"] == pytest.approx(
            reference["negative_log_likelihood"],
            rel=0.0,
            abs=1e-8,
        )
        assert result["negative_gradient"] == pytest.approx(
            reference["negative_gradient"], rel=0.0, abs=1e-8)

    payload = {
        "name": "static_equicorr_internal_thread_scaling",
        "workload": {"T": 400_000, "d": 2},
        "seconds": {str(key): value for key, value in timings.items()},
        "speedup": {
            str(key): measured.median_ratio(1, key) for key in evaluators
        },
    }
    print("PYSCA_BENCHMARK " + json.dumps(
        payload, sort_keys=True), flush=True)


@pytest.mark.benchmark
def test_student_rows_internal_thread_scaling_benchmark():
    _benchmark_enabled()
    copula, u = _student_case(T=700, d=80, seed=712)
    workers = (1, 2, 4, 8)
    calls = {
        count: lambda count=count: multivariate_native.log_pdf_and_dlog_rows(
            copula, u, 6.0, n_threads=count)
        for count in workers
    }
    for call in calls.values():
        call()
    measured = interleaved_timings(calls, repeats=5)
    timings = measured.medians
    reference = measured.results[1]
    for count in workers[1:]:
        result = measured.results[count]
        np.testing.assert_array_equal(result[0], reference[0])
        np.testing.assert_array_equal(result[1], reference[1])

    payload = {
        "name": "student_rows_internal_thread_scaling",
        "workload": {"T": 700, "d": 80},
        "seconds": {str(key): value for key, value in timings.items()},
        "speedup": {
            str(key): measured.median_ratio(1, key) for key in workers
        },
    }
    print("PYSCA_BENCHMARK " + json.dumps(
        payload, sort_keys=True), flush=True)
