"""Correctness and timing baselines for multivariate parallelism."""

from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
import os
import platform
import sys

import numpy as np
import pytest

from tools.benchmark_timing import interleaved_timings
from pyscarcopula import EquicorrGaussianCopula, StochasticStudentCopula
from pyscarcopula.numerical import multivariate_native


_SMALL_BENCHMARKS = [
    pytest.param("equicorr", 50_000, 40, 64, id="equicorr-T50000-d40-K64"),
    pytest.param("student", 700, 80, 40, id="student-T700-d80-K40"),
]

_LARGE_BENCHMARKS = [
    pytest.param("equicorr", 100_000, 100, 100,
                 id="equicorr-T100000-d100-K100"),
    pytest.param("student", 1_000, 100, 100,
                 id="student-T1000-d100-K100"),
]


def _equicorrelation(d, rho=0.2):
    correlation = np.full((d, d), rho, dtype=np.float64)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def _make_case(family, *, T, d, K, seed=20260721):
    rng = np.random.default_rng(seed + T + 10 * d + 100 * K)
    u = rng.uniform(0.02, 0.98, (T, d))
    grid = np.linspace(-2.5, 2.5, K)
    if family == "equicorr":
        return EquicorrGaussianCopula(d=d), u, grid, None
    copula = StochasticStudentCopula(d=d, R=_equicorrelation(d))
    cache = copula.prepare_emission_cache(u)
    return copula, u, grid, cache


def _grid_block(copula, u, grid, start, stop, cache):
    if cache is None:
        return copula.pdf_and_grad_on_grid_batch(u[start:stop], grid)
    return copula.pdf_and_grad_on_grid_batch(
        u[start:stop], grid, t_index=start, cache=cache)


def _chunked_grid(copula, u, grid, cache, n_workers):
    if n_workers == 1:
        return _grid_block(copula, u, grid, 0, len(u), cache)

    edges = np.linspace(0, len(u), n_workers + 1, dtype=np.int64)
    blocks = [
        (int(edges[index]), int(edges[index + 1]))
        for index in range(n_workers)
        if edges[index] < edges[index + 1]
    ]
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        parts = list(executor.map(
            lambda bounds: _grid_block(
                copula, u, grid, bounds[0], bounds[1], cache),
            blocks,
        ))
    return tuple(
        np.concatenate([part[item] for part in parts], axis=0)
        for item in (0, 1)
    )


def _measure_calls(calls, repeats):
    for call in calls.values():
        call()
    return interleaved_timings(calls, repeats=repeats)


def _peak_rss_bytes():
    """Return process peak RSS where the host exposes it, else ``None``."""
    if sys.platform == "win32":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        process = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb)
        return int(counters.PeakWorkingSetSize) if ok else None

    try:
        import resource
    except ImportError:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux and the BSDs report KiB.
    return int(value if sys.platform == "darwin" else value * 1024)


@pytest.mark.parametrize("family", ["equicorr", "student"])
def test_row_chunking_is_bitwise_equivalent(family):
    copula, u, grid, cache = _make_case(family, T=41, d=8, K=17)
    expected = _chunked_grid(copula, u, grid, cache, 1)

    for n_workers in (2, 4, 8):
        actual = _chunked_grid(copula, u, grid, cache, n_workers)
        assert np.array_equal(actual[0], expected[0])
        assert np.array_equal(actual[1], expected[1])


def test_student_cache_is_not_reused_for_equal_shape_rolling_window():
    d = 5
    first = np.random.default_rng(101).uniform(0.02, 0.98, (32, d))
    second = np.random.default_rng(202).uniform(0.02, 0.98, (32, d))
    grid = np.linspace(-1.5, 1.5, 9)
    correlation = _equicorrelation(d)
    rolling = StochasticStudentCopula(d=d, R=correlation)

    first_cache = rolling.prepare_emission_cache(first)
    rolling.pdf_and_grad_on_grid_batch(first, grid, cache=first_cache)
    second_cache = rolling.prepare_emission_cache(second)
    actual = rolling.pdf_and_grad_on_grid_batch(
        second, grid, cache=second_cache)

    fresh = StochasticStudentCopula(d=d, R=correlation)
    expected = fresh.pdf_and_grad_on_grid_batch(second, grid)

    assert second_cache.version != first_cache.version
    assert np.array_equal(second_cache.u_snapshot, second)
    np.testing.assert_allclose(actual[0], expected[0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual[1], expected[1], rtol=0.0, atol=0.0)


def test_repeated_rolling_mle_matches_fresh_model():
    d = 3
    correlation = _equicorrelation(d)
    windows = [
        np.random.default_rng(seed).uniform(0.02, 0.98, (48, d))
        for seed in (301, 302, 303)
    ]
    rolling = StochasticStudentCopula(d=d, R=correlation)

    for u in windows:
        rolling_result = rolling.fit(
            u, method="mle", maxiter=30, maxfun=60)
        fresh = StochasticStudentCopula(d=d, R=correlation)
        fresh_result = fresh.fit(
            u, method="mle", maxiter=30, maxfun=60)

        assert rolling_result.success == fresh_result.success
        assert rolling_result.copula_param == fresh_result.copula_param
        np.testing.assert_array_equal(
            rolling.log_pdf_rows(u, rolling_result.copula_param),
            fresh.log_pdf_rows(u, fresh_result.copula_param),
        )


def _fit_independent_window(payload):
    u, correlation = payload
    copula = StochasticStudentCopula(d=u.shape[1], R=correlation)
    result = copula.fit(u, method="mle", maxiter=30, maxfun=60)
    return result.copula_param, copula.log_pdf_rows(u, result.copula_param)


def test_independent_models_are_reentrant_in_external_thread_pool():
    d = 3
    correlation = _equicorrelation(d)
    windows = [
        np.random.default_rng(seed).uniform(0.02, 0.98, (48, d))
        for seed in (401, 402, 403, 404)
    ]
    payloads = [(u, correlation) for u in windows]
    expected = [_fit_independent_window(payload) for payload in payloads]

    with ThreadPoolExecutor(max_workers=4) as executor:
        actual = list(executor.map(_fit_independent_window, payloads))

    for actual_item, expected_item in zip(actual, expected):
        assert actual_item[0] == expected_item[0]
        np.testing.assert_array_equal(actual_item[1], expected_item[1])


def test_student_kernel_diagnostics_cover_cached_and_exact_ppf_paths():
    T, d, K = 3, 4, 5
    copula, u, grid, cache = _make_case(
        "student", T=T, d=d, K=K, seed=501)

    _, _, cached = multivariate_native.pdf_and_grad_grid_info(
        copula, u, grid, cache=cache)
    assert cached["student_ppf_cache_values"] == T * K * d
    assert cached["student_ppf_exact_values"] == 0
    assert cached["student_ppf_asymptotic_values"] == 0
    assert cached["student_ppf_total_values"] == T * K * d
    assert cached["student_workspace_growth_events"] == 2
    assert cached["student_parallel_blocks"] == 1
    assert cached["student_workspace_peak_bytes"] >= 2 * d * 8

    _, _, exact = multivariate_native.log_pdf_and_dlog_rows_info(
        copula, u, 5.0)
    assert exact["student_ppf_cache_values"] == 0
    assert exact["student_ppf_exact_values"] == T * d
    assert exact["student_ppf_asymptotic_values"] == 0
    assert exact["student_ppf_total_values"] == T * d
    assert exact["student_workspace_growth_events"] == 2
    assert exact["student_workspace_peak_bytes"] >= 2 * d * 8


def test_student_kernel_diagnostics_cover_large_df_asymptotic_path():
    T, d, K = 3, 4, 2
    copula, u, _, _ = _make_case(
        "student", T=T, d=d, K=K, seed=502)
    large_df_grid = np.array([2_500_000.0, 3_000_000.0])

    _, _, diagnostics = multivariate_native.pdf_and_grad_grid_info(
        copula, u, large_df_grid)

    assert diagnostics["student_ppf_cache_values"] == 0
    assert diagnostics["student_ppf_exact_values"] == 0
    assert diagnostics["student_ppf_asymptotic_values"] == T * K * d
    assert diagnostics["student_ppf_total_values"] == T * K * d


def test_student_kernel_diagnostics_are_invocation_local_across_threads():
    T, d = 4, 5
    copula, u, grid, cache = _make_case(
        "student", T=T, d=d, K=3, seed=503)

    with ThreadPoolExecutor(max_workers=2) as executor:
        cached_future = executor.submit(
            multivariate_native.pdf_and_grad_grid_info,
            copula, u, grid, cache=cache)
        exact_future = executor.submit(
            multivariate_native.log_pdf_and_dlog_rows_info,
            copula, u, 5.0)
        cached = cached_future.result()[2]
        exact = exact_future.result()[2]

    assert cached["student_ppf_cache_values"] == T * len(grid) * d
    assert cached["student_ppf_exact_values"] == 0
    assert exact["student_ppf_cache_values"] == 0
    assert exact["student_ppf_exact_values"] == T * d


def test_student_internal_threads_are_bitwise_equivalent():
    T, d, K = 41, 8, 17
    copula, u, grid, cache = _make_case(
        "student", T=T, d=d, K=K, seed=504)
    expected = multivariate_native.pdf_and_grad_grid_info(
        copula, u, grid, cache=cache, n_threads=1)

    for n_threads in (2, 4, 8):
        actual = multivariate_native.pdf_and_grad_grid_info(
            copula, u, grid, cache=cache, n_threads=n_threads)
        assert np.array_equal(actual[0], expected[0])
        assert np.array_equal(actual[1], expected[1])
        assert actual[2]["student_ppf_total_values"] == T * K * d
        assert actual[2]["n_threads_requested"] == n_threads
        assert 1 < actual[2]["student_parallel_blocks"] <= n_threads
        assert actual[2]["student_workspace_growth_events"] == (
            2 * actual[2]["student_parallel_blocks"])


def test_student_small_grid_uses_sequential_fast_path():
    from pyscarcopula.numerical import _cpp_extension

    copula, u, grid, cache = _make_case(
        "student", T=8, d=6, K=9, seed=508)
    module = _cpp_extension.load()
    batches_before = dict(module._parallel_runtime_info())[
        "batches_submitted"]

    _, _, diagnostics = multivariate_native.pdf_and_grad_grid_info(
        copula, u, grid, cache=cache, n_threads=8)

    batches_after = dict(module._parallel_runtime_info())[
        "batches_submitted"]
    assert diagnostics["n_threads_requested"] == 8
    assert diagnostics["student_parallel_blocks"] == 1
    assert diagnostics["student_workspace_growth_events"] == 2
    assert batches_after == batches_before


def test_equicorr_internal_threads_are_bitwise_equivalent():
    T, d, K = 4_160, 40, 64
    copula, u, grid, _ = _make_case(
        "equicorr", T=T, d=d, K=K, seed=510)
    expected = multivariate_native.pdf_and_grad_grid_info(
        copula, u, grid, n_threads=1)

    for n_threads in (2, 4, 8):
        actual = multivariate_native.pdf_and_grad_grid_info(
            copula, u, grid, n_threads=n_threads)
        assert np.array_equal(actual[0], expected[0])
        assert np.array_equal(actual[1], expected[1])
        assert actual[2]["n_threads_requested"] == n_threads
        assert actual[2]["student_parallel_blocks"] == 0
        assert 1 < actual[2]["equicorr_parallel_blocks"] <= n_threads


def test_equicorr_small_grid_uses_sequential_fast_path():
    from pyscarcopula.numerical import _cpp_extension

    copula, u, grid, _ = _make_case(
        "equicorr", T=128, d=40, K=64, seed=511)
    module = _cpp_extension.load()
    batches_before = dict(module._parallel_runtime_info())[
        "batches_submitted"]

    _, _, diagnostics = multivariate_native.pdf_and_grad_grid_info(
        copula, u, grid, n_threads=8)

    batches_after = dict(module._parallel_runtime_info())[
        "batches_submitted"]
    assert diagnostics["n_threads_requested"] == 8
    assert diagnostics["equicorr_parallel_blocks"] == 1
    assert batches_after == batches_before


def test_equicorr_parallel_failure_index_matches_sequential():
    from pyscarcopula.numerical import _cpp_copula, _cpp_extension

    T, d, K = 4_160, 40, 64
    copula = EquicorrGaussianCopula(d=d)
    u = np.full((T, d), 1e-8, dtype=np.float64)
    u[:5] = 0.5
    grid = np.linspace(-2.5, 2.5, K)
    module = _cpp_extension.load()
    spec = _cpp_copula.make_multivariate_spec(module, copula, cache=None)

    sequential = dict(module.multivariate_pdf_and_grad_grid(
        spec, u, grid, 0, 1))
    parallel = dict(module.multivariate_pdf_and_grad_grid(
        spec, u, grid, 0, 4))

    assert sequential["status"] == module.SCAR_NUMERICAL_FAILURE
    assert parallel["status"] == module.SCAR_NUMERICAL_FAILURE
    assert sequential["failure_index"] == parallel["failure_index"] == 5
    np.testing.assert_array_equal(
        parallel["pdf"][:5], sequential["pdf"][:5])
    np.testing.assert_array_equal(parallel["pdf"][6:], 0.0)
    np.testing.assert_array_equal(parallel["d_pdf_dx"][6:], 0.0)


@pytest.mark.parametrize("n_threads", [0, 257, True, 1.5])
def test_student_internal_threads_reject_invalid_values(n_threads):
    copula, u, grid, cache = _make_case(
        "student", T=10, d=4, K=5, seed=505)
    with pytest.raises(ValueError, match="n_threads"):
        copula.pdf_and_grad_on_grid_batch(
            u, grid, cache=cache, n_threads=n_threads)


@pytest.mark.parametrize("n_threads", [0, 257, True, 1.5])
def test_equicorr_internal_threads_reject_invalid_values(n_threads):
    copula, u, grid, _ = _make_case(
        "equicorr", T=10, d=4, K=5, seed=514)
    with pytest.raises(ValueError, match="n_threads"):
        copula.pdf_and_grad_on_grid_batch(
            u, grid, n_threads=n_threads)


def test_student_parallel_failure_index_matches_sequential():
    from pyscarcopula.numerical import _cpp_copula, _cpp_extension

    T, d, K = 24, 4, 7
    copula, u, grid, _ = _make_case(
        "student", T=T, d=d, K=K, seed=506)
    u[:5] = 0.5
    module = _cpp_extension.load()
    spec = _cpp_copula.make_multivariate_spec(module, copula, cache=None)
    spec.l_inv = (np.eye(d) * 1e200).reshape(-1).tolist()

    sequential = dict(module.multivariate_pdf_and_grad_grid(
        spec, u, grid, 0, 1))
    parallel = dict(module.multivariate_pdf_and_grad_grid(
        spec, u, grid, 0, 4))

    assert sequential["status"] == module.SCAR_NUMERICAL_FAILURE
    assert parallel["status"] == module.SCAR_NUMERICAL_FAILURE
    assert sequential["failure_index"] == parallel["failure_index"] == 5
    np.testing.assert_array_equal(
        parallel["pdf"][:5], sequential["pdf"][:5])
    np.testing.assert_array_equal(parallel["pdf"][6:], 0.0)
    np.testing.assert_array_equal(parallel["d_pdf_dx"][6:], 0.0)


def _benchmark_enabled(large):
    if os.environ.get("PYSCA_RUN_BENCHMARKS") != "1":
        pytest.skip("set PYSCA_RUN_BENCHMARKS=1 to run benchmark baselines")
    if large and os.environ.get("PYSCA_RUN_LARGE_BENCHMARKS") != "1":
        pytest.skip(
            "set PYSCA_RUN_LARGE_BENCHMARKS=1 to run large benchmarks")


def _run_scaling_benchmark(family, T, d, K, *, large):
    _benchmark_enabled(large)
    copula, u, grid, cache = _make_case(family, T=T, d=d, K=K)
    _grid_block(copula, u, grid, 0, min(T, 8), cache)

    workers = [count for count in (1, 2, 4, 8)
               if count <= (os.cpu_count() or 1)]
    measured = _measure_calls(
        {
            count: lambda count=count: _chunked_grid(
                copula, u, grid, cache, count)
            for count in workers
        },
        repeats=3 if family == "student" else 5,
    )
    timings = measured.medians
    reference = measured.results[workers[0]]
    for count in workers[1:]:
        result = measured.results[count]
        assert np.array_equal(result[0], reference[0])
        assert np.array_equal(result[1], reference[1])

    payload = {
        "name": "multivariate_external_thread_scaling_baseline",
        "family": family,
        "workload": {"T": T, "d": d, "K": K},
        "seconds": {str(key): value for key, value in timings.items()},
        "speedup": {
            str(key): measured.median_ratio(1, key) for key in workers
        },
        "execution": "python_thread_chunks",
        "peak_rss_bytes": _peak_rss_bytes(),
        "ppf_cache_bytes": (
            None
            if cache is None or cache.ppf_table is None
            else int(cache.ppf_table.nbytes)
        ),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "timer": "perf_counter interleaved median",
    }
    if family == "student":
        _, _, diagnostics = multivariate_native.pdf_and_grad_grid_info(
            copula, u, grid, cache=cache)
        payload["student_kernel_diagnostics"] = diagnostics
    print("PYSCA_BENCHMARK " + json.dumps(
        payload, sort_keys=True), flush=True)


@pytest.mark.benchmark
@pytest.mark.parametrize(("family", "T", "d", "K"), _SMALL_BENCHMARKS)
def test_external_thread_scaling_benchmark(family, T, d, K):
    _run_scaling_benchmark(family, T, d, K, large=False)


@pytest.mark.benchmark
@pytest.mark.parametrize(("family", "T", "d", "K"), _LARGE_BENCHMARKS)
def test_large_external_thread_scaling_benchmark(family, T, d, K):
    _run_scaling_benchmark(family, T, d, K, large=True)


@pytest.mark.benchmark
def test_student_internal_thread_scaling_benchmark():
    _benchmark_enabled(large=False)
    T, d, K = 700, 80, 40
    copula, u, grid, cache = _make_case(
        "student", T=T, d=d, K=K, seed=507)
    max_workers = min(8, os.cpu_count() or 1)
    copula.pdf_and_grad_on_grid_batch(
        u, grid, cache=cache, n_threads=max_workers)

    workers = [count for count in (1, 2, 4, 8)
               if count <= (os.cpu_count() or 1)]
    measured = _measure_calls(
        {
            count: (
                lambda count=count: multivariate_native.pdf_and_grad_grid_info(
                copula,
                u,
                grid,
                cache=cache,
                n_threads=count,
                )
            )
            for count in workers
        },
        repeats=5,
    )
    timings = measured.medians
    reference = measured.results[workers[0]]
    diagnostics = {
        count: measured.results[count][2] for count in workers
    }
    for count in workers[1:]:
        result = measured.results[count]
        assert np.array_equal(result[0], reference[0])
        assert np.array_equal(result[1], reference[1])

    payload = {
        "name": "student_internal_thread_scaling",
        "family": "student",
        "workload": {"T": T, "d": d, "K": K},
        "seconds": {str(key): value for key, value in timings.items()},
        "speedup": {
            str(key): measured.median_ratio(1, key) for key in workers
        },
        "kernel_diagnostics": {
            str(key): value for key, value in diagnostics.items()
        },
        "peak_rss_bytes": _peak_rss_bytes(),
        "logical_cpus": os.cpu_count(),
        "timer": "perf_counter interleaved median",
    }
    print("PYSCA_BENCHMARK " + json.dumps(
        payload, sort_keys=True), flush=True)


@pytest.mark.benchmark
def test_student_scar_matrix_internal_thread_benchmark():
    from pyscarcopula.numerical import _cpp_scar_ou
    from pyscarcopula.numerical._scar_ou_config import AutoTMConfig

    _benchmark_enabled(large=False)
    T, d, K = 300, 40, 40
    copula, u, _, _ = _make_case(
        "student", T=T, d=d, K=K, seed=509)
    workers = [count for count in (1, 2, 4, 8)
               if count <= (os.cpu_count() or 1)]
    prepared = {
        count: _cpp_scar_ou.prepare_objective(
            u,
            copula,
            AutoTMConfig(
                transition_method="matrix",
                K=K,
                adaptive=False,
                max_K=K,
                n_threads=count,
            ),
        )
        for count in workers
    }
    prepared[max(workers)].neg_loglik_with_grad_info(1.0, 0.1, 0.5)

    measured = _measure_calls(
        {
            count: lambda count=count: prepared[
                count].neg_loglik_with_grad_info(1.0, 0.1, 0.5)
            for count in workers
        },
        repeats=5,
    )
    timings = measured.medians
    reference = measured.results[workers[0]]
    for count in workers[1:]:
        result = measured.results[count]
        assert result[0] == reference[0]
        np.testing.assert_array_equal(result[1], reference[1])

    payload = {
        "name": "student_scar_matrix_gradient_internal_thread_scaling",
        "family": "student",
        "workload": {"T": T, "d": d, "K": K},
        "seconds": {str(key): value for key, value in timings.items()},
        "speedup": {
            str(key): measured.median_ratio(1, key) for key in workers
        },
        "logical_cpus": os.cpu_count(),
        "timer": "perf_counter interleaved median",
    }
    print("PYSCA_BENCHMARK " + json.dumps(
        payload, sort_keys=True), flush=True)


@pytest.mark.benchmark
def test_equicorr_internal_thread_scaling_benchmark():
    _benchmark_enabled(large=False)
    T, d, K = 50_000, 40, 64
    copula, u, grid, _ = _make_case(
        "equicorr", T=T, d=d, K=K, seed=512)
    max_workers = min(8, os.cpu_count() or 1)
    copula.pdf_and_grad_on_grid_batch(
        u, grid, n_threads=max_workers)

    workers = [count for count in (1, 2, 4, 8)
               if count <= (os.cpu_count() or 1)]
    measured = _measure_calls(
        {
            count: (
                lambda count=count: multivariate_native.pdf_and_grad_grid_info(
                    copula, u, grid, n_threads=count)
            )
            for count in workers
        },
        repeats=5,
    )
    timings = measured.medians
    reference = measured.results[workers[0]]
    diagnostics = {
        count: measured.results[count][2] for count in workers
    }
    for count in workers[1:]:
        result = measured.results[count]
        assert np.array_equal(result[0], reference[0])
        assert np.array_equal(result[1], reference[1])

    payload = {
        "name": "equicorr_internal_thread_scaling",
        "family": "equicorr",
        "workload": {"T": T, "d": d, "K": K},
        "seconds": {str(key): value for key, value in timings.items()},
        "speedup": {
            str(key): measured.median_ratio(1, key) for key in workers
        },
        "kernel_diagnostics": {
            str(key): value for key, value in diagnostics.items()
        },
        "logical_cpus": os.cpu_count(),
        "timer": "perf_counter interleaved median",
    }
    print("PYSCA_BENCHMARK " + json.dumps(
        payload, sort_keys=True), flush=True)


@pytest.mark.benchmark
def test_equicorr_scar_matrix_internal_thread_benchmark():
    from pyscarcopula.numerical import _cpp_scar_ou
    from pyscarcopula.numerical._scar_ou_config import AutoTMConfig

    _benchmark_enabled(large=False)
    T, d, K = 8_000, 40, 40
    copula, u, _, _ = _make_case(
        "equicorr", T=T, d=d, K=K, seed=513)
    workers = [count for count in (1, 2, 4, 8)
               if count <= (os.cpu_count() or 1)]
    prepared = {
        count: _cpp_scar_ou.prepare_objective(
            u,
            copula,
            AutoTMConfig(
                transition_method="matrix",
                K=K,
                adaptive=False,
                max_K=K,
                n_threads=count,
            ),
        )
        for count in workers
    }
    prepared[max(workers)].neg_loglik_with_grad_info(1.0, 0.1, 0.5)

    measured = _measure_calls(
        {
            count: lambda count=count: prepared[
                count].neg_loglik_with_grad_info(1.0, 0.1, 0.5)
            for count in workers
        },
        repeats=5,
    )
    timings = measured.medians
    reference = measured.results[workers[0]]
    for count in workers[1:]:
        result = measured.results[count]
        assert result[0] == reference[0]
        np.testing.assert_array_equal(result[1], reference[1])

    payload = {
        "name": "equicorr_scar_matrix_gradient_internal_thread_scaling",
        "family": "equicorr",
        "workload": {"T": T, "d": d, "K": K},
        "seconds": {str(key): value for key, value in timings.items()},
        "speedup": {
            str(key): measured.median_ratio(1, key) for key in workers
        },
        "logical_cpus": os.cpu_count(),
        "timer": "perf_counter interleaved median",
    }
    print("PYSCA_BENCHMARK " + json.dumps(
        payload, sort_keys=True), flush=True)
