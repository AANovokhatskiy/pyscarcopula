"""Optional performance checks for StochasticStudent fast paths."""

import os
import statistics
import time

import numpy as np
import pytest

from pyscarcopula._utils import pobs
from pyscarcopula.copula.multivariate.stochastic_student import (
    StochasticStudentCopula,
)
from pyscarcopula.copula.multivariate.corr_param import (
    _shrinkage_raw_corr_direction,
)
from pyscarcopula.numerical import _cpp_scar_ou
from pyscarcopula.numerical._scar_ou_config import AutoTMConfig
from pyscarcopula.numerical.gas_filter import gas_filter


auto_neg_loglik = _cpp_scar_ou.neg_loglik
auto_neg_loglik_with_grad = _cpp_scar_ou.neg_loglik_with_grad


_SCAR_WORKLOADS = [
    pytest.param(80, 3, 16, "matrix", id="matrix-T80-d3-K16"),
    pytest.param(160, 3, 20, "local", id="local-T160-d3-K20"),
    pytest.param(160, 3, 20, "spectral", id="spectral-T160-d3-K20"),
    pytest.param(80, 10, 16, "matrix", id="matrix-T80-d10-K16"),
]

_COLD_CACHE_WORKLOADS = [
    pytest.param(80, 3, 16, id="T80-d3-K16"),
    pytest.param(400, 10, 24, id="T400-d10-K24"),
]

_JOINT_WORKLOADS = [
    pytest.param("shrinkage", 3, id="shrinkage-d3"),
    pytest.param("cholesky", 3, id="cholesky-d3"),
    pytest.param("cholesky", 10, id="cholesky-d10"),
]

_LARGE_JOINT_WORKLOADS = [
    pytest.param(1000, 10, 24, id="T1000-d10-K24"),
    pytest.param(500, 15, 20, id="T500-d15-K20"),
    pytest.param(300, 20, 16, id="T300-d20-K16"),
]


def _skip_unless_enabled():
    if os.environ.get("PYSCA_RUN_BENCHMARKS") != "1":
        pytest.skip("set PYSCA_RUN_BENCHMARKS=1 to run benchmark checks")


def _skip_unless_large_enabled():
    if os.environ.get("PYSCA_RUN_LARGE_BENCHMARKS") != "1":
        pytest.skip(
            "set PYSCA_RUN_LARGE_BENCHMARKS=1 to run large benchmark checks")


def _skip_unless_cpp_available():
    if not _cpp_scar_ou.available():
        pytest.skip("StochasticStudent SCAR benchmarks require the C++ extension")


def _example_student(d=10, T=600, corr_mode="fixed"):
    rng = np.random.default_rng(15_000 + 100 * d + T)
    raw = rng.standard_t(df=5.0, size=(T, d))
    u = pobs(raw)
    if corr_mode != "fixed":
        return StochasticStudentCopula(
            d=d,
            corr_mode=corr_mode,
            allow_large_cholesky=True,
        ), u
    R = np.full((d, d), 0.35)
    np.fill_diagonal(R, 1.0)
    return StochasticStudentCopula(d=d, R=R), u


def _median_elapsed(call, repeats=5):
    elapsed = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = call()
        elapsed.append(time.perf_counter() - start)
    return statistics.median(elapsed), result


def _print_benchmark(name, **fields):
    values = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"BENCH {name} {values}", flush=True)


@pytest.mark.benchmark
def test_stochastic_student_gas_cpp_fast_path_speed_smoke():
    _skip_unless_enabled()
    copula, u = _example_student()
    params = (0.08, 0.04, 0.92)

    # Build and transfer the full-sample cache before measuring steady-state
    # throughput inside optimizer loops.
    gas_filter(*params, u[:8], copula, scaling="unit")
    gas_filter(*params, u, copula, scaling="unit")

    t0 = time.perf_counter()
    g_fast, r_fast, ll_fast = gas_filter(
        *params, u, copula, scaling="unit")
    fast_elapsed = time.perf_counter() - t0

    assert np.all(np.isfinite(g_fast))
    assert np.all(np.isfinite(r_fast))
    assert np.isfinite(ll_fast)
    assert fast_elapsed < 5.0


@pytest.mark.benchmark
@pytest.mark.parametrize(("T", "d", "K"), _COLD_CACHE_WORKLOADS)
def test_stochastic_student_cpp_cold_cache_benchmark(T, d, K):
    _skip_unless_enabled()
    _skip_unless_cpp_available()
    _, u = _example_student(d=d, T=T)
    R = np.full((d, d), 0.35)
    np.fill_diagonal(R, 1.0)
    config = AutoTMConfig(
        K=K,
        max_K=K,
        adaptive=False,
        transition_method="matrix",
    )
    params = (1.1, 0.7, 0.9)

    def cold_call():
        copula = StochasticStudentCopula(d=d, R=R)
        return _cpp_scar_ou.neg_loglik(*params, u, copula, config)

    warm_copula = StochasticStudentCopula(d=d, R=R)
    warm_value = _cpp_scar_ou.neg_loglik(
        *params, u, warm_copula, config)
    cold_elapsed, cold_value = _median_elapsed(cold_call, repeats=3)
    warm_elapsed, measured_warm_value = _median_elapsed(
        lambda: _cpp_scar_ou.neg_loglik(
            *params, u, warm_copula, config),
        repeats=7,
    )

    np.testing.assert_allclose(
        cold_value, warm_value, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        measured_warm_value, warm_value, rtol=0.0, atol=1e-12)
    _print_benchmark(
        "scar_cold_cache",
        T=T,
        d=d,
        K=K,
        transition="matrix",
        cold_ms=f"{1e3 * cold_elapsed:.3f}",
        warm_ms=f"{1e3 * warm_elapsed:.3f}",
        cold_over_warm=f"{cold_elapsed / warm_elapsed:.2f}",
    )


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("T", "d", "K", "transition_method"), _SCAR_WORKLOADS)
def test_stochastic_student_warm_likelihood_benchmark(
        T, d, K, transition_method):
    _skip_unless_enabled()
    _skip_unless_cpp_available()
    copula, u = _example_student(d=d, T=T)
    config = AutoTMConfig(
        K=K,
        max_K=K,
        adaptive=False,
        transition_method=transition_method,
        gh_order=5,
    )
    params = (1.1, 0.7, 0.9)

    wrapper_value = auto_neg_loglik(*params, u, copula, config)
    cpp_value = _cpp_scar_ou.neg_loglik(*params, u, copula, config)
    wrapper_elapsed, measured_wrapper = _median_elapsed(
        lambda: auto_neg_loglik(*params, u, copula, config))
    cpp_elapsed, measured_cpp = _median_elapsed(
        lambda: _cpp_scar_ou.neg_loglik(*params, u, copula, config))

    np.testing.assert_allclose(
        cpp_value, wrapper_value, rtol=0.0, atol=5e-4)
    np.testing.assert_allclose(
        measured_wrapper, wrapper_value, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        measured_cpp, cpp_value, rtol=0.0, atol=1e-12)
    _print_benchmark(
        "scar_warm_likelihood",
        T=T,
        d=d,
        K=K,
        transition=transition_method,
        wrapper_ms=f"{1e3 * wrapper_elapsed:.3f}",
        cpp_ms=f"{1e3 * cpp_elapsed:.3f}",
        wrapper_overhead=f"{wrapper_elapsed / cpp_elapsed:.2f}",
    )


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("T", "d", "K", "transition_method"), _SCAR_WORKLOADS)
def test_stochastic_student_warm_gradient_benchmark(
        T, d, K, transition_method):
    _skip_unless_enabled()
    _skip_unless_cpp_available()
    copula, u = _example_student(d=d, T=T)
    config = AutoTMConfig(
        K=K,
        max_K=K,
        adaptive=False,
        transition_method=transition_method,
        gh_order=5,
    )
    params = (1.1, 0.7, 0.9)

    wrapper_result = auto_neg_loglik_with_grad(
        *params, u, copula, config)
    cpp_result = _cpp_scar_ou.neg_loglik_with_grad(
        *params, u, copula, config)
    wrapper_elapsed, measured_wrapper = _median_elapsed(
        lambda: auto_neg_loglik_with_grad(
            *params, u, copula, config))
    cpp_elapsed, measured_cpp = _median_elapsed(
        lambda: _cpp_scar_ou.neg_loglik_with_grad(
            *params, u, copula, config))

    np.testing.assert_allclose(
        cpp_result[0], wrapper_result[0], rtol=0.0, atol=5e-4)
    np.testing.assert_allclose(
        cpp_result[1], wrapper_result[1], rtol=0.0, atol=5e-3)
    np.testing.assert_allclose(
        measured_wrapper[0], wrapper_result[0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        measured_cpp[0], cpp_result[0], rtol=0.0, atol=1e-12)
    _print_benchmark(
        "scar_warm_gradient",
        T=T,
        d=d,
        K=K,
        transition=transition_method,
        wrapper_ms=f"{1e3 * wrapper_elapsed:.3f}",
        cpp_ms=f"{1e3 * cpp_elapsed:.3f}",
        wrapper_overhead=f"{wrapper_elapsed / cpp_elapsed:.2f}",
    )


@pytest.mark.benchmark
def test_stochastic_student_prepared_spectral_directional_benchmark():
    _skip_unless_enabled()
    _skip_unless_cpp_available()
    d = 10
    T = 120
    n_calls = 20
    _, u = _example_student(d=d, T=T)
    corr_base = np.full((d, d), 0.25, dtype=np.float64)
    np.fill_diagonal(corr_base, 1.0)
    copula = StochasticStudentCopula(
        d=d,
        R=corr_base,
        corr_mode="shrinkage",
        corr_base=corr_base,
        allow_large_cholesky=True,
    )
    raw = np.array([0.2], dtype=np.float64)
    copula._set_corr_from_params(raw)
    direction = _shrinkage_raw_corr_direction(raw, copula._corr_base)
    config = AutoTMConfig(
        transition_method="spectral",
        basis_order=16,
        quad_order=48,
    )
    params = (1.1, 0.7, 0.9)
    prepared = _cpp_scar_ou.prepare_objective(u, copula, config)

    def functional_calls():
        result = None
        for _ in range(n_calls):
            result = _cpp_scar_ou.neg_loglik_with_grad_and_corr_directional_info(
                *params, u, copula, direction, config)
        return result

    def prepared_calls():
        result = None
        for _ in range(n_calls):
            result = prepared.neg_loglik_with_grad_and_corr_directional_info(
                *params, direction)
        return result

    functional_result = functional_calls()
    prepared_result = prepared_calls()
    functional_elapsed, measured_functional = _median_elapsed(
        functional_calls, repeats=3)
    prepared_elapsed, measured_prepared = _median_elapsed(
        prepared_calls, repeats=3)

    np.testing.assert_allclose(
        prepared_result[0], functional_result[0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        prepared_result[1], functional_result[1], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        prepared_result[2], functional_result[2], rtol=0.0, atol=0.0)
    assert prepared_result[2].shape == (1,)
    assert measured_functional[3]["backend"] == "spectral"
    assert measured_prepared[3]["backend"] == "spectral"
    _print_benchmark(
        "scar_prepared_spectral_directional",
        T=T,
        d=d,
        calls=n_calls,
        functional_ms=f"{1e3 * functional_elapsed:.3f}",
        prepared_ms=f"{1e3 * prepared_elapsed:.3f}",
        speedup=f"{functional_elapsed / prepared_elapsed:.2f}",
    )


@pytest.mark.benchmark
@pytest.mark.parametrize(("corr_mode", "d"), _JOINT_WORKLOADS)
def test_stochastic_student_joint_fit_benchmark(corr_mode, d):
    _skip_unless_enabled()
    _skip_unless_cpp_available()
    _, u = _example_student(d=d, T=80)
    fit_kwargs = {
        "method": "scar-tm-ou",
        "alpha0": np.array([1.0, 1.0, 0.8]),
        "K": 16,
        "max_K": 16,
        "adaptive": False,
        "transition_method": "matrix",
        "maxiter": 2,
        "maxfun": 200,
        "analytical_grad": True,
        "smart_init": False,
    }

    def fit_native():
        copula = StochasticStudentCopula(
            d=d,
            corr_mode=corr_mode,
            allow_large_cholesky=True,
        )
        return copula.fit(u, **fit_kwargs)

    # Exclude one-time imports/JIT compilation from the measured fit.
    fit_native()
    native_elapsed, result = _median_elapsed(fit_native, repeats=3)

    assert np.isfinite(result.log_likelihood)
    assert result.diagnostics["joint_gradient"] == "analytical"
    assert result.diagnostics["selected_engine"] == "cpp"
    assert result.diagnostics["correlation_fd_evaluations"] == 0
    assert (
        result.diagnostics[
            "native_correlation_gradient_evaluations"] > 0)
    _print_benchmark(
        "scar_joint_fit",
        T=80,
        d=d,
        K=16,
        transition="matrix",
        corr_mode=corr_mode,
        n_corr=result.diagnostics["corr_n_params"],
        native_ms=f"{1e3 * native_elapsed:.3f}",
        objectives=result.diagnostics["objective_evaluations"],
    )


@pytest.mark.benchmark
@pytest.mark.parametrize(("T", "d", "K"), _LARGE_JOINT_WORKLOADS)
def test_stochastic_student_large_cholesky_native_gradient_benchmark(
        T, d, K, monkeypatch):
    _skip_unless_large_enabled()
    _skip_unless_cpp_available()
    _, u = _example_student(d=d, T=T)
    fallback_timings = {
        "fd_seconds": 0.0,
        "ou_gradient_seconds": 0.0,
        "fd_calls": 0,
        "ou_gradient_calls": 0,
    }
    native_timings = {"seconds": 0.0, "calls": 0}
    original_value = _cpp_scar_ou.neg_loglik_info
    original_gradient = _cpp_scar_ou.neg_loglik_with_grad_info
    original_native = _cpp_scar_ou.neg_loglik_with_grad_and_corr_info
    original_prepare = _cpp_scar_ou.prepare_objective

    def timed_value(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original_value(*args, **kwargs)
        finally:
            fallback_timings["fd_seconds"] += time.perf_counter() - start
            fallback_timings["fd_calls"] += 1

    def timed_gradient(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original_gradient(*args, **kwargs)
        finally:
            fallback_timings[
                "ou_gradient_seconds"] += time.perf_counter() - start
            fallback_timings["ou_gradient_calls"] += 1

    def unsupported_native(*args, **kwargs):
        raise _cpp_scar_ou.CppUnsupported("benchmark finite-difference path")

    def unsupported_prepare(*args, **kwargs):
        raise _cpp_scar_ou.CppUnsupported("benchmark module-level path")

    def timed_native(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original_native(*args, **kwargs)
        finally:
            native_timings["seconds"] += time.perf_counter() - start
            native_timings["calls"] += 1

    def fit_model():
        copula = StochasticStudentCopula(
            d=d,
            corr_mode="cholesky",
            allow_large_cholesky=True,
        )
        return copula.fit(
            u,
            method="scar-tm-ou",
            alpha0=np.array([1.0, 1.0, 0.8]),
            K=K,
            max_K=K,
            adaptive=False,
            transition_method="matrix",
            maxiter=1,
            maxfun=500,
            analytical_grad=True,
            smart_init=False,
        )

    monkeypatch.setattr(_cpp_scar_ou, "neg_loglik_info", timed_value)
    monkeypatch.setattr(
        _cpp_scar_ou, "neg_loglik_with_grad_info", timed_gradient)
    monkeypatch.setattr(
        _cpp_scar_ou,
        "neg_loglik_with_grad_and_corr_info",
        unsupported_native,
    )
    monkeypatch.setattr(
        _cpp_scar_ou, "prepare_objective", unsupported_prepare)
    start = time.perf_counter()
    fallback_result = fit_model()
    fallback_seconds = time.perf_counter() - start

    monkeypatch.setattr(_cpp_scar_ou, "neg_loglik_info", original_value)
    monkeypatch.setattr(
        _cpp_scar_ou, "neg_loglik_with_grad_info", original_gradient)
    monkeypatch.setattr(
        _cpp_scar_ou, "neg_loglik_with_grad_and_corr_info", timed_native)
    monkeypatch.setattr(
        _cpp_scar_ou, "prepare_objective", unsupported_prepare)
    start = time.perf_counter()
    native_result = fit_model()
    native_seconds = time.perf_counter() - start
    monkeypatch.setattr(
        _cpp_scar_ou, "prepare_objective", original_prepare)

    fallback = fallback_result.diagnostics
    native = native_result.diagnostics
    backend_seconds = (
        fallback_timings["fd_seconds"]
        + fallback_timings["ou_gradient_seconds"]
    )
    fd_backend_share = fallback_timings["fd_seconds"] / backend_seconds
    speedup = fallback_seconds / native_seconds

    assert np.isfinite(fallback_result.log_likelihood)
    assert np.isfinite(native_result.log_likelihood)
    assert fallback["corr_n_params"] == d * (d - 1) // 2
    assert native["corr_n_params"] == d * (d - 1) // 2
    assert fallback_timings["fd_calls"] == (
        fallback["correlation_fd_evaluations"])
    assert fallback_timings["ou_gradient_calls"] == (
        fallback["hybrid_gradient_evaluations"])
    assert (
        fallback_timings["fd_calls"]
        + fallback_timings["ou_gradient_calls"]
        == fallback["objective_evaluations"]
    )
    assert native["correlation_fd_evaluations"] == 0
    assert native_timings["calls"] == (
        native["native_correlation_gradient_evaluations"])
    assert native["objective_evaluations"] == native_timings["calls"]
    assert native["objective_evaluations"] < fallback["objective_evaluations"]
    _print_benchmark(
        "scar_large_cholesky_native_corr_gradient",
        T=T,
        d=d,
        K=K,
        n_corr=native["corr_n_params"],
        fallback_ms=f"{1e3 * fallback_seconds:.3f}",
        native_ms=f"{1e3 * native_seconds:.3f}",
        speedup=f"{speedup:.2f}",
        fallback_objectives=fallback["objective_evaluations"],
        native_objectives=native["objective_evaluations"],
        fd_calls=fallback_timings["fd_calls"],
        native_calls=native_timings["calls"],
        fd_backend_ms=f"{1e3 * fallback_timings['fd_seconds']:.3f}",
        native_backend_ms=f"{1e3 * native_timings['seconds']:.3f}",
        fd_backend_share=f"{fd_backend_share:.3f}",
    )
