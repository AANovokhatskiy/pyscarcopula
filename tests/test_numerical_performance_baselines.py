"""Opt-in timing reports for numerical workloads.

Correctness is asserted here; timings are observational and intentionally have
no host-dependent wall-clock gate.
"""

import json
import os
import platform
import statistics
import sys
import time

import numpy as np
import pytest
import scipy

from tools.benchmark_timing import interleaved_timings
from pyscarcopula import GaussianCopula, GumbelCopula, StudentCopula
from pyscarcopula.copula.multivariate.correlation_policy import (
    CorrelationPolicy,
)
from pyscarcopula.copula.multivariate.stochastic_student import (
    StochasticStudentCopula,
)
from pyscarcopula._native import scar_ou as _cpp_scar_ou
from pyscarcopula._native import static as static_likelihood
from pyscarcopula._native import jacobi as jacobi_native
from pyscarcopula.strategy.mle import MLEStrategy


def _enabled():
    if os.environ.get("PYSCA_RUN_BENCHMARKS") != "1":
        pytest.skip("set PYSCA_RUN_BENCHMARKS=1 to run benchmark baselines")


def _median_elapsed(call, repeats=5):
    elapsed = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = call()
        elapsed.append(time.perf_counter() - start)
    return statistics.median(elapsed), result


def _report(name, elapsed, *, workload, cache_state):
    payload = {
        "name": name,
        "seconds": elapsed,
        "workload": workload,
        "cache_state": cache_state,
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "native_extension": "required",
        "timer": "perf_counter median",
    }
    print("NUMERICAL_BENCH " + json.dumps(payload, sort_keys=True), flush=True)


@pytest.mark.benchmark
def test_bivariate_mle_objective_benchmark_report():
    _enabled()
    u = np.random.default_rng(20260622).uniform(0.01, 0.99, (20_000, 2))
    copula = GumbelCopula(rotate=180)
    strategy = MLEStrategy()
    alpha = np.array([1.8])
    strategy.objective(copula, u[:8], alpha)

    elapsed, value = _median_elapsed(
        lambda: strategy.objective(copula, u, alpha)
    )

    assert np.isfinite(value)
    _report(
        "bivariate_mle_objective",
        elapsed,
        workload={"T": len(u), "family": "gumbel", "rotation": 180},
        cache_state="warm",
    )


@pytest.mark.benchmark
def test_gaussian_mle_objective_benchmark_report():
    _enabled()
    d = 5
    u = np.random.default_rng(20260623).uniform(0.01, 0.99, (20_000, d))
    correlation = np.full((d, d), 0.2)
    np.fill_diagonal(correlation, 1.0)
    copula = GaussianCopula()
    copula.corr = correlation
    copula._nll(u[:8])

    elapsed, result = _median_elapsed(lambda: copula._nll(u))

    assert np.isfinite(result)
    _report(
        "gaussian_mle_objective",
        elapsed,
        workload={"T": len(u), "dimension": d},
        cache_state="warm",
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("T,d", [(200, 3), (1_000, 10), (3_000, 20)])
@pytest.mark.parametrize("corr_mode", ["fixed", "shrinkage", "cholesky"])
@pytest.mark.parametrize("family", ["gaussian", "student"])
def test_static_elliptical_correlation_mode_benchmark_matrix(
        T, d, corr_mode, family):
    """Report dense static workloads without host-specific time limits."""
    _enabled()
    rng = np.random.default_rng(20260801 + T + d)
    u = rng.uniform(0.02, 0.98, (T, d))
    base = np.full((d, d), 0.15, dtype=np.float64)
    np.fill_diagonal(base, 1.0)

    if corr_mode == "fixed":
        policy = CorrelationPolicy.create(
            mode="fixed",
            estimator="supplied",
            dimension=d,
            supplied_correlation=base,
        )
        raw = np.empty(0, dtype=np.float64)
    else:
        # Construction also exercises the public high-dimensional override.
        model_type = GaussianCopula if family == "gaussian" else StudentCopula
        model_type(
            d=d,
            corr_mode=corr_mode,
            allow_large_cholesky=(corr_mode == "cholesky" and d > 10),
        )
        policy = CorrelationPolicy.create(
            mode=corr_mode,
            estimator="joint_mle",
            dimension=d,
            base_correlation=base,
        )
        raw = policy.initial_raw_parameters()
    correlation = policy.trial_correlation(raw)

    calls = {}
    for n_threads in (1, 2, 4):
        prepare = (
            static_likelihood.prepare_gaussian
            if family == "gaussian"
            else static_likelihood.prepare_student)
        evaluator = prepare(correlation, u, n_threads=n_threads)
        if corr_mode == "fixed":
            parameter = 0.0 if family == "gaussian" else 6.0
            calls[n_threads] = (
                lambda evaluator=evaluator, parameter=parameter:
                evaluator.log_likelihood(parameter)
            )
        else:
            def call(evaluator=evaluator):
                if family == "gaussian":
                    value, corr_gradient = (
                        evaluator.gaussian_objective_and_gradient(correlation))
                    parameter_gradient = np.empty(0, dtype=np.float64)
                else:
                    value, parameter_gradient, corr_gradient = (
                        evaluator.objective_and_joint_gradient(6.0))
                raw_gradient = policy.raw_gradient(
                    raw, correlation, corr_gradient)
                return value, np.concatenate(
                    [parameter_gradient, raw_gradient])
            calls[n_threads] = call

    for call in calls.values():
        call()
    measured = interleaved_timings(calls, repeats=3)
    reference = measured.results[1]
    for n_threads in (1, 2, 4):
        result = measured.results[n_threads]
        if corr_mode == "fixed":
            assert np.isfinite(result)
            np.testing.assert_allclose(result, reference, rtol=1e-12, atol=1e-10)
        else:
            assert np.isfinite(result[0])
            assert np.all(np.isfinite(result[1]))
            np.testing.assert_allclose(
                result[0], reference[0], rtol=1e-12, atol=1e-10)
            np.testing.assert_allclose(
                result[1], reference[1], rtol=2e-11, atol=2e-9)
        _report(
            "static_elliptical_correlation_mode",
            measured.medians[n_threads],
            workload={
                "T": T,
                "dimension": d,
                "family": family,
                "corr_mode": corr_mode,
                "n_threads": n_threads,
                "large_cholesky_override": (
                    corr_mode == "cholesky" and d > 10),
            },
            cache_state="prepared_scores",
        )

    if corr_mode == "fixed":
        parameter = 0.0 if family == "gaussian" else 6.0
        reconstruction_call = lambda: prepare(
            correlation, u, n_threads=1).log_likelihood(parameter)
    elif family == "gaussian":
        reconstruction_call = lambda: prepare(
            correlation, u, n_threads=1).gaussian_objective_and_gradient(
                correlation)
    else:
        reconstruction_call = lambda: prepare(
            correlation, u, n_threads=1).objective_and_joint_gradient(6.0)
    reconstruction_elapsed, reconstruction_result = _median_elapsed(
        reconstruction_call, repeats=3)
    if corr_mode == "fixed":
        assert np.isfinite(reconstruction_result)
    else:
        assert np.isfinite(reconstruction_result[0])
    _report(
        "static_elliptical_correlation_reconstruction",
        reconstruction_elapsed,
        workload={
            "T": T,
            "dimension": d,
            "family": family,
            "corr_mode": corr_mode,
        },
        cache_state="reconstructed_each_call",
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("family", ["gaussian", "student"])
def test_static_factor_large_dimension_benchmark_report(family):
    _enabled()
    T, d, rank = 200, 1_000, 4
    rng = np.random.default_rng(20260802)
    u = rng.uniform(0.02, 0.98, (T, d))
    loadings = rng.normal(scale=0.08, size=(d, rank))
    model_type = GaussianCopula if family == "gaussian" else StudentCopula
    model = model_type(
        d=d,
        corr_mode="factor",
        factor_rank=rank,
        factor_loadings=loadings,
    )
    if family == "student":
        model.df = 6.0

    calls = {
        n_threads: lambda n_threads=n_threads: model.log_likelihood(
            u, n_threads=n_threads)
        for n_threads in (1, 2, 4)
    }
    for call in calls.values():
        call()
    measured = interleaved_timings(calls, repeats=3)
    reference = measured.results[1]
    for n_threads in (1, 2, 4):
        value = measured.results[n_threads]
        assert np.isfinite(value)
        np.testing.assert_allclose(value, reference, rtol=1e-12, atol=1e-9)
        if family == "gaussian":
            assert model.corr is None
        else:
            assert model.correlation is None
        _report(
            "static_factor_large_dimension",
            measured.medians[n_threads],
            workload={
                "T": T,
                "dimension": d,
                "family": family,
                "factor_rank": rank,
                "n_threads": n_threads,
            },
            cache_state="compact_factor_operator",
        )


@pytest.mark.benchmark
def test_jacobi_prepared_evaluator_setup_benchmark_report():
    _enabled()
    u = np.random.default_rng(20260624).uniform(0.01, 0.99, (1_000, 2))
    copula = GumbelCopula(rotate=180)

    def prepare_and_filter(observations):
        evaluator = jacobi_native.PreparedScarJacobiEvaluator(
            observations,
            copula,
            basis_order=4,
            quad_order=64,
            transition_method="local",
            gh_order=3,
        )
        return evaluator.filter(1.2, 0.4, 0.25)

    prepare_and_filter(u[:8])

    elapsed, result = _median_elapsed(
        lambda: prepare_and_filter(u)
    )

    assert result["emissions"].shape == (len(u), 64)
    _report(
        "jacobi_prepared_evaluator_setup",
        elapsed,
        workload={"T": len(u), "K": 64, "family": "gumbel"},
        cache_state="cold_preparation",
    )


@pytest.mark.benchmark
def test_multivariate_student_grid_emission_benchmark_report():
    _enabled()
    T = 300
    d = 5
    K = 40
    rng = np.random.default_rng(20260626)
    u = rng.uniform(0.02, 0.98, (T, d))
    correlation = np.full((d, d), 0.2)
    np.fill_diagonal(correlation, 1.0)
    x_grid = np.linspace(-3.0, 3.0, K)
    copula = StochasticStudentCopula(d=d, R=correlation)
    cache = copula.prepare_emission_cache(u)
    copula.copula_grid_batch(u[:8], x_grid, cache=cache)

    elapsed, result = _median_elapsed(
        lambda: copula.copula_grid_batch(u, x_grid, cache=cache),
        repeats=3,
    )

    assert result.shape == (T, K)
    assert np.all(np.isfinite(result))
    _report(
        "multivariate_student_grid_emission",
        elapsed,
        workload={"T": T, "d": d, "K": K},
        cache_state="warm_ppf_cache",
    )
