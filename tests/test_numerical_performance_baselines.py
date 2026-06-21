"""Optional reproducible performance baselines for numerical workloads."""

import json
import os
import platform
import statistics
import sys
import time

import numpy as np
import pytest
import scipy

from pyscarcopula import GaussianCopula, GumbelCopula
from pyscarcopula.copula.multivariate.stochastic_student import (
    StochasticStudentCopula,
)
from pyscarcopula.numerical import _cpp_scar_ou
from pyscarcopula.numerical.jacobi_tm import _emission_grid
from pyscarcopula.numerical.mc_samplers import p_sampler_loglik
from pyscarcopula.numerical.ou_kernels import calculate_dwt
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
        "native_extension": _cpp_scar_ou.available(),
        "timer": "perf_counter median",
    }
    print("WP0_BENCH " + json.dumps(payload, sort_keys=True), flush=True)


@pytest.mark.benchmark
def test_bivariate_mle_objective_baseline():
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
def test_gaussian_mle_objective_baseline():
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
def test_jacobi_emission_construction_baseline():
    _enabled()
    u = np.random.default_rng(20260624).uniform(0.01, 0.99, (1_000, 2))
    tau = np.linspace(0.01, 0.95, 64)
    copula = GumbelCopula(rotate=180)
    _emission_grid(u[:8], copula, tau)

    elapsed, result = _median_elapsed(
        lambda: _emission_grid(u, copula, tau)
    )

    assert result[0].shape == (len(u), len(tau))
    _report(
        "jacobi_emission_construction",
        elapsed,
        workload={"T": len(u), "K": len(tau), "family": "gumbel"},
        cache_state="warm",
    )


@pytest.mark.benchmark
def test_mc_copula_density_accumulation_baseline():
    _enabled()
    T = 200
    n_tr = 2_000
    u = np.random.default_rng(20260625).uniform(0.01, 0.99, (T, 2))
    dwt = calculate_dwt(T, n_tr, seed=20260625)
    copula = GumbelCopula(rotate=180)
    call = lambda: p_sampler_loglik(1.1, 0.3, 0.8, u, dwt, copula, True)
    call()

    elapsed, value = _median_elapsed(call, repeats=3)

    assert np.isfinite(value)
    _report(
        "mc_copula_density_accumulation",
        elapsed,
        workload={"T": T, "n_tr": n_tr, "family": "gumbel"},
        cache_state="warm_fixed_dwt",
    )


@pytest.mark.benchmark
def test_multivariate_student_grid_emission_baseline():
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
