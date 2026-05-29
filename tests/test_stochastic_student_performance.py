"""Optional performance checks for StochasticStudent fast paths."""

import os
import time

import numpy as np
import pytest

from pyscarcopula._utils import pobs
from pyscarcopula.copula.experimental.stochastic_student import (
    StochasticStudentCopula,
)
from pyscarcopula.numerical.gas_filter import (
    _gas_filter_multivariate,
    gas_filter,
)


def _skip_unless_enabled():
    if os.environ.get("PYSCA_RUN_BENCHMARKS") != "1":
        pytest.skip("set PYSCA_RUN_BENCHMARKS=1 to run benchmark checks")


def _example_student(d=10, T=600):
    rng = np.random.default_rng(42)
    raw = rng.standard_t(df=5.0, size=(T, d))
    u = pobs(raw)
    R = np.full((d, d), 0.35)
    np.fill_diagonal(R, 1.0)
    return StochasticStudentCopula(d=d, R=R), u


@pytest.mark.benchmark
def test_stochastic_student_gas_numba_fast_path_speed_smoke():
    _skip_unless_enabled()
    copula, u = _example_student()
    params = (0.08, 0.04, 0.92)

    # Compile the Numba path and build the full-sample cache before measuring
    # steady-state throughput inside optimizer loops.
    gas_filter(*params, u[:8], copula, scaling="unit")
    gas_filter(*params, u, copula, scaling="unit")

    t0 = time.perf_counter()
    g_fast, r_fast, ll_fast = gas_filter(
        *params, u, copula, scaling="unit")
    fast_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    g_ref, r_ref, ll_ref = _gas_filter_multivariate(
        *params, u, copula, scaling="unit")
    generic_elapsed = time.perf_counter() - t0

    np.testing.assert_allclose(g_fast, g_ref, atol=3e-3, rtol=2e-3)
    np.testing.assert_allclose(r_fast, r_ref, atol=3e-3, rtol=2e-3)
    np.testing.assert_allclose(ll_fast, ll_ref, atol=1e-2, rtol=2e-3)
    assert fast_elapsed < generic_elapsed * 0.5
