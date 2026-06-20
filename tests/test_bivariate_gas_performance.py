"""Optional performance smoke test for the native bivariate GAS filter."""

import os
import time

import numpy as np
import pytest

from pyscarcopula.copula.gumbel import GumbelCopula
from pyscarcopula.numerical import _cpp_gas


@pytest.mark.benchmark
def test_native_bivariate_gas_filter_speed_smoke():
    if os.environ.get("PYSCA_RUN_BENCHMARKS") != "1":
        pytest.skip("set PYSCA_RUN_BENCHMARKS=1 to run benchmark checks")

    u = np.random.default_rng(20260612).uniform(
        0.01, 0.99, size=(50_000, 2))
    copula = GumbelCopula(rotate=180)
    params = (0.04, 0.12, 0.82)

    _cpp_gas.filter(*params, u[:8], copula, "unit")
    start = time.perf_counter()
    g_path, r_path, log_likelihood = _cpp_gas.filter(
        *params, u, copula, "unit")
    elapsed = time.perf_counter() - start

    assert g_path.shape == (len(u),)
    assert r_path.shape == (len(u),)
    assert np.isfinite(log_likelihood)
    assert elapsed < 5.0
