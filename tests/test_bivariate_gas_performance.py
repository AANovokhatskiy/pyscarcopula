"""Opt-in benchmark reporter for the native bivariate GAS filter."""

import os
import time

import numpy as np
import pytest

from pyscarcopula.copula.gumbel import GumbelCopula
from pyscarcopula._native import gas as _cpp_gas
from tools.benchmark_timing import interleaved_timings


@pytest.mark.benchmark
def test_native_bivariate_gas_filter_benchmark_report():
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
    # Measurement-only benchmark: absolute wall-clock gates are not portable.
    print(f"BENCH bivariate_gas_filter elapsed_ms={1e3 * elapsed:.3f}",
          flush=True)


@pytest.mark.benchmark
def test_native_bivariate_gas_sampling_benchmark_report():
    if os.environ.get("PYSCA_RUN_BENCHMARKS") != "1":
        pytest.skip("set PYSCA_RUN_BENCHMARKS=1 to run benchmark checks")

    draws = np.random.default_rng(20260830).uniform(
        0.01, 0.99, size=(10_000, 2))
    copula = GumbelCopula(rotate=180)
    params = (0.04, 0.12, 0.82)
    adapter = copula._native_adapter()

    def stepwise():
        state = _cpp_gas.initial_state(*params, copula, "unit")
        g_t = state.g
        r_t = state.parameter
        output = np.empty_like(draws)
        for index, draw in enumerate(draws):
            row = adapter.sample_from_uniforms(
                copula,
                draw.reshape(1, 2),
                np.array([r_t]),
            )[0]
            output[index] = row
            if index + 1 < len(draws):
                update = _cpp_gas.update_one(
                    *params, g_t, row, copula, "unit")
                g_t = update.g_next
                r_t = update.r_next
        return output

    _cpp_gas.sample_bivariate(*params, draws[:16], copula, "unit")
    expected = stepwise()
    fused = _cpp_gas.sample_bivariate(
        *params, draws, copula, "unit")
    np.testing.assert_allclose(fused, expected, rtol=0.0, atol=0.0)

    measured = interleaved_timings({
        "fused": lambda: _cpp_gas.sample_bivariate(
            *params, draws, copula, "unit"),
        "stepwise": stepwise,
    }, repeats=4)
    fused_median = measured.medians["fused"]
    stepwise_median = measured.medians["stepwise"]
    assert fused_median < stepwise_median
    print(
        "BENCH bivariate_gas_sampling "
        f"n={len(draws)} fused_ms={1e3 * fused_median:.3f} "
        f"stepwise_ms={1e3 * stepwise_median:.3f} "
        f"speedup={stepwise_median / fused_median:.2f}",
        flush=True,
    )
