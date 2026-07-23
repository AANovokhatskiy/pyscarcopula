"""Opt-in Phase-8 scaling gates for compact equicorrelation preparation."""

from __future__ import annotations

import json
import os
import platform
import time

import numpy as np
import pytest

from pyscarcopula import EquicorrGaussianCopula


def _enabled(*, large: bool) -> None:
    if os.environ.get("PYSCA_RUN_BENCHMARKS") != "1":
        pytest.skip("set PYSCA_RUN_BENCHMARKS=1 to run benchmark gates")
    if large and os.environ.get("PYSCA_RUN_LARGE_BENCHMARKS") != "1":
        pytest.skip(
            "set PYSCA_RUN_LARGE_BENCHMARKS=1 to run d=1e6 gate")


def _median(callable_, repeats=5):
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = callable_()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples)), result


def _run_gate(*, n_obs: int, dimension: int, large: bool) -> None:
    _enabled(large=large)
    rng = np.random.default_rng(801)
    u = rng.uniform(0.01, 0.99, size=(n_obs, dimension))
    model = EquicorrGaussianCopula(d=dimension)
    workers = [
        count for count in (1, 2, 4, 8)
        if count <= (os.cpu_count() or 1)
    ]

    # Warm native code and page in the input before measuring.
    model.prepare_sufficient_statistics(
        u[:1], dimension_tile=4096, n_threads=1)
    timings = {}
    reference = None
    diagnostics = None
    for count in workers:
        elapsed, prepared = _median(
            lambda count=count: model.prepare_sufficient_statistics(
                u,
                batch_rows=n_obs,
                dimension_tile=4096,
                n_threads=count,
            ),
            repeats=3 if large else 5,
        )
        timings[count] = elapsed
        if reference is None:
            reference = prepared
        else:
            np.testing.assert_array_equal(
                prepared.sum_z, reference.sum_z)
            np.testing.assert_array_equal(
                prepared.sum_z2, reference.sum_z2)
        diagnostics = dict(prepared.diagnostics)

    speedup = {
        count: timings[1] / elapsed
        for count, elapsed in timings.items()
    }
    efficiency_4 = (
        speedup[4] / 4.0 if 4 in speedup else None)
    target_met = efficiency_4 is None or efficiency_4 >= 0.60
    payload = {
        "name": "phase8_equicorr_preparation_scaling",
        "workload": {"T": n_obs, "d": dimension},
        "seconds": {str(key): value for key, value in timings.items()},
        "speedup": {str(key): value for key, value in speedup.items()},
        "parallel_efficiency_4": efficiency_4,
        "parallel_efficiency_target": 0.60,
        "target_met": target_met,
        "peak_temporary_values": diagnostics["peak_temporary_values"],
        "prepared_bytes": 2 * n_obs * np.dtype(np.float64).itemsize,
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "timer": "perf_counter median",
    }
    print(
        "PHASE8_EQUICORR_PREP_BENCH "
        + json.dumps(payload, sort_keys=True),
        flush=True,
    )

    if os.environ.get("PYSCA_ENFORCE_PERFORMANCE_GATES") == "1":
        assert target_met, payload


@pytest.mark.benchmark
@pytest.mark.parametrize("dimension", [10_000, 100_000])
def test_equicorr_preparation_scaling_gate(dimension):
    _run_gate(n_obs=32, dimension=dimension, large=False)


@pytest.mark.benchmark
def test_equicorr_preparation_million_dimension_gate():
    _run_gate(n_obs=1, dimension=1_000_000, large=True)
