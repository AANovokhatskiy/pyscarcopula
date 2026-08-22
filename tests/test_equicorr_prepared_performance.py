"""Opt-in scaling gates for compact equicorrelation preparation."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import numpy as np
import pytest

from benchmark_timing import interleaved_timings
from pyscarcopula import EquicorrGaussianCopula


def _enabled(*, large: bool) -> None:
    if os.environ.get("PYSCA_RUN_BENCHMARKS") != "1":
        pytest.skip("set PYSCA_RUN_BENCHMARKS=1 to run benchmark gates")
    if large and os.environ.get("PYSCA_RUN_LARGE_BENCHMARKS") != "1":
        pytest.skip(
            "set PYSCA_RUN_LARGE_BENCHMARKS=1 to run d=1e6 gate")


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
    calls = {
        count: (
            lambda count=count: model.prepare_sufficient_statistics(
                u,
                batch_rows=n_obs,
                dimension_tile=4096,
                n_threads=count,
            )
        )
        for count in workers
    }
    for call in calls.values():
        call()
    measured = interleaved_timings(
        calls, repeats=3 if large else 5)
    timings = measured.medians
    reference = measured.results[workers[0]]
    for count in workers:
        prepared = measured.results[count]
        if count != workers[0]:
            np.testing.assert_array_equal(
                prepared.sum_z, reference.sum_z)
            np.testing.assert_array_equal(
                prepared.sum_z2, reference.sum_z2)
    diagnostics = dict(measured.results[workers[-1]].diagnostics)

    speedup = {
        count: measured.median_ratio(1, count)
        for count in workers
    }
    efficiency_4 = (
        speedup[4] / 4.0 if 4 in speedup else None)
    target_applicable = n_obs * dimension >= 320_000
    target_met = (
        not target_applicable
        or (efficiency_4 is not None and efficiency_4 >= 0.60))
    payload = {
        "name": "equicorr_preparation_scaling",
        "workload": {"T": n_obs, "d": dimension},
        "seconds": {str(key): value for key, value in timings.items()},
        "speedup": {str(key): value for key, value in speedup.items()},
        "parallel_efficiency_4": efficiency_4,
        "parallel_efficiency_target": 0.60,
        "parallel_efficiency_target_applicable": target_applicable,
        "target_met": target_met,
        "peak_temporary_values": diagnostics["peak_temporary_values"],
        "prepared_bytes": 2 * n_obs * np.dtype(np.float64).itemsize,
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "timer": "perf_counter interleaved median",
    }
    print(
        "PYSCA_BENCHMARK "
        + json.dumps(payload, sort_keys=True),
        flush=True,
    )
    output = os.environ.get("PYSCA_EQUICORR_BENCHMARK_OUTPUT")
    if output:
        path = Path(output)
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            report = {"status": "passed", "workloads": []}
        report["workloads"].append(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if os.environ.get("PYSCA_ENFORCE_PERFORMANCE_GATES") == "1":
        assert target_met, payload


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("n_obs", "dimension"),
    [
        (1, 100_000),
        (32, 10_000),
        (32, 100_000),
        (1000, 10_000),
    ],
)
def test_equicorr_preparation_scaling_gate(n_obs, dimension):
    _run_gate(n_obs=n_obs, dimension=dimension, large=False)


@pytest.mark.benchmark
def test_equicorr_preparation_million_dimension_gate():
    _run_gate(n_obs=1, dimension=1_000_000, large=True)
