"""Measure time and Python-heap peaks for the four migrated TM endpoints."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import threading
import time
import tracemalloc

import numpy as np

from pyscarcopula import ClaytonCopula, EquicorrGaussianCopula
from pyscarcopula.copula.multivariate import StochasticStudentCopula
from pyscarcopula.numerical import _cpp_scar_ou
from pyscarcopula.numerical._scar_ou_config import AutoTMConfig


def _output_bytes(value):
    if isinstance(value, tuple):
        return sum(_output_bytes(item) for item in value)
    return int(np.asarray(value).nbytes)


def _current_rss_bytes():
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
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
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
        return int(counters.WorkingSetSize) if ok else 0

    statm = "/proc/self/statm"
    if os.path.exists(statm):
        with open(statm, encoding="ascii") as stream:
            resident_pages = int(stream.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    return 0


def _monitored_call(call):
    baseline = _current_rss_bytes()
    peak = [baseline]
    stopped = threading.Event()

    def monitor():
        while not stopped.wait(0.001):
            peak[0] = max(peak[0], _current_rss_bytes())

    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    try:
        value = call()
    finally:
        peak[0] = max(peak[0], _current_rss_bytes())
        stopped.set()
        worker.join()
    return value, baseline, peak[0]


def _measure(call, repeats):
    call()
    elapsed = []
    peaks = []
    rss_peaks = []
    rss_deltas = []
    output_bytes = 0
    for _ in range(repeats):
        tracemalloc.start()
        started = time.perf_counter()
        value, rss_before, rss_peak = _monitored_call(call)
        elapsed.append(time.perf_counter() - started)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak)
        rss_peaks.append(rss_peak)
        rss_deltas.append(max(rss_peak - rss_before, 0))
        output_bytes = _output_bytes(value)
    return {
        "best_seconds": min(elapsed),
        "median_seconds": float(np.median(elapsed)),
        "peak_python_heap_bytes": max(peaks),
        "peak_process_rss_bytes": max(rss_peaks),
        "peak_process_rss_delta_bytes": max(rss_deltas),
        "output_bytes": output_bytes,
    }


def run_benchmark(T=256, K=257, dimension=4, repeats=3):
    rng = np.random.default_rng(20260817)
    pair_u = rng.uniform(0.02, 0.98, size=(T, 2))
    multi_u = rng.uniform(0.02, 0.98, size=(T, dimension))
    correlation = np.full((dimension, dimension), 0.15, dtype=np.float64)
    np.fill_diagonal(correlation, 1.0)
    student = StochasticStudentCopula(d=dimension, R=correlation)
    config = AutoTMConfig(
        K=K,
        grid_range=4.0,
        adaptive=False,
        transition_method="local",
        max_K=None,
        gh_order=7,
    )
    params = (0.05, 0.0, 1.0)
    calls = {
        "bivariate_rosenblatt": lambda: _cpp_scar_ou.forward_rosenblatt(
            *params, pair_u, ClaytonCopula(), config),
        "gaussian_gof": lambda: _cpp_scar_ou.gaussian_rosenblatt(
            *params, multi_u, EquicorrGaussianCopula(d=dimension), config),
        "student_gof": lambda: _cpp_scar_ou.student_rosenblatt(
            *params, multi_u, student, config),
        "student_smoothing": (
            lambda: _cpp_scar_ou.smoothed_state_distribution(
                *params, multi_u, student, config)
        ),
    }
    return {
        "configuration": {
            "T": T,
            "K": K,
            "dimension": dimension,
            "repeats": repeats,
            "transition_method": "local",
            "gh_order": 7,
        },
        "results": {
            name: _measure(call, repeats)
            for name, call in calls.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=256)
    parser.add_argument("--K", type=int, default=257)
    parser.add_argument("--dimension", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.T < 2 or args.K < 2 or args.dimension < 2 or args.repeats < 1:
        parser.error("T, K, dimension and repeats must satisfy 2, 2, 2, 1")
    print(json.dumps(
        run_benchmark(args.T, args.K, args.dimension, args.repeats),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
