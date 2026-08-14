"""Write reproducible conditional-sampling benchmark artifacts.

Wall time is measurement-only: this runner never enforces a throughput
threshold.  Correctness remains covered by pytest; the benchmark records only
cheap invariants so timing is not polluted by analytical-oracle work.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from typing import Callable, Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.conditional._high_dimensional import (
    DIMENSION,
    FREE_COUNTS,
    configured_gaussian,
    configured_student,
    high_dimensional_gaussian_vine,
    scattered_given,
    suffix_given,
)


DEFAULT_JSON = (
    ROOT / "benchmark_artifacts" /
    "conditional_sampling_stage8_benchmark.json"
)
DEFAULT_CSV = (
    ROOT / "benchmark_artifacts" /
    "conditional_sampling_stage8_benchmark.csv"
)


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    value = completed.stdout.strip()
    return value or None


def _version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def _process_memory_bytes() -> tuple[int | None, int | None]:
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = (
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            )
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            process = kernel32.GetCurrentProcess()
            ok = psapi.GetProcessMemoryInfo(
                process, ctypes.byref(counters), counters.cb
            )
            if not ok:
                return None, None
            return (
                int(counters.WorkingSetSize),
                int(counters.PeakWorkingSetSize),
            )
        except (AttributeError, OSError):
            return None, None
    try:
        import resource

        usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        peak = usage if sys.platform == "darwin" else usage * 1024
        return None, peak
    except (ImportError, OSError):
        return None, None


def _assert_output(samples: np.ndarray, given: dict[int, float]) -> None:
    if samples.ndim != 2 or samples.shape[1] != DIMENSION:
        raise AssertionError(
            f"expected a two-dimensional d={DIMENSION} sample, "
            f"received {samples.shape}"
        )
    if np.any(~np.isfinite(samples)):
        raise AssertionError("sample contains non-finite values")
    if np.any((samples <= 0.0) | (samples >= 1.0)):
        raise AssertionError("sample leaves the open unit interval")
    for index, value in given.items():
        np.testing.assert_array_equal(
            samples[:, index],
            np.full(len(samples), value, dtype=np.float64),
        )


def _measure(
        *,
        case: str,
        path: str,
        k_free: int,
        n_draws: int,
        repeats: int,
        warmups: int,
        seed: int,
        n_threads: int,
        given: dict[int, float],
        draw: Callable[[int], tuple[np.ndarray, dict[str, object]]],
        metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    if repeats <= 0 or warmups < 0:
        raise ValueError("repeats must be positive and warmups non-negative")
    for offset in range(warmups):
        samples, _diagnostics = draw(seed - 10_000 - offset)
        _assert_output(samples, given)

    wall_seconds = []
    last_samples: np.ndarray | None = None
    last_diagnostics: dict[str, object] = {}
    for offset in range(repeats):
        started = time.perf_counter()
        last_samples, last_diagnostics = draw(seed + offset)
        wall_seconds.append(time.perf_counter() - started)
        _assert_output(last_samples, given)

    tracemalloc.start()
    rss_current_before, rss_peak_before = _process_memory_bytes()
    memory_samples, memory_diagnostics = draw(seed + repeats + 1_000)
    _current, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_current_after, rss_peak_after = _process_memory_bytes()
    _assert_output(memory_samples, given)
    if last_samples is None:
        raise AssertionError("benchmark produced no sample")
    median = statistics.median(wall_seconds)
    diagnostics = memory_diagnostics or last_diagnostics
    warning_codes = [
        item.get("code")
        for item in diagnostics.get("warnings", [])
        if isinstance(item, dict)
    ]
    return {
        "case": case,
        "path": path,
        "dimension": DIMENSION,
        "k_free": k_free,
        "k_given": DIMENSION - k_free,
        "n_draws": n_draws,
        "n_threads": n_threads,
        "seed": seed,
        "warmups": warmups,
        "repeats": repeats,
        "wall_seconds": wall_seconds,
        "wall_seconds_median": median,
        "throughput_rows_per_second": n_draws / median,
        "output_bytes": int(last_samples.nbytes),
        "python_tracemalloc_peak_bytes": int(python_peak),
        "rss_current_before_bytes": rss_current_before,
        "rss_current_after_bytes": rss_current_after,
        "rss_peak_before_bytes": rss_peak_before,
        "rss_peak_after_bytes": rss_peak_after,
        "fixed_columns_bit_exact": True,
        "finite_open_unit_output": True,
        "conditional_method": diagnostics.get("conditional_method"),
        "warning_codes": warning_codes,
        "metadata": metadata or {},
    }


def _static_records(
        *,
        free_counts: Iterable[int],
        n_draws: int,
        repeats: int,
        warmups: int,
        n_threads: int,
        base_seed: int,
) -> list[dict[str, object]]:
    records = []
    cases = (
        (
            "gaussian-dense-ar1",
            "exact",
            lambda: configured_gaussian("ar1"),
            lambda model, n, given, rng: model.sample_conditional(
                n, given, rng=rng, n_threads=n_threads
            ),
            {},
        ),
        (
            "gaussian-factor-rank=3",
            "exact-factor",
            lambda: configured_gaussian("factor", 3),
            lambda model, n, given, rng: model.sample_conditional(
                n, given, rng=rng, n_threads=n_threads
            ),
            {"factor_rank": 3},
        ),
        (
            "student-dense-df=8",
            "exact",
            lambda: configured_student(df=8.0),
            lambda model, n, given, rng: model.sample_conditional(
                n, given, rng=rng, n_threads=n_threads
            ),
            {"df": 8.0},
        ),
        (
            "student-factor-rank=3-df=10",
            "exact-factor",
            lambda: configured_student(df=10.0, rank=3),
            lambda model, n, given, rng: model.sample_conditional(
                n, given, rng=rng, n_threads=n_threads
            ),
            {"df": 10.0, "factor_rank": 3},
        ),
    )
    for case_index, (case, path, factory, sampler, metadata) in enumerate(cases):
        model = factory()
        for k_free in free_counts:
            given = scattered_given(k_free)
            seed = base_seed + 100 * case_index + k_free

            def draw(
                    draw_seed: int,
                    model=model,
                    given=given,
                    sampler=sampler,
            ) -> tuple[np.ndarray, dict[str, object]]:
                samples = sampler(
                    model,
                    n_draws,
                    given,
                    np.random.default_rng(draw_seed),
                )
                return samples, {}

            records.append(_measure(
                case=case,
                path=path,
                k_free=k_free,
                n_draws=n_draws,
                repeats=repeats,
                warmups=warmups,
                seed=seed,
                n_threads=n_threads,
                given=given,
                draw=draw,
                metadata=metadata,
            ))
    return records


def _vine_exact_records(
        *,
        kinds: Iterable[str],
        free_counts: Iterable[int],
        n_draws: int,
        repeats: int,
        warmups: int,
        base_seed: int,
) -> list[dict[str, object]]:
    records = []
    for kind_index, kind in enumerate(kinds):
        vine = high_dimensional_gaussian_vine(kind)
        for k_free in free_counts:
            given = suffix_given(vine, k_free)
            seed = base_seed + 1_000 + 100 * kind_index + k_free

            def draw(
                    draw_seed: int,
                    vine=vine,
                    given=given,
            ) -> tuple[np.ndarray, dict[str, object]]:
                return vine.predict(
                    n_draws,
                    given=given,
                    return_diagnostics=True,
                    rng=np.random.default_rng(draw_seed),
                )

            records.append(_measure(
                case=f"gaussian-{kind}",
                path="exact-suffix",
                k_free=k_free,
                n_draws=n_draws,
                repeats=repeats,
                warmups=warmups,
                seed=seed,
                n_threads=1,
                given=given,
                draw=draw,
                metadata={"structure": kind},
            ))
    return records


def _mcmc_records(
        *,
        free_counts: Iterable[int],
        n_draws: int,
        base_seed: int,
) -> list[dict[str, object]]:
    records = []
    vine = high_dimensional_gaussian_vine("d-vine")
    had_override = "_suffix_sampling_state" in vine.__dict__
    original_override = vine.__dict__.get("_suffix_sampling_state")
    try:
        vine._suffix_sampling_state = lambda _given: None
        for k_free in free_counts:
            given = suffix_given(vine, k_free)
            seed = base_seed + 2_000 + k_free

            def draw(
                    draw_seed: int,
                    vine=vine,
                    given=given,
                    k_free=k_free,
            ) -> tuple[np.ndarray, dict[str, object]]:
                return vine.predict(
                    n_draws,
                    given=given,
                    mcmc_steps=max(20, 4 * k_free),
                    mcmc_burnin=max(10, 2 * k_free),
                    return_diagnostics=True,
                    rng=np.random.default_rng(draw_seed),
                )

            records.append(_measure(
                case="gaussian-d-vine-forced-mcmc",
                path="dag-mcmc",
                k_free=k_free,
                n_draws=n_draws,
                repeats=1,
                warmups=0,
                seed=seed,
                n_threads=1,
                given=given,
                draw=draw,
                metadata={
                    "structure": "d-vine",
                    "correctness_gating": False,
                    "mcmc_steps": max(20, 4 * k_free),
                    "mcmc_burnin": max(10, 2 * k_free),
                },
            ))
    finally:
        if had_override:
            vine._suffix_sampling_state = original_override
        else:
            del vine.__dict__["_suffix_sampling_state"]
    return records


def run_benchmark(
        *,
        profile: str = "full",
        n_draws: int = 1_024,
        mcmc_draws: int = 8,
        repeats: int = 3,
        warmups: int = 1,
        n_threads: int = 4,
        base_seed: int = 20268100,
        include_mcmc: bool = False,
) -> dict[str, object]:
    if profile not in {"smoke", "full"}:
        raise ValueError("profile must be 'smoke' or 'full'")
    if min(n_draws, mcmc_draws, repeats, n_threads) <= 0 or warmups < 0:
        raise ValueError(
            "draws, repeats, and threads must be positive; warmups non-negative"
        )
    free_counts = (3,) if profile == "smoke" else FREE_COUNTS
    vine_kinds = ("d-vine",) if profile == "smoke" else (
        "c-vine", "d-vine", "r-vine"
    )
    started = time.perf_counter()
    records = _static_records(
        free_counts=free_counts,
        n_draws=n_draws,
        repeats=repeats,
        warmups=warmups,
        n_threads=n_threads,
        base_seed=base_seed,
    )
    records.extend(_vine_exact_records(
        kinds=vine_kinds,
        free_counts=free_counts,
        n_draws=n_draws,
        repeats=repeats,
        warmups=warmups,
        base_seed=base_seed,
    ))
    if include_mcmc:
        records.extend(_mcmc_records(
            free_counts=free_counts,
            n_draws=mcmc_draws,
            base_seed=base_seed,
        ))
    return {
        "schema_version": 1,
        "stage": 8,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "python_compiler": platform.python_compiler(),
            "numpy": np.__version__,
            "scipy": _version("scipy"),
            "pyscarcopula": _version("pyscarcopula"),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "configuration": {
            "profile": profile,
            "dimension": DIMENSION,
            "free_counts": list(free_counts),
            "n_draws": n_draws,
            "mcmc_draws": mcmc_draws,
            "repeats": repeats,
            "warmups": warmups,
            "n_threads": n_threads,
            "base_seed": base_seed,
            "include_mcmc": include_mcmc,
        },
        "measurement_policy": {
            "wall_clock_gating": False,
            "mcmc_correctness_gating": False,
            "comparison_rule": "compare medians only on the same runner",
            "memory_note": (
                "tracemalloc covers Python allocations; process RSS includes "
                "native allocations and may be a high-water measurement"
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
    }


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def write_artifacts(
        report: dict[str, object],
        json_output: Path,
        csv_output: Path) -> None:
    if not json_output.is_absolute():
        json_output = ROOT / json_output
    if not csv_output.is_absolute():
        csv_output = ROOT / csv_output
    json_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    records = list(report["records"])
    fieldnames = sorted({key for record in records for key in record})
    with csv_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({
                key: _csv_value(record.get(key)) for key in fieldnames
            })


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--n-draws", type=int, default=1_024)
    parser.add_argument("--mcmc-draws", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=20268100)
    parser.add_argument("--include-mcmc", action="store_true")
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_benchmark(
        profile=args.profile,
        n_draws=args.n_draws,
        mcmc_draws=args.mcmc_draws,
        repeats=args.repeats,
        warmups=args.warmups,
        n_threads=args.n_threads,
        base_seed=args.base_seed,
        include_mcmc=args.include_mcmc,
    )
    write_artifacts(report, args.json_output, args.csv_output)
    print(
        f"wrote {len(report['records'])} benchmark records to "
        f"{args.json_output} and {args.csv_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
