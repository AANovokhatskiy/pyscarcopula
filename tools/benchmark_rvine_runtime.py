"""Record a reproducible Python-oracle baseline for the R-vine runtime.

This is a measurement harness, not a wall-clock test.  It records timings,
Python allocation counts, peak traced memory, output checksums, RNG state and
the complete workload matrix.  Cases deliberately bounded by the selected
profile remain visible as ``not_run`` records with a reason.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from typing import Callable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
for directory in (ROOT, TESTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from pyscarcopula import (
    BivariateGaussianCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    IndependentCopula,
    JoeCopula,
)
from pyscarcopula.numerical import _cpp_extension
from pyscarcopula.stattests import student_rosenblatt_transform
from pyscarcopula.vine._edge_adapter import edge_copula
from pyscarcopula.vine._rvine_dag import (
    build_runtime_rvine_dag,
    plan_conditional_sample,
)
from rvine_runtime_cases import (
    configured_mixed_family_vine,
    configured_static_dvine,
    fitted_pair,
    scalar_parameters,
)


DEFAULT_OUTPUT = (
    ROOT / "benchmark_artifacts" / "rvine_runtime_python_baseline.json"
)
BACKEND_ENV = "PYSCARCOPULA_TEST_RVINE_BACKEND"


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


def _checksum(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype=np.float64)
    digest = hashlib.blake2b(memoryview(array).cast("B"), digest_size=16)
    return digest.hexdigest()


def _rng_checksum(rng: np.random.Generator) -> str:
    encoded = json.dumps(
        rng.bit_generator.state, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def _measure(
        *,
        operation: str,
        dimension: int,
        rows: int,
        parameter_mode: str,
        repeats: int,
        warmups: int,
        seed: int,
        call: Callable[[np.random.Generator], np.ndarray],
) -> dict[str, object]:
    for offset in range(warmups):
        call(np.random.default_rng(seed - 10_000 - offset))

    timings = []
    last = None
    last_rng = None
    for offset in range(repeats):
        last_rng = np.random.default_rng(seed + offset)
        started = time.perf_counter()
        last = np.asarray(call(last_rng), dtype=np.float64)
        timings.append(time.perf_counter() - started)

    memory_rng = np.random.default_rng(seed + 100_000)
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    memory_result = np.asarray(call(memory_rng), dtype=np.float64)
    after = tracemalloc.take_snapshot()
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    positive_count = sum(
        max(0, statistic.count_diff)
        for statistic in after.compare_to(before, "lineno")
    )
    positive_bytes = sum(
        max(0, statistic.size_diff)
        for statistic in after.compare_to(before, "lineno")
    )

    if last is None or last_rng is None:
        raise AssertionError("benchmark produced no result")
    if last.shape[0] != rows:
        raise AssertionError(
            f"{operation} returned {last.shape}, expected {rows} rows")
    if np.any(~np.isfinite(last)):
        raise AssertionError(f"{operation} returned non-finite values")
    median = statistics.median(timings)
    return {
        "status": "measured",
        "operation": operation,
        "dimension": dimension,
        "rows": rows,
        "parameter_mode": parameter_mode,
        "seed": seed,
        "warmups": warmups,
        "repeats": repeats,
        "wall_seconds": timings,
        "wall_seconds_median": median,
        "throughput_rows_per_second": rows / median if median else None,
        "output_shape": list(last.shape),
        "output_bytes": int(last.nbytes),
        "output_checksum": _checksum(last),
        "rng_state_checksum": _rng_checksum(last_rng),
        "python_tracemalloc_peak_bytes": int(peak),
        "python_positive_allocation_count": int(positive_count),
        "python_positive_allocation_bytes": int(positive_bytes),
        "memory_probe_checksum": _checksum(memory_result),
        "memory_probe_rng_state_checksum": _rng_checksum(memory_rng),
    }


def _not_run(
        operation: str,
        dimension: int,
        rows: int,
        parameter_mode: str,
        reason: str,
) -> dict[str, object]:
    return {
        "status": "not_run",
        "operation": operation,
        "dimension": dimension,
        "rows": rows,
        "parameter_mode": parameter_mode,
        "reason": reason,
    }


def _path_parameters(vine, rows: int, *, mixed: bool):
    parameters = scalar_parameters(vine)
    phase = np.linspace(0.0, 2.0 * np.pi, rows, endpoint=False)
    path_index = 0
    for key, edge in sorted(vine.pair_copulas.items()):
        if isinstance(edge_copula(edge), IndependentCopula):
            continue
        if mixed and path_index % 2:
            path_index += 1
            continue
        center = float(edge.param)
        parameters[key] = np.ascontiguousarray(
            np.clip(center + 0.04 * np.sin(phase), -0.8, 0.8),
            dtype=np.float64,
        )
        path_index += 1
    return parameters


def _ar1_correlation(dimension: int, rho: float = 0.35) -> np.ndarray:
    index = np.arange(dimension)
    return rho ** np.abs(index[:, None] - index[None, :])


def _vine_records(
        dimensions, row_counts, repeats, warmups, include_mcmc, profile):
    records = []
    seed = 202608140
    for dimension in dimensions:
        vine = configured_static_dvine(dimension)
        independent = configured_static_dvine(dimension, independent=True)
        peel = [
            int(vine.matrix[dimension - 1 - column, column])
            for column in range(dimension)
        ]
        suffix_given = {peel[-1]: 0.42}
        dag_given = {peel[0]: 0.57}
        dag = build_runtime_rvine_dag(vine.matrix, vine._edge_map)
        dag_plan = plan_conditional_sample(dag, dag_given, dimension)
        for rows in row_counts:
            cases = [
                (
                    "unconditional",
                    "scalar",
                    lambda rng, vine=vine, rows=rows: vine.sample(
                        rows,
                        rng=rng,
                        batch_rows=min(rows, 8192),
                    ),
                ),
                (
                    "unconditional_independence_heavy",
                    "none",
                    lambda rng, vine=independent, rows=rows: vine.sample(
                        rows,
                        rng=rng,
                        batch_rows=min(rows, 8192),
                    ),
                ),
                (
                    "suffix_conditional",
                    "scalar",
                    lambda rng, vine=vine, rows=rows, given=suffix_given:
                    vine.predict(rows, given=given, rng=rng),
                ),
            ]
            for operation, parameter_mode, call in cases:
                records.append(_measure(
                    operation=operation,
                    dimension=dimension,
                    rows=rows,
                    parameter_mode=parameter_mode,
                    repeats=repeats,
                    warmups=warmups,
                    seed=seed,
                    call=call,
                ))
                seed += 10

            bounded = dimension * rows <= (
                50_000 if profile == "full" else 500)
            for parameter_mode in ("row_path", "mixed_scalar_path"):
                operation = "unconditional_replayed_parameters"
                if not bounded:
                    records.append(_not_run(
                        operation,
                        dimension,
                        rows,
                        parameter_mode,
                        "bounded profile excludes large Python path buffers",
                    ))
                    continue
                paths = _path_parameters(
                    vine, rows, mixed=parameter_mode.startswith("mixed"))
                records.append(_measure(
                    operation=operation,
                    dimension=dimension,
                    rows=rows,
                    parameter_mode=parameter_mode,
                    repeats=repeats,
                    warmups=warmups,
                    seed=seed,
                    call=lambda rng, vine=vine, rows=rows, paths=paths:
                    vine._sample_with_r(rows, paths, rng),
                ))
                seed += 10

            if bounded:
                dag_parameters = scalar_parameters(vine)
                records.append(_measure(
                    operation="arbitrary_dag_initializer",
                    dimension=dimension,
                    rows=rows,
                    parameter_mode="scalar",
                    repeats=repeats,
                    warmups=warmups,
                    seed=seed,
                    call=lambda rng, vine=vine, rows=rows,
                    parameters=dag_parameters, given=dag_given, plan=dag_plan:
                    vine._sample_dag_given_with_r(
                        rows,
                        parameters,
                        rng,
                        given,
                        plan,
                        vine.pair_copulas,
                    ),
                ))
                seed += 10
            else:
                records.append(_not_run(
                    "arbitrary_dag_initializer",
                    dimension,
                    rows,
                    "scalar",
                    "bounded profile excludes large DAG workspace",
                ))

            mcmc_allowed = include_mcmc and dimension * rows <= 500
            if not mcmc_allowed:
                reason = (
                    "pass --include-mcmc to measure"
                    if not include_mcmc else
                    "bounded MCMC probe caps dimension*rows at 500"
                )
                records.append(_not_run(
                    "forced_coordinate_mcmc",
                    dimension,
                    rows,
                    "scalar",
                    reason,
                ))
            else:
                parameters = scalar_parameters(vine)
                initial = np.full((rows, dimension), 0.5, dtype=np.float64)
                for variable, value in dag_given.items():
                    initial[:, variable] = value
                records.append(_measure(
                    operation="forced_coordinate_mcmc",
                    dimension=dimension,
                    rows=rows,
                    parameter_mode="scalar",
                    repeats=1,
                    warmups=0,
                    seed=seed,
                    call=lambda rng, vine=vine, rows=rows,
                    parameters=parameters, given=dag_given, initial=initial:
                    vine._sample_arbitrary_given_mcmc(
                        rows,
                        parameters,
                        rng,
                        given,
                        initial=initial,
                        n_steps=max(8, 2 * (dimension - 1)),
                        burnin_steps=max(4, dimension - 1),
                    )[0],
                ))
                seed += 10
    return records


def _rotation_records(repeats, warmups):
    vine = configured_mixed_family_vine()
    records = [_measure(
        operation="unconditional_rotated_transposed_mixed_family",
        dimension=vine.d,
        rows=1_000,
        parameter_mode="mixed_scalar_path",
        repeats=repeats,
        warmups=warmups,
        seed=202608999,
        call=lambda rng: vine._sample_with_r(
            1_000,
            _path_parameters(vine, 1_000, mixed=True),
            rng,
        ),
    )]
    cases = [
        *[(ClaytonCopula, rotation, 0.8) for rotation in (0, 90, 180, 270)],
        *[(GumbelCopula, rotation, 1.6) for rotation in (0, 90, 180, 270)],
        *[(JoeCopula, rotation, 1.7) for rotation in (0, 90, 180, 270)],
        (FrankCopula, 0, 2.5),
        (BivariateGaussianCopula, 0, -0.4),
        (IndependentCopula, 0, 0.0),
    ]
    for index, (family, rotation, parameter) in enumerate(cases):
        pair_vine = configured_static_dvine(2)
        copula = (
            family()
            if family is IndependentCopula else family(rotate=rotation)
        )
        pair_vine.pair_copulas[(0, 0)] = fitted_pair(copula, parameter)
        records.append(_measure(
            operation="unconditional_public_pair_family",
            dimension=2,
            rows=1_000,
            parameter_mode=(
                f"{family.__name__}:rotation={rotation}:scalar"
            ),
            repeats=repeats,
            warmups=warmups,
            seed=202609000 + 10 * index,
            call=lambda rng, pair_vine=pair_vine:
            pair_vine.sample(1_000, rng=rng),
        ))
    return records


def _student_records(dimensions, row_counts, repeats, warmups, profile):
    records = []
    seed = 202609500
    for dimension in dimensions:
        correlation = _ar1_correlation(dimension)
        for rows in row_counts:
            observations = np.random.default_rng(seed).uniform(
                0.01, 0.99, size=(rows, dimension))
            records.append(_measure(
                operation="dense_student_rosenblatt",
                dimension=dimension,
                rows=rows,
                parameter_mode="scalar_df",
                repeats=repeats,
                warmups=warmups,
                seed=seed,
                call=lambda _rng, correlation=correlation,
                observations=observations:
                student_rosenblatt_transform(
                    correlation, 7.5, observations),
            ))
            seed += 10

            path_allowed = (
                dimension <= 10
                and rows <= (1_000 if profile == "full" else 32)
            )
            if not path_allowed:
                records.append(_not_run(
                    "dense_student_rosenblatt",
                    dimension,
                    rows,
                    "df_path_row_loop",
                    "bounded profile excludes the quadratic Python row loop",
                ))
                continue
            df_path = np.linspace(5.0, 12.0, rows)

            def df_path_transform(
                    _rng,
                    correlation=correlation,
                    observations=observations,
                    df_path=df_path):
                return student_rosenblatt_transform(
                    correlation, df_path, observations)

            records.append(_measure(
                operation="dense_student_rosenblatt",
                dimension=dimension,
                rows=rows,
                parameter_mode="df_path_row_loop",
                repeats=repeats,
                warmups=warmups,
                seed=seed,
                call=df_path_transform,
            ))
            seed += 10
    return records


def run_benchmark(
        *,
        profile: str,
        backend: str,
        repeats: int,
        warmups: int,
        include_mcmc: bool,
) -> dict[str, object]:
    if profile not in {"smoke", "full"}:
        raise ValueError("profile must be 'smoke' or 'full'")
    if backend not in {"python_executor", "native_strict"}:
        raise ValueError("backend must be python_executor or native_strict")
    if repeats <= 0 or warmups < 0:
        raise ValueError("repeats must be positive and warmups non-negative")
    os.environ[BACKEND_ENV] = backend
    dimensions = (5,) if profile == "smoke" else (5, 10, 50)
    row_counts = (1, 32) if profile == "smoke" else (1, 32, 1_000, 10_000)
    started = time.perf_counter()
    records = _vine_records(
        dimensions,
        row_counts,
        repeats,
        warmups,
        include_mcmc,
        profile,
    )
    records.extend(_rotation_records(repeats, warmups))
    records.extend(_student_records(
        dimensions, row_counts, repeats, warmups, profile))
    module = _cpp_extension.load()
    return {
        "schema_version": 1,
        "artifact_kind": "rvine_runtime_python_baseline",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "backend": backend,
        "configuration": {
            "profile": profile,
            "dimensions": list(dimensions),
            "row_counts": list(row_counts),
            "repeats": repeats,
            "warmups": warmups,
            "include_mcmc": include_mcmc,
        },
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_compiler_runtime": platform.python_compiler(),
            "numpy": np.__version__,
            "scipy": _version("scipy"),
            "pyscarcopula": _version("pyscarcopula"),
            "native_extension": str(getattr(module, "__file__", "unknown")),
            "native_rvine_symbols": sorted(
                name for name in dir(module) if "rvine" in name.lower()),
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
        "measurement_policy": {
            "wall_clock_gating": False,
            "comparison_rule": "compare medians only on the same runner",
            "allocation_scope": "Python allocations visible to tracemalloc",
            "native_peak_memory": "not measured by tracemalloc",
            "not_run_cases_are_explicit": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument(
        "--backend",
        choices=("python_executor", "native_strict"),
        default="python_executor",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--include-mcmc", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_benchmark(
        profile=args.profile,
        backend=args.backend,
        repeats=args.repeats,
        warmups=args.warmups,
        include_mcmc=args.include_mcmc,
    )
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    measured = sum(record["status"] == "measured" for record in report["records"])
    print(f"wrote {measured} measured records to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
