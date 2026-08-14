"""Calibrate conditional-sampling Monte Carlo gates on oracle-only draws.

The calibration deliberately does not construct a pyscarcopula model or call
any production sampler.  It generates Uniform, Gaussian, and Student draws
directly from their defining random-variable representations, then applies the
same statistical assertions used by ``tests/conditional``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Callable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.conditional._statistical_assertions import (
    assert_covariance_with_whitening,
    assert_mean_with_mc_error,
    assert_uniform_pit,
)


DEFAULT_OUTPUT = (
    ROOT / "benchmark_artifacts" /
    "conditional_sampling_stage8_calibration.json"
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


def _correlation(dimension: int) -> np.ndarray:
    indices = np.arange(dimension)
    matrix = 0.45 ** np.abs(indices[:, None] - indices[None, :])
    np.linalg.cholesky(matrix)
    return matrix


def _gaussian_draws(
        rng: np.random.Generator,
        n_draws: int,
        covariance: np.ndarray) -> np.ndarray:
    root = np.linalg.cholesky(covariance)
    return rng.standard_normal((n_draws, len(covariance))) @ root.T


def _student_draws_with_unit_covariance(
        rng: np.random.Generator,
        n_draws: int,
        dimension: int,
        df: float) -> np.ndarray:
    # Z / sqrt(V / df) has a multivariate t law with identity shape and
    # covariance df / (df - 2) * I.  The prefactor makes covariance exactly I.
    normal = rng.standard_normal((n_draws, dimension))
    chi_square = rng.chisquare(df, size=(n_draws, 1))
    return np.sqrt((df - 2.0) / df) * normal / np.sqrt(chi_square / df)


def _record_assertion(
        failures: dict[str, list[dict[str, object]]],
        gate: str,
        run: int,
        seed: int,
        assertion: Callable[[], None]) -> None:
    try:
        assertion()
    except AssertionError as error:
        failures[gate].append({
            "run": run,
            "seed": seed,
            "message": str(error),
        })


def run_calibration(
        *,
        runs: int = 20,
        base_seed: int = 20268000,
        uniform_draws: int = 4_000,
        gaussian_draws: int = 5_000,
        student_draws: int = 7_000,
        dimensions: Sequence[int] = (1, 3, 5, 10),
        student_dfs: Sequence[float] = (43.0, 48.0, 70.0),
        max_failure_rate: float = 0.01,
) -> dict[str, object]:
    """Return an empirical false-failure report for every statistical gate."""

    if runs <= 0:
        raise ValueError("runs must be positive")
    if min(uniform_draws, gaussian_draws, student_draws) <= 1:
        raise ValueError("draw counts must be greater than one")
    if not 0.0 <= max_failure_rate <= 1.0:
        raise ValueError("max_failure_rate must lie in [0, 1]")
    dimensions = tuple(int(value) for value in dimensions)
    student_dfs = tuple(float(value) for value in student_dfs)
    if not dimensions or min(dimensions) <= 0:
        raise ValueError("dimensions must contain positive integers")
    if not student_dfs or min(student_dfs) <= 2.0:
        raise ValueError("student_dfs must exceed two")

    gate_names = ["uniform-pit"]
    for dimension in dimensions:
        gate_names.extend((
            f"gaussian-mean-d={dimension}",
            f"gaussian-covariance-d={dimension}",
        ))
        for df in student_dfs:
            gate_names.extend((
                f"student-mean-d={dimension}-df={df:g}",
                f"student-covariance-d={dimension}-df={df:g}",
            ))
    failures: dict[str, list[dict[str, object]]] = {
        name: [] for name in gate_names
    }

    started = time.perf_counter()
    for run in range(runs):
        seed = base_seed + run
        rng = np.random.default_rng(seed)
        uniform = rng.uniform(size=uniform_draws)
        _record_assertion(
            failures,
            "uniform-pit",
            run,
            seed,
            lambda values=uniform: assert_uniform_pit(values),
        )

        for dimension in dimensions:
            covariance = _correlation(dimension)
            gaussian = _gaussian_draws(rng, gaussian_draws, covariance)
            mean_gate = f"gaussian-mean-d={dimension}"
            covariance_gate = f"gaussian-covariance-d={dimension}"
            _record_assertion(
                failures,
                mean_gate,
                run,
                seed,
                lambda values=gaussian, matrix=covariance:
                    assert_mean_with_mc_error(
                        values,
                        np.zeros(len(matrix)),
                        matrix,
                    ),
            )
            _record_assertion(
                failures,
                covariance_gate,
                run,
                seed,
                lambda values=gaussian, matrix=covariance:
                    assert_covariance_with_whitening(
                        values,
                        np.zeros(len(matrix)),
                        matrix,
                    ),
            )

            identity = np.eye(dimension)
            for df in student_dfs:
                student = _student_draws_with_unit_covariance(
                    rng, student_draws, dimension, df
                )
                mean_gate = f"student-mean-d={dimension}-df={df:g}"
                covariance_gate = (
                    f"student-covariance-d={dimension}-df={df:g}"
                )
                _record_assertion(
                    failures,
                    mean_gate,
                    run,
                    seed,
                    lambda values=student, matrix=identity:
                        assert_mean_with_mc_error(
                            values,
                            np.zeros(len(matrix)),
                            matrix,
                            sigma=8.0,
                            numerical_floor=0.006,
                        ),
                )
                _record_assertion(
                    failures,
                    covariance_gate,
                    run,
                    seed,
                    lambda values=student, matrix=identity:
                        assert_covariance_with_whitening(
                            values,
                            np.zeros(len(matrix)),
                            matrix,
                            sigma=10.0,
                            numerical_floor=0.035,
                        ),
                )

    gates = []
    for name in gate_names:
        gate_failures = failures[name]
        failure_rate = len(gate_failures) / runs
        gates.append({
            "gate": name,
            "runs": runs,
            "failure_count": len(gate_failures),
            "failure_rate": failure_rate,
            "passed": failure_rate <= max_failure_rate,
            "failures": gate_failures,
        })
    return {
        "schema_version": 1,
        "stage": 8,
        "oracle_only": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "configuration": {
            "runs": runs,
            "base_seed": base_seed,
            "uniform_draws": uniform_draws,
            "gaussian_draws": gaussian_draws,
            "student_draws": student_draws,
            "dimensions": list(dimensions),
            "student_dfs": list(student_dfs),
            "max_failure_rate": max_failure_rate,
        },
        "measurement_policy": {
            "purpose": "oracle-vs-oracle false-failure calibration",
            "production_sampler_used": False,
            "tolerance_selection_from_production_errors": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "passed": all(bool(gate["passed"]) for gate in gates),
        "gates": gates,
    }


def write_report(report: dict[str, object], output: Path) -> None:
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=20268000)
    parser.add_argument("--uniform-draws", type=int, default=4_000)
    parser.add_argument("--gaussian-draws", type=int, default=5_000)
    parser.add_argument("--student-draws", type=int, default=7_000)
    parser.add_argument("--max-failure-rate", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_calibration(
        runs=args.runs,
        base_seed=args.base_seed,
        uniform_draws=args.uniform_draws,
        gaussian_draws=args.gaussian_draws,
        student_draws=args.student_draws,
        max_failure_rate=args.max_failure_rate,
    )
    write_report(report, args.output)
    failed = [
        gate["gate"] for gate in report["gates"]
        if not gate["passed"]
    ]
    print(
        f"calibrated {len(report['gates'])} gates over {args.runs} runs; "
        f"failed={len(failed)}; output={args.output}"
    )
    if failed:
        print("failed gates: " + ", ".join(str(value) for value in failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
