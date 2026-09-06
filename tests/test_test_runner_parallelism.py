"""Contracts for opt-in parallel pytest execution."""

from pathlib import Path
import ast

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

import pytest

from tools import benchmark_timing
from tools.benchmark_timing import interleaved_timings


ROOT = Path(__file__).resolve().parents[1]


def _project_configuration():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def test_parallel_test_runner_is_available_but_not_enabled_by_default():
    config = _project_configuration()
    test_dependencies = config["project"]["optional-dependencies"]["test"]
    addopts = config["tool"]["pytest"]["ini_options"].get("addopts", "")

    assert any(
        dependency.startswith("pytest-xdist")
        for dependency in test_dependencies
    )
    assert "-n" not in addopts
    assert "--numprocesses" not in addopts


def test_parallel_test_runner_keeps_each_module_sequential():
    config = _project_configuration()
    addopts = config["tool"]["pytest"]["ini_options"]["addopts"]

    assert "--dist=loadscope" in addopts


def test_relative_timings_are_paired_and_order_balanced(monkeypatch):
    clock = iter((0.0, 2.0, 2.0, 3.0, 3.0, 4.0,
                  4.0, 6.0, 6.0, 10.0, 10.0, 12.0))
    order = []
    monkeypatch.setattr(
        benchmark_timing.time, "perf_counter", lambda: next(clock))

    measured = interleaved_timings(
        {
            "baseline": lambda: order.append("baseline"),
            "candidate": lambda: order.append("candidate"),
        },
        repeats=3,
    )

    assert order == [
        "baseline", "candidate",
        "candidate", "baseline",
        "baseline", "candidate",
    ]
    assert measured.medians == {
        "baseline": pytest.approx(2.0),
        "candidate": pytest.approx(1.0),
    }
    assert measured.median_ratio("baseline", "candidate") == 2.0


def test_benchmarks_have_no_direct_absolute_wall_clock_assertions():
    absolute_metric_names = {
        "duration",
        "elapsed",
        "elapsed_seconds",
        "runtime",
        "seconds",
        "wall_seconds",
    }
    violations = []

    for path in sorted((ROOT / "tests").rglob("*.py")):
        source = path.read_text(encoding="utf-8-sig")
        if "pytest.mark.benchmark" not in source:
            continue
        tree = ast.parse(source, filename=str(path))
        for assertion in (
                node for node in ast.walk(tree)
                if isinstance(node, ast.Assert)):
            compared_names = {
                node.id.lower()
                for node in ast.walk(assertion.test)
                if isinstance(node, ast.Name)
            }
            direct_absolute_names = compared_names & absolute_metric_names
            if direct_absolute_names:
                violations.append(
                    f"{path.relative_to(ROOT)}:{assertion.lineno}: "
                    f"{sorted(direct_absolute_names)}"
                )

    assert not violations, (
        "benchmark gates must compare relative speedup/ratio/efficiency, "
        "not absolute wall-clock values:\n" + "\n".join(violations)
    )
