"""Contracts for conditional-sampling CI and artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tomllib

from tools.benchmark_conditional_sampling import write_artifacts
from tools.calibrate_conditional_statistical_gates import (
    run_calibration,
    write_report,
)


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "conditional-sampling.yml"
GUIDE = ROOT / "docs" / "guide" / "conditional-sampling-validation.md"


def test_conditional_ci_markers_are_registered_for_strict_collection():
    configuration = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    markers = {
        entry.split(":", 1)[0]
        for entry in configuration["tool"]["pytest"]["ini_options"]["markers"]
    }
    assert {"validation", "benchmark", "external", "high_dimensional"} <= markers


def test_conditional_workflow_exposes_all_four_layers_and_exact_selections():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    required_jobs = (
        "pr-smoke:",
        "validation:",
        "nightly-external-high-dimensional:",
        "manual-benchmark:",
    )
    assert all(job in workflow for job in required_jobs)
    assert 'not validation and not benchmark and not external' in workflow
    assert 'validation and not benchmark and not external' in workflow
    assert '(external or high_dimensional) and not benchmark' in workflow
    assert "--runs 20 --max-failure-rate 0.01" in workflow
    assert workflow.count('--status "${{ job.status }}"') == 4
    assert "pyvinecopulib==0.7.5" not in workflow
    assert 'python -m pip install -e ".[test,external]"' in workflow
    assert "PYSCA_ENFORCE_PERFORMANCE_GATES" not in workflow


def test_conditional_validation_guide_documents_commands_and_triage():
    guide = GUIDE.read_text(encoding="utf-8")
    assert "## Failure triage" in guide
    assert "## Statistical stability policy" in guide
    assert "--strict-markers --run-validation" in guide
    assert "pyvinecopulib==0.7.5" in guide
    assert "Wall time is not gated" in guide


def test_oracle_calibration_report_has_per_gate_failure_rates(tmp_path):
    report = run_calibration(
        runs=2,
        base_seed=20268900,
        uniform_draws=64,
        gaussian_draws=64,
        student_draws=64,
        dimensions=(1,),
        student_dfs=(8.0,),
        max_failure_rate=1.0,
    )
    output = tmp_path / "calibration.json"
    write_report(report, output)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["oracle_only"] is True
    assert persisted["measurement_policy"]["production_sampler_used"] is False
    assert persisted["passed"] is True
    assert {gate["gate"] for gate in persisted["gates"]} == {
        "uniform-pit",
        "gaussian-mean-d=1",
        "gaussian-covariance-d=1",
        "student-mean-d=1-df=8",
        "student-covariance-d=1-df=8",
    }
    assert all(gate["runs"] == 2 for gate in persisted["gates"])


def test_benchmark_writer_preserves_reproducibility_fields(tmp_path):
    record = {
        "case": "gaussian-dense-ar1",
        "path": "exact",
        "seed": 20268101,
        "dimension": 50,
        "k_free": 3,
        "n_draws": 16,
        "n_threads": 2,
        "wall_seconds": [0.1, 0.09],
        "wall_seconds_median": 0.095,
        "metadata": {"correlation": "ar1"},
    }
    report = {
        "schema_version": 1,
        "git_commit": "0123456789abcdef",
        "environment": {
            "python": "3.12",
            "python_compiler": "test compiler",
            "processor": "test cpu",
        },
        "records": [record],
    }
    json_output = tmp_path / "benchmark.json"
    csv_output = tmp_path / "benchmark.csv"
    write_artifacts(report, json_output, csv_output)

    persisted = json.loads(json_output.read_text(encoding="utf-8"))
    assert persisted["git_commit"] == "0123456789abcdef"
    assert persisted["records"][0]["seed"] == 20268101
    with csv_output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["case"] == "gaussian-dense-ar1"
    assert rows[0]["n_threads"] == "2"
    assert json.loads(rows[0]["metadata"]) == {"correlation": "ar1"}
