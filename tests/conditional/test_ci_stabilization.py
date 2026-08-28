"""Contracts for conditional-sampling CI and artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

from tools.benchmark_conditional_sampling import write_artifacts


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
