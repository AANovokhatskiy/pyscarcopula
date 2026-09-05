"""Contracts for the native performance benchmark driver."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from tools import run_native_benchmarks as native_benchmark


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "benchmarks" / "native_performance_v3.json"


def _manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _record(n_threads: int, seconds: float) -> dict:
    return {
        "id": f"scale.t{n_threads}",
        "case": {
            "id": f"scale.t{n_threads}",
            "runner": "equicorr_grid",
            "model": "equicorr_gaussian",
            "family": "gaussian",
            "operation": "pdf_grid",
            "shape": {"n_obs": 128},
            "dimension": 8,
            "parameter_regime": "interior",
            "seed": 7,
            "n_threads": n_threads,
            "mode": "steady",
            "release_critical": True,
        },
        "resolved_n_threads": n_threads,
        "process_affinity": {
            "requested_cpus": list(range(n_threads)),
            "applied": True,
        },
        "session_median_seconds": seconds,
        "checksum": "stable",
        "domain_diagnostics": {"status": "ok"},
        "memory": {
            "python_allocation_count": 100,
            "python_allocated_bytes": 2**20,
            "python_peak_bytes": 2**24,
            "process_peak_rss_bytes": 2**26,
        },
    }


def _capture() -> dict:
    manifest = _manifest()
    return {
        "schema_version": native_benchmark.CAPTURE_SCHEMA_VERSION,
        "artifact_type": native_benchmark.CAPTURE_TYPE,
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": "synthetic-manifest",
        "protocol": manifest["protocol"],
        "valid_for_regression_check": True,
        "environment": {},
        "cases": [_record(1, 1.0), _record(4, 0.3)],
    }


def test_manifest_cases_are_complete_and_have_implemented_runners():
    manifest = _manifest()
    required = {
        "id", "runner", "model", "family", "operation", "shape",
        "dimension", "parameter_regime", "seed", "n_threads", "mode",
        "release_critical",
    }
    assert manifest["schema_version"] == 3
    assert manifest["protocol"]["paired_samples"] >= 4
    assert manifest["protocol"]["minimum_sample_seconds"] >= 0.02
    assert all(required <= set(case) for case in manifest["cases"])
    assert all(case["runner"] in native_benchmark.RUNNERS for case in manifest["cases"])
    assert all(case["release_critical"] is True for case in manifest["cases"])


def test_physical_cpu_pool_uses_distinct_core_representatives(monkeypatch):
    monkeypatch.setattr(
        native_benchmark, "_windows_physical_cpu_ids", lambda: [0, 2, 4, 6, 8])
    assert native_benchmark._parse_cpu_set("physical:4", 18) == [0, 2, 4, 6]
    assert native_benchmark._parse_cpu_set("physical", 18) == [0, 2, 4, 6, 8]


def test_calibration_requires_two_complete_batches(monkeypatch):
    elapsed_ns = 0
    calls = 0

    def clock():
        return elapsed_ns

    def call():
        nonlocal elapsed_ns, calls
        calls += 1
        elapsed_ns += 1_000_000

    monkeypatch.setattr(native_benchmark.time, "perf_counter_ns", clock)
    repetitions = native_benchmark._calibrate(call, 0.05)
    assert repetitions >= 63
    assert calls >= 2 * repetitions


def test_calibration_rejects_one_slow_probe(monkeypatch):
    elapsed_ns = 0
    calls = 0

    def clock():
        return elapsed_ns

    def call():
        nonlocal elapsed_ns, calls
        calls += 1
        elapsed_ns += 100_000_000 if calls == 1 else 1_000_000

    monkeypatch.setattr(native_benchmark.time, "perf_counter_ns", clock)
    repetitions = native_benchmark._calibrate(call, 0.05)
    assert repetitions >= 63
    assert calls > 2


def test_comparison_blocks_runtime_memory_checksum_diagnostics_and_scaling():
    baseline = _capture()
    assert native_benchmark.compare_benchmark_artifacts(
        baseline, copy.deepcopy(baseline))["passed"] is True

    mutations = []

    candidate = copy.deepcopy(baseline)
    candidate["cases"][0]["session_median_seconds"] *= 2.01
    mutations.append((candidate, "runtime ratio"))

    candidate = copy.deepcopy(baseline)
    candidate["cases"][0]["checksum"] = "changed"
    mutations.append((candidate, "checksum changed"))

    candidate = copy.deepcopy(baseline)
    candidate["cases"][0]["domain_diagnostics"]["status"] = "changed"
    mutations.append((candidate, "domain diagnostics changed"))

    for metric in (
        "python_allocation_count",
        "python_allocated_bytes",
        "python_peak_bytes",
        "process_peak_rss_bytes",
    ):
        candidate = copy.deepcopy(baseline)
        candidate["cases"][0]["memory"][metric] *= 3
        candidate["cases"][0]["memory"][metric] += 1
        mutations.append((candidate, metric))

    candidate = copy.deepcopy(baseline)
    candidate["cases"][0]["session_median_seconds"] *= 0.5
    candidate["cases"][1]["session_median_seconds"] *= 1.5
    mutations.append((candidate, "parallel scaling loss ratio"))

    for candidate, expected in mutations:
        comparison = native_benchmark.compare_benchmark_artifacts(baseline, candidate)
        assert comparison["passed"] is False
        assert any(expected in failure for failure in comparison["failures"])


def test_comparison_rejects_ineligible_or_incompatible_captures():
    baseline = _capture()
    candidate = copy.deepcopy(baseline)
    candidate["valid_for_regression_check"] = False
    result = native_benchmark.compare_benchmark_artifacts(baseline, candidate)
    assert "candidate is not an eligible" in " ".join(result["failures"])

    candidate = copy.deepcopy(baseline)
    candidate["manifest_sha256"] = "different"
    result = native_benchmark.compare_benchmark_artifacts(baseline, candidate)
    assert "different manifest content" in " ".join(result["failures"])


def test_comparison_rejects_obsolete_capture_format_for_either_side():
    for side in ("baseline", "candidate"):
        for field, value in (("schema_version", 0), ("artifact_type", "unknown")):
            artifacts = {"baseline": _capture(), "candidate": _capture()}
            artifacts[side][field] = value
            result = native_benchmark.compare_benchmark_artifacts(**artifacts)
            assert not result["passed"]
            assert any(f"{side} has an unsupported capture format" in failure
                       for failure in result["failures"])


def test_unknown_capture_schema_is_rejected_before_reading_its_fields():
    result = native_benchmark.compare_benchmark_artifacts({}, {})
    assert not result["passed"]
    assert len(result["failures"]) == 2
    assert result["cases"] == []


def test_unknown_manifest_schema_is_rejected_before_preparing_workloads(tmp_path):
    import pytest

    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"schema_version": 0}', encoding="utf-8")
    with pytest.raises(SystemExit, match="unsupported workload manifest format"):
        native_benchmark.main([
            "--manifest", str(manifest),
            "--artifact-root", str(tmp_path / "capture"),
        ])
