"""Contracts for the permanent FV6 Gate 1 benchmark driver."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from tools import run_gate1_benchmarks as gate1


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "benchmarks" / "gate1_manifest_v2.json"


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
    assert manifest["schema_version"] == 2
    assert manifest["protocol"]["paired_samples"] >= 4
    assert manifest["protocol"]["minimum_sample_seconds"] >= 0.02
    assert all(required <= set(case) for case in manifest["cases"])
    assert all(case["runner"] in gate1.RUNNERS for case in manifest["cases"])
    assert all(case["release_critical"] is True for case in manifest["cases"])


def test_physical_cpu_pool_uses_distinct_core_representatives(monkeypatch):
    monkeypatch.setattr(
        gate1, "_windows_physical_cpu_ids", lambda: [0, 2, 4, 6, 8])
    assert gate1._parse_cpu_set("physical:4", 18) == [0, 2, 4, 6]
    assert gate1._parse_cpu_set("physical", 18) == [0, 2, 4, 6, 8]


def test_calibration_requires_two_complete_batches(monkeypatch):
    elapsed_ns = 0
    calls = 0

    def clock():
        return elapsed_ns

    def call():
        nonlocal elapsed_ns, calls
        calls += 1
        elapsed_ns += 1_000_000

    monkeypatch.setattr(gate1.time, "perf_counter_ns", clock)
    repetitions = gate1._calibrate(call, 0.05)
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

    monkeypatch.setattr(gate1.time, "perf_counter_ns", clock)
    repetitions = gate1._calibrate(call, 0.05)
    assert repetitions >= 63
    assert calls > 2


def test_comparison_blocks_runtime_memory_checksum_diagnostics_and_scaling():
    baseline = _capture()
    assert gate1.compare_benchmark_artifacts(
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
        comparison = gate1.compare_benchmark_artifacts(baseline, candidate)
        assert comparison["passed"] is False
        assert any(expected in failure for failure in comparison["failures"])


def test_comparison_rejects_ineligible_or_incompatible_captures():
    baseline = _capture()
    candidate = copy.deepcopy(baseline)
    candidate["valid_for_regression_check"] = False
    result = gate1.compare_benchmark_artifacts(baseline, candidate)
    assert "candidate is not an eligible" in " ".join(result["failures"])

    candidate = copy.deepcopy(baseline)
    candidate["manifest_sha256"] = "different"
    result = gate1.compare_benchmark_artifacts(baseline, candidate)
    assert "different manifest content" in " ".join(result["failures"])
