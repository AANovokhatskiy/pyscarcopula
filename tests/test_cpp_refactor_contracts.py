"""Regression contracts for the C++ architecture refactor."""

from __future__ import annotations

import copy
from dataclasses import fields
import json
from pathlib import Path

import numpy as np
import pytest

from pyscarcopula import EquicorrGaussianCopula, StochasticStudentCopula
from pyscarcopula._types import NumericalConfig
from pyscarcopula.numerical import multivariate_native, static_likelihood
from tools import (
    benchmark_cpp_refactor,
    capture_cpp_refactor_goldens,
    write_cpp_refactor_inventory,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "benchmarks" / "cpp_refactor_manifest.json"
INVENTORY_PATH = ROOT / "benchmarks" / "cpp_refactor_inventory.json"
BASELINE_PATH = (
    ROOT / "benchmark_artifacts" / "cpp_refactor_baseline.json"
)
CANDIDATE_PATH = (
    ROOT / "benchmark_artifacts" / "cpp_refactor_candidate.json"
)
GOLDEN_PATH = (
    ROOT / "tests" / "fixtures" / "cpp_refactor_goldens_v1.json"
)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_benchmark_uses_native_compiler_identity_and_compute_manifest(
    monkeypatch,
):
    class NativeExtension:
        __cpp_compiler__ = "GCC 16.1.0"

    assert benchmark_cpp_refactor._extension_compiler_identity(
        NativeExtension()) == "GCC 16.1.0"

    class LegacyExtension:
        pass

    monkeypatch.setattr(
        benchmark_cpp_refactor.platform,
        "python_compiler",
        lambda: "legacy-python-compiler",
    )
    assert benchmark_cpp_refactor._extension_compiler_identity(
        LegacyExtension()) == "legacy-python-compiler"

    paths = {
        path.relative_to(ROOT).as_posix()
        for path in benchmark_cpp_refactor._compute_source_paths()
    }
    assert "setup.py" not in paths
    assert not any("/_cpp/src/bindings/" in f"/{path}" for path in paths)
    assert any(path.endswith("/_cpp/src/copula/core.cpp") for path in paths)
    assert any(path.endswith("/_cpp/include/scar/copula.hpp") for path in paths)


def test_benchmark_manifest_has_protocol_and_unique_complete_cases():
    manifest = _json(MANIFEST_PATH)
    protocol = manifest["protocol"]
    assert manifest["schema_version"] == 1
    assert protocol["paired_samples"] >= 4
    assert protocol["minimum_sample_seconds"] >= 0.02
    assert protocol["order"] == "A/B/B/A repeating"
    policy = protocol["regression_policy"]
    assert policy["maximum_runtime_ratio"] == 2.0
    assert policy["maximum_python_allocation_count_ratio"] == 2.0
    assert policy["maximum_python_allocated_bytes_ratio"] == 2.0
    assert policy["maximum_python_peak_memory_ratio"] == 2.0
    assert policy["maximum_process_peak_rss_ratio"] == 2.0
    assert policy["maximum_parallel_scaling_loss_ratio"] == 2.0
    assert policy["minimum_python_peak_memory_increase_bytes"] >= 2**20
    assert policy["require_checksum_match"] is True
    assert policy["require_domain_diagnostics_match"] is True
    assert {
        "seconds",
        "checksum",
        "python_allocation_count",
        "python_allocated_bytes",
        "python_peak_bytes",
        "process_peak_rss_bytes",
        "domain_diagnostics",
    } <= set(protocol["required_metrics"])

    cases = manifest["cases"]
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))
    required = {
        "id", "runner", "model", "family", "operation", "shape",
        "dimension", "parameter_regime", "seed", "n_threads", "mode",
        "release_critical",
    }
    assert all(required <= set(case) for case in cases)
    assert all(case["runner"] in benchmark_cpp_refactor.RUNNERS for case in cases)
    assert all(case["release_critical"] is True for case in cases)
    latency_cases = [
        case for case in cases
        if case.get("workload_class") == "small_call_latency"
    ]
    assert latency_cases
    assert all(case["shape"]["n_obs"] <= 16 for case in latency_cases)


def test_benchmark_manifest_covers_models_rotations_transforms_and_threads():
    cases = _json(MANIFEST_PATH)["cases"]
    pair_cells = {
        (case["family"], case["rotation"])
        for case in cases
        if case["runner"] == "pair_grid"
    }
    expected_cells = {
        ("independent", 0),
        ("frank", 0),
        ("gaussian", 0),
        *((family, rotation)
          for family in ("clayton", "gumbel", "joe")
          for rotation in (0, 90, 180, 270)),
    }
    assert pair_cells == expected_cells

    transform_cells = {
        (case["family"], case["transform"])
        for case in cases
        if case["runner"] == "transform"
    }
    assert transform_cells == {
        (family, transform)
        for family in ("clayton", "frank", "gumbel", "joe")
        for transform in ("softplus", "xtanh", "exp", "logistic")
    }

    for runner in (
        "equicorr_grid", "student_dense_grid", "student_factor_grid"):
        threads = {
            case["n_threads"] for case in cases if case["runner"] == runner
        }
        assert threads == {1, 2, 4, "physical"}

    assert {
        case["runner"] for case in cases if case["model"] == "rvine_static"
    } == {"vine_density", "vine_rosenblatt", "vine_sampling", "vine_mcmc"}
    assert {
        (case["backend"], case["mode"])
        for case in cases if case["model"] == "scar_ou_pair"
    } == {
        (backend, mode)
        for backend in ("matrix", "local", "spectral")
        for mode in ("cold_preparation", "prepared_repeated")
    }
    assert {case["model"] for case in cases} >= {
        "pair", "dense_gaussian", "factor_gaussian", "equicorr_gaussian",
        "dense_student", "factor_student", "gas_pair", "gas_dense_student",
        "scar_ou_pair", "rvine_static",
    }


def test_benchmark_baseline_supports_coarse_regression_detection():
    manifest = _json(MANIFEST_PATH)
    baseline = _json(BASELINE_PATH)
    assert baseline["artifact_type"] == "cpp-refactor-benchmark-capture"
    assert baseline["manifest_id"] == manifest["manifest_id"]
    assert baseline["smoke_override"] is False
    assert baseline["summary"]["total"] == len(manifest["cases"])
    assert {case["id"] for case in baseline["cases"]} == {
        case["id"] for case in manifest["cases"]
    }
    assert baseline["valid_for_regression_check"] is True
    assert baseline["eligibility_failures"] == []
    environment = baseline["environment"]
    assert environment["reference_label"]
    assert environment["compute_worktree_dirty"] is False
    assert environment["build_freshness"][
        "extension_not_older_than_compute_sources"] is True
    assert all(
        case["process_affinity"]["applied"] for case in baseline["cases"])

    gate_candidate = _json(CANDIDATE_PATH)
    assert gate_candidate["artifact_type"] == baseline["artifact_type"]
    assert gate_candidate["manifest_id"] == baseline["manifest_id"]
    assert gate_candidate["smoke_override"] is False
    assert gate_candidate["valid_for_regression_check"] is True
    assert gate_candidate["eligibility_failures"] == []
    assert gate_candidate["summary"]["total"] == len(manifest["cases"])
    assert gate_candidate["environment"]["compiler_identity"] == (
        environment["compiler_identity"]
    )
    # Repository captures are immutable historical Stage 0 artifacts.  They
    # were made at two different source revisions, so freeze each provenance
    # explicitly rather than comparing either one with the current tree.
    assert environment["compute_source_sha256"] == (
        "c492bb11b5f231e91b52c860fd54e934288a9bd46a2fc7a4d2c30722a235d9ea"
    )
    assert gate_candidate["environment"]["compute_source_sha256"] == (
        "037a1d762fae997e274b6afe1c9d2ac8373dcd4a4c68fab0c50cb8ff3aea4360"
    )
    gate_comparison = gate_candidate["comparison"]
    assert gate_comparison["passed"] is True
    assert gate_comparison["failures"] == []
    assert gate_comparison["environment_mismatches"] == []
    assert all(case["checksum_match"] for case in gate_comparison["cases"])
    assert all(
        case["domain_diagnostics_match"]
        for case in gate_comparison["cases"]
    )

    comparison = benchmark_cpp_refactor.compare_benchmark_artifacts(
        baseline, copy.deepcopy(baseline))
    assert comparison["passed"] is True

    candidate = copy.deepcopy(baseline)
    candidate["cases"][0]["session_median_seconds"] *= 2.01
    comparison = benchmark_cpp_refactor.compare_benchmark_artifacts(
        baseline, candidate)
    assert comparison["passed"] is False
    assert "runtime ratio" in comparison["failures"][0]

    candidate = copy.deepcopy(baseline)
    candidate["environment"]["cpu_model"] = "different CPU"
    comparison = benchmark_cpp_refactor.compare_benchmark_artifacts(
        baseline, candidate)
    assert comparison["passed"] is False
    assert comparison["environment_mismatches"][0]["field"] == "cpu_model"

    candidate = copy.deepcopy(baseline)
    candidate["cases"][0]["memory"]["python_allocation_count"] += 1000
    comparison = benchmark_cpp_refactor.compare_benchmark_artifacts(
        baseline, candidate)
    assert comparison["passed"] is False
    assert "python_allocation_count" in comparison["failures"][0]

    candidate = copy.deepcopy(baseline)
    rss = candidate["cases"][0]["memory"]["process_peak_rss_bytes"]
    candidate["cases"][0]["memory"]["process_peak_rss_bytes"] = (
        rss * 3 + 2**27)
    comparison = benchmark_cpp_refactor.compare_benchmark_artifacts(
        baseline, candidate)
    assert comparison["passed"] is False
    assert "process_peak_rss_bytes" in comparison["failures"][0]

    candidate = copy.deepcopy(baseline)
    candidate["cases"][0]["domain_diagnostics"]["audit_mutation"] = True
    comparison = benchmark_cpp_refactor.compare_benchmark_artifacts(
        baseline, candidate)
    assert comparison["passed"] is False
    assert "domain diagnostics changed" in comparison["failures"][0]

    candidate = copy.deepcopy(baseline)
    by_id = {case["id"]: case for case in candidate["cases"]}
    by_id["grid.equicorr.t1"]["session_median_seconds"] *= 0.5
    by_id["grid.equicorr.t4"]["session_median_seconds"] *= 1.5
    comparison = benchmark_cpp_refactor.compare_benchmark_artifacts(
        baseline, candidate)
    assert comparison["passed"] is False
    assert any(
        "parallel scaling loss ratio" in failure
        for failure in comparison["failures"])


@pytest.mark.sanitizer_numerical
def test_pair_kernel_goldens_are_unchanged():
    capture_cpp_refactor_goldens.check_fixture(GOLDEN_PATH)


def test_pair_kernel_golden_checker_has_no_superseded_outputs():
    assert not hasattr(
        capture_cpp_refactor_goldens, "SUPERSEDED_GOLDEN_OUTPUTS")


def test_pair_kernel_golden_tolerances_cannot_be_widened(tmp_path):
    fixture = _json(GOLDEN_PATH)
    fixture["cases"][0]["cross_platform_atol"] = 1.0
    modified = tmp_path / "modified-goldens.json"
    modified.write_text(json.dumps(fixture), encoding="utf-8")
    with pytest.raises(AssertionError, match="cross_platform_atol changed"):
        capture_cpp_refactor_goldens.check_fixture(modified)


def test_stage0_inventory_is_frozen_and_current_config_mapping_is_complete():
    inventory = _json(INVENTORY_PATH)
    assert inventory["inventory_id"] == "cpp-architecture-refactor-v1"
    assert inventory["source_commit"] == (
        "88560231c5208d6282128cfb3b68af33c2310319")
    assert inventory["compute_source_sha256"] == (
        "037a1d762fae997e274b6afe1c9d2ac8373dcd4a4c68fab0c50cb8ff3aea4360")

    current = write_cpp_refactor_inventory.build_payload()
    mappings = current["configuration_contracts"]["numerical_config_mappings"]
    assert {entry["old_field"] for entry in mappings} == {
        field.name for field in fields(NumericalConfig)
    }
    assert all(entry["target_owner"] for entry in mappings)
    assert all(entry["target_field_or_constant"] for entry in mappings)
    constants = current["configuration_contracts"]["named_constant_mappings"]
    assert len({entry["old"] for entry in constants}) == len(constants)
    assert all(entry["target_owner"] and entry["semantic"] for entry in constants)
    contracts = current["configuration_contracts"]
    complete = contracts["complete_constant_mappings"]
    assert len(complete) == (
        len(contracts["discovered_python_constants"])
        + len(contracts["discovered_cpp_constants"])
        + sum(
            entry["kind"] == "historical-python" for entry in complete)
    )
    assert all(
        entry["target_owner"] and entry["target"] and entry["semantic"]
        for entry in complete)

    changed_constant = copy.deepcopy(current)
    frozen_name = (
        "pyscarcopula.numerical.multivariate_native."
        "_DENSE_STUDENT_NATIVE_MIN_DF"
    )
    for entry in changed_constant["configuration_contracts"][
            "complete_constant_mappings"]:
        if entry["old"] == frozen_name:
            entry["value"] = 0.2
            break
    else:  # pragma: no cover - inventory construction contract
        raise AssertionError(f"missing frozen constant {frozen_name}")
    assert write_cpp_refactor_inventory._gate3_drift(
        current, changed_constant
    ) == [
        "Python constant "
        f"{frozen_name} changed from 0.1 to 0.2"
    ]

    changed = copy.deepcopy(current)
    changed["dependencies"]["include_graph"] = {}
    assert (
        write_cpp_refactor_inventory._contract_view(current)
        != write_cpp_refactor_inventory._contract_view(changed)
    )


def test_inventories_cover_build_dependencies_api_and_breaking_changes():
    inventory = _json(INVENTORY_PATH)
    build = inventory["build_matrix"]
    current_ids = {
        item["id"] for item in build["current_no_regression_matrix"]
    }
    assert current_ids == {
        "linux-gcc-py310",
        "linux-gcc-py314",
        "linux-clang-py312",
        "windows-msvc-py312",
        "windows-mingw64-py312",
        "macos-arm64-clang-py312",
    }
    assert build["required_addition"]["id"] == "windows-mingw64-py312"
    assert build["required_addition"]["status"].startswith("implemented")
    assert build["cxx_standard"] == "C++17"

    dependencies = inventory["dependencies"]
    layers = {
        item["name"]
        for item in dependencies["allowed_target_graph"]["layers"]
    }
    assert layers == {
        "foundation", "copula_models", "static", "gas", "scar_ou",
        "vine", "gas_rvine_composition", "python_bindings",
    }
    assert dependencies["include_graph"]

    compatibility = inventory["extension_compatibility"]
    assert {
        "CopulaSpec", "GasConfig", "OuNumericalConfig",
        "StaticCopulaEvaluator", "rvine_sample", "rvine_log_pdf_rows",
    } <= set(compatibility["symbols"])

    breaking = inventory["breaking_changes"]
    for key in (
        "public_imports_and_protocols",
        "backend_selectors",
        "fallback_paths",
        "docs_to_update",
        "tests_to_replace_with_removal_contracts",
        "migration_note_entries",
    ):
        assert breaking[key]
    assert {
        "pyscarcopula.vine.vine.VineCopula._sample_suffix_given_with_r_python",
        "pyscarcopula.vine.vine.VineCopula._sample_dag_given_with_r_python",
    } <= set(breaking["fallback_paths"])
    assert any(
        entry.startswith("tests/test_cpp_rvine_conditional.py")
        for entry in breaking["tests_to_replace_with_removal_contracts"])
    assert all((ROOT / path).is_file() for path in breaking["docs_to_update"])


@pytest.mark.sanitizer_numerical
def test_reduction_order_sensitive_native_kernels_preserve_current_thread_contracts():
    rng = np.random.default_rng(2026082390)

    dimension = 12
    correlation = np.full((dimension, dimension), 0.15, dtype=np.float64)
    np.fill_diagonal(correlation, 1.0)
    observations = rng.uniform(0.01, 0.99, size=(768, dimension))
    static_values = [
        static_likelihood.prepare_gaussian(
            correlation, observations, n_threads=n_threads
        ).log_likelihood(0.0)
        for n_threads in (1, 2, 4, 8)
    ]
    # Current static reductions change their final rounding with the worker
    # partition.  Exact per-thread values are frozen in the golden fixture;
    # only this pre-existing tiny cross-thread envelope is asserted here.
    np.testing.assert_allclose(
        static_values,
        np.full(4, static_values[0]),
        rtol=0.0,
        atol=5e-12,
    )

    grid = np.linspace(-1.8, 1.8, 17)
    equicorr = EquicorrGaussianCopula(dimension)
    equicorr_results = [
        multivariate_native.pdf_and_grad_grid(
            equicorr, observations, grid, n_threads=n_threads)
        for n_threads in (1, 2, 4, 8)
    ]
    for actual in equicorr_results[1:]:
        np.testing.assert_array_equal(actual[0], equicorr_results[0][0])
        np.testing.assert_array_equal(actual[1], equicorr_results[0][1])

    student = StochasticStudentCopula(d=dimension, R=correlation)
    cache = student.prepare_emission_cache(observations[:96])
    student_results = [
        multivariate_native.pdf_and_grad_grid(
            student,
            observations[:96],
            grid,
            cache=cache,
            n_threads=n_threads,
        )
        for n_threads in (1, 2, 4, 8)
    ]
    for actual in student_results[1:]:
        np.testing.assert_array_equal(actual[0], student_results[0][0])
        np.testing.assert_array_equal(actual[1], student_results[0][1])

    given_indices = np.array([0, 2, 5], dtype=np.int32)
    given_latent = rng.normal(size=(512, len(given_indices)))
    normal_draws = rng.normal(size=(512, dimension - len(given_indices)))
    conditional_results = [
        multivariate_native.gaussian_conditional_latent(
            correlation,
            given_indices,
            given_latent,
            normal_draws,
            n_threads=n_threads,
        )
        for n_threads in (1, 2, 4, 8)
    ]
    for actual in conditional_results[1:]:
        np.testing.assert_array_equal(actual, conditional_results[0])
