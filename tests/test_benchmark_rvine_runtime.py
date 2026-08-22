"""Structural smoke checks for the optional R-vine benchmark harness."""

from __future__ import annotations

from tools import benchmark_rvine_runtime


def test_full_incremental_mcmc_matrix_covers_stage_7_1_axes():
    workloads = benchmark_rvine_runtime._extended_mcmc_workloads("full")

    assert {workload[0] for workload in workloads} == {10, 20, 30, 50}
    assert {workload[1] for workload in workloads} == {1, 32, 100, 1_000}
    assert {workload[3] for workload in workloads} == {
        "dense", "independence_heavy", "mixed_family",
    }
    assert {workload[4] for workload in workloads} == {
        "scalar_parameters", "row_path",
    }
    assert {workload[5] for workload in workloads} == {
        "low", "medium", "high",
    }
    assert {
        (workload[3], workload[4]) for workload in workloads
    } == {
        ("dense", "scalar_parameters"),
        ("dense", "row_path"),
        ("independence_heavy", "scalar_parameters"),
        ("independence_heavy", "row_path"),
        ("mixed_family", "scalar_parameters"),
        ("mixed_family", "row_path"),
    }


def test_extended_disabled_workloads_remain_visible():
    records = benchmark_rvine_runtime._extended_workload_records(
        profile="smoke",
        backend="python_executor",
        repeats=1,
        warmups=0,
        enabled=False,
    )

    assert records
    assert all(record["status"] == "not_run" for record in records)
    assert {record["candidate"] for record in records} == {
        "factor_student_rosenblatt",
        "equicorr_rosenblatt",
        "dynamic_rvine_rosenblatt",
        "incremental_mcmc_density",
    }


def test_extended_python_smoke_records_cover_candidates(monkeypatch):
    monkeypatch.setenv(
        benchmark_rvine_runtime.BACKEND_ENV, "python_executor")
    records = benchmark_rvine_runtime._extended_workload_records(
        profile="smoke",
        backend="python_executor",
        repeats=1,
        warmups=0,
        enabled=True,
    )

    assert all(record["status"] == "measured" for record in records)
    assert all(
        record["workload_group"] == "extended_workloads"
        for record in records
    )
    assert all(record["output_checksum"] for record in records)

    dynamic_records = [
        record for record in records
        if record["candidate"] == "dynamic_rvine_rosenblatt"
    ]
    assert {record["dynamic_strategy"] for record in dynamic_records} == {
        "GAS", "SCAR",
    }
    assert {
        record["dynamic_edge_coverage"] for record in dynamic_records
    } == {"single", "all"}

    mcmc_records = [
        record for record in records
        if record["candidate"] == "incremental_mcmc_density"
    ]
    assert mcmc_records
    for record in mcmc_records:
        assert record["mcmc_density_algorithm"] == "full_recompute"
        assert record["mcmc_structure_mode"] in {
            "dense", "mixed_family",
        }
        assert record["mcmc_acceptance_mode"] == "medium"
        assert 0.0 <= record["mcmc_acceptance_rate"] <= 1.0
        assert record["final_log_pdf_checksum"]
        assert record["mcmc_diagnostics_checksum"]
        assert record["generated_draw_output_checksum"]
        assert record["generated_draw_initial_rng_state_checksum"]
        assert record["generated_draw_rng_state_checksum"]
        assert (
            record["generated_draw_rng_state_checksum"]
            != record["generated_draw_initial_rng_state_checksum"]
        )
        assert record["density_edge_operations_total"] > 0
        assert record["density_node_count"] > 0
        assert 0.0 < record[
            "incremental_affected_operation_fraction_mean"] <= 1.0
        assert record["incremental_cache_bytes_estimate"] > 0


def test_extended_native_strict_keeps_unimplemented_references_explicit(
        monkeypatch):
    monkeypatch.setenv(
        benchmark_rvine_runtime.BACKEND_ENV, "native_strict")
    records = benchmark_rvine_runtime._extended_workload_records(
        profile="smoke",
        backend="native_strict",
        repeats=1,
        warmups=0,
        enabled=True,
    )

    reference_candidates = {
        "factor_student_rosenblatt",
        "equicorr_rosenblatt",
        "dynamic_rvine_rosenblatt",
    }
    references = [
        record for record in records
        if record["candidate"] in reference_candidates
    ]
    assert references
    assert all(record["status"] == "not_run" for record in references)
    assert all(
        "not implemented" in record["reason"] for record in references)
    mcmc_records = [
        record for record in records
        if record["candidate"] == "incremental_mcmc_density"
    ]
    assert mcmc_records
    assert all(record["status"] == "measured" for record in mcmc_records)
    assert {record["mcmc_density_algorithm"] for record in mcmc_records} == {
        "full_recompute", "incremental",
    }
    paired = {}
    for record in mcmc_records:
        key = (
            record["dimension"],
            record["rows"],
            record["mcmc_given_mode"],
            record["mcmc_structure_mode"],
            record["parameter_mode"],
            record["mcmc_acceptance_mode"],
        )
        paired.setdefault(key, {})[
            record["mcmc_density_algorithm"]] = record
    assert paired
    for records in paired.values():
        assert set(records) == {"full_recompute", "incremental"}
        full = records["full_recompute"]
        incremental = records["incremental"]
        for key in (
                "output_checksum",
                "final_log_pdf_checksum",
                "mcmc_diagnostics_checksum",
                "generated_draw_output_checksum",
                "generated_draw_diagnostics_checksum",
                "generated_draw_initial_rng_state_checksum",
                "generated_draw_rng_state_checksum",
        ):
            assert incremental[key] == full[key]
        if full["mcmc_given_mode"] == "multiple_free":
            assert full["chunk_boundary_nonmultiple_n_free"]
            assert incremental["chunk_boundary_nonmultiple_n_free"]
