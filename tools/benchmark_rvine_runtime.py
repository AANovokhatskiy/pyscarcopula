"""Record reproducible R-vine runtime and extended candidate baselines.

This is a measurement harness, not a wall-clock test.  It records timings,
Python allocation counts, peak traced memory, output checksums, RNG state and
the complete workload matrix.  Cases deliberately bounded by the selected
profile remain visible as ``not_run`` records with a reason.  Optional
candidate workloads are enabled explicitly with ``--include-extended-workloads``
so the original Gate 0 baseline remains inexpensive and comparable.
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
    EquicorrGaussianCopula,
    FrankCopula,
    GumbelCopula,
    IndependentCopula,
    JoeCopula,
)
from pyscarcopula._types import (
    GASResult,
    LatentResult,
    MLEResult,
    gas_params,
    ou_params,
)
from pyscarcopula.copula.multivariate.factor_correlation import (
    FactorCorrelation,
)
from pyscarcopula.numerical import _cpp_extension
from pyscarcopula.stattests import (
    equicorr_rosenblatt_transform,
    factor_student_rosenblatt_transform,
    rvine_rosenblatt_transform,
    student_rosenblatt_transform,
)
from pyscarcopula.vine._edge_adapter import edge_copula
from pyscarcopula.vine._pair_copula import PairCopula
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


def _json_checksum(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
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


def _with_metadata(record: dict[str, object], **metadata) -> dict[str, object]:
    """Attach JSON-serializable workload metadata to one record."""
    return {**record, **metadata}


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


def _extended_factor_workloads(profile: str):
    if profile == "smoke":
        return ((50, 32, 3), (2049, 2, 1))
    return ((50, 1_000, 3), (250, 200, 5), (2049, 32, 3))


def _extended_equicorr_workloads(profile: str):
    if profile == "smoke":
        return ((50, 32), (2049, 2))
    return ((50, 1_000), (250, 200), (2049, 2))


def _extended_dynamic_workloads(profile: str):
    if profile == "smoke":
        return (
            ("gas", 5, 32, "single"),
            ("gas", 5, 32, "all"),
            ("scar", 5, 16, "single"),
            ("scar", 5, 16, "all"),
        )
    return (
        ("gas", 5, 3_000, "single"),
        ("gas", 5, 3_000, "all"),
        ("gas", 10, 1_000, "all"),
        ("scar", 5, 300, "single"),
        ("scar", 5, 300, "all"),
        ("scar", 10, 100, "all"),
    )


def _extended_mcmc_workloads(profile: str):
    if profile == "smoke":
        return (
            (10, 32, 12, "mixed_family", "scalar_parameters", "medium"),
            (20, 16, 12, "dense", "scalar_parameters", "medium"),
        )
    covering_cases = (
        (1, 13, "dense", "scalar_parameters", "low"),
        (32, 13, "dense", "row_path", "high"),
        (32, 13, "mixed_family", "row_path", "low"),
        (100, 30, "mixed_family", "scalar_parameters", "medium"),
        (100, 30, "independence_heavy", "scalar_parameters", "medium"),
        (1_000, 7, "independence_heavy", "row_path", "high"),
    )
    return tuple(
        (dimension, rows, steps, structure, parameters, acceptance)
        for dimension in (10, 20, 30, 50)
        for rows, steps, structure, parameters, acceptance in covering_cases
    )


def _configured_incremental_mcmc_vine(
        dimension: int, structure_mode: str, acceptance_mode: str):
    vine = configured_static_dvine(dimension)
    if structure_mode == "independence_heavy":
        return vine
    if structure_mode == "dense":
        rho = 0.85 if acceptance_mode == "low" else 0.05
        vine.pair_copulas = {
            key: fitted_pair(BivariateGaussianCopula(), rho)
            for key in vine.pair_copulas
        }
        return vine
    if structure_mode != "mixed_family":
        raise ValueError(f"unknown incremental MCMC structure {structure_mode}")

    families = (
        lambda: fitted_pair(BivariateGaussianCopula(), -0.25),
        lambda: fitted_pair(ClaytonCopula(rotate=90), 0.8),
        lambda: fitted_pair(GumbelCopula(rotate=180), 1.6),
        lambda: fitted_pair(FrankCopula(), 2.5),
        lambda: fitted_pair(IndependentCopula(), 0.0),
    )
    vine.pair_copulas = {
        key: families[index % len(families)]()
        for index, key in enumerate(sorted(vine.pair_copulas))
    }
    return vine


def _incremental_mcmc_parameters(vine, rows: int, parameter_mode: str):
    parameters = scalar_parameters(vine)
    if parameter_mode == "scalar_parameters":
        return parameters
    if parameter_mode != "row_path":
        raise ValueError(
            f"unknown incremental MCMC parameter mode {parameter_mode}")
    phase = np.linspace(0.0, 2.0 * np.pi, rows, endpoint=False)
    for key, edge in sorted(vine.pair_copulas.items()):
        if isinstance(edge_copula(edge), IndependentCopula):
            continue
        center = float(edge.param)
        amplitude = 0.02 * max(1.0, abs(center))
        parameters[key] = np.ascontiguousarray(
            center + amplitude * np.sin(phase), dtype=np.float64)
    return parameters


def _extended_reference_not_run(
        operation, dimension, rows, parameter_mode, reason, **metadata):
    return _with_metadata(
        _not_run(operation, dimension, rows, parameter_mode, reason),
        workload_group="extended_workloads",
        **metadata,
    )


def _density_dependency_profile(vine, parameters, rows, given):
    """Return the static dependency closure used to size incremental MCMC."""
    module = _cpp_extension.load()
    context, _parameter_pack = vine._native_density_context(
        module,
        vine.pair_copulas,
        vine._edge_map,
        parameters,
        rows,
    )
    if context is None:
        raise AssertionError("native density context is required for profiling")
    plan = context["plan"]
    dependencies = [set() for _ in range(int(plan.node_count))]
    for variable, node in enumerate(plan.input_nodes):
        dependencies[int(node)] = {int(variable)}

    operation_dependencies = []
    for position, (input1, input2) in enumerate(zip(
            plan.input1_nodes, plan.input2_nodes)):
        affected = dependencies[int(input1)] | dependencies[int(input2)]
        operation_dependencies.append(affected)
        output1 = int(plan.output1_nodes[position])
        output2 = int(plan.output2_nodes[position])
        if output1 >= 0:
            dependencies[output1] = set(affected)
        if output2 >= 0:
            dependencies[output2] = set(affected)

    free_variables = [
        variable for variable in range(vine.d) if variable not in given
    ]
    affected_counts = [
        sum(variable in affected for affected in operation_dependencies)
        for variable in free_variables
    ]
    operation_count = len(operation_dependencies)
    mean_affected = statistics.fmean(affected_counts)
    cache_values_per_row = int(plan.node_count) + operation_count
    return {
        "density_edge_operations_total": operation_count,
        "density_node_count": int(plan.node_count),
        "incremental_free_variables": free_variables,
        "incremental_affected_operations": affected_counts,
        "incremental_affected_operations_min": min(affected_counts),
        "incremental_affected_operations_max": max(affected_counts),
        "incremental_affected_operations_mean": mean_affected,
        "incremental_affected_operation_fraction_mean": (
            mean_affected / operation_count),
        "incremental_cache_values_per_row_estimate": cache_values_per_row,
        "incremental_cache_bytes_estimate": (
            rows * cache_values_per_row * np.dtype(np.float64).itemsize),
    }


def _extended_factor_records(
        profile, backend, repeats, warmups, enabled, seed):
    records = []
    reason = (
        "pass --include-extended-workloads to measure optional candidates"
        if not enabled else
        "native factor Student Rosenblatt candidate is not implemented"
    )
    for dimension, rows, rank in _extended_factor_workloads(profile):
        for parameter_mode in ("scalar_df", "df_path"):
            metadata = {
                "candidate": "factor_student_rosenblatt",
                "factor_rank": rank,
                "matrix_free_required": dimension > 2048,
                "implementation": "python_factor_reference",
            }
            if not enabled or backend != "python_executor":
                records.append(_extended_reference_not_run(
                    "extended_factor_student_rosenblatt",
                    dimension,
                    rows,
                    parameter_mode,
                    reason,
                    **metadata,
                ))
                continue

            rng = np.random.default_rng(seed)
            loadings = rng.normal(scale=0.04, size=(dimension, rank))
            correlation = FactorCorrelation(loadings).prepare()
            observations = rng.uniform(
                0.02, 0.98, size=(rows, dimension))
            df = (
                7.5 if parameter_mode == "scalar_df" else
                np.linspace(5.0, 12.0, rows)
            )
            record = _measure(
                operation="extended_factor_student_rosenblatt",
                dimension=dimension,
                rows=rows,
                parameter_mode=parameter_mode,
                repeats=repeats,
                warmups=warmups,
                seed=seed,
                call=lambda _rng, correlation=correlation, df=df,
                observations=observations:
                factor_student_rosenblatt_transform(
                    correlation, df, observations),
            )
            records.append(_with_metadata(
                record,
                workload_group="extended_workloads",
                **metadata,
            ))
            seed += 10
    return records, seed


def _extended_equicorr_records(
        profile, backend, repeats, warmups, enabled, seed):
    records = []
    reason = (
        "pass --include-extended-workloads to measure optional candidates"
        if not enabled else
        "native equicorrelation Rosenblatt candidate is not implemented"
    )
    for dimension, rows in _extended_equicorr_workloads(profile):
        for method in ("MLE", "GAS"):
            parameter_mode = (
                "mle_scalar_rho" if method == "MLE" else "gas_rho_path")
            metadata = {
                "candidate": "equicorr_rosenblatt",
                "estimation_method": method,
                "implementation": "python_vectorized_reference",
            }
            if not enabled or backend != "python_executor":
                records.append(_extended_reference_not_run(
                    "extended_equicorr_rosenblatt",
                    dimension,
                    rows,
                    parameter_mode,
                    reason,
                    **metadata,
                ))
                continue

            rng = np.random.default_rng(seed)
            observations = rng.uniform(
                0.02, 0.98, size=(rows, dimension))
            copula = EquicorrGaussianCopula(dimension)
            if method == "MLE":
                fit_result = MLEResult(
                    log_likelihood=0.0,
                    method="MLE",
                    copula_name=copula.name,
                    success=True,
                    copula_param=0.1,
                )
            else:
                fit_result = GASResult(
                    log_likelihood=0.0,
                    method="GAS",
                    copula_name=copula.name,
                    success=True,
                    params=gas_params(0.0, 0.05, 0.8),
                    scaling="unit",
                    r_last=0.0,
                )
            record = _measure(
                operation="extended_equicorr_rosenblatt",
                dimension=dimension,
                rows=rows,
                parameter_mode=parameter_mode,
                repeats=repeats,
                warmups=warmups,
                seed=seed,
                call=lambda _rng, copula=copula,
                observations=observations, fit_result=fit_result:
                equicorr_rosenblatt_transform(
                    copula, observations, fit_result),
            )
            records.append(_with_metadata(
                record,
                workload_group="extended_workloads",
                **metadata,
            ))
            seed += 10
    return records, seed


def _configured_extended_dynamic_vine(dimension, strategy, coverage):
    """Build one/all-edge dynamic workloads without fitting noise."""
    vine = configured_static_dvine(dimension)
    edge_keys = sorted(vine.pair_copulas)
    dynamic_keys = edge_keys[:1] if coverage == "single" else edge_keys
    for key in dynamic_keys:
        copula = BivariateGaussianCopula()
        if strategy == "gas":
            result = GASResult(
                log_likelihood=0.0,
                method="GAS",
                copula_name=copula.name,
                success=True,
                params=gas_params(0.0, 0.6, 0.0),
                scaling="unit",
                r_last=0.0,
            )
        else:
            result = LatentResult(
                log_likelihood=0.0,
                method="SCAR-TM-OU",
                copula_name=copula.name,
                success=True,
                params=ou_params(1.0, 0.0, 0.4),
                K=20,
                grid_range=3.0,
            )
        vine.pair_copulas[key] = PairCopula(
            copula=copula,
            param=0.0,
            log_likelihood=0.0,
            nfev=0,
            tau=0.0,
            fit_result=result,
        )
    vine.method = "MIXED"
    return vine, len(dynamic_keys)


def _extended_dynamic_records(
        profile, backend, repeats, warmups, enabled, seed):
    records = []
    reason = (
        "pass --include-extended-workloads to measure optional candidates"
        if not enabled else
        "native dynamic R-vine Rosenblatt candidate is not implemented"
    )
    for strategy, dimension, rows, coverage in _extended_dynamic_workloads(
            profile):
        vine, dynamic_edges = _configured_extended_dynamic_vine(
            dimension,
            strategy,
            coverage,
        )
        metadata = {
            "candidate": "dynamic_rvine_rosenblatt",
            "dynamic_strategy": strategy.upper(),
            "dynamic_edge_coverage": coverage,
            "dynamic_edges": dynamic_edges,
            "total_edges": len(vine.pair_copulas),
            "implementation": "python_executor_reference",
        }
        parameter_mode = f"{strategy}_{coverage}_dynamic_edges"
        if not enabled or backend != "python_executor":
            records.append(_extended_reference_not_run(
                "extended_dynamic_rvine_rosenblatt",
                vine.d,
                rows,
                parameter_mode,
                reason,
                **metadata,
            ))
            continue

        observations = np.random.default_rng(seed).uniform(
            0.02, 0.98, size=(rows, vine.d))
        record = _measure(
            operation="extended_dynamic_rvine_rosenblatt",
            dimension=vine.d,
            rows=rows,
            parameter_mode=parameter_mode,
            repeats=repeats,
            warmups=warmups,
            seed=seed,
            call=lambda _rng, vine=vine, observations=observations:
            rvine_rosenblatt_transform(vine, observations),
        )
        records.append(_with_metadata(
            record,
            workload_group="extended_workloads",
            **metadata,
        ))
        seed += 10
    return records, seed


def _extended_incremental_mcmc_records(
        profile, backend, repeats, warmups, enabled, seed):
    records = []
    for (
            dimension,
            rows,
            coordinate_steps,
            structure_mode,
            parameter_mode,
            acceptance_mode,
    ) in _extended_mcmc_workloads(profile):
        if not enabled:
            records.append(_extended_reference_not_run(
                "extended_mcmc_full_recompute",
                dimension,
                rows,
                parameter_mode,
                "pass --include-extended-workloads to measure optional "
                "candidates",
                candidate="incremental_mcmc_density",
                mcmc_density_algorithm="full_recompute",
                coordinate_steps=coordinate_steps,
                mcmc_structure_mode=structure_mode,
                mcmc_acceptance_mode=acceptance_mode,
            ))
            continue

        vine = _configured_incremental_mcmc_vine(
            dimension, structure_mode, acceptance_mode)
        parameters = _incremental_mcmc_parameters(
            vine, rows, parameter_mode)
        base_given_values = {
            variable: 0.2 + 0.6 * (variable + 1) / (dimension + 1)
            for variable in range(dimension)
        }
        given_sets = {
            "multiple_free": {0: 0.35, 2: 0.65},
            "edge_free": {
                variable: value
                for variable, value in base_given_values.items()
                if variable != dimension - 1
            },
            "central_free": {
                variable: value
                for variable, value in base_given_values.items()
                if variable != dimension // 2
            },
        }
        algorithms = (
            ("full_recompute",)
            if backend == "python_executor" else
            ("full_recompute", "incremental")
        )
        draw_chunk_steps = 5
        for given_mode, given in given_sets.items():
            rng = np.random.default_rng(seed)
            initial = rng.uniform(0.02, 0.98, size=(rows, dimension))
            for variable, value in given.items():
                initial[:, variable] = value
            if acceptance_mode == "high":
                for variable in range(dimension):
                    if variable not in given:
                        initial[:, variable] = 0.5
                proposal_draws = rng.uniform(
                    0.495, 0.505, size=(coordinate_steps, rows))
            elif acceptance_mode == "low":
                for variable in range(dimension):
                    if variable not in given:
                        initial[:, variable] = 0.5
                proposal_draws = np.empty(
                    (coordinate_steps, rows), dtype=np.float64)
                proposal_draws[0::2] = rng.uniform(
                    0.005, 0.02, size=proposal_draws[0::2].shape)
                proposal_draws[1::2] = rng.uniform(
                    0.98, 0.995, size=proposal_draws[1::2].shape)
            else:
                proposal_draws = rng.uniform(
                    0.01, 0.99, size=(coordinate_steps, rows))
            acceptance_draws = rng.uniform(
                0.01, 0.99, size=(coordinate_steps, rows))
            random_draws = np.stack(
                (proposal_draws, acceptance_draws), axis=2)
            dependency_profile = _density_dependency_profile(
                vine, parameters, rows, given)
            _probe_samples, probe_diagnostics = (
                vine._sample_arbitrary_given_mcmc(
                    rows,
                    parameters,
                    np.random.default_rng(seed + 1_000_000),
                    given,
                    initial=initial,
                    n_steps=coordinate_steps,
                    burnin_steps=0,
                    random_draws=random_draws,
                    density_algorithm="full_recompute",
                )
            )
            total_proposed = sum(probe_diagnostics["proposed"].values())
            total_accepted = sum(probe_diagnostics["accepted"].values())
            observed_acceptance_rate = (
                total_accepted / total_proposed if total_proposed else 0.0)
            for algorithm in algorithms:
                validation_rng = np.random.default_rng(seed + 2_000_000)
                validation_samples, validation_diagnostics = (
                    vine._sample_arbitrary_given_mcmc(
                        rows,
                        parameters,
                        validation_rng,
                        given,
                        initial=initial,
                        n_steps=coordinate_steps,
                        burnin_steps=0,
                        random_draws=random_draws,
                        density_algorithm=algorithm,
                        chunk_steps=draw_chunk_steps,
                    )
                )
                final_log_pdf = vine._log_pdf_rows_with_r(
                    validation_samples, parameters)

                generated_rng = np.random.default_rng(seed + 3_000_000)
                generated_initial_rng_state_checksum = _rng_checksum(
                    generated_rng)
                generated_samples, generated_diagnostics = (
                    vine._sample_arbitrary_given_mcmc(
                        rows,
                        parameters,
                        generated_rng,
                        given,
                        initial=initial,
                        n_steps=coordinate_steps,
                        burnin_steps=0,
                        density_algorithm=algorithm,
                        chunk_steps=draw_chunk_steps,
                    )
                )
                operation = f"extended_mcmc_{algorithm}"
                record = _measure(
                    operation=operation,
                    dimension=dimension,
                    rows=rows,
                    parameter_mode=parameter_mode,
                    repeats=repeats,
                    warmups=warmups,
                    seed=seed,
                    call=lambda call_rng, vine=vine, rows=rows,
                    parameters=parameters, given=given, initial=initial,
                    coordinate_steps=coordinate_steps,
                    random_draws=random_draws, algorithm=algorithm:
                    vine._sample_arbitrary_given_mcmc(
                        rows,
                        parameters,
                        call_rng,
                        given,
                        initial=initial,
                        n_steps=coordinate_steps,
                        burnin_steps=0,
                        random_draws=random_draws,
                        density_algorithm=algorithm,
                        chunk_steps=draw_chunk_steps,
                    )[0],
                )
                free_count = dimension - len(given)
                records.append(_with_metadata(
                    record,
                    workload_group="extended_workloads",
                    candidate="incremental_mcmc_density",
                    mcmc_density_algorithm=algorithm,
                    mcmc_given_mode=given_mode,
                    mcmc_structure_mode=structure_mode,
                    mcmc_acceptance_mode=acceptance_mode,
                    mcmc_acceptance_rate=float(observed_acceptance_rate),
                    coordinate_steps=coordinate_steps,
                    chunk_steps=draw_chunk_steps,
                    chunk_boundary_nonmultiple_n_free=bool(
                        coordinate_steps > draw_chunk_steps
                        and free_count > 1
                        and draw_chunk_steps % free_count != 0),
                    final_log_pdf_checksum=_checksum(final_log_pdf),
                    final_log_pdf_source="full_density_recheck",
                    mcmc_diagnostics=validation_diagnostics,
                    mcmc_diagnostics_checksum=_json_checksum(
                        validation_diagnostics),
                    generated_draw_output_checksum=_checksum(
                        generated_samples),
                    generated_draw_diagnostics_checksum=_json_checksum(
                        generated_diagnostics),
                    generated_draw_initial_rng_state_checksum=(
                        generated_initial_rng_state_checksum),
                    generated_draw_rng_state_checksum=_rng_checksum(
                        generated_rng),
                    **dependency_profile,
                ))
            seed += 10
    return records, seed


def _extended_workload_records(
        *, profile, backend, repeats, warmups, enabled):
    """Measure optional migrations and the current full-MCMC baseline."""
    records = []
    seed = 202610000
    factor, seed = _extended_factor_records(
        profile, backend, repeats, warmups, enabled, seed)
    records.extend(factor)
    equicorr, seed = _extended_equicorr_records(
        profile, backend, repeats, warmups, enabled, seed)
    records.extend(equicorr)
    dynamic, seed = _extended_dynamic_records(
        profile, backend, repeats, warmups, enabled, seed)
    records.extend(dynamic)
    mcmc, _seed = _extended_incremental_mcmc_records(
        profile, backend, repeats, warmups, enabled, seed)
    records.extend(mcmc)
    return records


def run_benchmark(
        *,
        profile: str,
        backend: str,
        repeats: int,
        warmups: int,
        include_mcmc: bool,
        include_extended_workloads: bool = False,
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
    records.extend(_extended_workload_records(
        profile=profile,
        backend=backend,
        repeats=repeats,
        warmups=warmups,
        enabled=include_extended_workloads,
    ))
    module = _cpp_extension.load()
    return {
        "schema_version": 2,
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
            "include_extended_workloads": include_extended_workloads,
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
    parser.add_argument(
        "--include-extended-workloads",
        action="store_true",
        help=(
            "measure optional factor/equicorr/dynamic Rosenblatt candidates "
            "and the current full-recompute MCMC scaling baseline"),
    )
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
        include_extended_workloads=args.include_extended_workloads,
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
