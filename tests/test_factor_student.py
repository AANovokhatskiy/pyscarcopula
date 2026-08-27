"""Factor Student likelihood adapter contracts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import inspect
import json
import subprocess
import sys

import numpy as np
import pytest

from pyscarcopula import (
    FactorCorrelation,
    FactorStudentEvaluation,
    FactorStudentEvaluator,
    FactorStudentGridEvaluation,
)
from pyscarcopula.copula.multivariate.factor_student import (
    _raise_native_status,
)
from pyscarcopula.copula.multivariate.factor_estimation import (
    FactorLoadingParameterization,
)
from pyscarcopula._native import static as static_likelihood
from pyscarcopula._native.errors import NativeError, NativeUnsupported


def _problem(dimension=10, rank=3, rows=24, seed=1201):
    rng = np.random.default_rng(seed)
    factor = FactorCorrelation(
        rng.normal(scale=0.07, size=(dimension, rank)))
    observations = rng.uniform(
        0.01, 0.99, size=(rows, dimension))
    return factor, observations


@pytest.mark.parametrize("df", [2.01, 4.5, 30.0])
def test_scalar_likelihood_and_gradient_match_dense_student(df):
    factor, observations = _problem()
    evaluator = FactorStudentEvaluator(factor, observations)
    result = evaluator.evaluate(df)
    dense = static_likelihood.prepare_student(
        factor.to_dense(), observations)
    dense_result = dense.result(df)

    assert isinstance(result, FactorStudentEvaluation)
    np.testing.assert_allclose(
        result.log_pdf,
        dense.log_pdf_rows(df),
        rtol=2e-12,
        atol=2e-12,
    )
    assert result.log_likelihood == pytest.approx(
        -dense_result["negative_log_likelihood"],
        rel=2e-12,
        abs=2e-12,
    )
    assert result.dlog_likelihood_ddf == pytest.approx(
        -dense_result["negative_gradient"],
        rel=2e-11,
        abs=2e-11,
    )


def test_analytical_row_gradient_matches_finite_difference():
    factor, observations = _problem(rows=8, seed=1202)
    evaluator = FactorStudentEvaluator(factor.prepare(), observations)
    df = 7.0
    step = 1e-5
    result = evaluator.evaluate(df)
    finite_difference = (
        evaluator.log_pdf_rows(df + step)
        - evaluator.log_pdf_rows(df - step)
    ) / (2.0 * step)

    np.testing.assert_allclose(
        result.dlog_ddf,
        finite_difference,
        rtol=2e-6,
        atol=2e-7,
    )


def test_row_specific_df_matches_dense_one_row_evaluators():
    factor, observations = _problem(rows=7, seed=1203)
    evaluator = FactorStudentEvaluator(factor, observations)
    df = np.linspace(2.1, 25.0, len(observations))
    result = evaluator.evaluate(df)
    dense_correlation = factor.to_dense()
    expected_log_pdf = []
    expected_gradient = []
    for row, row_df in zip(observations, df, strict=True):
        dense = static_likelihood.prepare_student(
            dense_correlation, row[None, :])
        dense_result = dense.result(float(row_df))
        expected_log_pdf.append(
            -dense_result["negative_log_likelihood"])
        expected_gradient.append(
            -dense_result["negative_gradient"])

    np.testing.assert_allclose(
        result.log_pdf, expected_log_pdf, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        result.dlog_ddf,
        expected_gradient,
        rtol=2e-11,
        atol=2e-11,
    )
    with pytest.raises(ValueError, match="common scalar"):
        _ = result.dlog_likelihood_ddf


def test_thread_results_are_exact_and_concurrent_calls_are_safe():
    factor, observations = _problem(
        dimension=64, rank=5, rows=64, seed=1204)
    evaluator = FactorStudentEvaluator(factor, observations)
    sequential = evaluator.evaluate(8.5, n_threads=1)
    parallel = evaluator.evaluate(8.5, n_threads=4)

    np.testing.assert_array_equal(
        parallel.log_pdf, sequential.log_pdf)
    np.testing.assert_array_equal(
        parallel.dlog_ddf, sequential.dlog_ddf)
    assert parallel.diagnostics["row_parallel_blocks"] == 4

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(evaluator.evaluate, 8.5, n_threads=4)
            for _ in range(2)
        ]
    for future in futures:
        concurrent = future.result()
        np.testing.assert_array_equal(
            concurrent.log_pdf, sequential.log_pdf)
        np.testing.assert_array_equal(
            concurrent.dlog_ddf, sequential.dlog_ddf)


def test_evaluator_owns_read_only_observations_and_optimizer_contract():
    factor, observations = _problem(rows=5, seed=1205)
    original = observations.copy()
    evaluator = FactorStudentEvaluator(factor, observations)
    observations[:] = 0.5

    np.testing.assert_array_equal(evaluator.observations, original)
    assert evaluator.observations.flags.writeable is False
    result = evaluator.evaluate(6.0)
    assert result.log_pdf.flags.writeable is False
    assert result.dlog_ddf.flags.writeable is False
    objective, gradient = evaluator.objective_and_gradient(6.0)
    assert objective == pytest.approx(-result.log_likelihood)
    np.testing.assert_allclose(
        gradient, [-result.dlog_likelihood_ddf])


def test_parameterized_penalized_objective_matches_composed_native_results():
    factor, observations = _problem(
        dimension=7, rank=2, rows=31, seed=1206)
    parameterization, parameters = (
        FactorLoadingParameterization.from_loadings(
            factor.loadings, uniqueness_min=1e-8))
    evaluator = FactorStudentEvaluator(factor, observations)
    penalty = 2.5e-5
    df = 6.25

    result = evaluator.penalized_parameterized_objective_and_gradient(
        df,
        parameters,
        parameterization,
        penalty=penalty,
        condition_max=1e12,
        n_threads=4,
    )
    joint = FactorStudentEvaluator(
        FactorCorrelation(result.loadings), observations
    ).joint_likelihood_and_gradient(df, n_threads=4)
    expected_loading_gradient = (
        -joint.dlog_likelihood_dloadings
        + 2.0 * penalty * result.loadings
    )
    expected_gradient = np.concatenate((
        np.asarray([-joint.dlog_likelihood_ddf]),
        parameterization.pullback(parameters, expected_loading_gradient),
    ))

    assert result.objective == pytest.approx(
        -joint.log_likelihood
        + penalty * float(np.sum(result.loadings ** 2)),
        rel=2e-13,
        abs=2e-13,
    )
    assert result.log_likelihood == pytest.approx(
        joint.log_likelihood, rel=0.0, abs=0.0)
    np.testing.assert_allclose(
        result.gradient, expected_gradient, rtol=2e-13, atol=2e-13)
    assert result.gradient.flags.writeable is False
    assert result.loadings.flags.writeable is False


def test_factor_student_python_adapter_contains_no_aggregate_or_grid_math():
    evaluation_source = inspect.getsource(FactorStudentEvaluation)
    grid_source = inspect.getsource(FactorStudentGridEvaluation)

    assert "np.sum" not in evaluation_source
    assert "np.exp" not in grid_source
    assert "* self.dlog_ddf" not in grid_source


def test_large_dimension_uses_linear_worker_workspace():
    dimension = 100_000
    rank = 4
    factor = FactorCorrelation(
        np.full((dimension, rank), 1e-3))
    evaluator = FactorStudentEvaluator(
        factor, np.full((1, dimension), 0.5))
    result = evaluator.evaluate(7.0)

    assert np.isfinite(result.log_pdf[0])
    assert np.isfinite(result.dlog_ddf[0])
    assert (
        result.diagnostics["worker_workspace_peak_bytes"]
        == (3 * dimension + rank) * 8
    )
    assert not hasattr(evaluator.correlation, "correlation_matrix")


@pytest.mark.parametrize(
    ("observations", "match"),
    [
        (np.ones((0, 4)), "n >= 1"),
        (np.ones((3, 3)), r"shape \(n, 4\)"),
        (
            np.array([
                [0.2, 0.3, 0.4, 0.5],
                [0.2, np.nan, 0.4, 0.5],
            ]),
            "finite",
        ),
    ],
)
def test_invalid_observations_are_rejected(observations, match):
    factor = FactorCorrelation(np.full((4, 1), 0.1))
    with pytest.raises(ValueError, match=match):
        FactorStudentEvaluator(factor, observations)


@pytest.mark.parametrize(
    "df",
    [2.0, np.nan, np.array([5.0, 6.0])],
)
def test_invalid_df_is_rejected(df):
    factor, observations = _problem(rows=3)
    evaluator = FactorStudentEvaluator(factor, observations)
    with pytest.raises(ValueError, match="df"):
        evaluator.evaluate(df)


def test_invalid_correlation_type_is_rejected():
    with pytest.raises(TypeError, match="FactorCorrelation"):
        FactorStudentEvaluator(np.eye(3), np.ones((2, 3)))


def test_default_evaluation_does_not_initialize_parallel_runtime():
    code = (
        "import json, numpy as np\n"
        "from pyscarcopula import FactorCorrelation, FactorStudentEvaluator\n"
        "from pyscarcopula._native import _extension as _cpp_extension\n"
        "m = _cpp_extension.load()\n"
        "before = dict(m._parallel_runtime_info())\n"
        "factor = FactorCorrelation(np.full((1024, 4), 0.01))\n"
        "e = FactorStudentEvaluator(factor, np.full((3, 1024), 0.5))\n"
        "e.evaluate(7.0)\n"
        "e.evaluate_grid([5.0, 7.0], dimension_tile=128)\n"
        "after = dict(m._parallel_runtime_info())\n"
        "print(json.dumps({'before': before, 'after': after}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = json.loads(completed.stdout)
    assert payload["before"]["initialized"] is False
    assert payload["after"]["initialized"] is False


@pytest.mark.parametrize("dimension_tile", [1, 4, 64])
def test_tiled_grid_matches_repeated_row_evaluation(dimension_tile):
    factor, observations = _problem(
        dimension=13, rank=3, rows=6, seed=1210)
    evaluator = FactorStudentEvaluator(factor, observations)
    grid = np.array([2.05, 3.5, 8.0, 40.0])
    result = evaluator.evaluate_grid(
        grid, dimension_tile=dimension_tile)
    expected = [evaluator.evaluate(df) for df in grid]

    assert isinstance(result, FactorStudentGridEvaluation)
    np.testing.assert_allclose(
        result.log_pdf,
        np.column_stack([item.log_pdf for item in expected]),
        rtol=3e-12,
        atol=3e-12,
    )
    np.testing.assert_allclose(
        result.dlog_ddf,
        np.column_stack([item.dlog_ddf for item in expected]),
        rtol=3e-11,
        atol=3e-11,
    )
    assert result.diagnostics["dimension_tiles"] == (
        factor.dimension + dimension_tile - 1) // dimension_tile
    assert (
        result.diagnostics["ppf_exact_values"]
        == len(observations) * len(grid) * factor.dimension
    )


def test_tiled_grid_cell_parallelism_is_thread_exact():
    factor, observations = _problem(
        dimension=40, rank=4, rows=8, seed=1211)
    evaluator = FactorStudentEvaluator(factor, observations)
    grid = np.linspace(2.1, 20.0, 8)
    sequential = evaluator.evaluate_grid(
        grid, dimension_tile=7, n_threads=1)
    parallel = evaluator.evaluate_grid(
        grid, dimension_tile=7, n_threads=4)

    np.testing.assert_array_equal(
        parallel.log_pdf, sequential.log_pdf)
    np.testing.assert_array_equal(
        parallel.dlog_ddf, sequential.dlog_ddf)
    assert parallel.diagnostics["parallel_axis"] == "cells"
    assert parallel.diagnostics["parallel_blocks"] == 4


def test_tiled_grid_dimension_parallelism_is_thread_exact():
    factor, observations = _problem(
        dimension=128, rank=4, rows=1, seed=1212)
    evaluator = FactorStudentEvaluator(factor, observations)
    grid = np.array([7.0])
    sequential = evaluator.evaluate_grid(
        grid, dimension_tile=16, n_threads=1)
    parallel = evaluator.evaluate_grid(
        grid, dimension_tile=16, n_threads=4)

    np.testing.assert_array_equal(
        parallel.log_pdf, sequential.log_pdf)
    np.testing.assert_array_equal(
        parallel.dlog_ddf, sequential.dlog_ddf)
    assert parallel.diagnostics["parallel_axis"] == "dimension_tiles"
    assert parallel.diagnostics["partial_workspace_peak_bytes"] == (
        8 * ((4 + 2 * factor.rank) * 8 + 1)
    )


def test_grid_density_gradient_and_row_batches():
    factor, observations = _problem(rows=11, seed=1213)
    evaluator = FactorStudentEvaluator(factor, observations)
    grid = np.array([3.0, 6.0, 12.0])
    full = evaluator.evaluate_grid(grid, dimension_tile=3)
    density, gradient = full.pdf_and_gradient()
    np.testing.assert_allclose(density, np.exp(full.log_pdf))
    np.testing.assert_allclose(
        gradient, density * full.dlog_ddf)

    blocks = list(evaluator.evaluate_grid_batches(
        grid,
        batch_rows=4,
        dimension_tile=3,
    ))
    assert [block.log_pdf.shape for block in blocks] == [
        (4, 3), (4, 3), (3, 3)]
    np.testing.assert_array_equal(
        np.vstack([block.log_pdf for block in blocks]),
        full.log_pdf,
    )
    np.testing.assert_array_equal(
        np.vstack([block.dlog_ddf for block in blocks]),
        full.dlog_ddf,
    )


def test_grid_memory_budget_covers_output_and_native_workspace():
    factor, observations = _problem(rows=7, seed=1214)
    evaluator = FactorStudentEvaluator(factor, observations)
    grid = np.array([3.0, 5.0, 9.0, 20.0])
    reference = evaluator.evaluate_grid(
        grid, dimension_tile=4, n_threads=4)
    required = reference.diagnostics["peak_bytes_required"]

    with pytest.raises(MemoryError, match="evaluate_grid_batches"):
        evaluator.evaluate_grid(
            grid,
            dimension_tile=4,
            n_threads=4,
            memory_budget_bytes=required - 1,
        )

    batch_required = FactorStudentEvaluator(
        factor, observations[:3]).evaluate_grid(
            grid,
            dimension_tile=4,
            n_threads=4,
        ).diagnostics["peak_bytes_required"]
    with pytest.raises(MemoryError, match="reduce batch_rows"):
        list(evaluator.evaluate_grid_batches(
            grid,
            batch_rows=3,
            dimension_tile=4,
            n_threads=4,
            memory_budget_bytes=batch_required - 1,
        ))


def test_large_dimension_grid_has_no_full_ppf_cache():
    dimension = 100_000
    rank = 4
    factor = FactorCorrelation(
        np.full((dimension, rank), 1e-3))
    evaluator = FactorStudentEvaluator(
        factor, np.full((1, dimension), 0.5))
    grid = np.array([4.0, 8.0])
    result = evaluator.evaluate_grid(
        grid, dimension_tile=4096, n_threads=1)

    assert result.log_pdf.shape == (1, 2)
    assert result.diagnostics["ppf_exact_values"] == (
        dimension * len(grid))
    assert result.diagnostics["worker_workspace_peak_bytes"] == (
        (2 * (4 + 2 * rank) + rank) * 8
    )
    assert result.diagnostics["partial_workspace_peak_bytes"] == 0


@pytest.mark.parametrize(
    ("grid", "kwargs"),
    [
        ([], {}),
        ([2.0, 5.0], {}),
        ([np.nan], {}),
        ([5.0], {"dimension_tile": 0}),
        ([5.0], {"n_threads": 0}),
    ],
)
def test_invalid_grid_contract_is_rejected(grid, kwargs):
    factor, observations = _problem(rows=2)
    evaluator = FactorStudentEvaluator(factor, observations)
    with pytest.raises((TypeError, ValueError)):
        evaluator.evaluate_grid(grid, **kwargs)


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (2, ValueError),
        (3, NativeUnsupported),
        (7, FloatingPointError),
        (1, NativeError),
    ],
)
def test_native_factor_status_is_translated_by_python_adapter(status, error):
    with pytest.raises(error, match=rf"status={status} .*index=4"):
        _raise_native_status(
            {"status": status, "failure_index": 4},
            "contract test",
        )


def test_native_factor_success_status_is_accepted():
    _raise_native_status({"status": 0}, "contract test")
