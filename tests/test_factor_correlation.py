"""Independent FactorCorrelation and native Woodbury contracts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys

import numpy as np
import pytest

from pyscarcopula import FactorCorrelation, PreparedFactorCorrelation


def _factor(d=12, rank=3, seed=901):
    rng = np.random.default_rng(seed)
    return FactorCorrelation(
        rng.normal(scale=0.08, size=(d, rank)))


def test_value_object_is_normalized_owned_and_read_only():
    source = np.array([
        [0.2, 0.1],
        [0.1, -0.2],
        [0.05, 0.15],
        [-0.1, 0.1],
    ])
    factor = FactorCorrelation(source)
    source[0, 0] = 99.0

    assert factor.dimension == 4
    assert factor.rank == 2
    assert factor.loadings[0, 0] == 0.2
    assert factor.loadings.flags.writeable is False
    assert factor.uniqueness.flags.writeable is False
    dense = factor.to_dense()
    np.testing.assert_array_equal(np.diag(dense), np.ones(4))
    assert np.linalg.eigvalsh(dense).min() > 0.0


def test_unconstrained_transform_respects_uniqueness_floor():
    values = np.array([
        [0.0, 0.0],
        [1e100, -1e100],
        [2.0, 3.0],
        [-4.0, 1.0],
    ])
    factor = FactorCorrelation.from_unconstrained(
        values, uniqueness_min=1e-4)

    assert np.min(factor.uniqueness) >= 1e-4
    assert factor.diagnostics["source"] == (
        "unconstrained_row_transform")


def test_woodbury_operations_match_dense_reference():
    rng = np.random.default_rng(902)
    factor = _factor(d=18, rank=4)
    prepared = factor.prepare()
    dense = factor.to_dense()
    rows = rng.normal(size=(35, factor.dimension))

    np.testing.assert_allclose(
        prepared.logdet,
        np.linalg.slogdet(dense)[1],
        rtol=2e-14,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        prepared.matvec(rows),
        rows @ dense,
        rtol=2e-14,
        atol=2e-14,
    )
    expected_solve = np.linalg.solve(dense, rows.T).T
    np.testing.assert_allclose(
        prepared.solve(rows),
        expected_solve,
        rtol=5e-14,
        atol=5e-14,
    )
    np.testing.assert_allclose(
        prepared.quadratic_forms(rows),
        np.einsum("ij,ij->i", rows, expected_solve),
        rtol=5e-14,
        atol=5e-14,
    )
    np.testing.assert_allclose(
        prepared.matvec(rows[0]),
        dense @ rows[0],
        rtol=2e-14,
        atol=2e-14,
    )
    assert prepared.quadratic_form(rows[0]) == pytest.approx(
        rows[0] @ expected_solve[0], rel=5e-14, abs=5e-14)


def test_row_parallelism_is_exact_and_prepared_operator_is_thread_safe():
    rng = np.random.default_rng(903)
    prepared = _factor(d=128, rank=6).prepare()
    rows = rng.normal(size=(64, 128))
    sequential = (
        prepared.matvec(rows, n_threads=1),
        prepared.solve(rows, n_threads=1),
        prepared.quadratic_forms(rows, n_threads=1),
    )
    parallel = (
        prepared.matvec(rows, n_threads=4),
        prepared.solve(rows, n_threads=4),
        prepared.quadratic_forms(rows, n_threads=4),
    )
    for actual, expected in zip(parallel, sequential, strict=True):
        np.testing.assert_array_equal(actual, expected)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(prepared.solve, rows, n_threads=4)
            for _ in range(2)
        ]
    for future in futures:
        np.testing.assert_array_equal(future.result(), sequential[1])


def test_normal_sampling_matches_factor_covariance_and_is_thread_exact():
    factor = _factor(d=6, rank=2, seed=904)
    prepared = factor.prepare()
    dense = factor.to_dense()
    one = prepared.sample_normal(
        60_000, rng=np.random.default_rng(905), n_threads=1)
    many = prepared.sample_normal(
        60_000, rng=np.random.default_rng(905), n_threads=4)

    np.testing.assert_array_equal(many, one)
    np.testing.assert_allclose(
        np.cov(one, rowvar=False),
        dense,
        atol=0.018,
    )


def test_sampling_batches_and_dense_materialization_enforce_budgets():
    prepared = _factor(d=20, rank=3).prepare()
    required = 4 * (20 + 3) * 8
    blocks = list(prepared.sample_normal_batches(
        11,
        batch_rows=4,
        rng=np.random.default_rng(906),
        memory_budget_bytes=required,
    ))
    assert [block.shape for block in blocks] == [
        (4, 20), (4, 20), (3, 20)]

    with pytest.raises(MemoryError, match="sample_normal_batches"):
        prepared.sample_normal(4, memory_budget_bytes=required - 1)
    with pytest.raises(MemoryError, match="reduce batch_rows"):
        list(prepared.sample_normal_batches(
            4, batch_rows=4, memory_budget_bytes=required - 1))
    with pytest.raises(MemoryError, match="disabled"):
        prepared.to_dense(max_dimension=10)
    with pytest.raises(MemoryError, match="requires"):
        prepared.to_dense(
            max_dimension=20,
            memory_budget_bytes=20 * 20 * 8 - 1,
        )


def test_npz_and_mmap_round_trip(tmp_path):
    factor = _factor(d=15, rank=4)
    npz_path = factor.save_npz(tmp_path / "factor.npz")
    mmap_path = factor.save_mmap(tmp_path / "factor-mmap")
    portable = FactorCorrelation.load_npz(npz_path)
    mapped = FactorCorrelation.load_mmap(mmap_path)

    for restored in (portable, mapped):
        np.testing.assert_array_equal(
            restored.loadings, factor.loadings)
        np.testing.assert_array_equal(
            restored.uniqueness, factor.uniqueness)
        assert restored.prepare().logdet == pytest.approx(
            factor.prepare().logdet, rel=0.0, abs=0.0)
    assert isinstance(mapped.loadings, np.memmap)


def test_large_dimension_storage_is_linear_and_dense_is_not_implicit():
    dimension = 100_000
    rank = 4
    loadings = np.full((dimension, rank), 1e-3)
    factor = FactorCorrelation(loadings)
    prepared = factor.prepare()

    assert factor.storage_bytes == dimension * (rank + 1) * 8
    assert (
        prepared.diagnostics["prepared_storage_bytes"]
        == (2 * dimension * rank + 2 * dimension + rank * rank) * 8
    )
    assert not hasattr(prepared, "correlation_matrix")
    with pytest.raises(MemoryError, match="disabled"):
        factor.to_dense()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: FactorCorrelation(np.ones((1, 1))),
        lambda: FactorCorrelation(np.ones((3, 3))),
        lambda: FactorCorrelation(np.array([[0.1], [np.nan]])),
        lambda: FactorCorrelation(
            np.array([[1.0], [0.1]]), uniqueness_min=1e-8),
        lambda: FactorCorrelation(np.zeros((3, 1)), uniqueness_min=0.0),
    ],
)
def test_invalid_factor_contract_is_rejected(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_invalid_operator_inputs_are_rejected():
    prepared = _factor(d=5, rank=2).prepare()
    with pytest.raises(ValueError):
        prepared.solve(np.ones((3, 4)))
    with pytest.raises(ValueError):
        prepared.matvec(np.array([1.0, 2.0, 3.0, 4.0, np.nan]))
    with pytest.raises(ValueError):
        prepared.quadratic_form(np.ones((1, 5)))
    with pytest.raises(ValueError, match="n_threads"):
        prepared.solve(np.ones(5), n_threads=0)


def test_default_operator_calls_do_not_initialize_parallel_runtime():
    code = (
        "import json, numpy as np\n"
        "from pyscarcopula import FactorCorrelation\n"
        "from pyscarcopula._native import _extension as _cpp_extension\n"
        "m = _cpp_extension.load()\n"
        "before = dict(m._parallel_runtime_info())\n"
        "op = FactorCorrelation(np.full((1024, 4), 0.01)).prepare()\n"
        "op.solve(np.ones((16, 1024)))\n"
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


def test_public_prepared_type_requires_factor_value_object():
    with pytest.raises(TypeError):
        PreparedFactorCorrelation(np.zeros((4, 1)))
