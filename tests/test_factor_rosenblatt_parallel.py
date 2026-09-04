"""Factor Rosenblatt row dispatch and preserved numerical failure output."""

import numpy as np
import pytest

from pyscarcopula._native import _extension


@pytest.fixture
def native():
    module = _extension.load()
    module._parallel_runtime_shutdown()
    try:
        yield module
    finally:
        module._parallel_runtime_shutdown()


@pytest.mark.parametrize("family", ["gaussian", "student"])
@pytest.mark.parametrize("rows,rank", [(65, 1), (257, 4), (65, 33)])
def test_factor_rosenblatt_uses_one_row_batch(native, family, rows, rank):
    rng = np.random.default_rng(9823)
    loadings = rng.normal(size=(40, rank))
    loadings *= 0.3 / np.linalg.norm(loadings, axis=1, keepdims=True)
    operator = native._FactorCorrelationOperator(loadings, 1e-8)
    observations = rng.uniform(0.02, 0.98, (rows, 40))
    df = np.linspace(3.25, 12.0, rows)
    function = getattr(native, f"factor_{family}_rosenblatt_transform")
    arguments = (operator, observations, df) if family == "student" else (
        operator, observations)
    reference = function(*arguments, 1)
    assert reference["status"] == native.SCAR_OK
    assert not native._parallel_runtime_info()["initialized"]
    for request in (4, 17, 32):
        before = dict(native._parallel_runtime_info())
        actual = function(*arguments, request)
        after = dict(native._parallel_runtime_info())
        assert actual["status"] == native.SCAR_OK
        assert actual["residuals"].tobytes() == reference["residuals"].tobytes()
        blocks = min(request, (rows + 15) // 16)
        assert after["batches_submitted"] - before["batches_submitted"] == 1
        assert after["tasks_submitted"] - before["tasks_submitted"] == blocks
        assert actual["diagnostics"]["parallel_blocks"] == request


@pytest.mark.parametrize("n_threads,stopped_end", [(1, 65), (4, 33), (17, 26), (32, 26)])
def test_real_numerical_failure_preserves_original_block_prefix(native, n_threads, stopped_end):
    # This valid-input failure was reproduced on the implementation preceding
    # row batching: the large-df CDF becomes non-finite at row17/coordinate3.
    operator = native._FactorCorrelationOperator(np.zeros((40, 1)), 1e-8)
    observations = np.full((65, 40), 0.5)
    observations[17, 3] = 0.7
    df = np.full(65, 5.7)
    df[17] = 1e308
    actual = native.factor_student_rosenblatt_transform(operator, observations, df, n_threads)
    assert actual["status"] == native.SCAR_NUMERICAL_FAILURE
    assert actual["failure_index"] == 17
    assert actual["failure_coordinate"] == 3
    expected = np.zeros((65, 40))
    expected[:, :3] = 0.5
    expected[:17, 3] = 0.5
    expected[stopped_end:, 3] = 0.5
    np.testing.assert_array_equal(actual["residuals"], expected)
    info = native._parallel_runtime_info()
    assert info["batches_submitted"] == int(n_threads > 1)
    assert info["tasks_submitted"] == (0 if n_threads == 1 else min(n_threads, 5))


def test_zero_loading_student_transform_retains_conditional_df(native):
    operator = native._FactorCorrelationOperator(np.zeros((40, 1)), 1e-8)
    observations = np.full((65, 40), 0.7)
    observations[::2, 0] = 0.99
    actual = native.factor_student_rosenblatt_transform(
        operator, observations, np.array([3.25]), 4)
    reference = native.dense_student_rosenblatt_transform(
        np.eye(40), observations, np.array([3.25]), 1)
    assert actual["status"] == reference["status"] == native.SCAR_OK
    np.testing.assert_allclose(actual["residuals"], reference["residuals"],
                               rtol=0.0, atol=3e-11)
    assert np.any(np.abs(actual["residuals"][:, 1] - observations[:, 1]) > 1e-3)
