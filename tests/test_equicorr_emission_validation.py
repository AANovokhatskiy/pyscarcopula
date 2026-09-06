"""Boundary validation must not depend on native emission work being needed."""

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from pyscarcopula import EquicorrGaussianCopula
from pyscarcopula._native import multivariate


ROWS = (
    "log_pdf_rows",
    "dlog_pdf_dr_rows",
    "log_pdf_and_dlog_dr_rows",
)
U = np.random.default_rng(82031).uniform(0.1, 0.9, (12, 4))


@pytest.mark.parametrize("entry", ROWS)
@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("offset", [-1, np.int64(-3), True, np.bool_(False),
                                    1.5, 1.0, "3", np.array(3)])
def test_invalid_row_offsets_fail_before_native(entry, prepared, offset, monkeypatch):
    model = EquicorrGaussianCopula(4)
    data = model.prepare_sufficient_statistics(U) if prepared else U

    def unexpected_native_call(*args, **kwargs):
        raise AssertionError("invalid t_index reached the native adapter")

    monkeypatch.setattr(multivariate, "log_pdf_and_dlog_rows", unexpected_native_call)
    with pytest.raises((TypeError, ValueError), match="t_index"):
        getattr(model, entry)(data, 0.25, t_index=offset)


@pytest.mark.parametrize("entry", ROWS)
@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("offset", [None, 0, 3, np.int64(3), 10000])
@pytest.mark.parametrize("vector_parameter", [False, True])
def test_valid_offsets_preserve_sliced_density_and_gradient(
        entry, prepared, offset, vector_parameter):
    model = EquicorrGaussianCopula(4)
    block = U[3:9]
    data = model.prepare_sufficient_statistics(block) if prepared else block
    rho = (np.linspace(-0.2, 0.55, len(U))[3:9] if vector_parameter
           else np.full(len(block), 0.25))
    parameter = rho if vector_parameter else float(rho[0])
    z = norm.ppf(block)

    def independent_log_density(parameters):
        return np.array([
            multivariate_normal.logpdf(row, cov=np.eye(4) * (1 - r) + r)
            - norm.logpdf(row).sum()
            for row, r in zip(z, parameters)
        ])

    expected_log = independent_log_density(rho)
    h = 1e-6
    expected_gradient = (
        independent_log_density(rho + h) - independent_log_density(rho - h)
    ) / (2 * h)
    actual = getattr(model, entry)(data, parameter, t_index=offset, n_threads=1)
    if entry == "log_pdf_rows":
        np.testing.assert_allclose(actual, expected_log, atol=2e-12)
    elif entry == "dlog_pdf_dr_rows":
        np.testing.assert_allclose(actual, expected_gradient, rtol=2e-6, atol=2e-7)
    else:
        np.testing.assert_allclose(actual[0], expected_log, atol=2e-12)
        np.testing.assert_allclose(actual[1], expected_gradient, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("n_threads", [0, -1, True, np.bool_(False), 1.5, "1"])
@pytest.mark.parametrize("n_rows", [0, 3])
def test_grid_batches_validate_threads_independently_of_row_count(n_threads, n_rows):
    model = EquicorrGaussianCopula(4)
    with pytest.raises((ValueError, TypeError), match="n_threads"):
        list(model.pdf_and_grad_on_grid_batches(U[:n_rows], [0.1], n_threads=n_threads))


@pytest.mark.parametrize("shape", [(0,), (0, 3), (0, 5), (0, 4, 1)])
def test_empty_grid_batches_reject_malformed_dimension(shape):
    model = EquicorrGaussianCopula(4)
    with pytest.raises(ValueError, match="2D|columns"):
        list(model.pdf_and_grad_on_grid_batches(np.empty(shape), [0.1]))


def test_empty_grid_batches_reject_complex_input():
    model = EquicorrGaussianCopula(4)
    with pytest.raises(TypeError, match="complex"):
        list(model.pdf_and_grad_on_grid_batches(np.empty((0, 4), dtype=complex), [0.1]))


@pytest.mark.parametrize("n_threads", [1, np.int64(2)])
def test_valid_empty_grid_batches_do_not_enter_native(n_threads, monkeypatch):
    model = EquicorrGaussianCopula(4)

    def unexpected_native_call(*args, **kwargs):
        raise AssertionError("valid empty batches do not require a native grid")

    monkeypatch.setattr(multivariate, "pdf_and_grad_grid", unexpected_native_call)
    assert list(model.pdf_and_grad_on_grid_batches(
        np.empty((0, 4)), [-0.2, 0.3], batch_rows=2,
        n_threads=n_threads, memory_budget_bytes=0,
    )) == []


def test_prepared_grid_batches_reject_wrong_model_dimension():
    prepared = EquicorrGaussianCopula(3).prepare_sufficient_statistics(U[:, :3])
    with pytest.raises(ValueError, match="dimension"):
        list(EquicorrGaussianCopula(4).pdf_and_grad_on_grid_batches(prepared, [0.1]))


@pytest.mark.parametrize("representation", ["float32", "memmap"])
def test_nonempty_grid_validation_preserves_block_memory(representation, tmp_path, monkeypatch):
    model = EquicorrGaussianCopula(4)
    data = U.astype(np.float32)
    if representation == "memmap":
        data = np.memmap(tmp_path / "observations.bin", dtype=np.float32,
                         mode="w+", shape=U.shape)
        data[:] = U
    grid = np.array([-0.3, 0.2, 0.7])
    expected = model.pdf_and_grad_on_grid_batch(data, grid)
    original = multivariate.pdf_and_grad_grid
    blocks = []

    def observe(copula, observations, x_grid, **kwargs):
        assert observations.dtype == np.float32
        assert len(observations) <= 3
        assert np.shares_memory(observations, data)
        blocks.append(observations)
        return original(copula, observations, x_grid, **kwargs)

    monkeypatch.setattr(multivariate, "pdf_and_grad_grid", observe)
    output = list(model.pdf_and_grad_on_grid_batches(
        data, grid, batch_rows=3, n_threads=1,
        memory_budget_bytes=3 * len(grid) * 16,
    ))
    assert len(blocks) == 4
    for column in (0, 1):
        np.testing.assert_array_equal(np.concatenate([b[column] for b in output]),
                                      expected[column])
