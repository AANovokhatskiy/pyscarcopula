"""Centered sufficient statistics survive preparation and persistence."""

import numpy as np
import pytest
from scipy.special import ndtri

from pyscarcopula import EquicorrGaussianCopula, EquicorrPreparedData


@pytest.mark.parametrize("storage", ["npz", "mmap"])
@pytest.mark.parametrize("legacy", [False, True])
def test_optional_centered_statistics_round_trip(tmp_path, storage, legacy):
    model = EquicorrGaussianCopula(3)
    prepared = model.prepare_sufficient_statistics(np.full((2, 3), 0.9))
    if legacy:
        prepared = EquicorrPreparedData(
            prepared.sum_z, prepared.sum_z2, 2, 3)
    if storage == "npz":
        restored = EquicorrPreparedData.load_npz(
            prepared.save_npz(tmp_path / "statistics.npz"))
    else:
        restored = EquicorrPreparedData.load_mmap(
            prepared.save_mmap(tmp_path / "statistics"))
    np.testing.assert_array_equal(restored.sum_z, prepared.sum_z)
    np.testing.assert_array_equal(restored.sum_z2, prepared.sum_z2)
    if legacy:
        assert restored.centered_squares is None
    else:
        np.testing.assert_array_equal(restored.centered_squares, [0.0, 0.0])
        assert not restored.centered_squares.flags.writeable
        if storage == "mmap":
            assert isinstance(restored.centered_squares, np.memmap)
        rho = np.nextafter(1.0, 0.0)
        np.testing.assert_array_equal(
            model.log_pdf_rows(restored, rho), model.log_pdf_rows(prepared, rho))


def test_centered_preparation_preserves_small_variance_across_tiles_and_threads():
    u = np.tile([0.9 - 1e-8, 0.9, 0.9 + 1e-8], (4, 2048))
    model = EquicorrGaussianCopula(u.shape[1])
    z = ndtri(u)
    expected = np.sum((z - z.mean(axis=1, keepdims=True))**2, axis=1)
    sequential = model.prepare_sufficient_statistics(
        u, dimension_tile=128, n_threads=1)
    parallel = model.prepare_sufficient_statistics(
        u, dimension_tile=128, n_threads=4)
    np.testing.assert_array_equal(
        sequential.centered_squares, parallel.centered_squares)
    np.testing.assert_allclose(
        sequential.centered_squares, expected, rtol=2e-7, atol=0.0)


@pytest.mark.parametrize("centered", [[-1.0], [np.nan], [np.inf], [0.0, 0.0], [0.5]])
def test_prepared_rejects_invalid_centered_statistics(centered):
    with pytest.raises(ValueError):
        EquicorrPreparedData([0.0], [1.0], 1, 3, centered_squares=centered)
