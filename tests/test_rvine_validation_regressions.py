"""Regression tests for RVine public-boundary validation and memory bounds."""

import numpy as np
import pytest

from pyscarcopula import (
    BivariateGaussianCopula,
    IndependentCopula,
    RVineCopula,
)
from pyscarcopula.vine._rvine_dissmann import select_rvine


def _independent_vine():
    data = np.array([
        [0.15, 0.25],
        [0.35, 0.45],
        [0.55, 0.65],
        [0.75, 0.85],
    ])
    return RVineCopula(candidates=[IndependentCopula]).fit(data)


def _gaussian_vine():
    rng = np.random.default_rng(12)
    first = rng.standard_normal(250)
    second = 0.7 * first + np.sqrt(1.0 - 0.7 ** 2) * rng.standard_normal(250)
    order_first = np.argsort(np.argsort(first)) + 1
    order_second = np.argsort(np.argsort(second)) + 1
    data = np.column_stack((order_first, order_second)) / 251.0
    return RVineCopula(
        candidates=[BivariateGaussianCopula],
        allow_rotations=False,
    ).fit(
        data,
        copulas=[[(BivariateGaussianCopula, 0)]],
    )


@pytest.mark.parametrize(
    "bad_data",
    [
        np.array([[0.2, np.nan], [0.4, 0.5]]),
        np.array([[0.2, np.inf], [0.4, 0.5]]),
        np.array([[-0.1, 0.2], [0.4, 0.5]]),
        np.array([[1.1, 0.2], [0.4, 0.5]]),
        np.array([[0.0, 0.2], [0.4, 0.5]]),
        np.array([[1.0, 0.2], [0.4, 0.5]]),
        np.empty((0, 2)),
    ],
)
def test_fit_rejects_invalid_pseudo_observations_before_selection(bad_data):
    with pytest.raises(ValueError):
        RVineCopula(candidates=[IndependentCopula]).fit(bad_data)


def test_fit_rejects_complex_data_without_lossy_cast():
    data = np.array([
        [0.2 + 1.0j, 0.3],
        [0.4, 0.5],
    ])
    with pytest.raises(TypeError, match="real values"):
        RVineCopula(candidates=[IndependentCopula]).fit(data)


@pytest.mark.parametrize(
    "bad_data",
    [
        np.array([[0.2, np.nan]]),
        np.array([[0.2, np.inf]]),
        np.array([[-0.1, 0.2]]),
        np.array([[1.1, 0.2]]),
        np.empty((0, 2)),
    ],
)
def test_explicit_likelihood_rejects_invalid_pseudo_observations(bad_data):
    with pytest.raises(ValueError):
        _independent_vine().log_likelihood(bad_data)


def test_predict_rejects_invalid_explicit_history():
    vine = _independent_vine()
    with pytest.raises(ValueError, match="finite"):
        vine.predict(4, u=np.array([[0.2, np.nan]]))
    with pytest.raises(ValueError, match="pseudo-observations"):
        vine.predict(4, u=np.array([[-0.1, 0.2]]))


def test_direct_selector_rejects_invalid_data():
    with pytest.raises(ValueError, match="finite"):
        select_rvine(np.array([[0.2, np.inf], [0.4, 0.5]]))
    with pytest.raises(ValueError, match="at least one row"):
        select_rvine(np.empty((0, 2)))


def test_static_sampling_batches_without_changing_seeded_output():
    vine = _independent_vine()
    expected = vine.sample(
        31,
        rng=np.random.default_rng(20260726),
        batch_rows=31,
    )
    actual = vine.sample(
        31,
        rng=np.random.default_rng(20260726),
        batch_rows=7,
    )
    np.testing.assert_array_equal(actual, expected)


def test_static_sampling_passes_scalar_edge_parameter_paths(monkeypatch):
    vine = _gaussian_vine()
    original = vine._sample_with_r
    observed = []

    def recording_sample(n, r_all, rng, **kwargs):
        observed.append((
            n,
            tuple(np.asarray(value).size for value in r_all.values()),
        ))
        return original(n, r_all, rng, **kwargs)

    monkeypatch.setattr(vine, "_sample_with_r", recording_sample)
    samples = vine.sample(
        17,
        rng=np.random.default_rng(3),
        batch_rows=6,
    )

    assert samples.shape == (17, 2)
    assert [rows for rows, _ in observed] == [6, 6, 5]
    assert all(sizes == (1,) for _, sizes in observed)


def test_sample_memory_budget_is_checked_before_allocation():
    vine = _independent_vine()
    with pytest.raises(MemoryError, match="memory_budget_bytes"):
        vine.sample(
            100,
            batch_rows=10,
            memory_budget_bytes=100,
            rng=np.random.default_rng(4),
        )


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_sample_rejects_invalid_batch_rows(value):
    vine = _independent_vine()
    with pytest.raises((TypeError, ValueError)):
        vine.sample(10, batch_rows=value)
