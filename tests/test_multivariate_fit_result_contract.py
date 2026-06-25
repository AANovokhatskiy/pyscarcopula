"""Unified typed-result contract for static multivariate MLE."""

import numpy as np
import pytest

from pyscarcopula import (
    EquicorrGaussianCopula,
    GaussianCopula,
    MultivariateMLEResult,
    StochasticStudentCopula,
    StudentCopula,
)
from pyscarcopula._utils import pobs


def _u(n=50, d=3):
    return pobs(np.random.default_rng(20260624).normal(size=(n, d)))


@pytest.mark.parametrize(
    ("factory", "fit_kwargs", "expected_parameters"),
    [
        (GaussianCopula, {}, 3),
        (StudentCopula, {}, 4),
        (
            lambda: EquicorrGaussianCopula(d=3),
            {"method": "mle", "maxiter": 5},
            1,
        ),
        (
            lambda: StochasticStudentCopula(d=3),
            {"method": "mle", "maxiter": 5},
            4,
        ),
    ],
)
def test_static_multivariate_fit_returns_one_typed_contract(
        factory, fit_kwargs, expected_parameters):
    model = factory()
    observations = _u()

    result = model.fit(observations, **fit_kwargs)

    assert isinstance(result, MultivariateMLEResult)
    assert result is model.fit_result
    assert result.method == "MLE"
    assert result.n_observations == len(observations)
    assert result.parameter_count == expected_parameters
    assert result.correlation_matrix.shape == (3, 3)
    assert np.all(np.isfinite(result.correlation_matrix))
    assert np.isfinite(result.log_likelihood)
    assert result.aic == pytest.approx(
        2.0 * result.n_params - 2.0 * result.log_likelihood)
    assert result.bic == pytest.approx(
        np.log(len(observations)) * result.n_params
        - 2.0 * result.log_likelihood)


@pytest.mark.parametrize("factory", [GaussianCopula, StudentCopula])
def test_static_multivariate_result_persistence_roundtrip(factory, tmp_path):
    model = factory()
    result = model.fit(_u())
    path = tmp_path / f"{type(model).__name__}.json"

    model.save(path)
    loaded = type(model).load(path)

    assert isinstance(loaded.fit_result, MultivariateMLEResult)
    assert loaded.fit_result.parameter_count == result.parameter_count
    assert loaded.fit_result.n_observations == result.n_observations
    assert loaded.fit_result.aic == pytest.approx(result.aic)
    assert loaded.fit_result.bic == pytest.approx(result.bic)
    np.testing.assert_allclose(
        loaded.fit_result.correlation_matrix,
        result.correlation_matrix,
    )


@pytest.mark.parametrize("factory", [GaussianCopula, StudentCopula])
def test_static_sampling_reads_parameters_from_typed_result(factory):
    model = factory()
    result = model.fit(_u())
    expected = model.sample(12, rng=np.random.default_rng(7))

    if isinstance(model, GaussianCopula):
        model.corr = np.eye(3)
    else:
        model.shape = np.eye(3)
        model.df = 100.0

    actual = model.sample(12, rng=np.random.default_rng(7))

    assert model.fit_result is result
    np.testing.assert_array_equal(actual, expected)
