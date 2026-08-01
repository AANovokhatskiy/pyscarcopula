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
    ("factory", "fit_kwargs", "expected_parameters", "dense_correlation"),
    [
        (GaussianCopula, {}, 3, True),
        (StudentCopula, {}, 4, True),
        (
            lambda: EquicorrGaussianCopula(d=3),
            {"method": "mle", "maxiter": 5},
            1,
            False,
        ),
        (
            lambda: StochasticStudentCopula(d=3),
            {"method": "mle", "maxiter": 5},
            4,
            True,
        ),
    ],
)
def test_static_multivariate_fit_returns_one_typed_contract(
        factory, fit_kwargs, expected_parameters, dense_correlation):
    model = factory()
    observations = _u()

    result = model.fit(observations, **fit_kwargs)

    assert isinstance(result, MultivariateMLEResult)
    assert result is model.fit_result
    assert result.method == "MLE"
    assert result.n_observations == len(observations)
    assert result.parameter_count == expected_parameters
    if dense_correlation:
        assert result.correlation_matrix.shape == (3, 3)
        assert np.all(np.isfinite(result.correlation_matrix))
    else:
        assert result.correlation_matrix is None
        assert (
            result.diagnostics["correlation_representation"]
            == "equicorrelation_scalar"
        )
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


def test_zero_parameter_gaussian_persistence_and_information_criteria(
        tmp_path):
    correlation = np.array([
        [1.0, 0.35, -0.1],
        [0.35, 1.0, 0.2],
        [-0.1, 0.2, 1.0],
    ])
    model = GaussianCopula(R=correlation)
    result = model.fit(_u())
    path = tmp_path / "zero-parameter-gaussian.json"

    model.save(path)
    loaded = GaussianCopula.load(path)
    loaded_result = loaded.fit_result

    assert loaded_result.parameter_count == 0
    assert loaded_result.aic == pytest.approx(-2.0 * result.log_likelihood)
    assert loaded_result.bic == pytest.approx(-2.0 * result.log_likelihood)
    assert loaded_result.diagnostics["corr_effective_n_params"] == 0
    assert loaded_result.diagnostics["corr_initialization_source"] == (
        "supplied")
    assert loaded_result.diagnostics["final_validation_passed"] is True


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
