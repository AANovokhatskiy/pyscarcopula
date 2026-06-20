"""Public contracts for multivariate copula models."""

import numpy as np
import pytest

from pyscarcopula import (
    ClaytonCopula,
    EquicorrGaussianCopula,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    IndependentCopula,
    JoeCopula,
    StochasticStudentCopula,
    StudentCopula,
    MultivariateMLEResult,
)
from pyscarcopula._utils import pobs
from pyscarcopula.copula.elliptical import BivariateGaussianCopula


def test_gaussian_fit_return_and_state_contract():
    u = pobs(np.random.default_rng(20260620).standard_normal((60, 3)))
    copula = GaussianCopula()

    returned = copula.fit(u)

    assert isinstance(returned, MultivariateMLEResult)
    assert returned is copula.fit_result
    assert returned.correlation_matrix is copula.corr
    assert returned.correlation_matrix.shape == (3, 3)
    assert returned.copula_param is None
    assert returned.method == "MLE"
    assert returned.parameter_count == 3
    assert np.isfinite(returned.log_likelihood)
    assert np.isfinite(returned.aic)
    assert np.isfinite(returned.bic)


def test_student_fit_return_and_state_contract():
    u = pobs(np.random.default_rng(20260621).standard_normal((50, 3)))
    copula = StudentCopula()

    returned = copula.fit(u)

    assert isinstance(returned, MultivariateMLEResult)
    assert returned is copula.fit_result
    assert returned.correlation_matrix is copula.shape
    assert returned.copula_param == copula.df
    assert copula.shape.shape == (3, 3)
    assert returned.model_parameters["df"] == copula.df
    assert returned.method == "MLE"
    assert returned.parameter_count == 4
    assert np.isfinite(returned.log_likelihood)
    assert np.isfinite(returned.aic)
    assert np.isfinite(returned.bic)


@pytest.mark.parametrize(
    ("factory", "sample"),
    [
        (
            lambda: IndependentCopula(),
            lambda copula, rng: copula.sample_at_parameter(24, rng=rng),
        ),
        (
            lambda: ClaytonCopula(rotate=90),
            lambda copula, rng: copula.sample_at_parameter(24, 1.4, rng=rng),
        ),
        (
            lambda: FrankCopula(),
            lambda copula, rng: copula.sample_at_parameter(24, 3.0, rng=rng),
        ),
        (
            lambda: GumbelCopula(rotate=180),
            lambda copula, rng: copula.sample_at_parameter(24, 1.8, rng=rng),
        ),
        (
            lambda: JoeCopula(rotate=270),
            lambda copula, rng: copula.sample_at_parameter(24, 1.7, rng=rng),
        ),
        (
            lambda: BivariateGaussianCopula(),
            lambda copula, rng: copula.sample_at_parameter(24, 0.35, rng=rng),
        ),
        (
            lambda: EquicorrGaussianCopula(d=3),
            lambda copula, rng: copula.sample_at_parameter(24, 0.25, rng=rng),
        ),
        (
            lambda: StochasticStudentCopula(d=3, R=np.eye(3)),
            lambda copula, rng: copula.sample_at_parameter(24, 6.0, rng=rng),
        ),
    ],
    ids=[
        "independent",
        "clayton",
        "frank",
        "gumbel",
        "joe",
        "bivariate-gaussian",
        "equicorr-gaussian",
        "stochastic-student",
    ],
)
def test_parameterized_sampling_uses_passed_rng(factory, sample):
    copula = factory()

    first = sample(copula, np.random.default_rng(9123))
    second = sample(copula, np.random.default_rng(9123))
    different = sample(copula, np.random.default_rng(9124))

    np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)
    assert not np.allclose(first, different)


@pytest.mark.parametrize("kind", ["gaussian", "student"])
def test_static_multivariate_sampling_uses_passed_rng(kind):
    if kind == "gaussian":
        copula = GaussianCopula()
        copula.corr = np.array(
            [[1.0, 0.35, -0.1], [0.35, 1.0, 0.2], [-0.1, 0.2, 1.0]]
        )
    else:
        copula = StudentCopula()
        copula.shape = np.array(
            [[1.0, 0.25, -0.1], [0.25, 1.0, 0.15], [-0.1, 0.15, 1.0]]
        )
        copula.df = 6.5

    first = copula.sample(24, rng=np.random.default_rng(8123))
    second = copula.sample(24, rng=np.random.default_rng(8123))
    different = copula.sample(24, rng=np.random.default_rng(8124))

    np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)
    assert not np.allclose(first, different)
