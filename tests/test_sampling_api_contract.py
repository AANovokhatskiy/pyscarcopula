"""Cross-model contracts for fitted and parameter-level sampling."""

import inspect

import numpy as np

from pyscarcopula import (
    CVineCopula,
    EquicorrGaussianCopula,
    GaussianCopula,
    GumbelCopula,
    RVineCopula,
    StochasticStudentCopula,
    StudentCopula,
)
from pyscarcopula.copula.base import BivariateCopula


SAMPLE_PARAMETERS = ("self", "n", "u", "rng")
PREDICT_PREFIX = (
    "self",
    "n",
    "u",
    "rng",
    "given",
    "horizon",
    "predictive_r_mode",
    "predict_config",
)


def test_sample_has_one_fitted_model_signature_across_model_types():
    classes = (
        GumbelCopula,
        GaussianCopula,
        StudentCopula,
        EquicorrGaussianCopula,
        StochasticStudentCopula,
        CVineCopula,
        RVineCopula,
    )

    for cls in classes:
        assert tuple(inspect.signature(cls.sample).parameters) == (
            SAMPLE_PARAMETERS)


def test_predict_has_common_arguments_in_common_order():
    classes = (
        GumbelCopula,
        GaussianCopula,
        StudentCopula,
        EquicorrGaussianCopula,
        StochasticStudentCopula,
        CVineCopula,
        RVineCopula,
    )

    for cls in classes:
        parameters = tuple(inspect.signature(cls.predict).parameters)
        assert parameters[:len(PREDICT_PREFIX)] == PREDICT_PREFIX


def test_bivariate_parameter_sampling_is_explicit():
    assert not hasattr(BivariateCopula, "sample_model")
    assert tuple(
        inspect.signature(BivariateCopula.sample_at_parameter).parameters
    ) == ("self", "n", "r", "rng")

    copula = GumbelCopula()
    first = copula.sample_at_parameter(
        32, 1.7, rng=np.random.default_rng(27))
    second = copula.sample_at_parameter(
        32, 1.7, rng=np.random.default_rng(27))

    np.testing.assert_array_equal(first, second)


def test_bivariate_sample_reproduces_fitted_model():
    copula = GumbelCopula()
    training = copula.sample_at_parameter(
        80, 1.6, rng=np.random.default_rng(31))
    copula.fit(training, method="mle")

    samples = copula.sample(20, rng=np.random.default_rng(32))

    assert samples.shape == (20, 2)
    assert np.all((samples > 0.0) & (samples < 1.0))
