"""Positive and negative mypy fixtures for static correlation APIs."""

import numpy as np
from typing_extensions import assert_type

from pyscarcopula import (
    CorrelationEstimator,
    CorrelationMode,
    CorrelationPolicy,
    FactorEstimation,
    FloatArray,
    GaussianCopula,
    MultivariateMLEResult,
    StochasticStudentCopula,
    StudentCopula,
)


observations: FloatArray = np.full((8, 3), 0.5, dtype=np.float64)
mode: CorrelationMode = "fixed"
estimator: CorrelationEstimator = "gaussian_score"
factor_estimation: FactorEstimation = "two-stage"

gaussian = GaussianCopula(d=3, corr_mode=mode)
student = StudentCopula(corr_mode=mode)
stochastic = StochasticStudentCopula(3, corr_mode=mode)

assert_type(gaussian.corr_mode, CorrelationMode)
assert_type(student.corr_mode, CorrelationMode)
assert_type(stochastic.corr_mode, CorrelationMode)
assert_type(gaussian.corr_estimator_, CorrelationEstimator)
assert_type(student.corr_estimator_, CorrelationEstimator)
assert_type(stochastic.factor_estimation, FactorEstimation)
assert_type(gaussian.fit(observations), MultivariateMLEResult)
assert_type(student.fit(observations), MultivariateMLEResult)
assert_type(
    CorrelationPolicy.create(
        mode=mode,
        estimator=estimator,
        dimension=3,
        base_correlation=np.eye(3),
    ),
    CorrelationPolicy,
)

# The ignores are intentional negative fixtures. ``warn_unused_ignores`` makes
# this gate fail if these constructor arguments ever become untyped.
GaussianCopula(corr_mode="invalid")  # type: ignore[arg-type]
StudentCopula(corr_mode="dense")  # type: ignore[arg-type]
StochasticStudentCopula(3, factor_estimation="profile")  # type: ignore[arg-type]
