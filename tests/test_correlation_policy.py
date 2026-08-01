"""Stage-1 contracts for typed static correlation policies."""

from dataclasses import FrozenInstanceError
from typing import get_type_hints

import numpy as np
import pytest

from pyscarcopula import (
    CorrelationPolicy,
    GaussianCopula,
    StochasticStudentCopula,
    StudentCopula,
)
from pyscarcopula.copula.multivariate.corr_param import (
    estimate_kendall_correlation,
)
from pyscarcopula.copula.multivariate.correlation_policy import (
    normalize_correlation_mode,
)


def _correlation(rho=0.4):
    return np.array(
        [[1.0, rho, -0.1], [rho, 1.0, 0.2], [-0.1, 0.2, 1.0]],
        dtype=np.float64,
    )


def test_mode_normalization_is_case_insensitive_and_dense_is_deprecated():
    assert normalize_correlation_mode("CHOLESKY") == "cholesky"
    with pytest.warns(DeprecationWarning, match="fixed"):
        model = GaussianCopula(corr_mode="DENSE")
    assert model.corr_mode == "fixed"

    with pytest.raises(TypeError, match="string"):
        normalize_correlation_mode(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="corr_mode"):
        normalize_correlation_mode("profile")


@pytest.mark.parametrize(
    "mode,estimator,factor_rank,factor_estimation,optimized,plugin",
    [
        ("fixed", "supplied", None, None, 0, 0),
        ("fixed", "gaussian_score", None, None, 0, 3),
        ("fixed", "kendall_plugin", None, None, 0, 3),
        ("shrinkage", "joint_mle", None, None, 1, 0),
        ("cholesky", "joint_mle", None, None, 3, 0),
        ("factor", "factor_two_stage", 2, "two-stage", 0, 5),
        ("factor", "factor_joint", 2, "joint", 5, 0),
    ],
)
def test_policy_derives_canonical_parameter_counts(
        mode, estimator, factor_rank, factor_estimation, optimized, plugin):
    policy = CorrelationPolicy.create(
        mode=mode,
        estimator=estimator,
        dimension=3,
        base_correlation=(
            None if mode == "factor" else _correlation()),
        factor_rank=factor_rank,
        factor_estimation=factor_estimation,
    )

    assert policy.optimized_n_params == optimized
    assert policy.plugin_n_params == plugin
    assert policy.effective_n_params == optimized + plugin
    assert policy.diagnostics()["corr_estimator"] == estimator


def test_direct_policy_construction_derives_counts_and_rejects_mismatch():
    policy = CorrelationPolicy(
        mode="fixed",
        estimator="kendall_plugin",
        dimension=3,
    )
    assert policy.optimized_n_params == 0
    assert policy.plugin_n_params == 3
    assert policy.effective_n_params == 3

    with pytest.raises(ValueError, match="incompatible"):
        CorrelationPolicy(
            mode="fixed",
            estimator="joint_mle",
            dimension=3,
        )


def test_policy_owns_read_only_arrays_and_uses_initialization_priority():
    observations = np.array(
        [[0.1, 0.2, 0.7], [0.3, 0.8, 0.4], [0.9, 0.6, 0.2]])
    preprocessing = estimate_kendall_correlation(observations)
    supplied = _correlation(0.25)
    base = _correlation(0.55)
    policy = CorrelationPolicy.create(
        mode="cholesky",
        estimator="joint_mle",
        dimension=3,
        supplied_correlation=supplied,
        base_correlation=base,
        preprocessing=preprocessing,
    )

    supplied[0, 1] = -0.9
    base[0, 1] = -0.8
    np.testing.assert_array_equal(policy.initial_correlation, _correlation(0.55))
    assert policy.base_correlation.flags.writeable is False
    assert policy.preprocessing.correlation.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        policy.base_correlation[0, 1] = 0.0
    with pytest.raises(FrozenInstanceError):
        policy.mode = "fixed"


@pytest.mark.parametrize("mode", ["shrinkage", "cholesky"])
def test_policy_trial_gradient_matches_central_difference(mode):
    policy = CorrelationPolicy.create(
        mode=mode,
        estimator="joint_mle",
        dimension=3,
        base_correlation=_correlation(),
    )
    raw = policy.initial_raw_parameters()
    if mode == "cholesky":
        raw = raw + np.array([0.04, -0.03, 0.02])
    gradient = np.array([0.7, -0.2, 0.4])

    analytical = policy.raw_gradient(
        raw, policy.trial_correlation(raw), gradient)
    step = 1e-6
    finite_difference = np.empty_like(raw)
    for index in range(raw.size):
        plus = raw.copy()
        minus = raw.copy()
        plus[index] += step
        minus[index] -= step
        upper = policy.trial_correlation(plus)[np.tril_indices(3, -1)]
        lower = policy.trial_correlation(minus)[np.tril_indices(3, -1)]
        finite_difference[index] = (
            np.dot(gradient, upper) - np.dot(gradient, lower)
        ) / (2.0 * step)

    np.testing.assert_allclose(
        analytical, finite_difference, rtol=0.0, atol=2e-10)


def test_static_models_publish_canonical_policy_metadata():
    rng = np.random.default_rng(401)
    observations = rng.uniform(0.05, 0.95, size=(30, 3))

    gaussian = GaussianCopula()
    gaussian_result = gaussian.fit(observations)
    assert gaussian.corr_mode == "fixed"
    assert gaussian.corr_estimator_ == "gaussian_score"
    assert gaussian_result.model_parameters["corr_mode"] == "fixed"
    assert gaussian_result.diagnostics["corr_effective_n_params"] == 3

    student = StudentCopula(corr_mode="FIXED")
    student_result = student.fit(observations)
    assert student.corr_estimator_ == "kendall_plugin"
    assert student_result.model_parameters["corr_estimator"] == (
        "kendall_plugin")
    assert student_result.diagnostics["corr_plugin_n_params"] == 3


def test_static_gaussian_accepts_joint_correlation_modes():
    assert GaussianCopula(corr_mode="shrinkage").corr_mode == "shrinkage"
    assert GaussianCopula(corr_mode="cholesky").corr_mode == "cholesky"


def test_static_student_accepts_joint_correlation_modes():
    assert StudentCopula(corr_mode="shrinkage").corr_mode == "shrinkage"
    assert StudentCopula(corr_mode="cholesky").corr_mode == "cholesky"


def test_stochastic_student_exposes_shared_typed_terminology():
    supplied = StochasticStudentCopula(3, R=_correlation())
    assert supplied.corr_estimator_ == "supplied"
    assert supplied.correlation_policy_.effective_n_params == 0

    plugin = StochasticStudentCopula(3)
    assert plugin.corr_estimator_ == "kendall_plugin"

    shrinkage = StochasticStudentCopula(3, corr_mode="SHRINKAGE")
    assert shrinkage.corr_mode == "shrinkage"
    assert shrinkage.corr_estimator_ == "joint_mle"
    assert shrinkage.correlation_policy_.optimized_n_params == 1

    factor = StochasticStudentCopula(
        4, corr_mode="factor", factor_rank=2,
        factor_estimation="JOINT")
    assert factor.factor_estimation == "joint"
    assert factor.corr_estimator_ == "factor_joint"
    assert factor.correlation_policy_.optimized_n_params == 7


@pytest.mark.parametrize(
    "callable_object",
    [
        GaussianCopula.__init__,
        GaussianCopula.fit,
        StudentCopula.__init__,
        StudentCopula.fit,
        StochasticStudentCopula.__init__,
        StochasticStudentCopula.fit,
        GaussianCopula.corr_mode.fget,
        StudentCopula.corr_estimator_.fget,
        StochasticStudentCopula.factor_estimation.fget,
    ],
)
def test_public_static_correlation_api_has_runtime_type_hints(callable_object):
    hints = get_type_hints(callable_object)
    assert hints
    assert "return" in hints
