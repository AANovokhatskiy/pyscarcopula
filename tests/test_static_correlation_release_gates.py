"""Final correctness, safety, and ownership gates for static correlation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import multivariate_t, norm, t as t_dist

from pyscarcopula import GaussianCopula, StudentCopula
from pyscarcopula._native import static as static_likelihood


def _valid_observations(rows=80, seed=1501):
    return np.random.default_rng(seed).uniform(0.02, 0.98, size=(rows, 3))


@pytest.mark.parametrize("model_type", [GaussianCopula, StudentCopula])
@pytest.mark.parametrize("layout", ["c", "f", "strided", "readonly"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_valid_numeric_layouts_are_owned_and_not_mutated(
        model_type, layout, dtype):
    base = _valid_observations().astype(dtype)
    if layout == "f":
        observations = np.asfortranarray(base)
    elif layout == "strided":
        backing = np.empty((len(base), 6), dtype=dtype)
        backing[:, ::2] = base
        backing[:, 1::2] = -1
        observations = backing[:, ::2]
    else:
        observations = np.ascontiguousarray(base)
    if layout == "readonly":
        observations.setflags(write=False)
    before = observations.copy()

    result = model_type().fit(observations)

    assert result.success
    np.testing.assert_array_equal(observations, before)
    assert result.correlation_matrix.dtype == np.float64


@pytest.mark.parametrize("model_type", [GaussianCopula, StudentCopula])
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_integer_boundary_data_is_accepted_without_mutation(model_type, dtype):
    observations = np.array([
        [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
    ], dtype=dtype)
    before = observations.copy()

    result = model_type().fit(observations)

    assert result.success
    assert np.isfinite(result.log_likelihood)
    np.testing.assert_array_equal(observations, before)


@pytest.mark.parametrize("model_type", [GaussianCopula, StudentCopula])
@pytest.mark.parametrize(
    "observations",
    [
        np.array(0.5),
        np.array([0.1, 0.2, 0.3]),
        np.empty((0, 3)),
        np.ones((5, 1)) * 0.5,
        np.array([[0.2, 0.3, 0.4]]),
        np.array([[0.2, np.nan], [0.4, 0.6]]),
        np.array([[0.2, np.inf], [0.4, 0.6]]),
        np.array([[0.2, -np.inf], [0.4, 0.6]]),
        np.array([[0.2, -0.01], [0.4, 0.6]]),
        np.array([[0.2, 1.01], [0.4, 0.6]]),
        np.full((5, 3), 0.5),
    ],
    ids=[
        "scalar", "one-dimensional", "empty", "one-column", "singleton",
        "nan", "positive-inf", "negative-inf", "below-zero", "above-one",
        "constant",
    ],
)
def test_invalid_shape_value_matrix_is_rejected_without_state(
        model_type, observations):
    model = model_type()

    with pytest.raises((TypeError, ValueError)):
        model.fit(observations)

    assert model.fit_result is None
    assert getattr(model, "_last_u", None) is None


@pytest.mark.parametrize("model_type", [GaussianCopula, StudentCopula])
@pytest.mark.parametrize("kind", ["object", "string", "complex", "bool"])
def test_non_real_numeric_dtype_matrix_is_rejected(model_type, kind):
    observations = _valid_observations(rows=12)
    if kind == "object":
        observations = observations.astype(object)
    elif kind == "string":
        observations = observations.astype(str)
    elif kind == "complex":
        observations = observations.astype(complex) + 1j
    else:
        observations = observations > 0.5

    with pytest.raises((TypeError, ValueError), match="numeric|real-valued"):
        model_type().fit(observations)


@pytest.mark.parametrize("model_type", [GaussianCopula, StudentCopula])
def test_duplicate_variables_are_rejected_before_fit(model_type):
    observations = _valid_observations()
    observations[:, 1] = observations[:, 0]

    with pytest.raises(ValueError, match="duplicate"):
        model_type(R=np.eye(3)).fit(observations)


@pytest.mark.parametrize(
    "correlation",
    [
        np.ones((2, 3)),
        np.array([[1.0, np.nan], [np.nan, 1.0]]),
    ],
    ids=["non-square", "nan"],
)
@pytest.mark.parametrize("model_type", [GaussianCopula, StudentCopula])
def test_unprojectable_supplied_correlation_matrix_is_rejected(
        correlation, model_type):
    with pytest.raises(ValueError):
        model_type(R=correlation)


@pytest.mark.parametrize(
    "correlation",
    [
        np.array([[1.0, 0.4], [0.2, 1.0]]),
        np.array([[1.0, 0.2], [0.2, 0.9]]),
        np.array([[1.0, 1.1], [1.1, 1.0]]),
    ],
    ids=["non-symmetric", "non-unit-diagonal", "non-spd"],
)
def test_gaussian_rejects_invalid_finite_supplied_correlation(correlation):
    with pytest.raises(ValueError):
        GaussianCopula(R=correlation)


@pytest.mark.parametrize(
    "correlation",
    [
        np.array([[1.0, 0.4], [0.2, 1.0]]),
        np.array([[1.0, 0.2], [0.2, 0.9]]),
        np.array([[1.0, 1.1], [1.1, 1.0]]),
    ],
    ids=["non-symmetric", "non-unit-diagonal", "non-spd"],
)
def test_student_projects_invalid_finite_supplied_correlation(correlation):
    model = StudentCopula(R=correlation)

    projected = model._supplied_correlation
    np.testing.assert_allclose(projected, projected.T, atol=1e-14)
    np.testing.assert_allclose(np.diag(projected), 1.0, atol=0.0)
    assert np.min(np.linalg.eigvalsh(projected)) > 0.0
    assert model._supplied_preprocessing.projection_applied


def _dependent_sample(model_type, rows, seed):
    correlation = np.array([
        [1.0, 0.48, 0.08],
        [0.48, 1.0, -0.22],
        [0.08, -0.22, 1.0],
    ])
    rng = np.random.default_rng(seed)
    if model_type is GaussianCopula:
        latent = rng.multivariate_normal(np.zeros(3), correlation, size=rows)
        return norm.cdf(latent)
    latent = multivariate_t.rvs(
        np.zeros(3), correlation, df=6.0, size=rows, random_state=rng)
    return t_dist.cdf(latent, df=6.0)


@pytest.mark.parametrize("model_type", [GaussianCopula, StudentCopula])
@pytest.mark.parametrize("corr_mode", ["shrinkage", "cholesky"])
@pytest.mark.parametrize("rows,seed", [(60, 1511), (180, 1512)])
def test_joint_fit_multi_seed_acceptance_spd_and_gradient_gate(
        model_type, corr_mode, rows, seed):
    observations = _dependent_sample(model_type, rows, seed)
    result = model_type(corr_mode=corr_mode).fit(
        observations, gtol=1e-5, maxiter=500)

    assert result.success, result.message
    correlation = result.correlation_matrix
    np.testing.assert_allclose(correlation, correlation.T, atol=1e-14)
    np.testing.assert_allclose(np.diag(correlation), 1.0, atol=0.0)
    assert np.min(np.linalg.eigvalsh(correlation)) > 0.0
    assert result.diagnostics["final_objective"] <= (
        result.diagnostics["initial_objective"] + 1e-8)
    assert result.diagnostics["final_gradient_inf_norm"] <= (
        result.diagnostics["gradient_gate"])


def test_student_cholesky_gaussian_limit_stays_numerically_conditioned():
    observations = _dependent_sample(StudentCopula, 60, 1511)
    result = StudentCopula(corr_mode="cholesky").fit(
        observations, gtol=1e-5, maxiter=500)

    assert result.success, result.message
    assert result.model_parameters["df"] <= 10_000.0
    assert result.diagnostics["final_gradient_inf_norm"] <= (
        result.diagnostics["gradient_gate"])


def test_gaussian_and_student_independence_limits():
    observations = _valid_observations(rows=45, seed=1513)
    identity = np.eye(3)
    gaussian = static_likelihood.prepare_gaussian(identity, observations)
    np.testing.assert_allclose(
        gaussian.log_pdf_rows(0.0), 0.0, rtol=0.0, atol=2e-13)

    student = static_likelihood.prepare_student(identity, observations)
    rows = student.log_pdf_rows(1e7)
    np.testing.assert_allclose(rows, 0.0, rtol=0.0, atol=2e-5)


@pytest.mark.parametrize("model_type", [GaussianCopula, StudentCopula])
def test_joint_trial_evaluation_never_publishes_candidate_state(
        model_type, monkeypatch):
    observations = _dependent_sample(model_type, 90, 1514)
    model = model_type(corr_mode="cholesky")
    first = model.fit(observations, maxiter=300)
    before = model.to_correlation_matrix()
    module_name = (
        "pyscarcopula.copula.multivariate.gaussian"
        if model_type is GaussianCopula
        else "pyscarcopula.copula.multivariate.student")

    def inspect_trial(problem, **kwargs):
        trial = np.asarray(problem.initial_parameters).copy()
        trial[-1] += 0.05
        problem.evaluate(trial)
        np.testing.assert_array_equal(model.to_correlation_matrix(), before)
        assert model.fit_result is first
        raise RuntimeError("stop after ownership check")

    monkeypatch.setattr(
        f"{module_name}.run_static_multivariate_mle", inspect_trial)
    with pytest.raises(RuntimeError, match="ownership check"):
        model.fit(observations)

    np.testing.assert_array_equal(model.to_correlation_matrix(), before)
    assert model.fit_result is first


@pytest.mark.parametrize("model_type", [GaussianCopula, StudentCopula])
def test_factor_refit_replaces_transient_operator_without_dense_state(
        model_type):
    observations = _valid_observations(rows=70, seed=1515)
    model = model_type(3, corr_mode="factor", factor_rank=1)
    first = model.fit(observations)
    first_operator = model._factor_operator

    second = model.fit(observations[::-1].copy())

    assert first.success and second.success
    assert model._factor_operator is not first_operator
    assert second.correlation_matrix is None
    if model_type is StudentCopula:
        assert model.correlation is None
    else:
        assert model.corr is None
