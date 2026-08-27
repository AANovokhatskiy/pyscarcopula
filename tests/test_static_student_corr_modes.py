from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import multivariate_t, t as t_dist

from pyscarcopula import StudentCopula
from pyscarcopula._types import NumericalConfig
from pyscarcopula.copula.multivariate.correlation_policy import CorrelationPolicy
from pyscarcopula._native import static as static_likelihood


def _sample(seed=17, n=350):
    rng = np.random.default_rng(seed)
    correlation = np.array([
        [1.0, 0.52, 0.12],
        [0.52, 1.0, -0.24],
        [0.12, -0.24, 1.0],
    ])
    latent = multivariate_t.rvs(
        np.zeros(3), correlation, df=6.0, size=n, random_state=rng)
    return t_dist.cdf(latent, df=6.0), correlation


def test_fixed_supplied_and_kendall_plugin_have_distinct_contracts():
    u, supplied = _sample()
    supplied_model = StudentCopula(R=supplied)
    supplied_result = supplied_model.fit(u)
    plugin_model = StudentCopula()
    plugin_result = plugin_model.fit(u)

    assert supplied_result.success
    assert plugin_result.success
    assert supplied_result.model_parameters["corr_estimator"] == "supplied"
    assert plugin_result.model_parameters["corr_estimator"] == "kendall_plugin"
    assert supplied_result.parameter_count == 1
    assert plugin_result.parameter_count == 4
    assert supplied_result.diagnostics["corr_initialization_source"] == (
        "supplied")
    assert plugin_result.diagnostics["corr_initialization_source"] == (
        "kendall")
    np.testing.assert_allclose(
        supplied_result.correlation_matrix, supplied, atol=1e-12)


@pytest.mark.parametrize("corr_mode", ["shrinkage", "cholesky"])
def test_dense_joint_modes_fit_and_do_not_lose_initial_likelihood(corr_mode):
    u, _ = _sample(seed=21)
    model = StudentCopula(corr_mode=corr_mode)
    result = model.fit(u, gtol=1e-4, maxiter=500)

    assert result.success, result.message
    assert result.model_parameters["corr_estimator"] == "joint_mle"
    assert result.diagnostics["not_worse_than_initial"]
    assert result.diagnostics["correlation_gradient"] == "analytical"
    assert result.diagnostics["gradient_mode"] == "analytical_joint"
    assert result.diagnostics["joint_static"] is True
    np.linalg.cholesky(result.correlation_matrix)
    assert np.allclose(np.diag(result.correlation_matrix), 1.0)


def test_cholesky_joint_gradient_matches_independent_central_difference():
    u, correlation = _sample(seed=9, n=120)
    policy = CorrelationPolicy.create(
        mode="cholesky", estimator="joint_mle", dimension=3,
        base_correlation=correlation)
    raw = policy.initial_raw_parameters()
    df = 6.5

    def objective(raw_parameters):
        trial = policy.trial_correlation(raw_parameters)
        return static_likelihood.prepare_student(
            trial, u).objective_and_gradient(df)[0]

    trial = policy.trial_correlation(raw)
    _, _, native_corr_gradient = static_likelihood.prepare_student(
        trial, u).objective_and_joint_gradient(df)
    analytical = policy.raw_gradient(raw, trial, native_corr_gradient)
    numerical = np.empty_like(raw)
    step = 1e-6
    for index in range(raw.size):
        plus = raw.copy()
        minus = raw.copy()
        plus[index] += step
        minus[index] -= step
        numerical[index] = (objective(plus) - objective(minus)) / (2 * step)
    np.testing.assert_allclose(analytical, numerical, rtol=2e-5, atol=2e-5)


def test_config_forwarding_unknown_kwargs_and_atomic_rollback(monkeypatch):
    u, _ = _sample(seed=31)
    model = StudentCopula(corr_mode="cholesky")
    first = model.fit(u, config=NumericalConfig(n_threads=2))
    assert first.success
    assert first.diagnostics["n_threads"] == 2
    before_shape = model.shape
    before_df = model.df
    before_result = model.fit_result
    before_u = model._last_u.copy()

    with pytest.raises(TypeError, match="unexpected MLE keyword"):
        model.fit(u, definitely_unknown=True)

    import pyscarcopula.copula.multivariate.student as student_module
    monkeypatch.setattr(
        student_module, "run_static_multivariate_mle",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        model.fit(u)
    np.testing.assert_array_equal(model.shape, before_shape)
    np.testing.assert_array_equal(model._last_u, before_u)
    assert model.df == before_df
    assert model.fit_result is before_result


def test_input_dimension_and_copy_contracts():
    u, _ = _sample()
    with pytest.raises(ValueError, match="real-valued"):
        StudentCopula().fit(u.astype(complex) + 1j)
    constant = u.copy()
    constant[:, 1] = 0.5
    with pytest.raises(ValueError, match="not identifiable"):
        StudentCopula().fit(constant)
    with pytest.raises(ValueError, match="pseudo-observations"):
        StudentCopula().fit(u * 2.0)

    model = StudentCopula(d=3)
    assert model.fit(u).success
    with pytest.raises(ValueError, match="3 columns"):
        model.fit(u[:, :2])
    exposed = model.shape
    exposed[0, 1] = -0.99
    assert model.shape[0, 1] != -0.99
    result_corr = model.fit_result.correlation_matrix
    result_corr[0, 1] = -0.88
    assert model.shape[0, 1] != -0.88


@pytest.mark.parametrize("estimation", ["two-stage", "joint"])
def test_factor_modes_return_compact_results(estimation):
    u, _ = _sample(seed=41, n=400)
    model = StudentCopula(
        d=3, corr_mode="factor", factor_rank=1,
        factor_estimation=estimation)
    result = model.fit(u, gtol=1e-4, maxiter=500)

    assert result.success, result.message
    assert result.correlation_matrix is None
    assert result.model_parameters["factor_rank"] == 1
    assert result.model_parameters["factor_estimation"] == estimation
    assert result.model_parameters["factor_loadings"].shape == (3, 1)
    assert result.model_parameters["factor_uniqueness"].shape == (3,)
    assert result.parameter_count == 4
    assert result.diagnostics["corr_effective_n_params"] == 3
    assert result.diagnostics["corr_plugin_n_params"] == (
        3 if estimation == "two-stage" else 0)
    assert result.diagnostics["gradient_mode"] == (
        "analytical_df"
        if estimation == "two-stage" else "analytical_joint_factor")
    assert result.diagnostics["joint_static"] is (estimation == "joint")
    assert model.to_correlation_matrix().shape == (3, 3)


def test_constructor_guards_run_before_factor_allocation():
    with pytest.raises(ValueError, match="d <= 2"):
        StudentCopula(d=3, corr_mode="cholesky", cholesky_d_max=2)
    with pytest.raises(ValueError, match="factor_joint_max_params"):
        StudentCopula(
            d=10, corr_mode="factor", factor_rank=4,
            factor_estimation="joint", factor_joint_max_params=5)
