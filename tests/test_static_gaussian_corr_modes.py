from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from pyscarcopula import GaussianCopula
from pyscarcopula._parallel import get_copula_constructor
from pyscarcopula._types import NumericalConfig
from pyscarcopula.numerical import static_likelihood


def _sample(seed=71, n=500):
    rng = np.random.default_rng(seed)
    correlation = np.array([
        [1.0, 0.55, 0.14],
        [0.55, 1.0, -0.28],
        [0.14, -0.28, 1.0],
    ])
    latent = rng.multivariate_normal(np.zeros(3), correlation, size=n)
    return norm.cdf(latent), correlation


def test_fixed_supplied_correlation_is_not_counted_as_fitted():
    u, supplied = _sample()
    model = GaussianCopula(R=supplied)
    result = model.fit(u, config=NumericalConfig(n_threads=2))

    assert result.success
    assert result.parameter_count == 0
    assert result.model_parameters["corr_estimator"] == "supplied"
    assert result.diagnostics["corr_n_params"] == 0
    assert result.diagnostics["corr_plugin_n_params"] == 0
    assert result.diagnostics["n_threads"] == 2
    np.testing.assert_array_equal(result.correlation_matrix, supplied)
    assert result.aic == pytest.approx(-2.0 * result.log_likelihood)


@pytest.mark.parametrize(
    "corr_mode,expected_count", [("shrinkage", 1), ("cholesky", 3)])
def test_joint_modes_fit_with_native_score_and_acceptance_gate(
        corr_mode, expected_count):
    u, _ = _sample(seed=73)
    score_result = GaussianCopula().fit(u)
    model = GaussianCopula(corr_mode=corr_mode)
    result = model.fit(u, gtol=1e-5, maxiter=500)

    assert result.success, result.message
    assert result.parameter_count == expected_count
    assert result.model_parameters["corr_estimator"] == "joint_mle"
    assert result.diagnostics["correlation_gradient"] == "analytical"
    assert result.diagnostics["not_worse_than_initial"]
    np.linalg.cholesky(result.correlation_matrix)
    np.testing.assert_allclose(np.diag(result.correlation_matrix), 1.0)
    if corr_mode == "cholesky":
        assert result.log_likelihood >= score_result.log_likelihood - 1e-8


def test_corr_base_precedes_supplied_correlation_for_joint_initialization():
    u, supplied = _sample(seed=79, n=220)
    base = np.array([
        [1.0, 0.12, 0.03],
        [0.12, 1.0, -0.08],
        [0.03, -0.08, 1.0],
    ])
    model = GaussianCopula(
        R=supplied, corr_base=base, corr_mode="cholesky")
    result = model.fit(u, maxiter=300)
    expected_initial = static_likelihood.prepare_gaussian(
        base, u).gaussian_objective_and_gradient(base)[0]

    assert result.diagnostics["initial_objective"] == pytest.approx(
        expected_initial, rel=2e-14, abs=2e-12)


def test_joint_fit_prepares_normal_scores_once(monkeypatch):
    u, _ = _sample(seed=83, n=260)
    real_prepare = static_likelihood.prepare_gaussian
    calls = {"prepare": 0, "trials": 0}

    def counted_prepare(*args, **kwargs):
        calls["prepare"] += 1
        evaluator = real_prepare(*args, **kwargs)
        real_trial = evaluator.gaussian_objective_and_gradient

        def counted_trial(*trial_args, **trial_kwargs):
            calls["trials"] += 1
            return real_trial(*trial_args, **trial_kwargs)

        evaluator.gaussian_objective_and_gradient = counted_trial
        return evaluator

    monkeypatch.setattr(static_likelihood, "prepare_gaussian", counted_prepare)
    result = GaussianCopula(corr_mode="cholesky").fit(
        u, maxiter=300)

    assert result.success
    assert calls["prepare"] == 1
    assert calls["trials"] > 1


def test_invalid_inputs_options_and_dimension_are_rejected_before_mutation():
    u, _ = _sample(seed=89)
    with pytest.raises(ValueError, match="real-valued"):
        GaussianCopula().fit(u.astype(complex) + 1j)
    with pytest.raises(ValueError, match="pseudo-observations"):
        GaussianCopula().fit(u * 2.0)
    with pytest.raises(TypeError, match="unexpected MLE keyword"):
        GaussianCopula().fit(u, unknown_option=True)
    with pytest.raises(TypeError, match="require corr_mode"):
        GaussianCopula().fit(u, maxiter=10)

    model = GaussianCopula(d=3, corr_mode="cholesky")
    assert model.fit(u).success
    before_corr = model.corr.copy()
    before_result = model.fit_result
    before_u = model._last_u.copy()
    with pytest.raises(ValueError, match="3 columns"):
        model.fit(u[:, :2])
    np.testing.assert_array_equal(model.corr, before_corr)
    np.testing.assert_array_equal(model._last_u, before_u)
    assert model.fit_result is before_result


def test_optimizer_exception_rolls_back_published_state(monkeypatch):
    u, _ = _sample(seed=97)
    model = GaussianCopula(corr_mode="shrinkage")
    first = model.fit(u)
    assert first.success
    before_corr = model.corr.copy()
    before_result = model.fit_result
    before_raw = model._corr_params_raw.copy()

    import pyscarcopula.copula.multivariate.gaussian as gaussian_module
    monkeypatch.setattr(
        gaussian_module, "run_static_multivariate_mle",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        model.fit(u)
    np.testing.assert_array_equal(model.corr, before_corr)
    np.testing.assert_array_equal(model._corr_params_raw, before_raw)
    assert model.fit_result is before_result


@pytest.mark.parametrize("corr_mode", ["fixed", "shrinkage", "cholesky"])
def test_worker_reconstruction_preserves_dense_correlation_policy(corr_mode):
    _, supplied = _sample()
    kwargs = {"corr_mode": corr_mode, "R": supplied}
    if corr_mode != "fixed":
        kwargs["corr_base"] = np.eye(3)
    source = GaussianCopula(**kwargs)
    cls, constructor_kwargs = get_copula_constructor(source)
    rebuilt = cls(**constructor_kwargs)

    assert rebuilt.corr_mode == corr_mode
    np.testing.assert_array_equal(rebuilt._constructor_R, supplied)
    if corr_mode != "fixed":
        np.testing.assert_array_equal(
            rebuilt._constructor_corr_base, np.eye(3))


def test_cholesky_and_factor_constructor_guards():
    with pytest.raises(ValueError, match="d <= 2"):
        GaussianCopula(d=3, corr_mode="cholesky", cholesky_d_max=2)
    with pytest.raises(NotImplementedError, match="factor-loading score"):
        GaussianCopula(
            d=4, corr_mode="factor", factor_rank=2,
            factor_estimation="joint")
