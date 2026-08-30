"""Regression cases from the multivariate static validation audit."""

import copy

import numpy as np
import pytest
from scipy.stats import multivariate_normal, multivariate_t, norm, t

from pyscarcopula import (
    EquicorrGaussianCopula,
    FactorCorrelation,
    GaussianCopula,
    StochasticStudentCopula,
    StudentCopula,
)
from pyscarcopula import api
from pyscarcopula._native import _descriptors, _extension, multivariate, static
from pyscarcopula.copula.multivariate.correlation_policy import CorrelationPolicy
from pyscarcopula.copula.multivariate.factor_estimation import FactorLoadingParameterization


FAMILIES = [GaussianCopula, StudentCopula, StochasticStudentCopula]


def _data(seed=713, rows=500):
    rng = np.random.default_rng(seed)
    loadings = np.array([0.85, 0.8, 0.7, 0.75])
    correlation = np.diag(1 - loadings**2) + np.outer(loadings, loadings)
    values = rng.standard_normal((rows, 4)) @ np.linalg.cholesky(correlation).T
    first = t.cdf(values / np.sqrt(rng.chisquare(6, rows)[:, None] / 6), 6)
    second = first.copy()
    second[:, [1, 3]] = 1 - second[:, [1, 3]]
    return first, second


def _fit(model, data, **kwargs):
    return model.fit(data, method="MLE", **kwargs)


def _model(family, mode, *, supplied=False):
    kwargs = {"corr_mode": mode}
    if mode == "factor":
        kwargs["factor_rank"] = 1
        if supplied:
            kwargs["factor_loadings"] = np.array([[0.4], [0.5], [0.3], [0.6]])
    elif supplied:
        kwargs["R" if mode == "fixed" else "corr_base"] = (
            0.6 * np.eye(4) + 0.4 * np.ones((4, 4)))
    return family(d=4, **kwargs)


def _result_correlation(result):
    if result.correlation_matrix is not None:
        return result.correlation_matrix
    loadings = result.model_parameters["factor_loadings"]
    return np.diag(1 - np.sum(loadings**2, axis=1)) + loadings @ loadings.T


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("mode", ["fixed", "shrinkage", "factor"])
@pytest.mark.parametrize("supplied", [False, True])
def test_refit_matches_fresh_fit_with_same_constructor(family, mode, supplied):
    first, second = _data()
    model = _model(family, mode, supplied=supplied)
    assert _fit(model, first).success
    repeated = _fit(model, second)
    fresh = _fit(_model(family, mode, supplied=supplied), second)
    assert repeated.success and fresh.success
    np.testing.assert_allclose(
        _result_correlation(repeated), _result_correlation(fresh), atol=2e-12)
    assert repeated.log_likelihood == pytest.approx(fresh.log_likelihood, abs=2e-9)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("mode", ["fixed", "factor"])
def test_explicit_old_result_uses_snapshot_and_independent_density(family, mode):
    first, second = _data()
    model = _model(family, mode)
    old = _fit(model, first)
    current = _fit(model, second)
    assert old.success and current.success
    correlation = _result_correlation(old)
    if family is GaussianCopula:
        quantiles = norm.ppf(first)
        rows = multivariate_normal.logpdf(quantiles, cov=correlation)
        rows -= norm.logpdf(quantiles).sum(axis=1)
    else:
        quantiles = t.ppf(first, old.copula_param)
        rows = multivariate_t.logpdf(quantiles, shape=correlation, df=old.copula_param)
        rows -= t.logpdf(quantiles, old.copula_param).sum(axis=1)
    actual = api.log_likelihood(model, first, old)
    assert actual == pytest.approx(rows.sum(), abs=2e-7)
    assert model.fit_result is current
    np.testing.assert_array_equal(model._last_u, second)


@pytest.mark.parametrize("family", [GaussianCopula, StudentCopula])
def test_api_forwards_iteration_limit(family):
    first, _ = _data()
    direct = _fit(family(d=4, corr_mode="cholesky"), first, maxiter=1)
    model = family(d=4, corr_mode="cholesky")
    through_api = api.fit(model, first, method="MLE", maxiter=1)
    assert not direct.success and not through_api.success
    assert through_api.nfev == direct.nfev
    assert through_api.log_likelihood == pytest.approx(direct.log_likelihood)
    assert model.fit_result is None


@pytest.mark.parametrize("family", FAMILIES + [EquicorrGaussianCopula])
def test_multivariate_mle_rejects_unsupported_initial_parameter(family):
    with pytest.raises(TypeError, match="alpha0"):
        api.fit(family(d=4), _data()[0], method="MLE", alpha0=[1.0])


@pytest.mark.parametrize("through_api", [False, True])
def test_failed_equicorr_fit_preserves_previous_result_and_training_data(through_api):
    first, second = _data()
    model = EquicorrGaussianCopula(d=4)
    accepted = _fit(model, first)
    assert accepted.success
    previous_u = model._last_u.copy()
    if through_api:
        rejected = api.fit(model, second, method="MLE", maxiter=1)
    else:
        rejected = _fit(model, second, maxiter=1)
    assert not rejected.success
    assert model.fit_result is accepted
    np.testing.assert_array_equal(model._last_u, previous_u)


@pytest.mark.parametrize("operation", ["sample", "predict", "sample_batches", "predict_batches"])
def test_dense_gaussian_budget_rejected_before_rng_changes(operation):
    model = GaussianCopula(R=np.eye(4))
    assert _fit(model, _data()[0]).success
    rng = np.random.default_rng(941)
    before = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(MemoryError):
        result = getattr(model, operation)(5, rng=rng, memory_budget_bytes=1)
        if operation.endswith("batches"):
            list(result)
    assert rng.bit_generator.state == before
    if operation.endswith("batches"):
        blocks = list(getattr(model, operation)(5, batch_rows=2, memory_budget_bytes=64))
        assert sum(len(block) for block in blocks) == 5
    else:
        assert getattr(model, operation)(5, memory_budget_bytes=160).shape == (5, 4)


@pytest.mark.parametrize("family", FAMILIES + [EquicorrGaussianCopula])
@pytest.mark.parametrize("object_dtype", [False, True])
def test_complex_data_is_rejected_by_fit_and_likelihood(family, object_dtype):
    data, _ = _data(rows=100)
    model = family(d=4)
    result = _fit(model, data)
    assert result.success
    invalid = data.astype(complex) + 0.1j
    if object_dtype:
        invalid = invalid.astype(object)
    with pytest.raises((TypeError, ValueError), match="real"):
        _fit(model, invalid)
    with pytest.raises((TypeError, ValueError), match="real"):
        static.prepare(model, invalid).log_likelihood(result.copula_param)
    with pytest.raises((TypeError, ValueError), match="real"):
        api.log_likelihood(model, invalid, result)
    with pytest.raises((TypeError, ValueError), match="real"):
        model.log_pdf_rows(invalid, result.copula_param)


@pytest.mark.parametrize("family", FAMILIES + [EquicorrGaussianCopula])
def test_api_static_fit_owns_training_data(family):
    data, _ = _data(rows=100)
    model = family(d=4)
    result = api.fit(model, data, method="MLE")
    assert result.success
    expected = data.copy()
    data[:] = 0.5
    np.testing.assert_array_equal(model._last_u, expected)


@pytest.mark.parametrize("object_dtype", [False, True])
def test_direct_native_static_evaluator_rejects_complex_data(object_dtype):
    module = _extension.load()
    spec = _descriptors.make_gaussian_static_spec(module, np.eye(4))
    invalid = np.full((5, 4), 0.5 + 0.1j)
    if object_dtype:
        invalid = invalid.astype(object)
    with pytest.raises((TypeError, ValueError), match="real"):
        module.StaticCopulaEvaluator(spec, invalid, 1)
    with pytest.raises((TypeError, ValueError), match="real"):
        module.multivariate_log_pdf_and_grad(spec, invalid, np.array([0.0]))
    factor = FactorCorrelation(np.full((4, 1), 0.3)).prepare()
    with pytest.raises((TypeError, ValueError), match="real"):
        module._factor_student_log_pdf_and_dlog_ddf(factor._native, invalid, np.array([6.0]))


@pytest.mark.parametrize("family", [StudentCopula, StochasticStudentCopula])
@pytest.mark.parametrize("dimension,rank", [(3, 2), (4, 2), (6, 3)])
def test_joint_factor_rejects_rank_outside_supported_regime(family, dimension, rank):
    with pytest.raises(ValueError, match="identifiability"):
        family(d=dimension, corr_mode="factor", factor_rank=rank, factor_estimation="joint")
    with pytest.raises(ValueError, match="identifiability"):
        FactorLoadingParameterization.from_loadings(
            np.full((dimension, rank), 0.1), uniqueness_min=1e-8)
    with pytest.raises(ValueError, match="invalid_parameter"):
        multivariate.factor_parameterization_from_loadings(
            np.full((dimension, rank), 0.1), 1e-8)


def test_two_stage_high_rank_count_is_correlation_dimension():
    policy = CorrelationPolicy.create(
        mode="factor", estimator="factor_two_stage", dimension=3, factor_rank=2)
    assert policy.effective_n_params == 3
    for family in FAMILIES:
        model = family(d=3, corr_mode="factor", factor_rank=2)
        result = _fit(model, _data()[0][:, :3])
        assert result.success
        count = 3 + (family is not GaussianCopula)
        assert result.parameter_count == count
        assert result.aic == pytest.approx(2 * count - 2 * result.log_likelihood)


@pytest.mark.parametrize("dimension,rank", [(3, 1), (5, 2), (7, 3)])
def test_supported_joint_coordinates_have_full_generic_correlation_rank(dimension, rank):
    loadings = np.random.default_rng(937).normal(scale=0.15, size=(dimension, rank))
    parameterization, raw = FactorLoadingParameterization.from_loadings(
        loadings, uniqueness_min=1e-8)
    off_diagonal = np.tril_indices(dimension, -1)
    columns = []
    for index in range(len(raw)):
        step = np.zeros_like(raw)
        step[index] = 1e-5
        plus = parameterization.loadings(raw + step)
        minus = parameterization.loadings(raw - step)
        columns.append(((plus @ plus.T - minus @ minus.T) / 2e-5)[off_diagonal])
    assert np.linalg.matrix_rank(np.column_stack(columns), tol=1e-8) == len(raw)


@pytest.mark.parametrize("n", [2.9, True, np.float64(2.0), -1])
@pytest.mark.parametrize("mode", ["fixed", "factor"])
@pytest.mark.parametrize("given", [{}, {0: 0.4}, {0: 0.4, 1: 0.5, 2: 0.6, 3: 0.7}])
def test_student_conditional_rejects_invalid_n_without_advancing_rng(n, mode, given):
    model = _model(StudentCopula, mode, supplied=True)
    assert _fit(model, _data()[0]).success
    rng = np.random.default_rng(913)
    before = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises((TypeError, ValueError), match="n"):
        model.sample_conditional(n, given, rng=rng)
    assert rng.bit_generator.state == before
    assert model.sample_conditional(0, given, rng=rng).shape == (0, 4)


@pytest.mark.parametrize("df", [999.0, 1000.0, 1001.0, 1e4, 1e6, 1e8])
def test_static_student_large_df_quantiles_and_derivatives(df):
    module = _extension.load()
    probabilities = np.array([1e-10, 1e-6, 0.1, 0.3, 0.5, 0.7, 0.9, 1 - 1e-10])
    values = np.array([module._student_quantile(float(p), df) for p in probabilities])
    joint = np.array([
        module._student_quantile_with_df_derivative(float(p), df) for p in probabilities])
    np.testing.assert_allclose(values, t.ppf(probabilities, df), rtol=3e-10, atol=2e-10)
    np.testing.assert_allclose(joint[:, 0], values, rtol=0, atol=0)
    step = df * 1e-3
    reference = (t.ppf(probabilities, df + step) - t.ppf(probabilities, df - step)) / (2 * step)
    np.testing.assert_allclose(joint[:, 1], reference, rtol=2e-5, atol=2e-12)


@pytest.mark.parametrize("df", [1000.0, 1e4, 1e6, 1e8])
def test_static_student_large_df_density_matches_scipy_and_normal_limit(df):
    dimension = 7
    correlation = 0.7 * np.eye(dimension) + 0.3 * np.ones((dimension, dimension))
    u = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]])
    model = StochasticStudentCopula(d=dimension, R=correlation)
    quantiles = t.ppf(u, df)
    expected = multivariate_t.logpdf(quantiles, shape=correlation, df=df)
    expected -= t.logpdf(quantiles, df).sum()
    actual = model.log_likelihood(u, df)
    assert actual == pytest.approx(expected, abs=1e-6)
    if df == 1e8:
        z = norm.ppf(u)
        limit = multivariate_normal.logpdf(z, cov=correlation) - norm.logpdf(z).sum()
        assert actual == pytest.approx(limit, abs=1e-6)
