"""Public parameter routing for static elliptical models."""

from copy import deepcopy

import numpy as np
import pytest
from scipy.stats import multivariate_t, norm, t

from pyscarcopula import (
    EquicorrGaussianCopula, GaussianCopula, StochasticStudentCopula,
    StudentCopula, api,
)
from pyscarcopula._native import static
from pyscarcopula._types import NumericalConfig, PredictConfig
from pyscarcopula.copula.multivariate.factor_student import FactorStudentEvaluator


CORRELATION = np.array([
    [1., .43, .17, -.12], [.43, 1., .21, .08],
    [.17, .21, 1., .3], [-.12, .08, .3, 1.],
])
LOADINGS = np.array([[.52], [.37], [-.29], [.41]])
DATA = norm.cdf(np.random.default_rng(317).multivariate_normal(
    np.zeros(4), CORRELATION, size=160))
VARIANTS = [
    (family, mode)
    for family in (GaussianCopula, StudentCopula)
    for mode in ("fixed", "shrinkage", "cholesky", "factor", "factor-joint")
    if family is StudentCopula or mode != "factor-joint"
]


@pytest.fixture(scope="module", params=VARIANTS,
                ids=[f"{family.__name__}-{mode}" for family, mode in VARIANTS])
def fitted(request):
    family, mode = request.param
    options = dict(d=4, corr_mode="factor" if mode == "factor-joint" else mode)
    if mode.startswith("factor"):
        options.update(factor_rank=1, factor_loadings=LOADINGS,
                       factor_estimation="joint" if mode == "factor-joint" else "two-stage")
    else:
        options["R"] = CORRELATION
        if mode != "fixed":
            options["corr_base"] = .6 * CORRELATION + .4 * np.eye(4)
    model = family(**options)
    result = model.fit(DATA)
    assert result.success, result.message
    return model, result


@pytest.mark.parametrize("key", ["unknown_option", "K", "gtol", "given"])
def test_density_rejects_unsupported_keywords(fitted, key):
    model, _ = fitted
    with pytest.raises(TypeError, match=key):
        model.log_pdf_rows(DATA, **{key: 23})


def test_objective_delivers_threads_to_native_evaluation(fitted, monkeypatch):
    model, _ = fitted
    records = []
    if isinstance(model, StudentCopula) and model.corr_mode == "factor":
        original = FactorStudentEvaluator.evaluate

        def evaluate(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            records.append(result.diagnostics)
            return result

        monkeypatch.setattr(FactorStudentEvaluator, "evaluate", evaluate)
    else:
        original = static.StaticLikelihoodEvaluator.value_result

        def value_result(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            records.append(result)
            return result

        monkeypatch.setattr(static.StaticLikelihoodEvaluator, "value_result", value_result)
    alpha = [] if isinstance(model, GaussianCopula) else [4.2]
    data = np.tile(DATA, (60, 1))
    reference = model.mlog_likelihood(alpha, data)
    records.clear()
    actual = model.mlog_likelihood(alpha, data, config=NumericalConfig(n_threads=4))
    assert actual == pytest.approx(reference, abs=1e-9, rel=1e-12)
    assert records
    assert all(record["n_threads_requested"] == 4 for record in records)
    blocks_key = ("row_parallel_blocks" if isinstance(model, StudentCopula)
                  and model.corr_mode == "factor" else "parallel_blocks")
    assert all(record[blocks_key] > 1 for record in records)


@pytest.mark.parametrize("alpha", [[4.2, .13], [[4.2]], [np.nan], [np.inf]])
def test_objective_rejects_invalid_parameter_vectors(fitted, alpha):
    model, _ = fitted
    with pytest.raises(ValueError, match="alpha"):
        model.mlog_likelihood(alpha, DATA)


def test_objective_uses_explicit_parameter_without_mutating_fit(fitted):
    model, result = fitted
    correlation = model.to_correlation_matrix().copy()
    if isinstance(model, GaussianCopula):
        assert model.mlog_likelihood([], DATA) == pytest.approx(-model.log_likelihood(DATA))
        for alpha in ([0.], [6.], [.1, .2, .3, .4, .5, .6]):
            with pytest.raises(ValueError, match="alpha"):
                model.mlog_likelihood(alpha, DATA)
    else:
        fitted_df = model.df
        values = []
        for df in (4.2, 17.3):
            quantiles = t.ppf(DATA, df)
            expected = -np.sum(multivariate_t.logpdf(quantiles, shape=correlation, df=df)
                               - np.sum(t.logpdf(quantiles, df), axis=1))
            actual = model.mlog_likelihood([df], DATA)
            assert actual == pytest.approx(expected, abs=3e-7)
            assert -model.log_likelihood(DATA, parameter=df) == pytest.approx(expected, abs=3e-7)
            values.append(actual)
        assert abs(values[0] - values[1]) > 1
        for df in (0., 2.):
            # Preserve native status translation: dense likelihood reports a
            # numerical failure here, while the factor adapter rejects df.
            with pytest.raises((ValueError, FloatingPointError)):
                model.mlog_likelihood([df], DATA)
        assert model.df == fitted_df
        with pytest.raises(ValueError, match="alpha"):
            model.mlog_likelihood([], DATA)
    assert model.fit_result is result
    np.testing.assert_array_equal(model.to_correlation_matrix(), correlation)
    np.testing.assert_array_equal(model._last_u, DATA)


def test_objective_does_not_retry_consumer_typeerror(fitted, monkeypatch):
    model, _ = fitted
    calls = []

    def fail(*args, **kwargs):
        calls.append(kwargs)
        raise TypeError("consumer failed")

    monkeypatch.setattr(model, "log_likelihood", fail)
    alpha = [] if isinstance(model, GaussianCopula) else [6.]
    with pytest.raises(TypeError, match="consumer failed"):
        model.mlog_likelihood(alpha, DATA)
    assert len(calls) == 1


INVALID_PREDICTION_OPTIONS = [
    dict(horizon="unknown"), dict(predictive_r_mode="unknown"),
    dict(predict_config=PredictConfig(return_diagnostics=True)),
    dict(predict_config=PredictConfig(dynamic_conditioning="given_only")),
    dict(predict_config=PredictConfig(mcmc_steps=7)),
    dict(predict_config=PredictConfig(mcmc_steps=0)),
    dict(predict_config=PredictConfig(mcmc_burnin=0)),
]


@pytest.mark.parametrize("options", INVALID_PREDICTION_OPTIONS)
@pytest.mark.parametrize("n", [0, 3])
def test_prediction_rejects_options_before_rng_use(fitted, options, n):
    model, result = fitted
    operations = [model.predict, lambda n, **kw: api.predict(model, DATA, result, n, **kw)]
    if isinstance(model, GaussianCopula):
        operations.append(model.predict_batches)
    for operation in operations:
        rng = np.random.default_rng(91)
        before = deepcopy(rng.bit_generator.state)
        with pytest.raises((ValueError, TypeError)):
            output = operation(n, rng=rng, **options)
            if not isinstance(output, np.ndarray):
                list(output)
        assert rng.bit_generator.state == before
    assert model.fit_result is result


@pytest.mark.parametrize("horizon", ["current", "next"])
def test_prediction_preserves_valid_static_config_and_overrides(fitted, horizon):
    model, result = fitted
    given = {1: .4}
    expected = model.sample_conditional(17, given=given, rng=np.random.default_rng(12))
    config = PredictConfig(given={0: .2}, horizon=horizon, predictive_r_mode="histogram")
    for operation in (model.predict, lambda n, **kw: api.predict(model, DATA, result, n, **kw)):
        actual = operation(17, given=given, predict_config=config, rng=np.random.default_rng(12))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    if isinstance(model, GaussianCopula):
        actual = np.concatenate(list(model.predict_batches(
            17, batch_rows=17, given=given, predict_config=config,
            rng=np.random.default_rng(12))))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    assert config.given == {0: .2}


def test_predictive_mean_requires_a_scalar_parameter(fitted):
    model, result = fitted
    if isinstance(model, GaussianCopula):
        with pytest.raises(NotImplementedError, match="scalar"):
            api.predictive_mean(model, DATA, result)
    else:
        np.testing.assert_array_equal(api.predictive_mean(model, DATA, result),
                                      np.full(len(DATA), result.copula_param))


def test_api_fit_rejects_prepared_evaluator_before_delegation(fitted, monkeypatch):
    model, result = fitted

    def must_not_fit(*args, **kwargs):
        pytest.fail("an unsupported evaluator must be rejected before model.fit")

    monkeypatch.setattr(model, "fit", must_not_fit)
    with pytest.raises(TypeError, match="_prepared_evaluator"):
        api.fit(model, DATA, method="mle", _prepared_evaluator=object())
    assert model.fit_result is result


@pytest.mark.parametrize("mode", ["fixed", "shrinkage", "cholesky"])
def test_joint_factor_estimation_requires_factor_mode(mode):
    with pytest.raises(ValueError, match="factor_estimation"):
        StudentCopula(d=4, corr_mode=mode, factor_estimation="joint")


@pytest.mark.parametrize("operation", [api.sample, api.predict])
@pytest.mark.parametrize("given", [None, {}, {1: .4}, dict(enumerate([.2, .4, .6, .8]))])
def test_gaussian_factor_api_enforces_workspace_budget(operation, given):
    model = GaussianCopula(d=4, corr_mode="factor", factor_rank=1, factor_loadings=LOADINGS)
    result = model.fit(DATA)
    assert result.success
    rng = np.random.default_rng(9)
    before = deepcopy(rng.bit_generator.state)
    with pytest.raises(MemoryError):
        operation(model, DATA, result, 17, given=given, rng=rng, memory_budget_bytes=17 * 4 * 8)
    assert rng.bit_generator.state == before
    expected = operation(model, DATA, result, 17, given=given, rng=np.random.default_rng(9))
    actual = operation(model, DATA, result, 17, given=given, rng=rng, memory_budget_bytes=4096)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("family", [GaussianCopula, StudentCopula])
@pytest.mark.parametrize("mode", ["fixed", "factor"])
@pytest.mark.parametrize("given", [None, {1: .4}])
def test_static_api_sampling_preserves_supported_budget_contract(family, mode, given):
    options = {"factor_rank": 1, "factor_loadings": LOADINGS} if mode == "factor" else {"R": CORRELATION}
    model = family(d=4, corr_mode=mode, **options)
    result = model.fit(DATA)
    assert result.success
    # Only Gaussian factor sampling needs more than the output array budget.
    budget = 4096 if family is GaussianCopula and mode == "factor" else 17 * 4 * 8
    for operation in (api.sample, api.predict):
        expected = operation(model, DATA, result, 17, given=given, rng=np.random.default_rng(18))
        actual = operation(model, DATA, result, 17, given=given, rng=np.random.default_rng(18),
                           memory_budget_bytes=budget)
        np.testing.assert_array_equal(actual, expected)
        rng = np.random.default_rng(18)
        before = deepcopy(rng.bit_generator.state)
        with pytest.raises(MemoryError):
            operation(model, DATA, result, 17, given=given, rng=rng, memory_budget_bytes=1)
        assert rng.bit_generator.state == before


@pytest.mark.parametrize("family, parameter", [(EquicorrGaussianCopula, .3),
                                               (StochasticStudentCopula, 4.2)])
def test_dynamic_multivariate_mle_objective_retains_natural_parameter(family, parameter):
    options = {} if family is EquicorrGaussianCopula else {"R": CORRELATION}
    model = family(d=4, **options)
    actual = model.mlog_likelihood([parameter], DATA, config=NumericalConfig(n_threads=4))
    assert actual == pytest.approx(-model.log_likelihood(DATA, parameter, n_threads=4))
