"""Regressions for result sampling, preparation boundaries and Student limits."""

from dataclasses import replace
from decimal import Decimal, localcontext
from itertools import product

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm, t

from pyscarcopula import (
    EquicorrGaussianCopula,
    FactorCorrelation,
    FactorStudentEvaluator,
    GaussianCopula,
    StochasticStudentCopula,
    StudentCopula,
    api,
)
from pyscarcopula._native import _extension, multivariate, static
from pyscarcopula.copula.multivariate.factor_estimation import (
    FactorLoadingParameterization,
    estimate_factor_loadings,
)


@pytest.mark.parametrize("family", [GaussianCopula, StudentCopula, StochasticStudentCopula])
@pytest.mark.parametrize("mode", ["fixed", "shrinkage", "cholesky", "factor"])
def test_sample_and_predict_keep_explicit_result_after_refit(family, mode):
    rng = np.random.default_rng(713)
    correlation = 0.4 * np.eye(4) + 0.6 * np.ones((4, 4))
    latent = rng.standard_normal((350, 4)) @ np.linalg.cholesky(correlation).T
    u = t.cdf(latent / np.sqrt(rng.chisquare(6, 350)[:, None] / 6), 6)
    v = u.copy()
    v[:, 1] = 1 - v[:, 1]
    options = {"factor_rank": 1} if mode == "factor" else {}
    model = family(d=4, corr_mode=mode, **options)
    old = model.fit(u, method="MLE")
    assert old.success
    givens = [None, {}, {1: 0.4}, dict(enumerate([0.2, 0.4, 0.6, 0.8]))]
    cases = list(product([api.sample, api.predict], [0, 40], givens))
    expected = [operation(model, u, old, n, given=given, rng=np.random.default_rng(15))
                for operation, n, given in cases]
    current = model.fit(v, method="MLE")
    assert current.success
    for (operation, n, given), before in zip(cases, expected):
        actual = operation(model, u, old, n, given=given, rng=np.random.default_rng(15))
        np.testing.assert_array_equal(actual, before)
        assert actual.shape == (n, 4)
        if given:
            for index, value in given.items():
                np.testing.assert_array_equal(actual[:, index], np.full(n, value))
    current_draws = api.sample(model, u, current, 40, rng=np.random.default_rng(15))
    old_draws = api.sample(model, u, old, 40, rng=np.random.default_rng(15))
    assert np.max(abs(current_draws - old_draws)) > 0.05
    assert model.fit_result is current
    np.testing.assert_array_equal(model._last_u, v)


@pytest.mark.parametrize("family", [GaussianCopula, StudentCopula, StochasticStudentCopula])
def test_factor_result_sampling_does_not_materialize_dense_or_use_prototype_policy(family, monkeypatch):
    rng = np.random.default_rng(50)
    u = rng.uniform(0.1, 0.9, (60, 4))
    model = family(d=4, corr_mode="factor", factor_rank=1,
                   factor_loadings=np.array([[0.8], [0.7], [0.6], [0.9]]))
    result = model.fit(u, method="MLE")
    assert result.success
    expected = api.sample(model, u, result, 30, rng=np.random.default_rng(32))
    prototype = family(d=4, corr_mode="factor", factor_rank=1, factor_uniqueness_min=0.8)
    from pyscarcopula.copula.multivariate.factor_correlation import PreparedFactorCorrelation

    def no_dense(*args, **kwargs):
        raise AssertionError("snapshot sampling must keep compact loadings")

    monkeypatch.setattr(PreparedFactorCorrelation, "to_dense", no_dense)
    actual = api.sample(prototype, u, result, 30, rng=np.random.default_rng(32))
    np.testing.assert_array_equal(actual, expected)
    assert prototype.fit_result is None


@pytest.mark.parametrize("family", [StudentCopula, StochasticStudentCopula])
@pytest.mark.parametrize("given", [None, {1: 0.3}])
def test_result_df_controls_sampling_independently_of_current_df(family, given):
    correlation = 0.6 * np.eye(3) + 0.4 * np.ones((3, 3))
    u = np.random.default_rng(70).uniform(0.1, 0.9, (60, 3))
    model = family(d=3, R=correlation)
    fitted = model.fit(u, method="MLE")
    assert fitted.success
    old = replace(fitted, copula_param=4.5)
    current = replace(fitted, copula_param=80.0)
    model.fit_result = current
    reference = family(d=3, R=correlation)
    if family is StudentCopula:
        model.df = 80.0
        reference.shape = correlation
        reference.df = 4.5
        expected = (reference.sample(50, rng=np.random.default_rng(13)) if given is None
                    else reference.sample_conditional(50, given, rng=np.random.default_rng(13)))
    else:
        expected = (reference.sample_at_parameter(50, 4.5, rng=np.random.default_rng(13))
                    if given is None else reference.sample_conditional(
                        50, r=4.5, given=given, rng=np.random.default_rng(13)))
    for operation in (api.sample, api.predict):
        actual = operation(model, u, old, 50, given=given, rng=np.random.default_rng(13))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    assert model.fit_result is current


@pytest.mark.parametrize("representation", ["complex", "object"])
@pytest.mark.parametrize("entry", ["equicorr", "equicorr_stream", "gaussian", "loadings", "estimate"])
def test_preparation_rejects_complex_before_lossy_cast(representation, entry):
    u = np.random.default_rng(1957).uniform(0.1, 0.9, (80, 4)).astype(complex) + 0.2j
    if representation == "object":
        u = u.astype(object)
    with pytest.raises((TypeError, ValueError), match="real|complex|numeric dtype"):
        if entry.startswith("equicorr"):
            blocks = u if entry == "equicorr" else iter([u[:40], u[40:]])
            EquicorrGaussianCopula(4).prepare_sufficient_statistics(blocks)
        elif entry == "gaussian":
            GaussianCopula(d=4, corr_mode="factor", factor_rank=1).initialize_factor(u)
        elif entry == "loadings":
            FactorLoadingParameterization.from_loadings(u[:4, :1], uniqueness_min=1e-8)
        else:
            estimate_factor_loadings(u, 1, uniqueness_min=1e-8,
                                    dimension_tile=4, seed=0, oversampling=2)


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("representation", ["complex", "object"])
@pytest.mark.parametrize("entry", ["equicorr", "loadings", "estimate", "projection"])
def test_native_preparation_rejects_complex(direct, representation, entry):
    valid = np.random.default_rng(31).uniform(0.1, 0.4, (12, 4))
    invalid = valid.astype(complex) + 0.2j
    if representation == "object":
        invalid = invalid.astype(object)
    module = _extension.load() if direct else multivariate
    with pytest.raises((TypeError, ValueError), match="real|complex"):
        if entry == "equicorr":
            prepare = (module.prepare_equicorr_sufficient_statistics if direct
                       else module.prepare_equicorr_statistics)
            prepare(invalid)
        elif entry == "loadings":
            prepare = (module.static_factor_parameterization_from_loadings if direct
                       else module.factor_parameterization_from_loadings)
            prepare(invalid[:4, :1], 1e-8)
        else:
            observations = invalid if entry == "estimate" else valid
            projection = invalid[:4, :2] if entry == "projection" else valid[:4, :2]
            if direct:
                module.static_estimate_factor_loadings(observations, 1, 1e-8, 4, projection)
            else:
                module.estimate_factor_loadings_from_projection(
                    observations, 1, uniqueness_min=1e-8, dimension_tile=4,
                    random_projection=projection)


def _decimal_student_log_pdf(u, correlation, df):
    """Independent Stirling oracle; scalar cancellation happens at 80 digits."""
    terms = [(1, 12, 1), (-1, 360, 3), (1, 1260, 5),
             (-1, 1680, 7), (1, 1188, 9), (-691, 360360, 11)]

    def log_gamma(z):
        return ((z - Decimal("0.5")) * z.ln() - z
                + sum(Decimal(a) / (Decimal(b) * z**power) for a, b, power in terms))

    q = t.ppf(u, df)
    quadratic = np.einsum("ij,ij->i", q, np.linalg.solve(correlation, q.T).T)
    with localcontext() as context:
        context.prec = 80
        v, d = Decimal(str(df)), Decimal(u.shape[1])
        constant = (log_gamma((v+d)/2) - d*log_gamma((v+1)/2)
                    + (d-1)*log_gamma(v/2) - Decimal(str(np.linalg.slogdet(correlation)[1]))/2)
        output = []
        for row, quad in zip(q, quadratic):
            joint = (1 + Decimal(str(quad))/v).ln()
            marginal = sum((1 + Decimal(str(x))**2/v).ln() for x in row)
            output.append(float(constant - (v+d)*joint/2 + (v+1)*marginal/2))
        return np.array(output)


@pytest.mark.parametrize("dimension", [2, 3, 7, 12])
@pytest.mark.parametrize("df", [31.999, 32.0, 32.001, 1000.0, 1e6, 1e8, 1e12, 1e16])
def test_student_density_matches_high_precision_normalization(dimension, df):
    u = np.random.default_rng(729).uniform(0.02, 0.98, (3, dimension))
    correlation = 0.7 * np.eye(dimension) + 0.3 * np.ones((dimension, dimension))
    expected = _decimal_student_log_pdf(u, correlation, df)
    evaluator = static.prepare_student(correlation, u)
    np.testing.assert_allclose(evaluator.log_pdf_rows(df), expected, rtol=0, atol=2e-11)
    factor = FactorCorrelation(np.full((dimension, 1), np.sqrt(0.3)))
    rows = FactorStudentEvaluator(factor, u).log_pdf_rows(df)
    np.testing.assert_allclose(rows, expected, rtol=0, atol=2e-11)
    model = StochasticStudentCopula(dimension, R=correlation)
    assert model.log_likelihood(u, df) == pytest.approx(expected.sum(), abs=6e-11)


@pytest.mark.parametrize("df", [31.999, 32.0, 32.001, 1e3, 1e5])
def test_student_score_matches_independent_finite_differences(df):
    u = np.random.default_rng(140).uniform(0.1, 0.9, (4, 7))
    correlation = 0.7 * np.eye(7) + 0.3 * np.ones((7, 7))
    step = df * 1e-4
    expected = (_decimal_student_log_pdf(u, correlation, df + step).sum()
                - _decimal_student_log_pdf(u, correlation, df - step).sum()) / (2*step)
    actual = -static.prepare_student(correlation, u).result(df)["negative_gradient"]
    assert actual == pytest.approx(expected, rel=3e-5, abs=2e-12)


@pytest.mark.parametrize("dimension", [2, 3, 7, 12])
@pytest.mark.parametrize("df", [1e8, 1e12, 1e16, 1e100])
def test_student_scaled_score_matches_first_normal_limit_correction(dimension, df):
    u = np.random.default_rng(173).uniform(0.05, 0.95, (5, dimension))
    correlation = 0.7 * np.eye(dimension) + 0.3 * np.ones((dimension, dimension))
    z = norm.ppf(u)
    precision_z = np.linalg.solve(correlation, z.T).T
    quadratic = np.einsum("ij,ij->i", z, precision_z)
    # Expanding the independent Student copula formula in 1/df gives A/df.
    quantile_correction = (z**3 + z)/4
    coefficient = (dimension*(dimension-1)/4 + quadratic**2/4 - dimension*quadratic/2
                   + (z**2/2 - z**4/4 + (z-precision_z)*quantile_correction).sum(axis=1))
    expected = -coefficient.sum()
    dense_score = -static.prepare_student(correlation, u).result(df)["negative_gradient"]
    factor = FactorCorrelation(np.full((dimension, 1), np.sqrt(0.3)))
    factor_score = FactorStudentEvaluator(factor, u).evaluate(df).dlog_likelihood_ddf
    for score in (dense_score, factor_score):
        assert (score * df) * df == pytest.approx(expected, rel=2e-6, abs=2e-6)


@pytest.mark.parametrize("df", [1e100, 1e300, np.finfo(float).max])
def test_student_density_is_finite_at_extreme_finite_df(df):
    u = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]])
    correlation = 0.7 * np.eye(7) + 0.3 * np.ones((7, 7))
    z = norm.ppf(u)
    expected = multivariate_normal.logpdf(z, cov=correlation) - norm.logpdf(z).sum()
    result = static.prepare_student(correlation, u).result(df)
    assert -result["negative_log_likelihood"] == pytest.approx(expected, abs=2e-13)
    assert np.isfinite(result["negative_gradient"])
