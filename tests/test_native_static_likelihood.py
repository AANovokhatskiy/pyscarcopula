"""Contracts for native static likelihood and MLE objectives."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import (
    multivariate_normal,
    multivariate_t,
    norm,
    t as t_dist,
)

from pyscarcopula import (
    BivariateGaussianCopula,
    ClaytonCopula,
    EquicorrGaussianCopula,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    JoeCopula,
    StochasticStudentCopula,
    StudentCopula,
)
from pyscarcopula._constants import PSEUDO_OBS_EPS
from pyscarcopula.numerical import _cpp_extension, static_likelihood
from pyscarcopula.copula.multivariate import stochastic_student
from pyscarcopula.copula.multivariate import student as static_student
from pyscarcopula.strategy import mle as mle_module
from pyscarcopula.strategy.mle import MLEStrategy


_BIVARIATE_CASES = [
    (lambda: ClaytonCopula(rotate=90), 0.8),
    (lambda: GumbelCopula(rotate=180), 1.7),
    (lambda: FrankCopula(), 2.2),
    (lambda: JoeCopula(rotate=270), 1.8),
    (lambda: BivariateGaussianCopula(), 0.35),
]


def _observations(n=80, d=2):
    return np.random.default_rng(20260614).uniform(0.05, 0.95, (n, d))


def _correlation():
    return np.array(
        [
            [1.0, 0.35, -0.15],
            [0.35, 1.0, 0.20],
            [-0.15, 0.20, 1.0],
        ],
        dtype=np.float64,
    )


def test_pybind_exports_static_likelihood_evaluator():
    module = _cpp_extension.load()
    assert hasattr(module, "StaticCopulaEvaluator")
    assert hasattr(module.CopulaFamily, "MultivariateGaussian")


@pytest.mark.parametrize("factory,parameter", _BIVARIATE_CASES)
def test_bivariate_objective_and_gradient_match_native_point_ops(
        factory, parameter):
    copula = factory()
    u = _observations()
    evaluator = static_likelihood.prepare(copula, u)
    result = evaluator.result(parameter)
    r = np.array([parameter], dtype=np.float64)

    expected_value = -np.sum(copula.log_pdf(u[:, 0], u[:, 1], r))
    expected_gradient = -np.sum(
        copula.dlog_pdf_dr(u[:, 0], u[:, 1], r))

    assert result["status"] == 0
    assert result["failure_index"] == -1
    assert result["negative_log_likelihood"] == pytest.approx(
        expected_value, rel=1e-13, abs=1e-13)
    assert result["negative_gradient"] == pytest.approx(
        expected_gradient, rel=1e-12, abs=1e-12)


def test_static_evaluator_reports_first_numerical_failure():
    evaluator = static_likelihood.prepare(
        ClaytonCopula(), np.array([[0.2, 0.3], [0.4, 0.7]]))
    result = evaluator.result(-1.0)

    assert result["status"] == _cpp_extension.load().SCAR_NUMERICAL_FAILURE
    assert result["failure_index"] == 0
    assert np.isinf(result["negative_log_likelihood"])


def test_strategy_failure_translation_preserves_fail_value():
    strategy = MLEStrategy()
    value = strategy.objective(
        BivariateGaussianCopula(),
        np.array([[0.2, 0.3], [0.4, 0.7]]),
        np.array([1.0]),
    )
    assert value == strategy.config.fail_value


def test_mle_fit_uses_one_prepared_native_evaluator(monkeypatch):
    u = _observations(120)
    copula = BivariateGaussianCopula()
    calls = {"prepare": 0, "objective": 0}
    real_prepare = static_likelihood.prepare

    def counted_prepare(copula_arg, u_arg):
        calls["prepare"] += 1
        evaluator = real_prepare(copula_arg, u_arg)
        real_objective = evaluator.objective_and_gradient

        def counted_objective(*args, **kwargs):
            calls["objective"] += 1
            return real_objective(*args, **kwargs)

        evaluator.objective_and_gradient = counted_objective
        return evaluator

    def legacy_fail(*args, **kwargs):
        raise AssertionError("legacy Python MLE kernel was called")

    monkeypatch.setattr(static_likelihood, "prepare", counted_prepare)
    monkeypatch.setattr(copula, "log_pdf", legacy_fail)
    monkeypatch.setattr(copula, "dlog_pdf_dr", legacy_fail)
    assert not hasattr(copula, "mle_objective_fused")

    result = MLEStrategy().fit(copula, u, maxiter=5)

    assert np.isfinite(result.log_likelihood)
    assert calls["prepare"] == 1
    assert calls["objective"] >= 1


def test_mle_explicit_alpha0_is_a_natural_parameter(monkeypatch):
    u = _observations(12)
    copula = GumbelCopula()
    evaluated = []

    class Evaluator:
        def objective_and_gradient(self, parameter, **kwargs):
            evaluated.append(float(parameter))
            return 0.0, np.array([0.0])

    def fake_minimize(fun, x0, *, jac, method, bounds, options):
        np.testing.assert_array_equal(x0, [2.25])
        value, gradient = fun(x0)
        assert value == 0.0
        np.testing.assert_array_equal(gradient, [0.0])
        return SimpleNamespace(
            x=np.asarray(x0),
            fun=0.0,
            success=True,
            nfev=1,
            message="ok",
        )

    monkeypatch.setattr(static_likelihood, "prepare", lambda *args: Evaluator())
    monkeypatch.setattr(mle_module, "minimize", fake_minimize)
    monkeypatch.setattr(
        copula,
        "transform",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("explicit MLE alpha0 must not be transformed")),
    )

    result = MLEStrategy().fit(copula, u, alpha0=np.array([2.25]))

    assert evaluated == [2.25]
    assert result.copula_param == 2.25


def test_static_student_mle_optimizes_natural_df(monkeypatch):
    u = _observations(15, 3)
    copula = StudentCopula()
    evaluated = []

    class Evaluator:
        def objective_and_gradient(self, parameter, **kwargs):
            evaluated.append(float(parameter))
            return 0.0, np.array([0.0])

    def fake_minimize(fun, x0, *, jac, method, bounds, options):
        assert jac is True
        assert x0[0] == 5.0
        assert bounds == [(2.001, np.inf)]
        trial = np.array([7.25])
        value, gradient = fun(trial)
        assert value == 0.0
        np.testing.assert_array_equal(gradient, [0.0])
        return SimpleNamespace(x=trial, fun=0.0, success=True)

    monkeypatch.setattr(static_likelihood, "prepare", lambda *args: Evaluator())
    monkeypatch.setattr(static_student, "minimize", fake_minimize)
    monkeypatch.setattr(copula, "_nll", lambda observations: 0.0)

    copula.fit(u)

    assert evaluated == [7.25]
    assert copula.df == 7.25


@pytest.mark.parametrize(
    ("corr_mode", "expected_jac"),
    [("fixed", True), ("shrinkage", True)],
)
def test_stochastic_student_mle_optimizes_natural_df(
        monkeypatch, corr_mode, expected_jac):
    u = _observations(15, 3)
    copula = StochasticStudentCopula(
        d=3,
        R=_correlation(),
        corr_mode=corr_mode,
    )
    evaluated = []

    class Evaluator:
        def objective_and_gradient(self, parameter, **kwargs):
            evaluated.append(float(parameter))
            return 0.0, np.array([0.0])

        def objective_and_joint_gradient(self, parameter, **kwargs):
            evaluated.append(float(parameter))
            n_corr = copula.d * (copula.d - 1) // 2
            return 0.0, np.array([0.0]), np.zeros(n_corr)

    def fake_minimize(
            fun, x0, *, method, bounds, options, jac=None):
        assert x0[0] == 5.0
        assert bounds[0] == (copula._df_offset, None)
        assert jac is expected_jac
        trial = np.asarray(x0, dtype=np.float64).copy()
        trial[0] = 6.75
        objective = fun(trial)
        if jac:
            value, gradient = objective
            assert value == 0.0
            np.testing.assert_array_equal(gradient, np.zeros_like(trial))
        else:
            assert objective == 0.0
        return SimpleNamespace(
            x=trial,
            fun=0.0,
            success=True,
            nfev=1,
            message="ok",
        )

    monkeypatch.setattr(static_likelihood, "prepare", lambda *args: Evaluator())
    monkeypatch.setattr(stochastic_student, "minimize", fake_minimize)
    monkeypatch.setattr(
        copula,
        "transform",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("static Student MLE must not transform df")),
    )
    monkeypatch.setattr(
        copula,
        "inv_transform",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("static Student MLE must not inverse-transform df")),
    )

    result = copula._fit_mle(u)

    assert evaluated == [6.75]
    assert result.copula_param == 6.75
    assert result.diagnostics["parameterization"] == "natural_df"
    assert result.diagnostics["gradient_mode"] == (
        "analytical_df" if corr_mode == "fixed" else "analytical_joint")
    assert result.diagnostics["optimizer_gradient"] == "analytical"


def test_static_student_native_correlation_gradient_matches_finite_difference():
    u = _observations(28, 3)
    correlation = _correlation()
    copula = StochasticStudentCopula(
        d=3, R=correlation, corr_mode="fixed")
    df = 6.5
    result = static_likelihood.prepare(copula, u).joint_result(df)
    analytical = np.asarray(
        result["negative_correlation_gradient"], dtype=np.float64)
    finite_difference = []
    step = 1e-6

    for i in range(1, 3):
        for j in range(i):
            plus = correlation.copy()
            minus = correlation.copy()
            plus[i, j] += step
            plus[j, i] += step
            minus[i, j] -= step
            minus[j, i] -= step
            plus_model = StochasticStudentCopula(d=3, R=plus)
            minus_model = StochasticStudentCopula(d=3, R=minus)
            plus_value = static_likelihood.prepare(
                plus_model, u).result(df)["negative_log_likelihood"]
            minus_value = static_likelihood.prepare(
                minus_model, u).result(df)["negative_log_likelihood"]
            finite_difference.append(
                (plus_value - minus_value) / (2.0 * step))

    np.testing.assert_allclose(
        analytical,
        finite_difference,
        rtol=0.0,
        atol=2e-7,
    )


def test_evaluator_owns_observation_state():
    u = _observations(20)
    evaluator = static_likelihood.prepare(BivariateGaussianCopula(), u)
    before = evaluator.result(0.4)
    u[:] = 0.5
    after = evaluator.result(0.4)
    np.testing.assert_array_equal(
        after.pop("negative_correlation_gradient"),
        before.pop("negative_correlation_gradient"),
    )
    assert after == before


def test_multivariate_gaussian_rows_and_reduction_match_scipy():
    u = _observations(45, 3)
    correlation = _correlation()
    copula = GaussianCopula()
    copula._set_dimension(3, allow_change=True)
    copula.corr = correlation

    x = norm.ppf(u)
    expected = (
        multivariate_normal.logpdf(
            x, mean=np.zeros(3), cov=correlation)
        - np.sum(norm.logpdf(x), axis=1)
    )
    evaluator = static_likelihood.prepare(copula, u)

    np.testing.assert_allclose(
        evaluator.log_pdf_rows(0.0), expected, rtol=0.0, atol=2e-8)
    assert evaluator.log_likelihood(0.0) == pytest.approx(
        np.sum(expected), abs=2e-7)


def test_multivariate_student_rows_objective_and_gradient():
    u = _observations(35, 3)
    correlation = _correlation()
    df = 6.0
    copula = StudentCopula()
    copula._set_dimension(3, allow_change=True)
    copula.shape = correlation
    copula.df = df
    evaluator = static_likelihood.prepare(copula, u)

    x = t_dist.ppf(u, df=df)
    expected = (
        multivariate_t.logpdf(
            x, loc=np.zeros(3), shape=correlation, df=df)
        - np.sum(t_dist.logpdf(x, df=df), axis=1)
    )
    result = evaluator.result(df)
    step = 1e-5
    finite_difference = (
        evaluator.result(df + step)["negative_log_likelihood"]
        - evaluator.result(df - step)["negative_log_likelihood"]
    ) / (2.0 * step)

    np.testing.assert_allclose(
        evaluator.log_pdf_rows(df), expected, rtol=0.0, atol=2e-10)
    assert result["negative_gradient"] == pytest.approx(
        finite_difference, abs=2e-7)


@pytest.mark.parametrize("family", ["gaussian", "student"])
def test_static_likelihood_quantile_boundaries_match_clipped_inputs(family):
    u = np.array(
        [
            [0.0, PSEUDO_OBS_EPS / 10.0, PSEUDO_OBS_EPS],
            [
                1.0,
                1.0 - PSEUDO_OBS_EPS / 10.0,
                1.0 - PSEUDO_OBS_EPS,
            ],
        ],
        dtype=np.float64,
    )
    clipped = np.clip(
        u, PSEUDO_OBS_EPS, 1.0 - PSEUDO_OBS_EPS)
    if family == "gaussian":
        copula = GaussianCopula()
        copula._set_dimension(3, allow_change=True)
        copula.corr = _correlation()
        parameter = 0.0
    else:
        copula = StudentCopula()
        copula._set_dimension(3, allow_change=True)
        copula.shape = _correlation()
        copula.df = 6.0
        parameter = copula.df

    boundary_rows = static_likelihood.prepare(
        copula, u).log_pdf_rows(parameter)
    clipped_rows = static_likelihood.prepare(
        copula, clipped).log_pdf_rows(parameter)

    np.testing.assert_allclose(
        boundary_rows, clipped_rows, rtol=0.0, atol=0.0)


def test_equicorr_objective_gradient_is_in_parameter_space():
    u = _observations(50, 4)
    evaluator = static_likelihood.prepare(
        EquicorrGaussianCopula(d=4), u)
    rho = 0.25
    step = 1e-6
    result = evaluator.result(rho)
    finite_difference = (
        evaluator.result(rho + step)["negative_log_likelihood"]
        - evaluator.result(rho - step)["negative_log_likelihood"]
    ) / (2.0 * step)

    assert result["negative_gradient"] == pytest.approx(
        finite_difference, abs=2e-7)


def test_stochastic_student_static_objective_uses_exact_native_quantiles():
    u = _observations(30, 3)
    copula = StochasticStudentCopula(d=3, R=_correlation())
    df = 6.0
    evaluator = static_likelihood.prepare(copula, u)

    expected = -copula.log_likelihood(u, df)
    assert evaluator.result(df)["negative_log_likelihood"] == pytest.approx(
        expected, rel=0.0, abs=2e-12)


def test_static_student_fit_and_likelihood_are_finite():
    source = StudentCopula()
    source.shape = _correlation()
    source.df = 7.0
    source._set_dimension(3, allow_change=True)
    u = source.sample(160, rng=np.random.default_rng(20260615))

    fitted = StudentCopula()
    fitted.fit(u)

    assert fitted.df > 2.0
    assert np.isfinite(fitted.log_likelihood(u))
