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
from pyscarcopula._native import _extension as _cpp_extension, static as static_likelihood
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


@pytest.mark.parametrize("model_type", [GaussianCopula, StudentCopula])
@pytest.mark.parametrize("corr_mode", ["fixed", "shrinkage", "cholesky"])
def test_public_likelihood_prepares_correlation_without_python_linalg(
        monkeypatch, model_type, corr_mode):
    model = model_type(d=3, R=_correlation(), corr_mode=corr_mode)
    observations = _observations(24, 3)
    options = (
        {} if model_type is GaussianCopula and corr_mode == "fixed"
        else {"maxiter": 100}
    )
    result = model.fit(observations, to_pobs=False, **options)
    assert result.success
    expected = model.log_likelihood(observations)

    def forbidden(*args, **kwargs):
        raise AssertionError("Python correlation factorization was called")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    monkeypatch.setattr(np.linalg, "inv", forbidden)
    monkeypatch.setattr(np.linalg, "slogdet", forbidden)
    assert model.log_likelihood(observations) == pytest.approx(
        expected, rel=1e-13, abs=1e-13)


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


@pytest.mark.parametrize("factory,parameter", _BIVARIATE_CASES)
def test_value_only_objective_matches_gradient_objective_exactly(
        factory, parameter):
    evaluator = static_likelihood.prepare(factory(), _observations())

    value_only = evaluator.value_result(parameter)
    with_gradient = evaluator.result(parameter)

    assert value_only["status"] == with_gradient["status"]
    assert value_only["failure_index"] == with_gradient["failure_index"]
    assert (
        value_only["negative_log_likelihood"]
        == with_gradient["negative_log_likelihood"]
    )


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


def test_static_evaluator_does_not_mask_invalid_or_nonfinite_native_results():
    evaluator = static_likelihood.StaticLikelihoodEvaluator.__new__(
        static_likelihood.StaticLikelihoodEvaluator)

    evaluator.result = lambda parameter: {
        "status": 6,
        "negative_log_likelihood": np.inf,
        "negative_gradient": 0.0,
    }
    with pytest.raises(ValueError, match="invalid_parameter"):
        evaluator.objective_and_gradient(0.2)

    evaluator.result = lambda parameter: {
        "status": 0,
        "negative_log_likelihood": np.nan,
        "negative_gradient": 0.0,
    }
    with pytest.raises(FloatingPointError, match="non-finite"):
        evaluator.objective_and_gradient(0.2)


def test_static_evaluator_uses_cpp_policy_only_for_numerical_failure():
    evaluator = static_likelihood.StaticLikelihoodEvaluator.__new__(
        static_likelihood.StaticLikelihoodEvaluator)
    evaluator.result = lambda parameter: {
        "status": 7,
        "negative_log_likelihood": np.inf,
        "negative_gradient": 0.0,
    }

    value, gradient = evaluator.objective_and_gradient(
        0.2, fail_value=321.0)

    assert value == 321.0
    np.testing.assert_array_equal(gradient, [0.0])


def test_mle_objective_propagates_unexpected_adapter_failures(monkeypatch):
    monkeypatch.setattr(
        static_likelihood,
        "prepare",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("unexpected adapter failure")),
    )

    with pytest.raises(RuntimeError, match="unexpected adapter failure"):
        MLEStrategy().objective(
            ClaytonCopula(), _observations(4), np.array([1.0]))


def test_mle_fit_uses_one_prepared_native_evaluator(monkeypatch):
    u = _observations(120)
    copula = BivariateGaussianCopula()
    calls = {"prepare": 0, "objective": 0, "native": 0}
    real_prepare = static_likelihood.prepare

    def counted_prepare(copula_arg, u_arg, **kwargs):
        calls["prepare"] += 1
        evaluator = real_prepare(copula_arg, u_arg, **kwargs)
        real_objective = evaluator.objective_and_gradient
        real_result = evaluator.result

        def counted_objective(*args, **kwargs):
            calls["objective"] += 1
            return real_objective(*args, **kwargs)

        def counted_result(parameter):
            calls["native"] += 1
            return real_result(parameter)

        evaluator.objective_and_gradient = counted_objective
        evaluator.result = counted_result
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
    assert calls["objective"] == result.nfev
    assert calls["native"] == result.nfev + 1


def test_mle_explicit_alpha0_is_a_natural_parameter(monkeypatch):
    u = _observations(12)
    copula = GumbelCopula()
    evaluated = []

    class Evaluator:
        def objective_and_gradient(self, parameter, **kwargs):
            return self.validated_objective_and_gradient(parameter)

        def validated_objective_and_gradient(self, parameter):
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

    monkeypatch.setattr(
        static_likelihood,
        "prepare",
        lambda *args, **kwargs: Evaluator(),
    )
    monkeypatch.setattr(mle_module, "minimize", fake_minimize)
    monkeypatch.setattr(
        copula,
        "transform",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("explicit MLE alpha0 must not be transformed")),
    )

    result = MLEStrategy().fit(copula, u, alpha0=np.array([2.25]))

    assert evaluated == [2.25, 2.25]
    assert result.copula_param == 2.25


@pytest.mark.parametrize(
    "alpha0",
    [
        np.array([]),
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([0.2, 0.3]),
        np.array([[0.2]]),
        np.array([0.2 + 1.0j]),
        np.array([1.0]),
    ],
)
def test_mle_rejects_invalid_alpha0_before_native_prepare(monkeypatch, alpha0):
    monkeypatch.setattr(
        static_likelihood,
        "prepare",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid alpha0 reached native preparation")),
    )

    with pytest.raises((TypeError, ValueError), match="alpha0"):
        MLEStrategy().fit(
            BivariateGaussianCopula(), _observations(12), alpha0=alpha0)


def test_mle_accepts_scalar_alpha0():
    result = MLEStrategy().fit(
        BivariateGaussianCopula(), _observations(30), alpha0=0.2)
    assert np.isfinite(result.copula_param)


def test_mle_strategy_rejects_unknown_keywords_before_native_prepare(
        monkeypatch):
    monkeypatch.setattr(
        static_likelihood,
        "prepare",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unknown keyword reached native preparation")),
    )

    with pytest.raises(
            TypeError,
            match="unexpected MLE keyword.*definitely_unknown"):
        MLEStrategy().fit(
            BivariateGaussianCopula(),
            _observations(12),
            definitely_unknown=True,
        )


def test_mle_strategy_constructor_rejects_unknown_keywords():
    with pytest.raises(
            TypeError,
            match="unexpected MLE keyword.*definitely_unknown"):
        MLEStrategy(definitely_unknown=True)


@pytest.mark.parametrize(
    ("optimizer_x", "optimizer_fun", "error", "message"),
    [
        (np.array([np.nan]), 0.0, ValueError, "MLE optimizer result"),
        (np.array([0.2]), np.inf, RuntimeError, "objective value"),
    ],
)
def test_mle_rejects_nonfinite_optimizer_output(
        monkeypatch, optimizer_x, optimizer_fun, error, message):
    class Evaluator:
        def objective_and_gradient(self, parameter, **kwargs):
            return 0.0, np.array([0.0])

    monkeypatch.setattr(
        static_likelihood,
        "prepare",
        lambda *args, **kwargs: Evaluator(),
    )
    monkeypatch.setattr(
        mle_module,
        "minimize",
        lambda *args, **kwargs: SimpleNamespace(
            x=optimizer_x,
            fun=optimizer_fun,
            success=True,
            nfev=1,
            message="invalid test result",
        ),
    )

    with pytest.raises(error, match=message):
        MLEStrategy().fit(BivariateGaussianCopula(), _observations(12))


@pytest.mark.parametrize(
    "final_result, error, message",
    [
        (dict(status=7, negative_log_likelihood=np.inf, negative_gradient=0.0,
              failure_index=0), FloatingPointError, "numerical_failure"),
        (dict(status=6, negative_log_likelihood=np.inf, negative_gradient=0.0),
         ValueError, "invalid_parameter"),
        (dict(status=0, negative_log_likelihood=np.nan, negative_gradient=0.0),
         FloatingPointError, "non-finite"),
        (dict(status=0, negative_log_likelihood=0.0, negative_gradient=np.nan),
         FloatingPointError, "non-finite"),
    ],
)
def test_mle_rejects_invalid_final_native_evaluation(
        monkeypatch, final_result, error, message):
    evaluator = static_likelihood.prepare(
        BivariateGaussianCopula(), _observations(12))

    def fake_minimize(fun, x0, **kwargs):
        value, _ = fun(x0)
        # The optimizer succeeded, but the final native check must still pass.
        monkeypatch.setattr(evaluator, "result", lambda parameter: final_result)
        return SimpleNamespace(
            x=x0, fun=value, success=True, nfev=1, message="ok")

    monkeypatch.setattr(mle_module, "minimize", fake_minimize)
    with pytest.raises(error, match=message):
        MLEStrategy().fit(
            BivariateGaussianCopula(), _observations(12),
            _prepared_evaluator=evaluator)


def test_mle_tolerates_numerical_failure_at_nonfinal_trial(monkeypatch):
    evaluator = static_likelihood.prepare(
        BivariateGaussianCopula(), _observations(12))
    original_result = evaluator.result
    strategy = MLEStrategy()

    def result(parameter):
        if parameter == 0.25:
            return dict(status=7, negative_log_likelihood=np.inf,
                        negative_gradient=0.0, failure_index=0)
        return original_result(parameter)

    def fake_minimize(fun, x0, **kwargs):
        penalty, gradient = fun(np.array([0.25]))
        assert penalty == strategy.config.fail_value
        np.testing.assert_array_equal(gradient, [0.0])
        value, _ = fun(x0)
        return SimpleNamespace(
            x=x0, fun=value, success=True, nfev=2, message="ok")

    monkeypatch.setattr(evaluator, "result", result)
    monkeypatch.setattr(mle_module, "minimize", fake_minimize)
    fitted = strategy.fit(
        BivariateGaussianCopula(), _observations(12), alpha0=0.2,
        _prepared_evaluator=evaluator)

    assert fitted.success
    assert fitted.log_likelihood == pytest.approx(
        evaluator.log_likelihood(fitted.copula_param))


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
        assert bounds == ((2.001, 10_000.0),)
        trial = np.array([7.25])
        value, gradient = fun(trial)
        assert value == 0.0
        np.testing.assert_array_equal(gradient, [0.0])
        return SimpleNamespace(x=trial, fun=0.0, success=True)

    monkeypatch.setattr(
        static_likelihood,
        "prepare_student",
        lambda *args, **kwargs: Evaluator(),
    )
    monkeypatch.setattr(
        "pyscarcopula.strategy.multivariate_mle.minimize",
        fake_minimize,
    )
    monkeypatch.setattr(copula, "_nll", lambda observations: 0.0)

    copula.fit(u)

    assert evaluated == [5.0, 7.25, 7.25]
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

    monkeypatch.setattr(
        static_likelihood,
        "prepare_student",
        lambda *args, **kwargs: Evaluator(),
    )
    monkeypatch.setattr(
        "pyscarcopula.strategy.multivariate_mle.minimize",
        fake_minimize,
    )
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

    assert evaluated == [5.0, 6.75, 6.75]
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


@pytest.mark.parametrize("family", ["gaussian", "equicorr", "student"])
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
    elif family == "equicorr":
        copula = EquicorrGaussianCopula(d=3)
        parameter = 0.25
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


@pytest.mark.parametrize("rho", [-0.2, 0.0, 0.65])
def test_equicorr_prepared_rows_match_direct_native_rows(rho):
    u = _observations(37, 4)
    copula = EquicorrGaussianCopula(d=4)
    evaluator = static_likelihood.prepare(copula, u)

    expected = copula.log_pdf_rows(u, rho)
    np.testing.assert_allclose(
        evaluator.log_pdf_rows(rho), expected, rtol=0.0, atol=2e-14)
    assert evaluator.result(rho)["negative_log_likelihood"] == pytest.approx(
        -np.sum(expected), rel=0.0, abs=2e-13)


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
