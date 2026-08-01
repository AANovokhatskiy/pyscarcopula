"""Stage-2 contracts for shared static multivariate MLE orchestration."""

from types import SimpleNamespace
import inspect

import numpy as np
import pytest

from pyscarcopula import (
    GaussianCopula,
    LBFGSBConfig,
    NumericalConfig,
    StochasticStudentCopula,
    StudentCopula,
)
from pyscarcopula.strategy import multivariate_mle
from pyscarcopula.strategy.multivariate_mle import (
    StaticMLEEvaluation,
    StaticMLEProblem,
    run_static_multivariate_mle,
)


def _problem(evaluate):
    return StaticMLEProblem(
        family="test",
        initial_parameters=np.array([0.0]),
        bounds=((None, None),),
        evaluate=evaluate,
    )


def test_evaluator_failure_supplies_nonzero_optimizer_gradient(monkeypatch):
    captured = {}

    def evaluate(parameters):
        raise ValueError("synthetic numerical failure")

    def fake_minimize(fun, x0, **kwargs):
        value, gradient = fun(x0)
        captured["value"] = value
        captured["gradient"] = gradient
        return SimpleNamespace(
            x=x0.copy(), fun=value, success=True, nfev=1, message="ok")

    monkeypatch.setattr(multivariate_mle, "minimize", fake_minimize)
    outcome = run_static_multivariate_mle(
        _problem(evaluate), optimizer_options={"gtol": 1e-4},
        fail_value=1e10)

    assert captured["value"] == 1e10
    assert np.linalg.norm(captured["gradient"]) > 0.0
    assert outcome.accepted is False
    assert outcome.evaluation is None


def test_unexpected_objective_error_propagates():
    def evaluate(parameters):
        raise RuntimeError("programming error")

    with pytest.raises(RuntimeError, match="programming error"):
        run_static_multivariate_mle(
            _problem(evaluate), optimizer_options={}, fail_value=1e10)


def test_final_objective_is_recomputed_and_mismatch_is_rejected(monkeypatch):
    calls = 0

    def evaluate(parameters):
        nonlocal calls
        calls += 1
        return StaticMLEEvaluation(
            objective=float(parameters[0] ** 2),
            gradient=np.array([2.0 * parameters[0]]),
        )

    def fake_minimize(fun, x0, **kwargs):
        trial = np.array([0.25])
        fun(trial)
        return SimpleNamespace(
            x=trial, fun=999.0, success=True, nfev=1, message="ok")

    monkeypatch.setattr(multivariate_mle, "minimize", fake_minimize)
    outcome = run_static_multivariate_mle(
        _problem(evaluate), optimizer_options={"gtol": 1.0},
        fail_value=1e10)

    assert calls == 3
    assert outcome.final_objective == pytest.approx(0.25 ** 2)
    assert outcome.objective_match is False
    assert outcome.accepted is False


@pytest.mark.parametrize("factory", [StudentCopula, lambda: StochasticStudentCopula(3)])
def test_failed_static_fit_does_not_publish_partial_state(monkeypatch, factory):
    rng = np.random.default_rng(7201)
    first = rng.uniform(0.05, 0.95, size=(40, 3))
    second = rng.uniform(0.05, 0.95, size=(35, 3))
    model = factory()
    original_result = model.fit(first, method="mle")
    original_last_u = model._last_u.copy()
    original_correlation = (
        model.shape.copy()
        if isinstance(model, StudentCopula) else model.R.copy())

    real_minimize = multivariate_mle.minimize

    def failed_minimize(fun, x0, **kwargs):
        value, _ = fun(x0)
        return SimpleNamespace(
            x=np.asarray(x0).copy(),
            fun=value,
            success=False,
            nfev=1,
            message="synthetic failure",
        )

    monkeypatch.setattr(multivariate_mle, "minimize", failed_minimize)
    failed = model.fit(second, method="mle")
    monkeypatch.setattr(multivariate_mle, "minimize", real_minimize)

    assert failed.success is False
    assert model.fit_result is original_result
    np.testing.assert_array_equal(model._last_u, original_last_u)
    actual_correlation = (
        model.shape if isinstance(model, StudentCopula) else model.R)
    np.testing.assert_array_equal(actual_correlation, original_correlation)


def test_static_student_optimizer_config_and_overrides_are_forwarded(
        monkeypatch):
    captured = {}
    real_minimize = multivariate_mle.minimize

    def spy_minimize(fun, x0, **kwargs):
        captured.update(kwargs["options"])
        return real_minimize(fun, x0, **kwargs)

    monkeypatch.setattr(multivariate_mle, "minimize", spy_minimize)
    config = NumericalConfig(
        static_student_optimizer=LBFGSBConfig(
            gtol=0.02, maxiter=80, maxls=12))
    observations = np.random.default_rng(7202).uniform(
        0.05, 0.95, size=(45, 3))

    result = StudentCopula().fit(
        observations, config=config, gtol=0.03, maxiter=40)

    assert result.success
    assert captured["gtol"] == pytest.approx(0.03)
    assert captured["maxiter"] == 40
    assert captured["maxls"] == 12


@pytest.mark.parametrize(
    "model",
    [GaussianCopula(), StudentCopula(), StochasticStudentCopula(3)],
)
def test_models_report_shared_static_strategy(model):
    observations = np.random.default_rng(7203).uniform(
        0.05, 0.95, size=(35, 3))
    result = model.fit(observations, method="mle")

    assert result.success
    assert result.diagnostics["static_mle_strategy"] == (
        "shared_multivariate")
    assert result.diagnostics["final_validation_passed"] is True
    assert result.diagnostics["objective_match"] is True


def test_model_modules_do_not_own_scipy_optimizer_loops():
    from pyscarcopula.copula.multivariate import gaussian, student
    from pyscarcopula.copula.multivariate import stochastic_student

    for module in (gaussian, student, stochastic_student):
        assert "minimize(" not in inspect.getsource(module)
