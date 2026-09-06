"""Contracts for shared static multivariate MLE orchestration."""

from types import SimpleNamespace

import numpy as np
import pytest

from pyscarcopula import (
    GaussianCopula,
    LBFGSBConfig,
    NumericalConfig,
    StochasticStudentCopula,
    StudentCopula,
)
from pyscarcopula._native import NativeUnsupported
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


@pytest.mark.parametrize('size,representation', [
    (0, 'complex'), (2, 'complex'), (2, 'object'), (2, 'nested')])
@pytest.mark.parametrize('imaginary', [0., .2])
def test_complex_initial_point_rejected_before_objective_or_optimizer(
        monkeypatch, size, representation, imaginary):
    values = np.full(size, 1. + imaginary * 1j)
    if representation == 'object':
        values = values.astype(object)
    elif representation == 'nested':
        outer = np.empty(size, dtype=object)
        for index, value in enumerate(values):
            outer[index] = np.array(value)
        values = outer

    def unexpected(*args, **kwargs):
        pytest.fail('complex initial point reached computation')

    monkeypatch.setattr(multivariate_mle, 'minimize', unexpected)
    problem = StaticMLEProblem('test', values, ((None, None),) * size, unexpected)
    with pytest.raises(TypeError, match='initial_parameters.*real'):
        run_static_multivariate_mle(problem, optimizer_options={}, fail_value=1e10)


def test_nested_real_initial_point_remains_supported():
    values = np.empty(2, dtype=object)
    values[0], values[1] = np.array(.7), np.array(-.4)
    values.setflags(write=False)
    problem = StaticMLEProblem(
        'quadratic', values, ((None, None),) * 2,
        lambda x: StaticMLEEvaluation(float(x @ x), 2 * x))
    outcome = run_static_multivariate_mle(problem, optimizer_options={}, fail_value=1e10)
    assert outcome.accepted
    np.testing.assert_allclose(outcome.parameters, [0., 0.], atol=1e-12)
    assert values[0].item() == .7 and values[1].item() == -.4
    assert not values.flags.writeable


@pytest.mark.parametrize('dtype', [np.float32, np.float64, object])
def test_real_initial_point_preserves_caller_storage(dtype):
    storage = np.array([1., 9., -1., 9.], dtype=dtype)
    values = storage[::2]
    values.setflags(write=False)
    before = storage.copy()
    problem = StaticMLEProblem(
        'quadratic', values, ((None, None),) * 2,
        lambda x: StaticMLEEvaluation(float(x @ x), 2 * x))
    outcome = run_static_multivariate_mle(
        problem, optimizer_options={}, fail_value=1e10)
    assert outcome.accepted
    np.testing.assert_allclose(outcome.parameters, [0., 0.], atol=1e-12)
    np.testing.assert_array_equal(storage, before)


@pytest.mark.parametrize('dimension', [0, 1])
@pytest.mark.parametrize('options,key', [
    ({'optimizer_typo': 3}, 'optimizer_typo'),
    ({'gtol': np.nan}, 'gtol'),
    ({'maxiter': 0}, 'maxiter'),
    ({'eps': np.complex128(1e-5 + .2j)}, 'eps'),
])
def test_invalid_options_rejected_before_initial_evaluation(dimension, options, key):
    calls = []

    def evaluate(parameters):
        calls.append(parameters)
        return StaticMLEEvaluation(float(np.sum(parameters ** 2)), 2 * parameters)

    problem = StaticMLEProblem('test', np.zeros(dimension),
                               ((None, None),) * dimension, evaluate)
    with pytest.raises((TypeError, ValueError), match=key):
        run_static_multivariate_mle(problem, optimizer_options=options, fail_value=1e10)
    assert calls == []


def test_typed_evaluator_failure_propagates_without_python_gradient():
    def evaluate(parameters):
        raise NativeUnsupported("synthetic typed evaluator failure")

    with pytest.raises(NativeUnsupported, match="typed evaluator failure"):
        run_static_multivariate_mle(
            _problem(evaluate), optimizer_options={"gtol": 1e-4},
            fail_value=1e10)


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


def test_abnormal_status_accepts_an_independently_valid_improved_point(
        monkeypatch):
    def evaluate(parameters):
        return StaticMLEEvaluation(
            objective=float(parameters[0] ** 2),
            gradient=np.array([2.0 * parameters[0]]),
        )

    def fake_minimize(fun, x0, **kwargs):
        final = np.array([0.0])
        value, _ = fun(final)
        return SimpleNamespace(
            x=final,
            fun=value,
            success=False,
            nfev=1,
            message="ABNORMAL: line search",
        )

    monkeypatch.setattr(multivariate_mle, "minimize", fake_minimize)
    problem = StaticMLEProblem(
        family="test",
        initial_parameters=np.array([1.0]),
        bounds=((None, None),),
        evaluate=evaluate,
    )
    outcome = run_static_multivariate_mle(
        problem, optimizer_options={"gtol": 1e-4}, fail_value=1e10)

    assert outcome.accepted is True
    assert outcome.optimizer_success is False
    assert "accepted by independent final validation" in outcome.message


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
