"""Contracts for the single native GAS numerical implementation."""

from types import SimpleNamespace

import numpy as np
import pytest

from pyscarcopula._types import GASResult, PredictiveState, gas_params
from pyscarcopula.copula.elliptical import BivariateGaussianCopula
from pyscarcopula.copula.gumbel import GumbelCopula
from pyscarcopula.copula.multivariate import EquicorrGaussianCopula
from pyscarcopula.numerical import _cpp_gas
from pyscarcopula.numerical._cpp_extension import CppUnavailable, CppUnsupported
from pyscarcopula.numerical.gas_filter import (
    gas_filter,
    gas_loglik,
    gas_mixture_h,
    gas_negloglik,
    gas_predict_param,
)
from pyscarcopula.strategy._base import get_strategy_for_result
from pyscarcopula.strategy.gas import GASStrategy


PARAMS = (0.03, 0.08, 0.72)


@pytest.fixture
def observations():
    return np.random.default_rng(20260613).uniform(
        0.01, 0.99, size=(12, 2))


def test_public_gas_operations_route_to_native_evaluator(
    monkeypatch,
    observations,
):
    copula = GumbelCopula()
    calls = []

    monkeypatch.setattr(
        _cpp_gas,
        "filter",
        lambda *args, **kwargs: (
            calls.append("filter") or (np.zeros(12), np.ones(12), 2.0)
        ),
    )
    monkeypatch.setattr(
        _cpp_gas,
        "log_likelihood",
        lambda *args, **kwargs: calls.append("loglik") or 3.0,
    )
    monkeypatch.setattr(
        _cpp_gas,
        "negative_log_likelihood",
        lambda *args, **kwargs: calls.append("negloglik") or -3.0,
    )
    monkeypatch.setattr(
        _cpp_gas,
        "predict_parameter",
        lambda *args, **kwargs: calls.append("predict") or 4.0,
    )
    monkeypatch.setattr(
        _cpp_gas,
        "h_path",
        lambda *args, **kwargs: calls.append("h") or np.full(12, 0.5),
    )

    assert gas_filter(*PARAMS, observations, copula)[2] == 2.0
    assert gas_loglik(*PARAMS, observations, copula) == 3.0
    assert gas_negloglik(*PARAMS, observations, copula) == -3.0
    assert gas_predict_param(*PARAMS, observations, copula) == 4.0
    np.testing.assert_allclose(
        gas_mixture_h(*PARAMS, observations, copula), 0.5)
    assert calls == ["filter", "loglik", "negloglik", "predict", "h"]


@pytest.mark.parametrize("backend", ["python", "cpp", "auto", "native", None])
def test_gas_backend_selector_is_removed(backend):
    with pytest.raises(TypeError, match="backend selection was removed"):
        GASStrategy(backend=backend)


def test_result_strategy_restoration_has_no_backend_state():
    result = GASResult(
        log_likelihood=0.0,
        method="GAS",
        copula_name="Gaussian",
        success=True,
        params=gas_params(*PARAMS),
    )

    strategy = get_strategy_for_result(result)

    assert isinstance(strategy, GASStrategy)
    assert not hasattr(strategy, "backend")
    assert not hasattr(result, "backend")


def test_unsupported_copula_fails_before_optimization(observations):
    custom = SimpleNamespace(
        name="custom-python-copula",
        _corr_num_params=lambda: 0,
    )

    with pytest.raises(CppUnsupported, match="custom-python-copula"):
        GASStrategy().fit(custom, observations, gamma0=np.asarray(PARAMS))


def test_missing_extension_has_no_python_fallback(monkeypatch, observations):
    def unavailable():
        raise CppUnavailable("compiled extension missing")

    monkeypatch.setattr(_cpp_gas._cpp_extension, "load", unavailable)

    with pytest.raises(CppUnavailable, match="compiled extension missing"):
        gas_filter(*PARAMS, observations, BivariateGaussianCopula())


def test_fit_checks_extension_before_optimization(monkeypatch, observations):
    def unavailable():
        raise CppUnavailable("compiled extension missing")

    def fail_minimize(*args, **kwargs):
        raise AssertionError("optimizer must not run without native GAS")

    monkeypatch.setattr(_cpp_gas._cpp_extension, "load", unavailable)
    monkeypatch.setattr(
        "pyscarcopula.strategy.gas.minimize", fail_minimize)

    with pytest.raises(CppUnavailable, match="compiled extension missing"):
        GASStrategy().fit(
            BivariateGaussianCopula(),
            observations,
            gamma0=np.asarray(PARAMS),
        )


def test_gas_diagnostics_distinguish_score_from_optimizer_gradient(
        monkeypatch, observations):
    captured = {}

    def fake_minimize(fun, x0, *, method, bounds, options):
        captured["value"] = fun(np.asarray(x0, dtype=np.float64))
        return SimpleNamespace(
            x=np.asarray(x0, dtype=np.float64),
            fun=float(captured["value"]),
            success=True,
            nfev=1,
            message="ok",
        )

    monkeypatch.setattr(
        "pyscarcopula.strategy.gas.minimize", fake_minimize)

    result = GASStrategy().fit(
        BivariateGaussianCopula(),
        observations,
        gamma0=np.asarray(PARAMS),
        maxiter=1,
    )

    assert np.isfinite(captured["value"])
    assert result.diagnostics["model_score"] == "native"
    assert result.diagnostics["optimizer_gradient"] == "numerical"
    assert result.diagnostics["gradient_kind"] == "numerical_optimizer"
    assert result.diagnostics["analytical_grad_used"] is False


def test_optimizer_objective_translates_native_failure(
    monkeypatch,
    observations,
):
    def fail(*args, **kwargs):
        raise FloatingPointError("native failure")

    monkeypatch.setattr(_cpp_gas, "negative_log_likelihood", fail)

    assert gas_negloglik(
        *PARAMS, observations, BivariateGaussianCopula()) == 1e10


def test_multivariate_filter_uses_native_evaluator(monkeypatch):
    copula = EquicorrGaussianCopula(d=4)
    observations = np.full((5, 4), 0.55, dtype=np.float64)
    calls = []

    def fake_filter(*args, **kwargs):
        calls.append("filter")
        return np.zeros(5), np.ones(5), 2.0

    monkeypatch.setattr(_cpp_gas, "filter", fake_filter)

    assert gas_filter(*PARAMS, observations, copula)[2] == 2.0
    assert calls == ["filter"]


@pytest.mark.parametrize(
    "copula",
    [BivariateGaussianCopula(), EquicorrGaussianCopula(d=4)],
)
def test_sampling_uses_native_initial_state_and_updates(monkeypatch, copula):
    result = GASResult(
        log_likelihood=0.0,
        method="GAS",
        copula_name=copula.name,
        success=True,
        params=gas_params(*PARAMS),
    )
    calls = []

    def fake_initial(*args, **kwargs):
        calls.append("initial")
        return _cpp_gas.GasStateOutput(g=0.0, parameter=0.0)

    def fake_update(*args, **kwargs):
        calls.append("update")
        return _cpp_gas.GasUpdateOutput(
            g_next=0.01,
            r=0.0,
            r_next=0.01,
            log_likelihood=0.0,
            score=0.0,
        )

    monkeypatch.setattr(_cpp_gas, "initial_state", fake_initial)
    monkeypatch.setattr(_cpp_gas, "update_one", fake_update)
    if not hasattr(copula, "sample"):
        pytest.skip("copula has no sampler")

    samples = GASStrategy().sample(
        copula,
        np.full((2, copula.d), 0.5),
        result,
        4,
        rng=np.random.default_rng(19),
    )

    assert samples.shape == (4, copula.d)
    assert calls == ["initial", "update", "update", "update"]


def test_model_and_conditioning_states_use_native_operations(monkeypatch):
    copula = BivariateGaussianCopula()
    result = GASResult(
        log_likelihood=0.0,
        method="GAS",
        copula_name=copula.name,
        success=True,
        params=gas_params(*PARAMS),
    )
    calls = []

    def fake_initial(*args, **kwargs):
        calls.append("initial")
        return _cpp_gas.GasStateOutput(g=0.2, parameter=0.05)

    def fake_update(*args, **kwargs):
        calls.append("update")
        return _cpp_gas.GasUpdateOutput(
            g_next=0.3,
            r=0.05,
            r_next=0.075,
            log_likelihood=0.0,
            score=0.0,
        )

    monkeypatch.setattr(_cpp_gas, "initial_state", fake_initial)
    monkeypatch.setattr(_cpp_gas, "update_one", fake_update)
    strategy = GASStrategy()

    model_state = strategy.model_sample_state(copula, result)
    conditioned = strategy.condition_state(
        copula,
        PredictiveState(
            method="GAS",
            horizon="next",
            kind="point",
            r=np.array([0.05]),
            metadata={"g": 0.2},
        ),
        np.array([[0.3, 0.7]]),
        result,
    )

    assert calls == ["initial", "update"]
    np.testing.assert_allclose(model_state.r, [0.05])
    assert model_state.metadata["g"] == pytest.approx(0.2)
    np.testing.assert_allclose(conditioned.r, [0.075])
    assert conditioned.metadata["g"] == pytest.approx(0.3)
