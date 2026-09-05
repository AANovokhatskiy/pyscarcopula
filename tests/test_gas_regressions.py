"""GAS starts, convergence validation, and Student objective consistency."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import Bounds
from scipy.special import ndtr

from pyscarcopula import EquicorrGaussianCopula, StochasticStudentCopula
from pyscarcopula._native import gas
from pyscarcopula.strategy.gas import GASStrategy, _fit_gas_starts


def test_joint_shrinkage_objective_uses_the_reported_ppf_cache():
    observations = np.random.default_rng(66).uniform(0.001, 0.999, (80, 4))
    model = StochasticStudentCopula(d=4, corr_mode="shrinkage")
    model._ensure_corr_initialized(observations)
    raw = -0.8
    model._set_corr_from_params(np.array([raw]))
    point = np.array([0.03, 0.02, 0.95])
    joint, _ = gas.negative_log_likelihood_and_gradient_shrinkage(
        *point, raw, model._corr_base, observations, model)
    reported = gas.negative_log_likelihood(*point, observations, model)
    assert joint == pytest.approx(reported, abs=1e-10, rel=0)


def test_equicorr_fit_preserves_the_nested_static_model():
    rng = np.random.default_rng(482)
    normal = 0.8 * rng.normal(size=(500, 1)) + 0.6 * rng.normal(size=(500, 6))
    observations = ndtr(normal)
    result = GASStrategy().fit(EquicorrGaussianCopula(d=6), observations)
    assert result.diagnostics["automatic_multistart"]
    assert {stage["stage"] for stage in result.diagnostics["optimizer_stages"]} >= {
        "standard", "nested_static"}
    assert result.log_likelihood >= result.diagnostics["nested_static_log_likelihood"] - 1e-3
    assert result.log_likelihood >= result.diagnostics["initial_static_log_likelihood"] - 1e-3
    assert result.diagnostics["objective_discrepancy"] == pytest.approx(0, abs=1e-6)


def test_false_optimizer_success_below_static_is_rejected():
    rng = np.random.default_rng(871)
    normal = rng.normal(size=(350, 1)) + 0.5 * rng.normal(size=(350, 6))
    observations = ndtr(normal)
    model = EquicorrGaussianCopula(d=6)
    point = np.array([0.01, 0.2, 0.95])
    value = gas.negative_log_likelihood(*point, observations, model)
    optimizer = SimpleNamespace(success=True, message="relative reduction", nfev=4,
                                fun=value, x=point, jac=np.ones(3))
    result = GASStrategy()._build_result(
        model, observations, optimizer, point, 1e-4, 20.0, 0.999)
    assert not result.success
    assert result.diagnostics["optimizer_success"]
    assert "below the nested static" in result.message


def test_better_nonconverged_start_is_not_replaced_by_worse_success(monkeypatch):
    calls = []

    def objective(point):
        return float(point[1]), np.array([0.0, 1.0, 0.0])

    def minimize(fun, point, **kwargs):
        value, gradient = fun(point)
        calls.append(point.copy())
        return SimpleNamespace(x=point.copy(), fun=value, jac=gradient,
                               success=bool(point[1]), nfev=4, message="stopped")

    monkeypatch.setattr("pyscarcopula.strategy.gas._minimize_gas_objective", minimize)
    result, diagnostics = _fit_gas_starts(
        objective, np.array([0.1, 0.05, 0.95]),
        bounds=Bounds([-np.inf, -20, -0.999], [np.inf, 20, 0.999]),
        options={"ftol": 1e-9}, automatic=True, refine=False)
    assert len(calls) == 2
    assert result.fun == 0.0
    assert not result.success
    assert result.nfev == 8
    assert diagnostics["automatic_multistart"]
    assert diagnostics["initial_static_log_likelihood"] == 0.0
