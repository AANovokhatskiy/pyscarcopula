"""GAS starts, convergence validation, and Student objective consistency."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import Bounds
from scipy.special import ndtr

from pyscarcopula import EquicorrGaussianCopula, StochasticStudentCopula
from pyscarcopula._native import gas
from pyscarcopula.strategy.gas import GASStrategy, _fit_gas_starts


def _fit_with_single_threaded_blas(family, observations):
    """Keep numerical regressions reproducible without a runtime dependency.

    BLAS reads these settings on import, so changing the parent environment
    after NumPy has loaded is insufficient. Use a fresh interpreter and retain
    sys.path so installed-wheel checks still exercise the same package.
    """
    import os
    import pickle
    import subprocess
    import sys

    env = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "OPENBLAS_DEFAULT_NUM_THREADS",
                 "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS"):
        env[name] = "1"
    script = """
import pickle
import sys

paths, family, observations = pickle.load(sys.stdin.buffer)
sys.path[:] = paths
import pyscarcopula
from pyscarcopula.strategy.gas import GASStrategy

model = getattr(pyscarcopula, family)(d=observations.shape[1])
result = GASStrategy().fit(model, observations)
pickle.dump(dict(log_likelihood=result.log_likelihood, success=result.success,
                 diagnostics=result.diagnostics, message=result.message),
            sys.stdout.buffer)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        input=pickle.dumps((sys.path, family.__name__, observations)),
        capture_output=True, env=env, timeout=180,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    return SimpleNamespace(**pickle.loads(completed.stdout))


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("family", [EquicorrGaussianCopula, StochasticStudentCopula])
def test_inherited_start_preserves_budget_and_explicit_start_precedence(monkeypatch, explicit, family):
    from pyscarcopula._types import gas_params
    from pyscarcopula.strategy import gas as strategy_module

    inherited = SimpleNamespace(params=gas_params(omega=.01, gamma=.02, beta=.8))
    calls = []
    original = strategy_module._minimize_gas_objective

    def recorded(objective, initial, *, bounds, options):
        calls.append(dict(options))
        return original(objective, initial, bounds=bounds, options=options)

    monkeypatch.setattr(strategy_module, "_minimize_gas_objective", recorded)
    observations = np.random.default_rng(924).uniform(.1, .9, (24, 3))
    result = GASStrategy().fit(
        family(d=3), observations,
        initial_gas_result=inherited,
        gamma0=np.array([.02, .03, .7]) if explicit else None,
        maxfun=40, maxiter=3, maxls=5)
    stages = result.diagnostics["optimizer_stages"]
    assert result.diagnostics["automatic_multistart"] is (not explicit)
    if explicit:
        assert all(stage["stage"] in {"standard", "refinement"} for stage in stages)
        np.testing.assert_array_equal(stages[0]["initial_params"], [.02, .03, .7])
    else:
        inherited_stage = next(stage for stage in stages if stage["stage"] == "inherited")
        np.testing.assert_array_equal(inherited_stage["initial_params"], inherited.params.values)
        assert any(stage["stage"] == "nested_static" for stage in stages)
    assert calls
    assert all((options["maxfun"], options["maxiter"], options["maxls"]) == (40, 3, 5)
               for options in calls)
    np.testing.assert_array_equal(inherited.params.values, [.01, .02, .8])


@pytest.mark.parametrize("override", [
    {"ftol": 1e-4}, {"eps": 1e-6}, {"finite_diff_rel_step": 1e-5},
])
def test_inherited_start_does_not_override_explicit_numerical_settings(override):
    from pyscarcopula._types import gas_params

    inherited = SimpleNamespace(params=gas_params(omega=.01, gamma=.02, beta=.8))
    observations = np.random.default_rng(924).uniform(.1, .9, (24, 3))
    result = GASStrategy().fit(
        EquicorrGaussianCopula(3), observations, initial_gas_result=inherited,
        maxfun=40, maxiter=3, **override)
    assert all(not stage["stage"].startswith("recovery_")
               for stage in result.diagnostics["optimizer_stages"])
    if "ftol" in override:
        assert "optimizer_refinement" not in result.diagnostics
    else:
        assert result.diagnostics["optimizer_gradient_eps"] == next(iter(override.values()))
        assert result.diagnostics["optimizer_gradient_relative"] is (
            "finite_diff_rel_step" in override)


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


def test_joint_shrinkage_workspace_does_not_mutate_the_source_or_leak_between_calls():
    observations = np.random.default_rng(66).uniform(0.001, 0.999, (80, 4))
    model = StochasticStudentCopula(d=4, corr_mode="shrinkage")
    model._ensure_corr_initialized(observations)
    point = np.array([0.03, 0.02, 0.95])
    reported = gas.negative_log_likelihood(*point, observations, model)

    def joint(raw):
        return gas.negative_log_likelihood_and_gradient_shrinkage(
            *point, raw, model._corr_base, observations, model)

    expected = joint(-0.8)
    joint(2.0)
    observed = joint(-0.8)
    assert observed[0] == expected[0]
    np.testing.assert_array_equal(observed[1], expected[1])
    assert gas.negative_log_likelihood(*point, observations, model) == reported


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


@pytest.mark.parametrize("point", [
    [1.0623299700264952, 0.034202001931613414, 0.9500386085738329],
    [-1.0623299700264952, 0.034202001931613414, 0.9500386085738329],
])
def test_saturated_equicorr_is_finite_but_not_an_optimizer_plateau(point):
    observations = np.array([[0.2, 0.7], [0.4, 0.6], [0.8, 0.3]])
    model = EquicorrGaussianCopula(d=2)
    # Stable density remains available for evaluation at extreme states.
    assert np.isfinite(gas.negative_log_likelihood(*point, observations, model))
    # Its transform has no representable sensitivity; returning a finite
    # objective with a zero gradient caused false relative-function stops.
    with pytest.raises(FloatingPointError, match="numerical_failure"):
        gas.negative_log_likelihood_and_gradient(*point, observations, model)


def test_relative_function_stop_requires_independent_stationarity(monkeypatch):
    def objective(point):
        return float(np.dot(point, point)), 2 * point

    def stopped(fun, point, **kwargs):
        value, gradient = fun(point)
        return SimpleNamespace(x=point.copy(), fun=value, jac=gradient,
                               success=True, nfev=4, message="relative reduction")

    monkeypatch.setattr("pyscarcopula.strategy.gas._minimize_gas_objective", stopped)
    result, diagnostics = _fit_gas_starts(
        objective, np.array([0.1, 0.05, 0.95]),
        bounds=Bounds([-np.inf, -20, -0.999], [np.inf, 20, 0.999]),
        options={"ftol": 1e-9, "gtol": 1e-3}, automatic=False, refine=False,
        validation_objective=objective)
    assert not result.success
    assert result.raw_optimizer_success
    assert not diagnostics["stationarity_validation"]["passed"]
    assert "stationarity" in result.message


def test_stationarity_rejects_disagreement_between_difference_scales(monkeypatch):
    def coarse(point):
        return 1.0, np.array([0.0008, 0.0, 0.0])

    def fine(point):
        return 1.0, np.array([-0.0008, 0.0, 0.0])

    coarse.finer_provider = fine

    def stopped(fun, point, **kwargs):
        value, gradient = fun(point)
        return SimpleNamespace(x=point.copy(), fun=value, jac=gradient,
                               success=True, nfev=4, message="relative reduction")

    monkeypatch.setattr("pyscarcopula.strategy.gas._minimize_gas_objective", stopped)
    result, diagnostics = _fit_gas_starts(
        coarse, np.array([0.1, 0.05, 0.95]),
        bounds=Bounds([-np.inf, -20, -0.999], [np.inf, 20, 0.999]),
        options={"ftol": 1e-9, "gtol": 1e-3}, automatic=False, refine=False,
        validation_objective=coarse)
    check = diagnostics["stationarity_validation"]
    assert check["projected_gradient_inf_norm"] < check["gtol"]
    assert check["gradient_step_discrepancy"] > check["gtol"]
    assert not result.success


def test_recovery_starts_from_better_retained_trial(monkeypatch):
    starts = []
    def objective(point):
        return float(np.dot(point, point)), 2 * point

    def stopped(fun, point, **kwargs):
        starts.append(point.copy())
        value, gradient = fun(point)
        if len(starts) == 1:
            fun(point / 2)  # line search found this, but its final stop lost it
            return SimpleNamespace(x=point.copy(), fun=value, jac=gradient,
                                   success=True, nfev=8, message="relative reduction")
        optimum = np.zeros(3)
        value, gradient = fun(optimum)
        return SimpleNamespace(x=optimum, fun=value, jac=gradient,
                               success=True, nfev=8, message="gradient tolerance")

    monkeypatch.setattr("pyscarcopula.strategy.gas._minimize_gas_objective", stopped)
    initial = np.array([0.1, 0.05, 0.95])
    result, diagnostics = _fit_gas_starts(
        objective, initial,
        bounds=Bounds([-np.inf, -20, -0.999], [np.inf, 20, 0.999]),
        options={"ftol": 1e-9, "gtol": 1e-3}, automatic=False, refine=False,
        recovery_objectives=[objective], validation_objective=objective)
    np.testing.assert_array_equal(starts[1], initial / 2)
    assert result.success
    assert result.fun == 0
    assert diagnostics["stationarity_validation"]["passed"]


def test_validation_uses_the_same_material_trial_selection_as_the_result(monkeypatch):
    starts = []

    def objective(point):
        return (0.9995 if point[0] else 1.0), np.zeros(3)

    def validation(point):
        return objective(point)[0], np.array([0.0 if point[0] else 1.0, 0.0, 0.0])

    def stopped(fun, point, **kwargs):
        starts.append(point.copy())
        value, gradient = fun(point)
        if len(starts) == 1:
            fun(np.array([1.0, 0.0, 0.0]))
        return SimpleNamespace(x=point.copy(), fun=value, jac=gradient,
                               success=True, nfev=8, message="relative reduction")

    monkeypatch.setattr("pyscarcopula.strategy.gas._minimize_gas_objective", stopped)
    result, diagnostics = _fit_gas_starts(
        objective, np.zeros(3),
        bounds=Bounds([-np.inf, -20, -0.999], [np.inf, 20, 0.999]),
        options={"ftol": 1e-9, "gtol": 1e-3}, automatic=False, refine=False,
        recovery_objectives=[objective], validation_objective=validation)
    assert len(starts) == 2  # tiny better trial cannot suppress recovery
    np.testing.assert_array_equal(starts[1], np.zeros(3))
    assert not diagnostics["retained_best_trial"]
    assert not result.success


def test_equal_objective_prefers_the_converged_candidate(monkeypatch):
    starts = []

    def objective(point):
        return 1.0, np.zeros(3)

    def stopped(fun, point, **kwargs):
        starts.append(point.copy())
        value, gradient = fun(point)
        return SimpleNamespace(x=point.copy(), fun=value, jac=gradient,
                               success=len(starts) > 1, nfev=4, message="stopped")

    monkeypatch.setattr("pyscarcopula.strategy.gas._minimize_gas_objective", stopped)
    result, _ = _fit_gas_starts(
        objective, np.array([0.1, 0.05, 0.95]),
        bounds=Bounds([-np.inf, -20, -0.999], [np.inf, 20, 0.999]),
        options={"ftol": 1e-9}, automatic=True, refine=False)
    assert result.success
    assert result.x[1] == 0.0


@pytest.mark.parametrize("loss,accepted", [(0.0005, True), (0.002, False)])
def test_stationary_recovery_may_only_sacrifice_immaterial_likelihood(
        monkeypatch, loss, accepted):
    starts = []

    def objective(point):
        return 1.0 + loss * point[0], np.array([0.0 if point[0] else 1.0, 0.0, 0.0])

    def stopped(fun, point, **kwargs):
        starts.append(point.copy())
        terminal = np.array([1.0, 0.0, 0.0]) if len(starts) > 1 else point.copy()
        value, gradient = fun(terminal)
        return SimpleNamespace(x=terminal, fun=value, jac=gradient,
                               success=True, nfev=4, message="stopped")

    monkeypatch.setattr("pyscarcopula.strategy.gas._minimize_gas_objective", stopped)
    result, diagnostics = _fit_gas_starts(
        objective, np.zeros(3),
        bounds=Bounds([-np.inf, -20, -0.999], [np.inf, 20, 0.999]),
        options={"ftol": 1e-9, "gtol": 1e-3}, automatic=False, refine=False,
        recovery_objectives=[objective], validation_objective=objective)
    assert result.success is accepted
    assert result.x[0] == float(accepted)
    assert diagnostics["stationary_selection_loglik_loss"] == pytest.approx(
        loss if accepted else 0.0)


@pytest.mark.parametrize("gradients,used,passed", [
    ([0.0015, 0.0008, 0.0003, 0.0002], 3, True),
    ([0.005, 0.0051, 0.0, 0.0], 2, False),
])
def test_bounded_difference_refinement_only_resolves_threshold_uncertainty(
        monkeypatch, gradients, used, passed):
    providers = []
    for value in gradients:
        def provider(point, derivative=value):
            return 1.0, np.array([derivative, 0.0, 0.0])
        providers.append(provider)
    providers[0].finer_provider = providers[1]
    providers[0].refinement_providers = providers[2:]

    def stopped(fun, point, **kwargs):
        value, gradient = fun(point)
        return SimpleNamespace(x=point.copy(), fun=value, jac=gradient,
                               success=True, nfev=4, message="stopped")

    monkeypatch.setattr("pyscarcopula.strategy.gas._minimize_gas_objective", stopped)
    result, diagnostics = _fit_gas_starts(
        providers[0], np.zeros(3),
        bounds=Bounds([-np.inf, -20, -0.999], [np.inf, 20, 0.999]),
        options={"ftol": 1e-9, "gtol": 1e-3}, automatic=False, refine=False,
        validation_objective=providers[0])
    assert result.success is passed
    check = diagnostics["stationarity_validation"]
    assert len(check["all_projected_gradient_norms"]) == used
    assert len(check["projected_gradient_norms"]) == 2


def test_bivariate_fits_preserve_the_existing_stationarity_policy(monkeypatch):
    from pyscarcopula import BivariateGaussianCopula

    def unexpected_validation():
        raise AssertionError("multivariate recovery must not change vine-edge fits")

    monkeypatch.setattr(gas, "optimizer_validation_steps", unexpected_validation)
    observations = np.random.default_rng(7281).uniform(0.03, 0.97, (30, 2))
    result = GASStrategy().fit(BivariateGaussianCopula(), observations)
    assert result.diagnostics["stationarity_validation"] is None
    assert not any(stage["stage"].startswith("recovery_")
                   for stage in result.diagnostics["optimizer_stages"])


@pytest.mark.data
def test_high_frequency_equicorr_default_fit_avoids_saturated_line_search():
    from pathlib import Path
    import pandas as pd
    from pyscarcopula._utils import pobs

    path = Path(__file__).resolve().parents[1] / "data" / "btc_eth_combined_30m.csv"
    if not path.exists():
        pytest.skip("high-frequency regression data unavailable")
    prices = pd.read_csv(path, index_col=0)[["BTC_close", "ETH_close"]]
    observations = pobs(np.log(prices / prices.shift(1)).iloc[1:12001].dropna().values)
    result = _fit_with_single_threaded_blas(EquicorrGaussianCopula, observations)
    # The regression stopped at 8199.377 with a raw gradient above 3700.
    assert result.log_likelihood >= 8279.35
    assert result.success
    check = result.diagnostics["stationarity_validation"]
    assert check["passed"]
    assert check["objective_scale"] == len(observations)


@pytest.mark.data
def test_student_default_fit_recovers_and_reports_stationarity(crypto_data_6d):
    result = _fit_with_single_threaded_blas(StochasticStudentCopula, crypto_data_6d)
    # A retained trial at 824.972 used to terminate the entire fit.
    assert result.log_likelihood >= 827.3
    assert any(stage["stage"].startswith("recovery_")
               for stage in result.diagnostics["optimizer_stages"])
    check = result.diagnostics["stationarity_validation"]
    assert result.success
    assert check["passed"]
    assert len(check["gradient_steps"]) == 2
    assert max(check["projected_gradient_norms"]) <= check["gtol"]
    assert check["gradient_step_discrepancy"] <= check["gtol"]


@pytest.mark.data
@pytest.mark.parametrize("dataset,minimum_loglik", [("hf", 8272.15), ("us6", 2232.299)])
def test_student_stationarity_is_robust_to_near_unit_persistence_and_noisy_score(
        dataset, minimum_loglik):
    from pathlib import Path
    import pandas as pd
    from pyscarcopula._utils import pobs

    root = Path(__file__).resolve().parents[1] / "data"
    path = root / ("btc_eth_combined_30m.csv" if dataset == "hf" else "us_equity_prices.csv")
    if not path.exists():
        pytest.skip("Student regression data unavailable")
    columns = (["BTC_close", "ETH_close"] if dataset == "hf"
               else ["AAPL", "JPM", "JNJ", "XOM", "PG", "BA"])
    prices = pd.read_csv(path, index_col=0, sep="," if dataset == "hf" else ";")[columns]
    returns = np.log(prices / prices.shift(1)).iloc[1:]
    if dataset == "hf":
        returns = returns.iloc[:12000]
    observations = pobs(returns.dropna().values)
    result = _fit_with_single_threaded_blas(StochasticStudentCopula, observations)
    assert result.log_likelihood >= minimum_loglik
    assert result.success
    check = result.diagnostics["stationarity_validation"]
    assert max(check["projected_gradient_norms"]) <= check["gtol"]
    assert check["gradient_step_discrepancy"] <= check["gtol"]
    assert len(check["projected_gradient_norms"]) == 2
    assert result.diagnostics["stationary_selection_loglik_loss"] <= 0.001
