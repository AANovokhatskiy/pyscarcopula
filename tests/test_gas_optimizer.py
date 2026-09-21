"""GAS finite differences preserve the SciPy optimizer provider contract."""

import numpy as np
import pytest
from scipy.optimize import Bounds, minimize
from scipy.optimize._numdiff import approx_derivative

from pyscarcopula import (
    BivariateGaussianCopula, ClaytonCopula, FrankCopula, GumbelCopula,
    JoeCopula, StochasticStudentCopula,
)
from pyscarcopula._native import gas
from pyscarcopula._types import GASResult, LBFGSBConfig, NumericalConfig, gas_params
from pyscarcopula.strategy.gas import GASStrategy, _minimize_gas_objective


OBSERVATIONS = np.random.default_rng(7281).uniform(0.03, 0.97, (17, 2))
BOUNDS = ([-np.inf, -0.2, -0.78], [np.inf, 0.2, 0.78])
POINTS = (
    (0.13, 0.08, 0.75),
    (-0.2, -0.04, -0.72),
    (0.0, 0.0, 0.6),
    (0.01, 0.2, 0.78),
    (-0.02, -0.2, -0.78),
)


@pytest.mark.parametrize("family", [
    BivariateGaussianCopula, ClaytonCopula, FrankCopula, GumbelCopula,
    JoeCopula,
])
@pytest.mark.parametrize("scaling", ["unit", "fisher"])
@pytest.mark.parametrize("point", POINTS)
@pytest.mark.parametrize("step,relative", [
    (1e-5, False), (1e-3, False), (0.02, True), (1e-30, False),
])
def test_gas_gradient_matches_scipy_two_point_provider(
        family, scaling, point, step, relative):
    copula = family()
    points = []

    def objective(values):
        points.append(np.asarray(values).copy())
        return gas.negative_log_likelihood(
            *values, OBSERVATIONS, copula, scaling)

    expected = approx_derivative(
        objective, np.asarray(point), method="2-point", bounds=BOUNDS,
        **({"rel_step": step} if relative else {"abs_step": step}),
    )
    value, observed = gas.negative_log_likelihood_and_gradient(
        *point, OBSERVATIONS, copula, scaling,
        optimizer_gradient_eps=step,
        optimizer_gradient_relative=relative,
        optimizer_bounds=BOUNDS,
    )
    assert value == objective(np.asarray(point))
    np.testing.assert_allclose(observed, expected, rtol=2e-11, atol=2e-8)
    assert np.all(np.asarray(points) >= np.asarray(BOUNDS[0]))
    assert np.all(np.asarray(points) <= np.asarray(BOUNDS[1]))


@pytest.mark.parametrize("relative", [False, True])
def test_joint_shrinkage_gradient_matches_scipy_two_point_provider(relative):
    observations = np.random.default_rng(661).uniform(0.1, 0.9, (9, 3))
    copula = StochasticStudentCopula(d=3, corr_mode="shrinkage")
    copula._ensure_corr_initialized(observations)
    base = np.asarray(copula._corr_base)
    point = np.array([0.03, 0.2, 0.78, -0.4])
    bounds = ([-np.inf, -0.2, -0.78, -np.inf],
              [np.inf, 0.2, 0.78, np.inf])

    def objective(values):
        # Different prepared Student-cache paths are not an interchangeable
        # objective. Differentiate the joint owner's values with SciPy.
        return gas.negative_log_likelihood_and_gradient_shrinkage(
            *values[:3], values[3], base, observations, copula)[0]

    expected = approx_derivative(
        objective, point, method="2-point", bounds=bounds,
        **({"rel_step": 0.003} if relative else {"abs_step": 1e-5}),
    )
    value, observed = gas.negative_log_likelihood_and_gradient_shrinkage(
        *point[:3], point[3], base, observations, copula,
        optimizer_gradient_eps=0.003 if relative else 1e-5,
        optimizer_gradient_relative=relative,
        optimizer_bounds=bounds,
    )
    assert value == pytest.approx(objective(point), abs=1e-12)
    np.testing.assert_allclose(observed, expected, rtol=2e-8, atol=2e-7)


@pytest.mark.parametrize("maxfun", [1, 3, 4, 5, 12, 23])
def test_gas_optimizer_scalar_evaluation_budget_matches_scipy(maxfun):
    copula = GumbelCopula()
    point = np.array([0.13, 0.08, 0.75])
    options = dict(maxfun=maxfun, maxiter=20, ftol=1e-9, gtol=1e-3)
    calls = []

    def scalar(values):
        calls.append(np.asarray(values).copy())
        return gas.negative_log_likelihood(*values, OBSERVATIONS, copula)

    expected = minimize(
        scalar, point, method="L-BFGS-B", bounds=Bounds(*BOUNDS),
        options={**options, "eps": 1e-5},
    )
    observed = _minimize_gas_objective(
        lambda values: gas.negative_log_likelihood_and_gradient(
            *values, OBSERVATIONS, copula, optimizer_bounds=BOUNDS),
        point, bounds=Bounds(*BOUNDS), options=options,
    )
    assert observed.nfev == expected.nfev == len(calls)
    assert observed.nit == expected.nit
    assert observed.message == expected.message
    np.testing.assert_array_equal(observed.x, expected.x)
    assert observed.fun == expected.fun


@pytest.mark.parametrize("bounds", [
    ([0.0], [1.0]),
    ([-np.inf, np.nan, -0.9], [np.inf, 0.2, 0.9]),
    ([-np.inf, 0.2, -0.9], [np.inf, -0.2, 0.9]),
])
def test_gas_gradient_rejects_invalid_bounds(bounds):
    with pytest.raises(ValueError, match="optimizer_bounds"):
        gas.negative_log_likelihood_and_gradient(
            0.03, 0.08, 0.72, OBSERVATIONS, GumbelCopula(),
            optimizer_bounds=bounds)


@pytest.mark.parametrize("fit_options", [
    {"eps": 0.003}, {"finite_diff_rel_step": 0.004},
    {"eps": 0.003, "finite_diff_rel_step": 0.004},
])
def test_explicit_fit_step_reaches_native_with_custom_bounds(
        monkeypatch, fit_options):
    captured = []
    native = gas.negative_log_likelihood_and_gradient

    def record(*args, **kwargs):
        captured.append(kwargs)
        return native(*args, **kwargs)

    monkeypatch.setattr(gas, "negative_log_likelihood_and_gradient", record)
    config = NumericalConfig(gas_optimizer=LBFGSBConfig(eps=0.08))
    result = GASStrategy(config=config).fit(
        GumbelCopula(), OBSERVATIONS,
        gamma0=np.array([0.03, 0.08, 0.72]),
        gamma_bound=0.2, beta_bound=0.78, maxiter=1, ftol=1e-9,
        **fit_options,
    )
    relative = "finite_diff_rel_step" in fit_options
    expected_step = fit_options.get("finite_diff_rel_step", fit_options.get("eps"))
    assert captured
    for kwargs in captured:
        assert kwargs["optimizer_gradient_eps"] == expected_step
        assert kwargs["optimizer_gradient_relative"] is relative
        np.testing.assert_array_equal(kwargs["optimizer_bounds"], BOUNDS)
    assert result.diagnostics["optimizer_gradient_eps"] == expected_step


def _result(copula):
    return GASResult(
        method="GAS", copula_name=copula.name, success=True,
        log_likelihood=0.0, params=gas_params(0.03, 0.08, 0.72),
        r_last=gas.predict_parameter(0.03, 0.08, 0.72, OBSERVATIONS, copula),
    )


@pytest.mark.parametrize("kwargs", [
    {"unknown_option": 1}, {"score_eps": 0.03}, {"scaling": "fisher"},
    {"horizon": "next"}, {"maxiter": 1},
])
def test_direct_gas_predictive_mean_rejects_unknown_and_wrong_context(kwargs):
    copula = GumbelCopula()
    with pytest.raises(TypeError, match="predictive_mean"):
        GASStrategy().predictive_mean(
            copula, OBSERVATIONS, _result(copula), **kwargs)


@pytest.mark.parametrize("history", [None, np.empty((0, 2))])
@pytest.mark.parametrize("horizon", ["current", 0, "0"])
@pytest.mark.parametrize("operation", ["predictive_state", "predictive_params", "predict"])
def test_current_gas_prediction_requires_history(history, horizon, operation):
    copula = GumbelCopula()
    strategy = GASStrategy()
    args = (copula, history, _result(copula))
    if operation != "predictive_state":
        args += (3,)
    with pytest.raises(ValueError, match="history.*current"):
        getattr(strategy, operation)(*args, horizon=horizon)


def test_gas_saved_next_state_and_current_history_have_distinct_timing():
    copula = GumbelCopula()
    strategy = GASStrategy()
    result = _result(copula)
    next_state = strategy.predictive_state(copula, None, result)
    current = strategy.predictive_state(
        copula, OBSERVATIONS, result, horizon="current")
    expected = strategy.predictive_mean(copula, OBSERVATIONS, result)[-1]
    assert next_state.horizon == "next"
    assert next_state.r[0] == result.r_last
    assert current.horizon == "current"
    assert current.r[0] == expected
    assert abs(current.r[0] - next_state.r[0]) > 1e-6


@pytest.mark.parametrize("threads", [0, True, 1.5, "2"])
def test_fused_gas_sample_validates_threads_before_rng(threads):
    copula = GumbelCopula()

    class NoDraws:
        def uniform(self, *args, **kwargs):
            raise AssertionError("RNG must not run for invalid n_threads")

    with pytest.raises((TypeError, ValueError), match="n_threads"):
        GASStrategy().sample(
            copula, None, _result(copula), 3, rng=NoDraws(),
            n_threads=threads)


@pytest.mark.parametrize("point", POINTS)
def test_native_three_point_gradient_matches_independent_provider(point):
    model = GumbelCopula()
    def objective(values):
        return gas.negative_log_likelihood(*values, OBSERVATIONS, model)
    expected = approx_derivative(
        objective, np.asarray(point), method="3-point", abs_step=1e-5,
        bounds=BOUNDS)
    value, gradient = gas.negative_log_likelihood_and_gradient(
        *point, OBSERVATIONS, model, optimizer_gradient_eps=1e-5,
        optimizer_gradient_central=True, optimizer_bounds=BOUNDS)
    assert value == objective(np.asarray(point))
    np.testing.assert_allclose(gradient, expected, rtol=1e-7, atol=1e-6)


def test_mean_state_recovery_preserves_physical_objective_and_gradient():
    target = np.array([1.3, 0.01, 0.995])

    def objective(point):
        mu = point[0] / (1.0 - point[2])
        delta = np.array([mu, point[1], point[2]]) - target
        gradient = 2.0 * delta
        gradient[0] /= 1.0 - point[2]
        gradient[2] += mu * gradient[0]
        return float(np.dot(delta, delta)), gradient

    objective.gas_mean_parameterization = True
    objective.objective_scale = 100.0
    result = _minimize_gas_objective(
        objective, np.array([0.05, 0.03, 0.95]),
        bounds=Bounds([-np.inf, -20, -0.999], [np.inf, 20, 0.999]),
        options={"gtol": 1e-8, "ftol": 1e-14, "maxfun": 1000})
    expected_point = target.copy()
    expected_point[0] *= 1.0 - expected_point[2]
    assert result.success
    np.testing.assert_allclose(result.x, expected_point, atol=1e-8)
    value, gradient = objective(result.x)
    assert result.fun == pytest.approx(value, abs=1e-14)
    np.testing.assert_allclose(result.jac, gradient, atol=1e-10)


@pytest.mark.parametrize("parameters,gradient,scale", [
    ([0.1, 0.2], [], 1.0),
    ([0.1, 0.2, 1.0], [], 1.0),
    ([0.1, 0.2, 0.9], [1.0, 2.0], 1.0),
    ([0.1, 0.2, 0.9], [], 0.0),
])
def test_native_optimizer_coordinate_mapping_rejects_invalid_domain(
        parameters, gradient, scale):
    with pytest.raises(ValueError):
        gas.optimizer_coordinates(parameters, gradient=gradient,
                                  objective_scale=scale)


def test_three_point_stencil_is_formed_directly_in_mean_coordinates():
    from pyscarcopula import EquicorrGaussianCopula

    model = EquicorrGaussianCopula(d=2)
    point = np.array([0.006, 0.013, 0.995])
    mean_point = point.copy()
    mean_point[0] /= 1.0 - mean_point[2]
    step = gas.optimizer_validation_steps()[0]

    def objective(values):
        physical = values.copy()
        physical[0] *= 1.0 - physical[2]
        return gas.negative_log_likelihood(*physical, OBSERVATIONS, model)

    expected = approx_derivative(objective, mean_point, method="3-point", abs_step=step)
    value, physical_gradient = gas.negative_log_likelihood_and_gradient(
        *point, OBSERVATIONS, model, optimizer_gradient_eps=step,
        optimizer_gradient_central=True, optimizer_gradient_mean_coordinates=True)
    mapped = gas.optimizer_coordinates(point, gradient=physical_gradient)
    assert value == objective(mean_point)
    np.testing.assert_allclose(mapped["gradient"], expected, rtol=1e-6, atol=1e-5)


def test_joint_shrinkage_rejects_unimplemented_mean_coordinate_option():
    model = StochasticStudentCopula(d=2, corr_mode="shrinkage")
    with pytest.raises(TypeError, match="optimizer_gradient_mean_coordinates"):
        gas.negative_log_likelihood_and_gradient_shrinkage(
            0.03, 0.08, 0.72, -0.8, np.eye(2), OBSERVATIONS, model,
            optimizer_gradient_mean_coordinates=True)
