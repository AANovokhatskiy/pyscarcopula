"""Direct latent-strategy contracts and optimizer derivative controls."""

from functools import wraps

import numpy as np
import pytest

from pyscarcopula import (
    BivariateGaussianCopula, ClaytonCopula, FrankCopula, GumbelCopula,
    JoeCopula, StochasticStudentCopula, api,
)
from pyscarcopula._native import jacobi as native_jacobi
from pyscarcopula._types import (
    LBFGSBConfig, LatentResult, NumericalConfig, jacobi_params, ou_params,
)
from pyscarcopula.strategy import scar_jacobi, scar_tm
from pyscarcopula.strategy._base import get_strategy_for_result
from pyscarcopula.vine._edge_adapter import (
    edge_mixture_h_pair, predict_r_path,
)
from pyscarcopula.vine import VineCopula


FAMILIES = (
    ClaytonCopula, FrankCopula, GumbelCopula, JoeCopula,
    BivariateGaussianCopula,
)
U = np.random.default_rng(231).uniform(.12, .88, (12, 2))
OU_ALPHA = np.array([1.2, .3, .5])
JACOBI_ALPHA = np.array([1.2, .35, .5])
METHODS = ("SCAR-TM-OU", "SCAR-TM-JACOBI")


def _result(copula, method):
    common = dict(
        method=method, copula_name=copula.name, log_likelihood=0.,
        success=True,
    )
    if method == "SCAR-TM-OU":
        return LatentResult(
            params=ou_params(*OU_ALPHA), K=24, max_K=24,
            adaptive=False, transition_method="matrix", **common)
    return LatentResult(
        params=jacobi_params(*JACOBI_ALPHA), spectral_basis_order=4,
        spectral_quad_order=16, transition_method="local_fixed", **common)


@pytest.mark.parametrize("family", FAMILIES, ids=lambda family: family.__name__)
@pytest.mark.parametrize("gradient", [False, True])
def test_jacobi_numerical_failure_uses_configured_penalty(family, gradient):
    copula = family()
    alpha = np.array([.08, .15, .3])
    options = dict(
        basis_order=8, quad_order=48, transition_method="spectral_matrix",
        clip_negative=False,
    )
    evaluator = native_jacobi.PreparedScarJacobiEvaluator(U, copula, **options)
    with pytest.raises(FloatingPointError, match="numerical_failure") as error:
        evaluator.neg_loglik(*alpha)
    assert error.value.status == 7

    config = NumericalConfig(fail_value=987654.)
    if gradient:
        strategy = scar_jacobi.SCARJacobiStrategy(config=config, **options)
        value, derivative = strategy._neg_loglik_with_grad(*alpha, U, copula)
        np.testing.assert_array_equal(derivative, np.zeros(3))
    else:
        value = copula.mlog_likelihood(
            alpha, U, method="SCAR-TM-JACOBI", config=config, **options)
    assert value == config.fail_value


@pytest.mark.parametrize("family", FAMILIES, ids=lambda family: family.__name__)
@pytest.mark.parametrize("gradient", [False, True])
@pytest.mark.parametrize("failure", ["numerical", "domain"])
def test_jacobi_fit_rejects_final_native_failure(family, gradient, failure):
    copula = family()
    if failure == "numerical":
        alpha = [.08, .15, .3]
        options = dict(
            basis_order=8, quad_order=48, transition_method="spectral_matrix",
            clip_negative=False)
        error_type, status = FloatingPointError, 7
    else:
        alpha = JACOBI_ALPHA
        options = dict(
            basis_order=4, quad_order=16, transition_method="local_fixed",
            stationary_shape_max=1.)
        error_type, status = FloatingPointError, 6
    evaluator = native_jacobi.PreparedScarJacobiEvaluator(U, copula, **options)
    with pytest.raises(error_type) as initial_error:
        evaluator.neg_loglik(*alpha)
    if failure == "numerical":
        assert initial_error.value.status == status
    else:
        assert not native_jacobi.shape_is_supported(*alpha, 1.)

    result = copula.fit(
        U, method="SCAR-TM-JACOBI", alpha0=alpha,
        analytical_grad=gradient, config=NumericalConfig(fail_value=987654.),
        maxiter=1, maxfun=20, **options)

    assert not result.success
    assert "final objective evaluation failed" in result.message
    assert result.diagnostics["final_evaluation_status"] == status
    assert not result.diagnostics["final_objective_consistent"]
    if failure == "numerical":
        with pytest.raises(error_type) as final_error:
            api.log_likelihood(copula, U, result)
        assert final_error.value.status == status
    else:
        # Public likelihood retains its -inf convention outside the domain.
        assert api.log_likelihood(copula, U, result) == -np.inf


@pytest.mark.parametrize("gradient", [False, True])
@pytest.mark.parametrize("penalty", [1e10, 987654., "objective"])
def test_jacobi_successful_fit_is_independent_of_penalty(gradient, penalty):
    copula = GumbelCopula()
    options = dict(
        basis_order=4, quad_order=16, transition_method="local_fixed")
    kwargs = dict(
        method="SCAR-TM-JACOBI", alpha0=JACOBI_ALPHA,
        analytical_grad=gradient, gtol=10., maxiter=1, maxfun=20, **options)
    reference = copula.fit(U, **kwargs)
    assert reference.success
    if penalty == "objective":
        penalty = -reference.log_likelihood
        assert penalty > 0.
    result = copula.fit(U, config=NumericalConfig(fail_value=penalty), **kwargs)

    assert result.success
    assert result.diagnostics["final_evaluation_status"] == 0
    assert result.diagnostics["final_objective_consistent"]
    np.testing.assert_array_equal(result.params.values, reference.params.values)
    assert result.log_likelihood == reference.log_likelihood
    assert api.log_likelihood(copula, U, result) == result.log_likelihood


@pytest.mark.parametrize("gradient", [False, True])
def test_jacobi_numerical_failure_triggers_vine_mle_fallback(gradient):
    vine = VineCopula.cvine(2)
    vine.fit(
        U, method="SCAR-TM-JACOBI", copulas=[[(GumbelCopula, 0)]],
        alpha0=[.08, .15, .3], analytical_grad=gradient,
        config=NumericalConfig(fail_value=987654.),
        basis_order=8, quad_order=48, transition_method="spectral_matrix",
        clip_negative=False, maxiter=100, maxfun=400,
        dynamic_failure_policy="fallback")

    pair = next(iter(vine.pair_copulas.values()))
    assert pair.fit_result.method == "MLE"
    assert pair.fit_result.success
    assert pair.fit_diagnostics["fallback_used"]
    assert vine.log_likelihood(U) == pytest.approx(
        0.21746888719808538, rel=1e-10, abs=1e-12)


def test_jacobi_final_validation_rejects_gradient_failure(monkeypatch):
    copula = GumbelCopula()
    options = dict(
        basis_order=4, quad_order=16, transition_method="local_fixed")
    kwargs = dict(
        method="SCAR-TM-JACOBI", alpha0=JACOBI_ALPHA,
        analytical_grad=True, gtol=10., maxiter=1, maxfun=20, **options)
    reference = copula.fit(U, **kwargs)
    assert reference.success
    failing = native_jacobi.PreparedScarJacobiEvaluator(
        U, copula, basis_order=8, quad_order=48,
        transition_method="spectral_matrix", clip_negative=False)
    with pytest.raises(FloatingPointError) as error:
        failing.neg_loglik(.08, .15, .3)

    def failing_gradient(*args, **kwargs):
        raise error.value

    monkeypatch.setattr(
        native_jacobi.PreparedScarJacobiEvaluator, "neg_loglik_with_grad",
        failing_gradient)
    # Even a penalty equal to the valid scalar objective cannot validate a
    # gradient evaluation that failed. The real optimizer stops on the plateau.
    result = copula.fit(
        U, config=NumericalConfig(fail_value=-reference.log_likelihood), **kwargs)
    assert not result.success
    assert result.diagnostics["final_evaluation_status"] == 7
    assert not result.diagnostics["final_objective_consistent"]
    assert result.log_likelihood == reference.log_likelihood


@pytest.mark.parametrize("error_type", [ValueError, FloatingPointError])
def test_jacobi_final_validation_preserves_unstructured_errors(
        monkeypatch, error_type):
    original = scar_jacobi.minimize

    def broken_final_evaluation(*args, **kwargs):
        raise error_type("unstructured final evaluation error")

    def minimize_then_break_final(*args, **kwargs):
        result = original(*args, **kwargs)
        monkeypatch.setattr(
            native_jacobi.PreparedScarJacobiEvaluator, "neg_loglik",
            broken_final_evaluation)
        return result

    monkeypatch.setattr(scar_jacobi, "minimize", minimize_then_break_final)
    with pytest.raises(error_type, match="unstructured final evaluation error"):
        GumbelCopula().fit(
            U, method="SCAR-TM-JACOBI", alpha0=JACOBI_ALPHA,
            basis_order=4, quad_order=16, transition_method="local_fixed",
            analytical_grad=False, gtol=10., maxiter=1, maxfun=20)


HELPER_CASES = [
    (method, operation, short)
    for method in METHODS
    for operation in (
        ("predictive_state", "condition_state", "sample_params", "model_sample_params")
        if method == "SCAR-TM-OU" else
        ("mixture_h", "mixture_h_pair", "predictive_state", "condition_state",
         "sample_params", "model_sample_params")
    )
    for short in ((False,) if operation in {"mixture_h", "mixture_h_pair"} else (False, True))
]


@pytest.mark.parametrize("family", FAMILIES, ids=lambda family: family.__name__)
@pytest.mark.parametrize("method,operation,short", HELPER_CASES)
def test_latent_direct_helpers_reject_unknown_keywords(family, method, operation, short):
    copula = family()
    result = _result(copula, method)
    strategy = get_strategy_for_result(result)
    state = strategy.predictive_state(copula, None if short else U, result)
    count = 0 if short else 3
    arguments = {
        "mixture_h": (copula, U, result),
        "mixture_h_pair": (copula, U, result),
        "predictive_state": (copula, None if short else U, result),
        "condition_state": (copula, state, None if short else U[:1], result),
        "sample_params": (copula, state, count),
        "model_sample_params": (copula, result, count),
    }[operation]
    rng = np.random.default_rng(18)
    before = rng.bit_generator.state
    kwargs = {"rng": rng} if operation.endswith("sample_params") else {}
    with pytest.raises(TypeError, match="unexpected.*keyword"):
        getattr(strategy, operation)(
            *arguments, predictive_r_mdoe="histogram", **kwargs)
    assert rng.bit_generator.state == before


@pytest.mark.parametrize("family", FAMILIES, ids=lambda family: family.__name__)
@pytest.mark.parametrize("method", METHODS)
def test_latent_helpers_keep_shared_prediction_and_vine_context(family, method):
    copula = family()
    result = _result(copula, method)
    strategy = get_strategy_for_result(result)
    state_cache, posterior_cache = {}, {}
    context = dict(
        given={0: .4}, horizon="current", predictive_r_mode="histogram",
        n_threads=1, memory_budget_bytes=1_000_000,
        state_cache=state_cache, cache_key="current", posterior_cache=posterior_cache,
    )
    actual = strategy.predictive_params(
        copula, U, result, 9, rng=np.random.default_rng(19), **context)
    expected = strategy.predictive_params(
        copula, U, result, 9, horizon="current", predictive_r_mode="histogram",
        rng=np.random.default_rng(19))
    np.testing.assert_array_equal(actual, expected)
    assert "current" in state_cache
    from_edge = predict_r_path(
        copula, result, 9, u_train_pair=U, horizon="current",
        predictive_r_mode="histogram", state_cache=state_cache,
        cache_key="current", posterior_cache=posterior_cache,
        rng=np.random.default_rng(19))
    np.testing.assert_array_equal(from_edge, expected)
    pair = edge_mixture_h_pair(
        copula, result, U, state_cache=state_cache,
        current_cache_key="current", next_cache_key="next",
        posterior_cache=posterior_cache)
    assert all(value.shape == (len(U),) for value in pair)
    assert "next" in state_cache


@pytest.mark.parametrize("family", FAMILIES, ids=lambda family: family.__name__)
def test_jacobi_model_sampler_keeps_given_and_diagnostics(family):
    copula = family()
    result = _result(copula, "SCAR-TM-JACOBI")
    diagnostics = {}
    samples = api.sample(
        copula, U, result, 9, given={1: .6}, sampling_diagnostics=diagnostics,
        rng=np.random.default_rng(20))
    np.testing.assert_array_equal(samples[:, 1], .6)
    assert diagnostics["sampling_method"] == "tm_grid"


@pytest.mark.parametrize("family", FAMILIES, ids=lambda family: family.__name__)
@pytest.mark.parametrize("n_threads", [0, True, 1.5, "2"])
def test_ou_direct_sample_rejects_threads_before_advancing_rng(family, n_threads):
    copula = family()
    result = _result(copula, "SCAR-TM-OU")
    strategy = get_strategy_for_result(result)
    rng = np.random.default_rng(30)
    before = rng.bit_generator.state
    with pytest.raises((TypeError, ValueError), match="n_threads"):
        strategy.sample(copula, U, result, 9, rng=rng, n_threads=n_threads)
    assert rng.bit_generator.state == before


def _fit_case(family, parameterization):
    copula = family()
    if parameterization == "jacobi":
        return copula, scar_jacobi, "SCAR-TM-JACOBI", JACOBI_ALPHA, dict(
            basis_order=4, quad_order=16, transition_method="local_fixed")
    return copula, scar_tm, "SCAR-TM-OU", OU_ALPHA, dict(
        K=24, max_K=24, adaptive=False, transition_method="matrix",
        log_stationary_scale_optimization=parameterization == "ou_log")


def _optimizer_probes(monkeypatch, family, parameterization, *, relative, absolute,
                      channel="kwargs", analytical=False):
    copula, module, method, alpha, options = _fit_case(family, parameterization)
    original = module.minimize
    probes, controls = [], []

    @wraps(original)
    def observed(fun, x0, *args, **kwargs):
        controls.append(kwargs)

        def objective(point):
            probes.append(np.array(point, copy=True))
            return fun(point)

        return original(objective, x0, *args, **kwargs)

    relative_config = relative if channel == "config" else (
        .05 if channel == "override" else None)
    optimizer = LBFGSBConfig(eps=absolute, finite_diff_rel_step=relative_config)
    config = NumericalConfig(
        scar_optimizer=optimizer, bivariate_scar_optimizer=optimizer,
        bivariate_log_scar_optimizer=optimizer)
    kwargs = {"finite_diff_rel_step": relative} if channel != "config" else {}
    with monkeypatch.context() as patch:
        patch.setattr(module, "minimize", observed)
        result = api.fit(
            copula, U, method=method, alpha0=alpha, config=config,
            analytical_grad=analytical, maxiter=1, maxfun=24, maxls=4,
            **options, **kwargs)
    assert controls
    return np.array(probes), controls[0], result


@pytest.mark.parametrize("family", FAMILIES, ids=lambda family: family.__name__)
@pytest.mark.parametrize("parameterization", ["ou_raw", "ou_log", "jacobi"])
@pytest.mark.parametrize("channel", ["kwargs", "config", "override"])
def test_latent_relative_step_controls_real_optimizer_probes(
        family, parameterization, channel, monkeypatch):
    for relative in (.001, .003):
        probes, controls, _ = _optimizer_probes(
            monkeypatch, family, parameterization, relative=relative,
            absolute=.02, channel=channel)
        assert controls.get("jac") == "2-point"
        assert controls["options"]["finite_diff_rel_step"] == relative
        assert len(probes) >= 4
        # Explicit relative steps use the optimizer coordinates. These
        # interior starts avoid SciPy's zero-coordinate/boundary fallback.
        np.testing.assert_allclose(
            probes[1:4] - probes[0], np.diag(relative * probes[0]),
            rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("family", FAMILIES, ids=lambda family: family.__name__)
@pytest.mark.parametrize("parameterization", ["ou_raw", "ou_log", "jacobi"])
def test_latent_absolute_step_remains_default(family, parameterization, monkeypatch):
    first, controls, _ = _optimizer_probes(
        monkeypatch, family, parameterization, relative=None, absolute=2e-5)
    second, _, _ = _optimizer_probes(
        monkeypatch, family, parameterization, relative=None, absolute=4e-5)
    assert controls.get("jac") is None
    np.testing.assert_allclose(
        second[1:4] - second[0], 2 * (first[1:4] - first[0]),
        rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("family", FAMILIES, ids=lambda family: family.__name__)
@pytest.mark.parametrize("parameterization", ["ou_raw", "ou_log", "jacobi"])
def test_latent_relative_step_does_not_replace_native_gradient(
        family, parameterization, monkeypatch):
    _, original_control, original = _optimizer_probes(
        monkeypatch, family, parameterization, relative=None, absolute=2e-5,
        analytical=True)
    _, changed_control, changed = _optimizer_probes(
        monkeypatch, family, parameterization, relative=.2, absolute=.04,
        analytical=True)
    assert original_control["jac"] is changed_control["jac"] is True
    np.testing.assert_array_equal(original.params.values, changed.params.values)
    assert original.log_likelihood == changed.log_likelihood


@pytest.mark.parametrize("log_stationary", [False, True])
def test_joint_ou_relative_step_reaches_optimizer(monkeypatch, log_stationary):
    copula = StochasticStudentCopula(
        d=3, corr_mode="shrinkage", R=np.eye(3) * .7 + .3)
    data = np.random.default_rng(24).uniform(.15, .85, (12, 3))
    original = scar_tm.minimize
    probes, controls = [], []

    @wraps(original)
    def observed(fun, x0, *args, **kwargs):
        controls.append(kwargs)

        def objective(point):
            probes.append(np.array(point, copy=True))
            return fun(point)

        return original(objective, x0, *args, **kwargs)

    monkeypatch.setattr(scar_tm, "minimize", observed)
    api.fit(
        copula, data, method="SCAR-TM-OU", alpha0=OU_ALPHA,
        analytical_grad=False, finite_diff_rel_step=.002, eps=.03,
        log_stationary_scale_optimization=log_stationary,
        transition_method="matrix", K=24, max_K=24, adaptive=False,
        maxiter=1, maxfun=24, maxls=4)
    assert controls[0].get("jac") == "2-point"
    values = np.array(probes)
    size = len(values[0])
    assert size > 3 and len(values) >= size + 1
    # Correlation raw coordinates may start at zero and use SciPy's fallback;
    # the nonzero OU block must still consume the specified relative step.
    np.testing.assert_allclose(
        values[1:4, :3] - values[0, :3], np.diag(.002 * values[0, :3]),
        rtol=1e-9, atol=1e-12)
