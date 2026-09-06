"""Fitted settings and explicit options retain their owners during GoF."""

from dataclasses import replace
from functools import wraps
from types import SimpleNamespace

import numpy as np
import pytest

from pyscarcopula import (
    BivariateGaussianCopula,
    ClaytonCopula,
    EquicorrGaussianCopula,
    GaussianCopula,
    StochasticStudentCopula,
    StudentCopula,
    VineCopula,
)
from pyscarcopula._types import (
    GASResult, LatentResult, NumericalConfig, gas_params, jacobi_params, ou_params,
)
from pyscarcopula.stattests import cvm_test, gof_test


def _observations():
    return np.random.default_rng(863).uniform(0.08, 0.92, (28, 2))


def _dynamic_result(method):
    common = dict(success=True, log_likelihood=0.0, copula_name="Gaussian")
    if method == "GAS":
        return GASResult(
            method=method, params=gas_params(0.2, 0.04, 0.65),
            scaling="unit", score_eps=0.02, **common)
    if method == "SCAR-TM-JACOBI":
        return LatentResult(
            method=method, params=jacobi_params(2.0, 0.4, 0.25),
            transition_method="local", spectral_basis_order=8,
            spectral_quad_order=24, **common)
    return LatentResult(
        method=method, params=ou_params(2.0, 0.4, 0.25),
        K=17, grid_range=3.4, adaptive=False,
        transition_method="matrix", **common)


@pytest.mark.parametrize("model_class", [GaussianCopula, StudentCopula])
@pytest.mark.parametrize("factor", [False, True])
def test_static_gof_accepts_explicit_result_on_unfitted_prototype(
        model_class, factor):
    u = _observations()
    options = dict(corr_mode="factor", factor_rank=1) if factor else {}
    fitted = model_class(2, **options)
    result = fitted.fit(u, to_pobs=False)
    expected = gof_test(fitted, u, to_pobs=False)
    prototype = model_class(2, **options)

    actual = gof_test(prototype, u, to_pobs=False, fit_result=result)

    assert actual.statistic == pytest.approx(expected.statistic, abs=1e-13)
    assert prototype.fit_result is None
    if factor:
        assert prototype.factor_loadings_ is None


@pytest.mark.parametrize("model_class", [GaussianCopula, StudentCopula])
def test_static_gof_explicit_result_overrides_attached_correlation(model_class):
    import pyscarcopula.stattests as st

    u = _observations()
    model = model_class(2)
    original = model.fit(u, to_pobs=False)
    correlation = np.array([[1.0, 0.85], [0.85, 1.0]])
    override = replace(original, correlation_matrix=correlation)
    transform = (
        st.gaussian_rosenblatt_transform(correlation, u)
        if model_class is GaussianCopula else
        st.student_rosenblatt_transform(correlation, override.copula_param, u))

    actual = gof_test(model, u, to_pobs=False, fit_result=override)

    assert actual.statistic == pytest.approx(cvm_test(transform).statistic)
    assert model.fit_result is original
    np.testing.assert_array_equal(
        model.corr if model_class is GaussianCopula else model.shape,
        original.correlation_matrix)


@pytest.mark.parametrize("model_class", [ClaytonCopula, BivariateGaussianCopula])
def test_gas_gof_preserves_fitted_fisher_score_step(model_class):
    from pyscarcopula.numerical.gas_filter import gas_rosenblatt

    u = _observations()
    model = model_class()
    result = replace(
        _dynamic_result("GAS"), params=gas_params(0.2, 0.12, 0.65),
        scaling="fisher", score_eps=0.2)
    expected = cvm_test(gas_rosenblatt(
        *result.params.values, u, model, scaling="fisher", score_eps=0.2))

    actual = gof_test(model, u, to_pobs=False, fit_result=result)

    assert actual.statistic == pytest.approx(expected.statistic, abs=1e-13)


@pytest.mark.parametrize("kind", ["equicorr", "student"])
def test_dynamic_gof_preserves_auto_threshold_at_native_consumer(
        kind, monkeypatch):
    from pyscarcopula._native import scar_ou as native

    model = (EquicorrGaussianCopula(2) if kind == "equicorr" else
             StochasticStudentCopula(2, R=np.eye(2)))
    result = replace(
        _dynamic_result("SCAR-TM-OU"), transition_method="auto",
        auto_small_kdt=0.3)
    seen = []
    original = native._call_vector

    def capture(evaluator, prefix, method, params, spec, u, config):
        seen.append(config.auto_small_kdt)
        return original(evaluator, prefix, method, params, spec, u, config)

    monkeypatch.setattr(native, "_call_vector", capture)
    actual = gof_test(
        model, _observations(), to_pobs=False, fit_result=result,
        K=17, grid_range=3.4)

    assert np.isfinite(actual.statistic)
    assert seen == [0.3]


@pytest.mark.parametrize("kind", ["equicorr", "student"])
@pytest.mark.parametrize("wrong_method", [True, False])
def test_dynamic_multivariate_gof_rejects_non_ou_process(kind, wrong_method):
    model = (EquicorrGaussianCopula(2) if kind == "equicorr" else
             StochasticStudentCopula(2, R=np.eye(2)))
    result = _dynamic_result("SCAR-TM-JACOBI")
    if not wrong_method:
        result = replace(result, method="SCAR-TM-OU")

    with pytest.raises(ValueError, match="SCAR-TM-OU result"):
        gof_test(model, _observations(), to_pobs=False, fit_result=result)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_bootstrap_rejects_failed_refits_after_bounded_retry(n_jobs):
    u = _observations()
    model = BivariateGaussianCopula()
    result = model.fit(u, to_pobs=False, method="mle")

    with pytest.raises(RuntimeError, match="refit did not converge after 2 attempts"):
        gof_test(
            model, u, to_pobs=False, fit_result=result, bootstrap=True,
            n_bootstrap=2, bootstrap_fit_kwargs={"maxiter": 1, "maxfun": 2},
            rng=221, n_jobs=n_jobs)


@pytest.mark.parametrize("bad", [
    {"unknown_option": 8}, {"K": 17}, {"tol": 1e-6}, {"to_pobs": False},
])
@pytest.mark.parametrize("refit", [False, True])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_bootstrap_rejects_misowned_kwargs_before_rng_or_workers(
        bad, refit, n_jobs, monkeypatch):
    import pyscarcopula.stattests as st

    u = _observations()
    model = BivariateGaussianCopula()
    result = model.fit(u, to_pobs=False, method="mle")

    def unexpected(*args, **kwargs):
        pytest.fail("invalid fit options reached bootstrap execution")

    monkeypatch.setattr(st, "spawn_seed_sequences", unexpected)
    monkeypatch.setattr(st, "_bootstrap_gof_worker", unexpected)
    with pytest.raises(TypeError):
        gof_test(
            model, u, to_pobs=False, fit_result=result, bootstrap=True,
            n_bootstrap=2, bootstrap_refit=refit, bootstrap_fit_kwargs=bad,
            rng=222, n_jobs=n_jobs)


@pytest.mark.parametrize("vine_type", ["cvine", "dvine", "rvine"])
@pytest.mark.parametrize("refit", [False, True])
def test_fitted_vine_bootstrap_reaches_owned_worker(vine_type, refit):
    constructor = getattr(VineCopula, vine_type)
    model = constructor(
        *(() if vine_type == "rvine" else (2,)),
        candidates=[BivariateGaussianCopula], allow_rotations=False)
    u = _observations()
    model.fit(u, to_pobs=False, method="mle")
    original = model.fit_result

    actual = gof_test(
        model, u, to_pobs=False, bootstrap=True, n_bootstrap=2,
        bootstrap_refit=refit, rng=24, n_jobs=2)

    assert np.all(np.isfinite(actual.bootstrap_statistics))
    assert all(row["bootstrap_refit"] is refit
               for row in actual.bootstrap_diagnostics)
    assert model.fit_result is original


@pytest.mark.parametrize("model_class", [
    EquicorrGaussianCopula, StochasticStudentCopula,
])
@pytest.mark.parametrize("key", ["alpha0", "_prepared_evaluator"])
@pytest.mark.parametrize("refit", [False, True])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_multivariate_mle_bootstrap_checks_model_keyword_owner(
        monkeypatch, model_class, key, refit, n_jobs):
    import pyscarcopula.stattests as st

    u = _observations()
    model = model_class(2)
    result = model.fit(u, method="mle", to_pobs=False)

    def unexpected(*args, **kwargs):
        pytest.fail("model-specific MLE option reached bootstrap RNG or worker")

    monkeypatch.setattr(st, "spawn_seed_sequences", unexpected)
    monkeypatch.setattr(st, "_bootstrap_gof_worker", unexpected)
    with pytest.raises(TypeError, match=key):
        gof_test(
            model, u, to_pobs=False, fit_result=result, bootstrap=True,
            n_bootstrap=2, bootstrap_refit=refit,
            bootstrap_fit_kwargs={key: [.2]}, rng=319, n_jobs=n_jobs)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_mle_rosenblatt_rejects_complex_parameter(dtype):
    from pyscarcopula.stattests import rosenblatt_transform_mle

    with pytest.raises((TypeError, ValueError), match="real|complex"):
        rosenblatt_transform_mle(
            BivariateGaussianCopula(), _observations(), dtype(0.3 + 0.7j))


@pytest.mark.parametrize("kind, method", [
    (kind, method)
    for kind in ("pair", "equicorr", "student")
    for method in ("GAS", "SCAR-TM-OU")
] + [("pair", "SCAR-TM-JACOBI")])
def test_bootstrap_constructor_overrides_reach_refit_strategy(
        kind, method, monkeypatch):
    from pyscarcopula.strategy.gas import GASStrategy
    from pyscarcopula.strategy.scar_jacobi import SCARJacobiStrategy
    from pyscarcopula.strategy.scar_tm import SCARTMStrategy

    model = {
        "pair": BivariateGaussianCopula(),
        "equicorr": EquicorrGaussianCopula(2),
        "student": StochasticStudentCopula(2, R=np.eye(2)),
    }[kind]
    options, strategy_class = {
        "GAS": ({"scaling": "fisher"}, GASStrategy),
        "SCAR-TM-OU": ({"K": 19, "grid_range": 3.2}, SCARTMStrategy),
        "SCAR-TM-JACOBI": (
            {"basis_order": 10, "quad_order": 32}, SCARJacobiStrategy),
    }[method]
    original = strategy_class.fit
    seen = []

    def capture(self, copula, u, **kwargs):
        seen.append({name: getattr(self, name) for name in options})
        assert not set(options).intersection(kwargs)
        # This checks constructor routing with an intentionally tiny fit
        # budget; convergence/failure handling has separate regression tests.
        return replace(original(self, copula, u, **kwargs), success=True)

    monkeypatch.setattr(strategy_class, "fit", wraps(original)(capture))
    actual = gof_test(
        model, _observations(), to_pobs=False,
        fit_result=_dynamic_result(method), K=17, grid_range=3.4,
        bootstrap=True, n_bootstrap=1, bootstrap_fit_kwargs={
            **options, "maxiter": 2, "maxfun": 20}, rng=821)

    assert np.isfinite(actual.statistic)
    assert seen == [options]


@pytest.mark.parametrize("kind", ["pair", "equicorr", "student"])
@pytest.mark.parametrize("warm_start", [False, True])
@pytest.mark.parametrize("overrides, expected", [
    ({}, 0.2),
    ({"score_eps": 0.3}, 0.3),
    ({"config": NumericalConfig(gas_score_eps=0.4)}, 0.4),
    ({"score_eps": 0.3, "config": NumericalConfig(gas_score_eps=0.4)}, 0.3),
    ({"score_eps": None, "config": NumericalConfig(gas_score_eps=0.4)}, 0.4),
    ({"score_eps": None}, 0.2),
], ids=["fitted", "explicit", "config", "explicit-over-config",
        "none-with-config", "none-with-fitted"])
def test_gas_bootstrap_refit_resolves_score_step_at_native_objective(
        kind, warm_start, overrides, expected, monkeypatch):
    from pyscarcopula._native import gas as native
    from pyscarcopula.strategy.gas import GASStrategy

    model = {
        "pair": BivariateGaussianCopula(),
        "equicorr": EquicorrGaussianCopula(2),
        "student": StochasticStudentCopula(2, R=np.eye(2)),
    }[kind]
    fitted = replace(_dynamic_result("GAS"), scaling="fisher", score_eps=0.2)
    objective_steps, fitted_steps = [], []
    original_objective = native.negative_log_likelihood_and_gradient
    original_fit = GASStrategy.fit

    @wraps(original_objective)
    def capture_objective(*args, **kwargs):
        objective_steps.append(args[6])
        return original_objective(*args, **kwargs)

    @wraps(original_fit)
    def capture_fit(self, copula, u, **kwargs):
        result = original_fit(self, copula, u, **kwargs)
        fitted_steps.append(result.score_eps)
        return replace(result, success=True)

    monkeypatch.setattr(
        native, "negative_log_likelihood_and_gradient", capture_objective)
    monkeypatch.setattr(GASStrategy, "fit", capture_fit)
    options = {"maxiter": 2, "maxfun": 20, "ftol": 1e-6, **overrides}
    if warm_start:
        options["gamma0"] = [0.1, 0.03, 0.6]
    actual = gof_test(
        model, _observations(), to_pobs=False, fit_result=fitted,
        bootstrap=True, n_bootstrap=1, bootstrap_fit_kwargs=options, rng=193)

    assert np.all(np.isfinite(actual.bootstrap_statistics))
    assert fitted_steps == [expected]
    assert objective_steps and set(objective_steps) == {expected}
    assert fitted.score_eps == 0.2


@pytest.mark.parametrize("kind, method, consumer", [
    ("student", "MLE", "student_sample_from_normal_uniforms"),
    ("factor_student", "MLE", "factor_student_sample_from_normal_uniforms"),
    ("equicorr", "MLE", "equicorr_gaussian_sample_from_normals"),
    ("equicorr", "GAS", "equicorr_gaussian_sample_from_normals"),
    ("equicorr", "SCAR-TM-OU", "equicorr_gaussian_sample_from_normals"),
    ("gaussian", "MLE", "gaussian_sample_from_normals"),
    ("dynamic_student", "GAS", "student_sample_from_normal_uniforms"),
])
@pytest.mark.parametrize("refit", [False, True])
def test_bootstrap_sampling_delivers_resolved_threads_to_native(
        kind, method, consumer, refit, monkeypatch):
    from pyscarcopula._native import multivariate as native

    u = _observations()
    model = {
        "student": StudentCopula(2),
        "factor_student": StudentCopula(2, corr_mode="factor", factor_rank=1),
        "equicorr": EquicorrGaussianCopula(2),
        "gaussian": GaussianCopula(2),
        "dynamic_student": StochasticStudentCopula(2, R=np.eye(2)),
    }[kind]
    result = (model.fit(u, to_pobs=False, method="mle") if method == "MLE"
              else _dynamic_result(method))
    seen = []
    original = getattr(native, consumer)

    @wraps(original)
    def capture(*args, **kwargs):
        seen.append(kwargs.get("n_threads", 1))
        return original(*args, **kwargs)

    monkeypatch.setattr(native, consumer, capture)
    # Keep the test focused on sampling thread routing. Small-budget refit
    # endpoints are intentional here and are not an optimizer acceptance test.
    import pyscarcopula.stattests as st

    def routing_refit(original_refit):
        def wrapper(*args, **kwargs):
            fitted_model, fitted_result = original_refit(*args, **kwargs)
            return fitted_model, replace(fitted_result, success=True)
        return wrapper

    for name, adapter in list(st._BOOTSTRAP_ADAPTERS.items()):
        monkeypatch.setitem(st._BOOTSTRAP_ADAPTERS, name, replace(
            adapter, refit=routing_refit(adapter.refit)))
    fit_kwargs = {"config": NumericalConfig(n_threads=2)}
    if kind != "gaussian":
        fit_kwargs["maxiter"] = 2
    actual = gof_test(
        model, u, to_pobs=False, fit_result=result, bootstrap=True,
        bootstrap_refit=refit, n_bootstrap=1, rng=194,
        bootstrap_fit_kwargs=fit_kwargs)

    assert actual.n_threads == 2
    assert np.all(np.isfinite(actual.bootstrap_statistics))
    assert seen and set(seen) == {2}


@pytest.mark.parametrize("model_class", [StudentCopula, EquicorrGaussianCopula])
def test_bootstrap_sampler_thread_count_preserves_seed_partition(model_class):
    model = model_class(2)
    u = _observations()
    fitted = model.fit(u, to_pobs=False, method="mle")
    options = dict(
        to_pobs=False, fit_result=fitted, bootstrap=True, bootstrap_refit=False,
        n_bootstrap=2, rng=195,
        bootstrap_fit_kwargs={"config": NumericalConfig(n_threads=2)})

    sequential = gof_test(model, u, n_jobs=1, **options)
    parallel = gof_test(model, u, n_jobs=2, **options)

    assert sequential.n_threads == 2 and parallel.n_threads == 1
    np.testing.assert_array_equal(
        sequential.bootstrap_statistics, parallel.bootstrap_statistics)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_bootstrap_worker_failure_retains_exception_chain(n_jobs):
    u = _observations()
    model = BivariateGaussianCopula()
    result = model.fit(u, to_pobs=False, method="mle")

    with pytest.raises(RuntimeError, match="bootstrap iteration") as caught:
        gof_test(
            model, u, to_pobs=False, fit_result=result, bootstrap=True,
            n_bootstrap=2, bootstrap_fit_kwargs={"maxiter": "invalid"},
            rng=927, n_jobs=n_jobs)

    assert caught.value.__cause__ is not None


@pytest.mark.parametrize("transition", [None, "matrix", "local", "spectral"])
def test_legacy_ou_result_restores_grid_defaults(transition):
    from pyscarcopula.stattests import _native_grid_config_from_result

    result = SimpleNamespace(method="SCAR-TM-OU", params=ou_params(2.0, 0.4, 0.25))
    if transition is not None:
        result.transition_method = transition
    config = _native_grid_config_from_result(result, 19, 3.2)

    assert config.K == 19 and config.grid_range == 3.2
    assert config.transition_method == (
        "auto" if transition == "spectral" else transition or "matrix")
    assert config.max_K is None
    assert config.grid_method == "auto" and config.adaptive is True
    assert config.small_kdt == 1e-2
    assert config.pts_per_sigma == 4 and config.gh_order == 5
