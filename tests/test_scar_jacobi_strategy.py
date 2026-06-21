import numpy as np
import pytest

from pyscarcopula import (
    BivariateGaussianCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    JoeCopula,
)
from pyscarcopula._types import LatentResult, jacobi_params
from pyscarcopula.api import (
    fit,
    log_likelihood,
    predict,
    predictive_mean,
    sample,
)
from pyscarcopula.stattests import gof_test
from pyscarcopula.strategy._base import (
    get_strategy,
    get_strategy_for_result,
    list_methods,
)
from pyscarcopula.strategy import scar_jacobi


def _u_sample():
    return np.array([
        [0.12, 0.22],
        [0.28, 0.40],
        [0.44, 0.53],
        [0.61, 0.70],
        [0.79, 0.85],
    ], dtype=np.float64)


def test_scar_jacobi_strategy_is_registered():
    assert 'SCAR-TM-JACOBI' in list_methods()
    strategy = get_strategy(
        'scar-tm-jacobi',
        basis_order=3,
        quad_order=16,
        smart_init=False,
    )
    assert strategy.basis_order == 3
    assert strategy.transition_method == 'auto'
    assert strategy.negative_mass_tol == pytest.approx(1e-5)


def test_scar_jacobi_accepts_spectral_order_aliases():
    strategy = get_strategy(
        'scar-tm-jacobi',
        spectral_basis_order=2,
        spectral_quad_order=18,
        smart_init=False,
    )

    assert strategy.basis_order == 2
    assert strategy.quad_order == 18


def test_scar_jacobi_transition_method_legacy_aliases_are_rejected():
    for legacy in ('matrix', 'spectral', 'gh', 'local_gh', 'fixed', 'fixed_grid',
                   'local_fixed_grid', 'coeff', 'coefficient'):
        with pytest.raises(ValueError, match='transition_method'):
            get_strategy('scar-tm-jacobi', transition_method=legacy)


def test_scar_jacobi_validates_optimizer_bounds():
    with pytest.raises(ValueError):
        get_strategy('scar-tm-jacobi', kappa_bounds=(1.0, 1.0))
    with pytest.raises(ValueError):
        get_strategy('scar-tm-jacobi', xi_bounds=(-1.0, 1.0))
    with pytest.raises(ValueError):
        get_strategy('scar-tm-jacobi', stationary_shape_max=0.0)


def test_scar_jacobi_fit_returns_latent_result():
    copula = GumbelCopula()
    u = _u_sample()

    result = fit(
        copula,
        u,
        method='scar-tm-jacobi',
        basis_order=3,
        quad_order=16,
        alpha0=np.array([1.0, 0.35, 0.2]),
        maxiter=3,
        maxfun=20,
    )

    assert isinstance(result, LatentResult)
    assert result.method == 'SCAR-TM-JACOBI'
    assert result.params.process_type == 'jacobi'
    assert result.params.kappa > 0.0
    assert 0.0 < result.params.m < 1.0
    assert result.params.xi > 0.0
    assert result.transition_method == 'auto'
    assert np.isfinite(result.log_likelihood)
    assert result.diagnostics["gradient_requested"] is False
    assert result.diagnostics["gradient_used"] is False
    assert result.diagnostics["optimizer_gradient"] == "numerical"
    assert result.diagnostics["gradient_kind"] == "numerical"
    assert result.diagnostics["transition_backend"] in {
        "local", "spectral_matrix"}


def test_scar_jacobi_fit_accepts_analytical_gradient():
    copula = GumbelCopula()
    u = _u_sample()

    result = fit(
        copula,
        u,
        method='scar-tm-jacobi',
        analytical_grad=True,
        basis_order=3,
        quad_order=18,
        alpha0=np.array([1.0, 0.35, 0.5]),
        maxiter=2,
        maxfun=12,
    )

    assert result.transition_method == 'auto'
    assert np.isfinite(result.log_likelihood)
    assert result.params.process_type == 'jacobi'
    assert result.diagnostics["gradient_requested"] is True
    assert result.diagnostics["gradient_used"] is True
    assert result.diagnostics["gradient_kind"] == "semi_analytical"
    assert (
        result.diagnostics["setup_derivative"]
        == "numerical_finite_difference"
    )
    assert result.diagnostics["filter_derivative"] == "analytical"
    assert result.diagnostics["transition_backend"] in {
        "local", "spectral_matrix"}


def test_scar_jacobi_initialization_records_mle_failure(monkeypatch):
    from pyscarcopula.strategy.mle import MLEStrategy

    def fail_mle(*args, **kwargs):
        raise ArithmeticError("static MLE failed")

    monkeypatch.setattr(MLEStrategy, "fit", fail_mle)
    result = fit(
        GumbelCopula(),
        _u_sample(),
        method='scar-tm-jacobi',
        basis_order=3,
        quad_order=18,
        maxiter=1,
        maxfun=8,
    )
    initialization = result.diagnostics["initialization"]

    assert initialization["requested_method"] == "static_mle_tau"
    assert initialization["selected_method"] == "m0_default"
    assert initialization["alpha0"] == [1.0, 0.5, 0.2]
    assert initialization["attempts"][0] == {
        "method": "static_mle_tau",
        "success": False,
        "error_type": "ArithmeticError",
        "error_message": "static MLE failed",
    }
    assert initialization["attempts"][1] == {
        "method": "m0_default",
        "success": True,
    }


def test_scar_jacobi_fit_accepts_spectral_matrix_analytical_gradient():
    copula = GumbelCopula()
    u = _u_sample()

    result = fit(
        copula,
        u,
        method='scar-tm-jacobi',
        analytical_grad=True,
        transition_method='spectral_matrix',
        basis_order=3,
        quad_order=18,
        alpha0=np.array([1.0, 0.35, 0.5]),
        maxiter=2,
        maxfun=12,
    )

    assert result.transition_method == 'spectral_matrix'
    assert np.isfinite(result.log_likelihood)
    assert result.diagnostics["gradient_kind"] == "semi_analytical"
    assert result.diagnostics["transition_backend"] == "spectral_matrix"


def test_scar_jacobi_local_fixed_reports_fully_analytical_gradient():
    result = fit(
        GumbelCopula(),
        _u_sample(),
        method='scar-tm-jacobi',
        analytical_grad=True,
        transition_method='local_fixed',
        basis_order=3,
        quad_order=18,
        alpha0=np.array([1.0, 0.35, 0.5]),
        maxiter=2,
        maxfun=12,
    )

    assert result.diagnostics["optimizer_gradient"] == "model_provided"
    assert result.diagnostics["gradient_kind"] == "analytical"
    assert result.diagnostics["setup_derivative"] == "analytical"
    assert result.diagnostics["filter_derivative"] == "analytical"
    assert result.diagnostics["transition_backend"] == "local_fixed"


def test_scar_jacobi_local_reports_semi_analytical_gradient():
    result = fit(
        GumbelCopula(),
        _u_sample(),
        method='scar-tm-jacobi',
        analytical_grad=True,
        transition_method='local',
        basis_order=3,
        quad_order=18,
        alpha0=np.array([1.0, 0.35, 0.5]),
        maxiter=2,
        maxfun=12,
    )

    assert result.diagnostics["gradient_kind"] == "semi_analytical"
    assert (
        result.diagnostics["setup_derivative"]
        == "numerical_finite_difference"
    )
    assert result.diagnostics["filter_derivative"] == "analytical"
    assert result.diagnostics["transition_backend"] == "local"


def test_scar_jacobi_spectral_coeff_rejects_requested_gradient():
    with pytest.raises(
            NotImplementedError,
            match="spectral_coeff Jacobi backend"):
        fit(
            GumbelCopula(),
            _u_sample(),
            method='scar-tm-jacobi',
            analytical_grad=True,
            transition_method='spectral_coeff',
            smart_init=False,
        )


def test_scar_jacobi_fit_clips_initial_point_to_bounds():
    copula = GumbelCopula()
    u = _u_sample()

    result = fit(
        copula,
        u,
        method='scar-tm-jacobi',
        kappa_bounds=(0.1, 2.0),
        xi_bounds=(0.05, 0.5),
        basis_order=3,
        quad_order=16,
        alpha0=np.array([50.0, 0.35, 20.0]),
        maxiter=1,
        maxfun=8,
    )

    assert 0.1 <= result.params.kappa <= 2.0
    assert 0.05 <= result.params.xi <= 0.5
    assert np.isfinite(result.log_likelihood)


@pytest.mark.data
def test_scar_jacobi_fit_crypto_notebook_smoke():
    import pandas as pd
    from pyscarcopula._utils import pobs

    prices = pd.read_csv("data/crypto_prices.csv", index_col=0, sep=";")
    tickers = ["BTC-USD", "ETH-USD"]
    returns = np.log(prices[tickers] / prices[tickers].shift(1))[1:181].values
    u = pobs(returns)

    result = fit(
        GumbelCopula(rotate=180),
        u,
        method='scar-tm-jacobi',
        basis_order=4,
        quad_order=32,
        alpha0=np.array([1.0, 0.65, 0.2]),
        maxiter=30,
        maxfun=100,
    )

    assert result.success
    assert result.transition_method == 'auto'
    assert result.log_likelihood > 0.0


def test_scar_jacobi_objective_penalizes_unsupported_stationary_shape():
    copula = GumbelCopula()
    u = _u_sample()
    strategy = get_strategy(
        'scar-tm-jacobi',
        stationary_shape_max=10.0,
        basis_order=3,
        quad_order=16,
    )

    assert strategy.objective(copula, u, np.array([1.0, 0.5, 0.1])) == 1e10


def test_scar_jacobi_fit_supports_legacy_spectral_coeff_backend():
    copula = GumbelCopula()
    u = _u_sample()

    result = fit(
        copula,
        u,
        method='scar-tm-jacobi',
        transition_method='spectral_coeff',
        basis_order=3,
        quad_order=16,
        alpha0=np.array([1.0, 0.35, 0.2]),
        maxiter=2,
        maxfun=12,
    )

    assert result.transition_method == 'spectral_coeff'
    assert result.gh_order is None
    assert np.isfinite(result.log_likelihood)
    assert result.diagnostics["gradient_requested"] is False
    assert result.diagnostics["gradient_used"] is False
    assert result.diagnostics["optimizer_gradient"] == "numerical"
    assert result.diagnostics["gradient_kind"] == "numerical"
    assert result.diagnostics["transition_backend"] == "spectral_coeff"


def test_scar_jacobi_fit_accepts_spectral_order_aliases():
    copula = GumbelCopula()
    u = _u_sample()

    result = fit(
        copula,
        u,
        method='scar-tm-jacobi',
        spectral_basis_order=1,
        spectral_quad_order=18,
        alpha0=np.array([1.0, 0.35, 0.2]),
        maxiter=2,
        maxfun=12,
    )

    assert result.spectral_basis_order == 1
    assert result.spectral_quad_order == 18


@pytest.mark.parametrize(
    "copula, min_param",
    [
        (ClaytonCopula(), 0.0),
        (FrankCopula(), 0.0),
        (GumbelCopula(), 1.0),
        (JoeCopula(), 1.0),
        (BivariateGaussianCopula(), 0.0),
    ],
)
def test_scar_jacobi_supported_copulas_fit_and_predict(copula, min_param):
    u = _u_sample()

    result = fit(
        copula,
        u,
        method='scar-tm-jacobi',
        basis_order=3,
        quad_order=16,
        alpha0=np.array([1.0, 0.35, 0.2]),
        maxiter=2,
        maxfun=12,
    )
    mean = predictive_mean(copula, u, result)

    assert result.method == 'SCAR-TM-JACOBI'
    assert np.isfinite(result.log_likelihood)
    assert mean.shape == (len(u),)
    assert np.all(mean > min_param)


def test_scar_jacobi_result_dispatches_predictive_functions():
    copula = GumbelCopula()
    u = _u_sample()
    result = LatentResult(
        log_likelihood=0.0,
        method='SCAR-TM-JACOBI',
        copula_name=copula.name,
        success=True,
        params=jacobi_params(1.2, 0.35, 0.25),
        spectral_basis_order=4,
        spectral_quad_order=20,
    )

    ll = log_likelihood(copula, u, result)
    mean = predictive_mean(copula, u, result)
    samples = predict(copula, u, result, n=8, rng=np.random.default_rng(123))

    assert np.isfinite(ll)
    assert mean.shape == (len(u),)
    assert np.all(mean >= 1.0)
    assert samples.shape == (8, 2)
    assert np.all((samples > 0.0) & (samples < 1.0))


def test_scar_jacobi_gof_uses_jacobi_strategy(monkeypatch):
    copula = GumbelCopula()
    u = _u_sample()
    result = LatentResult(
        log_likelihood=0.0,
        method='SCAR-TM-JACOBI',
        copula_name=copula.name,
        success=True,
        params=jacobi_params(1.2, 0.35, 0.25),
        transition_method='local',
        spectral_basis_order=4,
        spectral_quad_order=20,
    )
    calls = {'n': 0}

    def fake_rosenblatt_e2(self, copula_arg, u_arg, result_arg):
        calls['n'] += 1
        assert result_arg is result
        np.testing.assert_allclose(u_arg, u)
        return np.full(len(u_arg), 0.5, dtype=np.float64)

    monkeypatch.setattr(
        scar_jacobi.SCARJacobiStrategy,
        'rosenblatt_e2',
        fake_rosenblatt_e2,
    )

    gof = gof_test(copula, u, fit_result=result, to_pobs=False)

    assert calls['n'] == 1
    assert np.isfinite(gof.statistic)
    assert np.isfinite(gof.pvalue)


@pytest.mark.data
def test_scar_jacobi_gof_full_crypto_default_grid_is_finite():
    import pandas as pd
    from pyscarcopula._utils import pobs

    prices = pd.read_csv("data/crypto_prices.csv", index_col=0, sep=";")
    tickers = ["BTC-USD", "ETH-USD"]
    returns = np.log(prices[tickers] / prices[tickers].shift(1))[1:].values
    u = pobs(returns)
    copula = GumbelCopula(rotate=180)
    result = LatentResult(
        log_likelihood=1043.3019806319703,
        method='SCAR-TM-JACOBI',
        copula_name=copula.name,
        success=True,
        params=jacobi_params(49.9080, 0.6591, 2.4708),
        transition_method='auto',
        gh_order=5,
        spectral_basis_order=32,
        spectral_quad_order=None,
    )

    gof = gof_test(copula, u, fit_result=result, to_pobs=False)

    assert np.isfinite(gof.statistic)
    assert np.isfinite(gof.pvalue)


def test_scar_jacobi_get_strategy_for_result_restores_options():
    result = LatentResult(
        log_likelihood=0.0,
        method='SCAR-TM-JACOBI',
        copula_name='Gumbel copula',
        success=True,
        params=jacobi_params(1.2, 0.35, 0.25),
        transition_method='local',
        gh_order=7,
        spectral_basis_order=7,
        spectral_quad_order=31,
    )

    strategy = get_strategy_for_result(result)
    overridden = get_strategy_for_result(result, basis_order=5)

    assert strategy.basis_order == 7
    assert strategy.quad_order == 31
    assert strategy.transition_method == 'local'
    assert strategy.gh_order == 7
    assert overridden.basis_order == 5
    assert overridden.quad_order == 31


def test_scar_jacobi_objective_uses_physical_parameters():
    copula = GumbelCopula()
    u = _u_sample()
    strategy = get_strategy('scar-tm-jacobi', basis_order=3, quad_order=16)

    obj = strategy.objective(copula, u, np.array([1.2, 0.35, 0.25]))
    invalid = strategy.objective(copula, u, np.array([-1.0, 0.35, 0.25]))

    assert np.isfinite(obj)
    assert invalid == 1e10


def test_scar_jacobi_predictive_state_cache_reused(monkeypatch):
    copula = GumbelCopula()
    u = _u_sample()
    result = LatentResult(
        log_likelihood=0.0,
        method='SCAR-TM-JACOBI',
        copula_name=copula.name,
        success=True,
        params=jacobi_params(1.2, 0.35, 0.25),
        spectral_basis_order=4,
        spectral_quad_order=20,
    )
    strategy = get_strategy_for_result(result)
    calls = {'n': 0}
    original = scar_jacobi.jacobi_matrix_state_distribution

    def counted(*args, **kwargs):
        calls['n'] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(scar_jacobi, 'jacobi_matrix_state_distribution', counted)
    cache = {}

    first = strategy.predictive_state(
        copula, u, result, horizon='next', state_cache=cache, cache_key='edge')
    second = strategy.predictive_state(
        copula, u, result, horizon='next', state_cache=cache, cache_key='edge')

    assert calls['n'] == 1
    np.testing.assert_allclose(first.z_grid, second.z_grid)
    np.testing.assert_allclose(first.prob, second.prob)


def test_scar_jacobi_mixture_h_populates_state_cache():
    copula = GumbelCopula()
    u = _u_sample()
    result = LatentResult(
        log_likelihood=0.0,
        method='SCAR-TM-JACOBI',
        copula_name=copula.name,
        success=True,
        params=jacobi_params(1.2, 0.35, 0.25),
        spectral_basis_order=4,
        spectral_quad_order=20,
    )
    strategy = get_strategy_for_result(result)
    cache = {}

    h_mix = strategy.mixture_h(
        copula,
        u,
        result,
        state_cache=cache,
        current_cache_key='current',
        next_cache_key='next',
    )

    assert h_mix.shape == (len(u),)
    assert set(cache) == {'current', 'next'}
    for tau_grid, prob in cache.values():
        assert tau_grid.shape == prob.shape
        np.testing.assert_allclose(np.sum(prob), 1.0, rtol=1e-12, atol=1e-12)


def test_scar_jacobi_condition_state_reweights_grid_distribution():
    copula = GumbelCopula()
    u = _u_sample()
    result = LatentResult(
        log_likelihood=0.0,
        method='SCAR-TM-JACOBI',
        copula_name=copula.name,
        success=True,
        params=jacobi_params(1.2, 0.35, 0.25),
        spectral_basis_order=4,
        spectral_quad_order=20,
    )
    strategy = get_strategy_for_result(result)
    state = strategy.predictive_state(copula, u, result, horizon='next')

    updated = strategy.condition_state(copula, state, u[:1], result)

    assert updated.kind == 'grid'
    np.testing.assert_allclose(updated.z_grid, state.z_grid)
    np.testing.assert_allclose(np.sum(updated.prob), 1.0, rtol=1e-12, atol=1e-12)
    assert not np.allclose(updated.prob, state.prob)


def test_scar_jacobi_object_methods_dispatch():
    copula = GumbelCopula()
    u = _u_sample()

    result = copula.fit(
        u,
        method='scar-tm-jacobi',
        basis_order=3,
        quad_order=16,
        alpha0=np.array([1.0, 0.35, 0.2]),
        maxiter=2,
        maxfun=12,
    )
    samples = copula.predict(n=5, rng=np.random.default_rng(321))

    assert result.method == 'SCAR-TM-JACOBI'
    assert copula.fit_result is result
    assert samples.shape == (5, 2)
    assert np.all((samples > 0.0) & (samples < 1.0))


def test_scar_jacobi_fitted_sampling_is_explicitly_unimplemented():
    copula = GumbelCopula()
    u = _u_sample()
    result = LatentResult(
        log_likelihood=0.0,
        method='SCAR-TM-JACOBI',
        copula_name=copula.name,
        success=True,
        params=jacobi_params(1.2, 0.35, 0.25),
        spectral_basis_order=4,
        spectral_quad_order=20,
    )

    with pytest.raises(NotImplementedError):
        sample(copula, u, result, n=5, rng=np.random.default_rng(1))


@pytest.mark.parametrize("copula", [FrankCopula(), JoeCopula()])
def test_scar_jacobi_supports_frank_and_joe_tau_mapping(copula):
    result = fit(
        copula,
        _u_sample(),
        method='scar-tm-jacobi',
        basis_order=3,
        quad_order=16,
        alpha0=np.array([1.0, 0.35, 0.2]),
        maxiter=1,
        maxfun=6,
    )

    assert result.method == 'SCAR-TM-JACOBI'
    assert result.params.process_type == 'jacobi'
