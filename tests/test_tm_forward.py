from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pyscarcopula import FrankCopula, GumbelCopula, JoeCopula
from pyscarcopula._constants import H_FUNCTION_EPS
from pyscarcopula._types import (
    LBFGSBConfig,
    LatentResult,
    NumericalConfig,
    ou_params,
)
from pyscarcopula.api import fit
from pyscarcopula.numerical.tm_functions import (
    tm_forward_mixture_h,
    tm_forward_predictive_mean,
    tm_forward_rosenblatt,
    tm_loglik,
)
from pyscarcopula.numerical.hermite_tm import (
    hermite_loglik,
    hermite_loglik_with_grad,
)
from pyscarcopula.numerical._scar_ou_config import (
    AutoTMConfig,
    select_auto_backend,
)
from pyscarcopula.copula.elliptical import BivariateGaussianCopula
from pyscarcopula.copula.clayton import ClaytonCopula
from pyscarcopula.copula.multivariate import StochasticStudentCopula
from pyscarcopula.strategy import scar_tm
from pyscarcopula.strategy._base import get_strategy_for_result
from pyscarcopula.strategy.scar_tm import SCARTMStrategy
from pyscarcopula._native import scar_ou as _cpp_scar_ou
from pyscarcopula import stattests


def tm_loglik_with_grad(
        kappa, mu, nu, u, copula, **config_kwargs):
    config_values = {
        "transition_method": "matrix",
        "K": 300,
        "grid_range": 5.0,
        "grid_method": "auto",
        "adaptive": True,
        "pts_per_sigma": 4,
        "max_K": None,
        "r_gh": 3.0,
        "gh_order": 5,
    }
    config_values.update(config_kwargs)
    return _cpp_scar_ou.neg_loglik_with_grad(
        kappa,
        mu,
        nu,
        u,
        copula,
        AutoTMConfig(**config_values),
    )


def _direct_forward_density(grid, phi):
    """Direct quadrature for phi_next[i] = sum_j p(i|j) phi[j] w[j]."""
    z = grid.z
    means = grid.rho * z
    diff = z[np.newaxis, :] - means[:, np.newaxis]
    p = (
        np.exp(-0.5 * (diff / grid.sigma_cond) ** 2)
        / (grid.sigma_cond * np.sqrt(2.0 * np.pi))
    )
    return p.T @ (phi * grid.trap_w)


def _density_to_mass(grid, phi):
    mass = phi * grid.trap_w
    return mass / np.sum(mass)


def test_local_transition_loglik_is_finite():
    rng = np.random.default_rng(0)
    u = rng.uniform(0.05, 0.95, size=(40, 2))
    copula = BivariateGaussianCopula()

    value = tm_loglik(
        1.0,
        0.0,
        1.0,
        u,
        copula,
        K=40,
        adaptive=False,
        transition_method="local",
    )

    assert np.isfinite(value)


def test_gaussian_copula_batch_grid_matches_scalar_path():
    rng = np.random.default_rng(3)
    u = rng.uniform(0.05, 0.95, size=(7, 2))
    x_grid = np.linspace(-3.0, 3.0, 11)
    copula = BivariateGaussianCopula()

    fi_batch, dfi_batch = copula.pdf_and_grad_on_grid_batch(u, x_grid)
    fi_grid = copula.copula_grid_batch(u, x_grid)

    fi_expected = np.empty_like(fi_batch)
    dfi_expected = np.empty_like(dfi_batch)
    for idx, row in enumerate(u):
        fi_expected[idx], dfi_expected[idx] = copula.pdf_and_grad_on_grid(
            row, x_grid)

    np.testing.assert_allclose(fi_batch, fi_expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dfi_batch, dfi_expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fi_grid, fi_expected, rtol=1e-12, atol=1e-12)


def test_scar_tm_local_transition_keeps_analytical_gradient_enabled():
    assert SCARTMStrategy().analytical_grad is True
    assert SCARTMStrategy(transition_method="auto").analytical_grad is True
    assert SCARTMStrategy(max_K=100).analytical_grad is True


def test_log_stationary_optimizer_selection_is_optional_for_bivariate_model():
    bivariate = BivariateGaussianCopula()
    stochastic_student = StochasticStudentCopula(d=2, R=np.eye(2))

    assert bivariate._scar_log_optimizer_config == (
        "bivariate_log_scar_optimizer")
    assert bivariate._scar_optimizer_config == "bivariate_scar_optimizer"
    assert stochastic_student._scar_optimizer_config == (
        "stochastic_student_scar_optimizer")
    assert bivariate._scar_log_optimizer_config != (
        stochastic_student._scar_optimizer_config)
    assert bivariate._scar_stationary_scale_bounds == (0.001, 10_000.0)
    assert stochastic_student._scar_stationary_scale_bounds == (
        0.001, 10_000.0)
    assert not scar_tm._uses_log_stationary_scale(bivariate)
    assert scar_tm._uses_log_stationary_scale(bivariate, True)
    assert not scar_tm._uses_bounded_log_stationary_scale(bivariate)
    assert scar_tm._uses_bounded_log_stationary_scale(bivariate, True)
    assert scar_tm._uses_log_stationary_scale(stochastic_student)
    assert not scar_tm._uses_bounded_log_stationary_scale(
        stochastic_student)
    assert not scar_tm._uses_bounded_log_stationary_scale(
        stochastic_student, True)
    assert not scar_tm._uses_log_stationary_scale(
        stochastic_student, False)
    with pytest.raises(
            TypeError, match="log_stationary_scale_optimization"):
        SCARTMStrategy(log_stationary_scale_optimization="yes")


def test_scar_optimizer_config_selection_preserves_legacy_bivariate_policy():
    config = NumericalConfig(
        scar_optimizer=LBFGSBConfig(maxiter=7, maxls=9),
        bivariate_log_scar_optimizer=LBFGSBConfig(maxiter=11, maxls=13),
        stochastic_student_scar_optimizer=LBFGSBConfig(
            maxiter=17, maxls=19),
    )
    strategy = SCARTMStrategy(config=config)
    bivariate = BivariateGaussianCopula()
    stochastic_student = StochasticStudentCopula(d=2, R=np.eye(2))

    base = strategy._optimizer_config(
        bivariate, log_stationary=False)
    logged = strategy._optimizer_config(
        bivariate, log_stationary=True)
    student = strategy._optimizer_config(
        stochastic_student, log_stationary=True)

    assert (base.maxiter, base.maxls) == (7, 9)
    assert (logged.maxiter, logged.maxls) == (11, 13)
    assert (student.maxiter, student.maxls) == (17, 19)


def test_explicit_bivariate_scar_optimizer_can_override_legacy_fallback():
    config = NumericalConfig(
        scar_optimizer=LBFGSBConfig(maxiter=7),
        bivariate_scar_optimizer=LBFGSBConfig(maxiter=23),
    )
    strategy = SCARTMStrategy(config=config)

    selected = strategy._optimizer_config(
        BivariateGaussianCopula(), log_stationary=False)

    assert selected.maxiter == 23
    assert config.scar_optimizer.maxiter == 7


def test_bivariate_log_stationary_objective_wraps_gradient_and_uses_log_bounds(
        monkeypatch):
    alpha0 = np.array([2.0, 0.3, 1.4])
    physical_gradient = np.array([0.2, -0.4, 0.6])
    captured = {}

    def fake_evaluate(self, kappa, mu, nu, auto_config):
        captured.setdefault("physical_calls", []).append(
            np.array([kappa, mu, nu]))
        return 7.0, physical_gradient.copy(), {
            "backend": "spectral",
            "transition_method": "auto",
        }

    def fake_minimize(fun, x0, *, method, jac, bounds, options):
        value, gradient = fun(x0)
        captured.update({
            "x0": np.asarray(x0).copy(),
            "gradient": np.asarray(gradient).copy(),
            "bounds": bounds,
            "options": dict(options),
        })

        lower_point = np.asarray(bounds.lb).copy()
        lower_point[1] = x0[1]
        lower_value, _ = fun(lower_point)
        assert lower_value == pytest.approx(7.0)
        captured["lower_physical"] = captured["physical_calls"][-1]

        return SimpleNamespace(
            x=np.asarray(x0).copy(),
            fun=value,
            jac=np.asarray(gradient).copy(),
            success=True,
            message="test convergence",
            nfev=1,
        )

    monkeypatch.setattr(
        scar_tm._PreparedScarOuFitCache,
        "neg_loglik_with_grad_info",
        fake_evaluate,
    )
    monkeypatch.setattr(scar_tm, "minimize", fake_minimize)

    result = fit(
        BivariateGaussianCopula(),
        np.random.default_rng(20260809).uniform(0.01, 0.99, size=(20, 2)),
        method="scar-tm-ou",
        alpha0=alpha0,
        smart_init=False,
        log_stationary_scale_optimization=True,
    )

    expected_x0 = scar_tm._ou_to_log_stationary(alpha0)
    expected_gradient = scar_tm._ou_grad_to_log_stationary(
        alpha0, physical_gradient)
    np.testing.assert_allclose(captured["x0"], expected_x0)
    np.testing.assert_allclose(captured["gradient"], expected_gradient)
    np.testing.assert_allclose(captured["physical_calls"][0], alpha0)
    np.testing.assert_allclose(
        captured["bounds"].lb[[0, 2]], np.log([0.001, 0.001]))
    assert np.isneginf(captured["bounds"].lb[1])
    assert captured["bounds"].ub[2] == pytest.approx(np.log(10_000.0))
    assert captured["options"]["maxiter"] == 1000
    assert captured["options"]["maxls"] == 200
    assert captured["lower_physical"][0] == pytest.approx(0.001)
    assert 0.0 < captured["lower_physical"][2] < 0.001
    assert result.diagnostics["optimizer_parameterization"] == (
        "log_kappa_mu_log_stationary_sigma")
    assert result.diagnostics["optimizer_stationary_scale_bounds"] == (
        0.001, 10_000.0)
    np.testing.assert_allclose(result.params.values, alpha0)


def test_stochastic_student_uses_model_log_scale_bounds_by_default(monkeypatch):
    alpha0 = np.array([2.0, 0.3, 1.4])
    physical_gradient = np.array([0.2, -0.4, 0.6])
    captured = {}

    def fake_evaluate(self, kappa, mu, nu, auto_config):
        return 7.0, physical_gradient.copy(), {
            "backend": "spectral",
            "transition_method": "auto",
        }

    def fake_minimize(fun, x0, *, method, jac, bounds, options):
        value, gradient = fun(x0)
        captured.update({
            "x0": np.asarray(x0).copy(),
            "bounds": bounds,
            "options": dict(options),
        })
        return SimpleNamespace(
            x=np.asarray(x0).copy(),
            fun=value,
            jac=np.asarray(gradient).copy(),
            success=True,
            message="test convergence",
            nfev=1,
        )

    original_validate = SCARTMStrategy._validate_final_fit

    def capture_validate(self, *args, **kwargs):
        captured["final_lower"] = np.asarray(kwargs["lower"]).copy()
        return original_validate(self, *args, **kwargs)

    monkeypatch.setattr(
        scar_tm._PreparedScarOuFitCache,
        "neg_loglik_with_grad_info",
        fake_evaluate,
    )
    monkeypatch.setattr(scar_tm, "minimize", fake_minimize)
    monkeypatch.setattr(
        SCARTMStrategy, "_validate_final_fit", capture_validate)

    result = fit(
        StochasticStudentCopula(d=2, R=np.eye(2)),
        np.random.default_rng(20260810).uniform(0.01, 0.99, size=(20, 2)),
        method="scar-tm-ou",
        alpha0=alpha0,
        smart_init=False,
    )

    np.testing.assert_allclose(
        captured["x0"], scar_tm._ou_to_log_stationary(alpha0))
    assert captured["bounds"].lb[0] == pytest.approx(np.log(0.001))
    assert np.isneginf(captured["bounds"].lb[1])
    assert captured["bounds"].lb[2] == pytest.approx(np.log(0.001))
    assert captured["bounds"].ub[2] == pytest.approx(np.log(10_000.0))
    assert captured["options"]["maxiter"] == 1000
    assert captured["options"]["maxls"] == 200
    np.testing.assert_allclose(
        captured["final_lower"][[0, 2]], [0.001, 0.001])
    assert np.isneginf(captured["final_lower"][1])
    assert result.diagnostics["optimizer_parameterization"] == (
        "log_kappa_mu_log_stationary_sigma")
    assert result.diagnostics["optimizer_stationary_scale_bounds"] == (
        0.001, 10_000.0)


def test_scar_tm_auto_selects_spectral_except_narrow_local_fallback():
    cfg = AutoTMConfig(transition_method="auto", small_kdt=1e-3)

    assert select_auto_backend(0.05, 101, cfg) == "local"
    assert select_auto_backend(0.2, 101, cfg) == "spectral"
    assert select_auto_backend(6.0, 101, cfg) == "spectral"

    assert select_auto_backend(
        0.2, 101, AutoTMConfig(transition_method="matrix")) == "matrix"
    with pytest.raises(ValueError, match="transition_method"):
        select_auto_backend(
            0.2, 101, AutoTMConfig(transition_method="gh"))


def test_scar_tm_spectral_accepts_large_quad_order_on_hf_scale():
    rng = np.random.default_rng(20260612)
    u = rng.uniform(0.05, 0.95, size=(160, 2))
    copula = GumbelCopula(rotate=180)
    alpha = (58.8, 2.0, 15.2)

    value = hermite_loglik(
        *alpha, u, copula,
        basis_order=128,
        quad_order=384,
    )

    assert np.isfinite(value)


def test_scar_tm_strategy_defaults_to_auto_likelihood_with_auto_grid():
    strategy = SCARTMStrategy()
    assert strategy.transition_method == "auto"
    assert not hasattr(strategy, "backend")
    assert strategy.max_K == 1000
    assert strategy.auto_small_kdt == pytest.approx(1e-2)
    assert strategy.spectral_quad_order is None
    assert AutoTMConfig().small_kdt == pytest.approx(1e-2)
    assert AutoTMConfig().quad_order is None
    assert strategy._tm_kwargs()["transition_method"] == "auto"
    assert strategy._tm_kwargs()["max_K"] == 1000
    assert strategy._grid_tm_kwargs()["transition_method"] == "auto"
    assert strategy._grid_tm_kwargs()["max_K"] == 1000

    matrix_strategy = SCARTMStrategy(transition_method="matrix", max_K=None)
    assert matrix_strategy._tm_kwargs() == {}
    assert matrix_strategy._grid_tm_kwargs() == {}

    spectral_strategy = SCARTMStrategy(transition_method="spectral")
    assert spectral_strategy.transition_method == "spectral"
    assert spectral_strategy._grid_tm_kwargs()["transition_method"] == "auto"


def test_scar_tm_auto_spectral_basis_order_policy():
    strategy = SCARTMStrategy(spectral_basis_order="auto")
    assert SCARTMStrategy().spectral_basis_order == "auto"
    assert strategy.spectral_basis_order == "auto"
    assert strategy._spectral_basis_order_for(1.0, 101) == 128
    assert strategy._spectral_basis_order_for(2.0, 101) == 96
    assert strategy._spectral_basis_order_for(4.0, 101) == 64
    assert strategy._spectral_basis_order_for(10.0, 101) == 32
    assert SCARTMStrategy(spectral_basis_order="auto").spectral_basis_order == "auto"
    assert SCARTMStrategy(spectral_basis_order="64").spectral_basis_order == 64

    with pytest.raises(ValueError, match="spectral_basis_order"):
        SCARTMStrategy(spectral_basis_order="wide")
    for invalid in (True, np.bool_(False), 3.7):
        with pytest.raises(ValueError, match="spectral_basis_order"):
            SCARTMStrategy(spectral_basis_order=invalid)


@pytest.mark.parametrize("small_kdt", [0.0, -1e-3])
def test_scar_tm_rejects_nonpositive_auto_small_kdt_at_construction(
        small_kdt):
    with pytest.raises(ValueError, match="small_kdt.*positive finite"):
        SCARTMStrategy(auto_small_kdt=small_kdt)


def test_scar_tm_rejects_single_observation_before_optimizer(monkeypatch):
    optimizer_called = False

    def forbidden_optimizer(*args, **kwargs):
        nonlocal optimizer_called
        optimizer_called = True
        raise AssertionError("optimizer must not run")

    monkeypatch.setattr(scar_tm, "minimize", forbidden_optimizer)
    with pytest.raises(ValueError, match="at least two observations"):
        SCARTMStrategy(smart_init=False).fit(
            BivariateGaussianCopula(),
            np.array([[0.25, 0.75]]),
            alpha0=np.array([1.0, 0.0, 0.5]),
        )
    assert optimizer_called is False


def test_scar_tm_adaptive_spectral_basis_order_reaches_objective(monkeypatch):
    captured = {}

    def fake_objective(kappa, mu, nu, u_arg, copula_arg, config):
        captured["basis_order"] = config.basis_order
        captured["quad_order"] = config.quad_order
        return 12.0

    monkeypatch.setattr(_cpp_scar_ou, "neg_loglik", fake_objective)

    u = np.random.default_rng(20260701).uniform(0.05, 0.95, size=(101, 2))
    strategy = SCARTMStrategy(
        spectral_basis_order="auto",
        spectral_quad_order=None,
    )

    value = strategy.objective(
        BivariateGaussianCopula(),
        u,
        np.array([2.0, 0.0, 1.0]),
    )

    assert value == pytest.approx(12.0)
    assert captured == {"basis_order": 96, "quad_order": None}


def test_scar_tm_adaptive_spectral_basis_order_records_diagnostics(monkeypatch):
    calls = []

    def fake_objective(kappa, mu, nu, u_arg, copula_arg, config):
        calls.append(config.basis_order)
        value = (
            (float(kappa) - 1.0) ** 2
            + float(mu) ** 2
            + (float(nu) - 1.0) ** 2
            + 1.0
        )
        grad = np.array(
            [2.0 * (float(kappa) - 1.0), 2.0 * float(mu),
             2.0 * (float(nu) - 1.0)],
            dtype=np.float64,
        )
        info = {
            "backend": "spectral",
            "transition_method": "auto",
            "kappa_dt": float(kappa) / (len(u_arg) - 1),
            "n_obs": len(u_arg),
            "basis_order": config.basis_order,
        }
        return value, grad, info

    monkeypatch.setattr(
        _cpp_scar_ou, "neg_loglik_with_grad_info", fake_objective)
    monkeypatch.setattr(
        _cpp_scar_ou,
        "prepare_objective",
        lambda *args, **kwargs: (
            (_ for _ in ()).throw(
                _cpp_scar_ou.NativeUnsupported("test fallback"))),
    )

    u = np.random.default_rng(20260702).uniform(0.05, 0.95, size=(101, 2))
    result = SCARTMStrategy(
        smart_init=False,
        spectral_basis_order="auto",
    ).fit(
        BivariateGaussianCopula(),
        u,
        alpha0=np.array([1.0, 0.0, 1.0]),
        maxiter=1,
        maxfun=5,
    )

    assert calls
    assert result.spectral_basis_order == "auto"
    assert result.diagnostics["adaptive_spectral_basis_order"] is True
    assert result.diagnostics["auto_spectral_basis_order"] is True
    assert result.diagnostics["basis_order_128_evaluations"] >= 1
    assert result.diagnostics["last_spectral_basis_order"] == 128


def test_scar_tm_strategy_recovered_from_legacy_result_uses_matrix_path():
    result = LatentResult(
        log_likelihood=0.0,
        method="SCAR-TM-OU",
        copula_name="BivariateGaussian",
        success=True,
        params=ou_params(0.8, 0.1, 1.0),
        K=41,
        grid_range=3.0,
        pts_per_sigma=2,
    )

    strategy = get_strategy_for_result(result)

    assert strategy.K == 41
    assert strategy.grid_range == 3.0
    assert strategy.pts_per_sigma == 2
    assert strategy.transition_method == "matrix"
    assert strategy.max_K is None
    assert strategy._tm_kwargs() == {}


def test_scar_tm_strategy_recovered_from_result_preserves_explicit_no_cap():
    result = LatentResult(
        log_likelihood=0.0,
        method="SCAR-TM-OU",
        copula_name="BivariateGaussian",
        success=True,
        params=ou_params(0.8, 0.1, 1.0),
        K=41,
        grid_range=3.0,
        grid_method="sparse",
        adaptive=False,
        pts_per_sigma=2,
        transition_method="auto",
        max_K=None,
    )

    strategy = get_strategy_for_result(result)
    overridden = get_strategy_for_result(
        replace(result, max_K=500),
        max_K=None,
    )

    assert strategy.transition_method == "auto"
    assert strategy.max_K is None
    assert strategy.grid_method == "sparse"
    assert strategy.adaptive is False
    assert strategy._tm_kwargs()["max_K"] is None
    assert overridden.transition_method == "auto"
    assert overridden.max_K is None


def test_local_transition_analytical_gradient_matches_finite_difference():
    rng = np.random.default_rng(1)
    u = rng.uniform(0.1, 0.9, size=(20, 2))
    copula = BivariateGaussianCopula()
    alpha = np.array([0.8, 0.1, 1.2])
    kwargs = {
        "K": 45,
        "grid_range": 4.0,
        "adaptive": False,
        "transition_method": "local",
        "gh_order": 5,
    }

    value, grad = tm_loglik_with_grad(*alpha, u, copula, **kwargs)
    direct = tm_loglik(*alpha, u, copula, **kwargs)

    np.testing.assert_allclose(value, direct, rtol=1e-11, atol=1e-11)

    fd = np.empty(3)
    steps = np.array([1e-5, 1e-5, 1e-5])
    for idx, step in enumerate(steps):
        plus = alpha.copy()
        minus = alpha.copy()
        plus[idx] += step
        minus[idx] -= step
        fd[idx] = (
            tm_loglik(*plus, u, copula, **kwargs)
            - tm_loglik(*minus, u, copula, **kwargs)
        ) / (2.0 * step)

    np.testing.assert_allclose(grad, fd, rtol=2e-4, atol=2e-4)


def test_spectral_analytical_gradient_matches_finite_difference_on_weak_data():
    rng = np.random.default_rng(42)
    u = rng.uniform(0.05, 0.95, size=(30, 2))
    copula = BivariateGaussianCopula()
    alpha = np.array([5.0, 0.0, 0.6])
    kwargs = {"basis_order": 16}

    value, grad = hermite_loglik_with_grad(*alpha, u, copula, **kwargs)
    direct = -hermite_loglik(*alpha, u, copula, **kwargs)

    np.testing.assert_allclose(value, direct, rtol=1e-11, atol=1e-11)

    fd = np.empty(3)
    steps = np.array([1e-5, 1e-5, 1e-5])
    for idx, step in enumerate(steps):
        plus = alpha.copy()
        minus = alpha.copy()
        plus[idx] += step
        minus[idx] -= step
        fd[idx] = (
            -hermite_loglik(*plus, u, copula, **kwargs)
            + hermite_loglik(*minus, u, copula, **kwargs)
        ) / (2.0 * step)

    np.testing.assert_allclose(grad, fd, rtol=2e-5, atol=2e-5)


def test_scar_tm_failed_dispatch_objective_is_not_success(monkeypatch):
    def fail_objective(*args, **kwargs):
        return 1e10, np.zeros(3), {
            "backend": "spectral",
            "transition_method": "auto",
            "kappa_dt": 0.1,
            "n_obs": 10,
        }

    monkeypatch.setattr(
        _cpp_scar_ou, "neg_loglik_with_grad_info", fail_objective)
    monkeypatch.setattr(
        _cpp_scar_ou,
        "prepare_objective",
        lambda *args, **kwargs: (
            (_ for _ in ()).throw(
                _cpp_scar_ou.NativeUnsupported("test fallback"))),
    )
    u = np.random.default_rng(17).uniform(0.05, 0.95, size=(10, 2))
    copula = BivariateGaussianCopula()

    result = SCARTMStrategy(smart_init=False).fit(
        copula,
        u,
        alpha0=np.array([1.0, 0.0, 1.0]),
        maxiter=2,
        maxfun=5,
    )

    assert not result.success
    assert result.log_likelihood == pytest.approx(-1e10)
    assert "invalid objective value" in result.message
    assert result.diagnostics["initialization"] == {
        "requested_method": "user_provided",
        "selected_method": "user_provided",
        "alpha0": [1.0, 0.0, 1.0],
        "attempts": [{
            "method": "user_provided",
            "success": True,
        }],
        "success": True,
    }


def _validate_candidate(
        final_params,
        selected_value=12.0,
        selected_grad=None,
        validation_value=12.0,
        initial_params=None,
        optimizer_value=None,
        strict_gradient_policy=False):
    strategy = SCARTMStrategy(
        smart_init=False,
        strict_gradient_policy=strict_gradient_policy,
    )
    final_params = np.asarray(final_params, dtype=np.float64)
    if initial_params is None:
        initial_params = np.array([1.0, 0.0, 1.0])
    if selected_grad is None:
        selected_grad = np.zeros_like(final_params)
    result = SimpleNamespace(
        fun=selected_value if optimizer_value is None else optimizer_value,
        jac=np.asarray(selected_grad),
        success=True,
        message="optimizer success",
    )
    diagnostics = strategy._validate_final_fit(
        result=result,
        final_params=final_params,
        initial_params=np.asarray(initial_params, dtype=np.float64),
        lower=np.array([0.001, -np.inf, 0.001]),
        upper=np.array([np.inf, np.inf, np.inf]),
        selected_evaluator=lambda values: (
            selected_value, np.asarray(selected_grad)),
        validation_evaluator=(
            None if validation_value is None
            else lambda values: validation_value),
        selected_engine="cpp",
        validation_engine=(
            None if validation_value is None else "validation"),
        n_obs=20,
        optimizer_options={"gtol": 1e-3},
    )
    return result, diagnostics


def test_scar_tm_final_validation_rejects_nonfinite_parameters():
    result, diagnostics = _validate_candidate([np.nan, 0.0, 1.0])

    assert not result.success
    assert not diagnostics["final_validation_passed"]
    assert "parameters are not finite" in result.message


def test_scar_tm_final_validation_rejects_nonfinite_gradient():
    result, diagnostics = _validate_candidate(
        [1.0, 0.0, 1.0],
        selected_grad=[0.0, np.nan, 0.0],
    )

    assert not result.success
    assert not diagnostics["final_validation_passed"]
    assert "gradient is not finite" in result.message


def test_scar_tm_final_validation_rejects_optimizer_sentinel():
    result, diagnostics = _validate_candidate(
        [1.0, 0.0, 1.0],
        selected_value=12.0,
        optimizer_value=1e10,
    )

    assert not result.success
    assert not diagnostics["final_validation_passed"]
    assert "optimizer returned an invalid objective value" in result.message


def test_scar_tm_final_validation_rejects_backend_disagreement():
    result, diagnostics = _validate_candidate(
        [1.0, 0.0, 1.0],
        selected_value=12.0,
        validation_value=13.0,
    )

    assert not result.success
    assert diagnostics["final_backend_value_difference"] == pytest.approx(1.0)
    assert "backends disagree" in result.message


def test_scar_tm_final_validation_rejects_extreme_ou_solution():
    result, diagnostics = _validate_candidate(
        [1e10, 0.0, 1.0],
        initial_params=[1.0, 0.0, 1.0],
        validation_value=None,
    )

    assert not result.success
    assert diagnostics["final_rho"] == pytest.approx(0.0)
    assert "autocorrelation is degenerate" in result.message
    assert "initialization scale" in result.message


def test_scar_tm_final_validation_accepts_projected_boundary_solution():
    result, diagnostics = _validate_candidate(
        [0.001, 0.0, 0.001],
        selected_grad=[1.0, 0.0, 1.0],
        validation_value=None,
        initial_params=[1.0, 0.0, 1.0],
        strict_gradient_policy=True,
    )

    assert result.success
    assert diagnostics["final_validation_passed"]
    assert diagnostics["final_projected_gradient_norm"] == pytest.approx(0.0)
    assert diagnostics["final_boundary_flags"] == (True, False, True)


def test_scar_tm_gradient_policy_is_disabled_by_default():
    result, diagnostics = _validate_candidate(
        [1.0, 0.0, 1.0],
        selected_grad=[1.0, 0.0, 0.0],
        validation_value=None,
    )

    assert SCARTMStrategy().strict_gradient_policy is False
    assert result.success
    assert "final_projected_gradient_norm" not in diagnostics


def test_scar_tm_strict_gradient_policy_rejects_large_gradient():
    result, diagnostics = _validate_candidate(
        [1.0, 0.0, 1.0],
        selected_grad=[1.0, 0.0, 0.0],
        validation_value=None,
        strict_gradient_policy=True,
    )

    assert not result.success
    assert diagnostics["final_projected_gradient_norm"] == pytest.approx(1.0)
    assert "projected gradient exceeds" in result.message


def test_scar_tm_abnormal_recovery_uses_physical_gradient(monkeypatch):
    physical_gradient = np.array([1e-3, 2e-3, 3e-3])
    optimizer_gradient = {}

    def fake_evaluate(self, kappa, mu, nu, auto_config):
        return 12.0, physical_gradient.copy(), {}

    def fake_minimize(fun, x0, *, method, jac, bounds, options):
        value, gradient = fun(x0)
        optimizer_gradient["value"] = np.asarray(gradient)
        return SimpleNamespace(
            x=np.asarray(x0),
            fun=value,
            jac=np.asarray(gradient),
            success=False,
            message="ABNORMAL: test line search",
            nfev=1,
        )

    monkeypatch.setattr(
        scar_tm._PreparedScarOuFitCache,
        "neg_loglik_with_grad_info",
        fake_evaluate,
    )
    monkeypatch.setattr(scar_tm, "minimize", fake_minimize)

    strategy = SCARTMStrategy(
        smart_init=False,
        strict_gradient_policy=True,
    )
    result = strategy.fit(
        BivariateGaussianCopula(),
        np.random.default_rng(42).uniform(0.01, 0.99, size=(20, 2)),
        alpha0=np.array([100.0, 2.0, 20.0]),
    )

    assert np.max(np.abs(optimizer_gradient["value"])) == pytest.approx(0.1)
    assert result.success
    assert "physical_projected_grad=0.003" in result.message
    assert result.diagnostics[
        "final_projected_gradient_norm"] == pytest.approx(0.003)


def test_multivariate_ppf_cache_uses_source_identity():
    first = np.full((2, 2), 0.25)
    second = np.full((2, 2), 0.75)
    model = StochasticStudentCopula(d=2, R=np.eye(2))

    first_cache = model.prepare_emission_cache(first)
    second_cache = model.prepare_emission_cache(second)

    assert second_cache is not first_cache
    assert second_cache.source_ref() is second
    assert model.prepare_emission_cache(second) is second_cache


def test_scar_tm_accepts_flat_boundary_convergence():
    u = np.random.default_rng(16).uniform(0.001, 0.999, size=(60, 2))
    copula = BivariateGaussianCopula()

    result = fit(
        copula,
        u,
        method="scar-tm-ou",
        spectral_basis_order=16,
        maxiter=80,
        maxfun=160,
        gtol=1e-5,
        strict_gradient_policy=True,
    )

    assert result.success
    assert result.params.nu == pytest.approx(0.001)
    assert result.diagnostics["final_validation_passed"]
    assert np.isfinite(
        result.diagnostics["final_selected_backend_value"])
    assert np.isfinite(
        result.diagnostics["final_projected_gradient_norm"])


@pytest.mark.parametrize("transition_method", ["auto", "spectral"])
def test_native_forward_rosenblatt_grid_fallback_matches_auto(
        transition_method):
    u = np.random.default_rng(20260802).uniform(
        0.05, 0.95, size=(10, 2))
    copula = BivariateGaussianCopula()
    kwargs = {
        "K": 31,
        "grid_range": 3.0,
        "grid_method": "dense",
        "adaptive": False,
        "transition_method": transition_method,
        "gh_order": 5,
    }
    expected = tm_forward_rosenblatt(
        0.9, 0.1, 1.0, u, copula,
        **{**kwargs, "transition_method": "auto"})

    np.testing.assert_allclose(
        tm_forward_rosenblatt(0.9, 0.1, 1.0, u, copula, **kwargs),
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_native_forward_rosenblatt_applies_final_gof_clipping():
    u = np.array(
        [[1e-10, 0.2], [0.5, 0.8], [1.0 - 1e-10, 0.4]],
        dtype=np.float64,
    )
    result = tm_forward_rosenblatt(
        0.8,
        0.0,
        1.0,
        u,
        BivariateGaussianCopula(),
        K=21,
        grid_range=3.0,
        adaptive=False,
        transition_method="matrix",
    )

    module = _cpp_scar_ou._extension.load()
    assert result[0, 0] == module.ROSENBLATT_OUTPUT_EPS
    assert result[-1, 0] == 1.0 - module.ROSENBLATT_OUTPUT_EPS
    assert np.all(result >= module.ROSENBLATT_OUTPUT_EPS)
    assert np.all(result <= 1.0 - module.ROSENBLATT_OUTPUT_EPS)


def test_bivariate_gof_passes_stored_transition_options(monkeypatch):
    rng = np.random.default_rng(6)
    u = rng.uniform(0.05, 0.95, size=(6, 2))
    copula = BivariateGaussianCopula()
    result = LatentResult(
        log_likelihood=0.0,
        method="SCAR-TM-OU",
        copula_name=copula.name,
        success=True,
        params=ou_params(0.8, 0.1, 1.0),
        transition_method="auto",
        max_K=123,
        r_gh=2.5,
        gh_order=7,
        pts_per_sigma=6,
    )
    captured = {}

    def fake_rosenblatt(kappa, mu, nu, u_arg, copula_arg, config):
        captured.update(vars(config))
        return u_arg[:, 1]

    monkeypatch.setattr(
        "pyscarcopula._native.scar_ou.mixture_h",
        fake_rosenblatt,
    )
    monkeypatch.setattr(
        "pyscarcopula._native.scar_ou.prepare_objective",
        lambda *args, **kwargs: (
            (_ for _ in ()).throw(
                _cpp_scar_ou.NativeUnsupported("test fallback"))),
    )

    stattests._bivariate_rosenblatt_from_result(copula, u, result)

    assert captured["transition_method"] == "auto"
    assert captured["max_K"] == 123
    assert captured["r_gh"] == 2.5
    assert captured["gh_order"] == 7
    assert captured["pts_per_sigma"] == 6


def test_top_level_api_uses_stored_scar_tm_options(monkeypatch):
    rng = np.random.default_rng(7)
    u = rng.uniform(0.05, 0.95, size=(6, 2))
    copula = BivariateGaussianCopula()
    result = LatentResult(
        log_likelihood=0.0,
        method="SCAR-TM-OU",
        copula_name=copula.name,
        success=True,
        params=ou_params(0.8, 0.1, 1.0),
        K=37,
        grid_range=2.5,
        grid_method="sparse",
        adaptive=False,
        pts_per_sigma=6,
        transition_method="auto",
        max_K=None,
        r_gh=2.25,
        gh_order=7,
    )
    captured = {}

    def fake_predictive_mean(kappa, mu, nu, u_arg, copula_arg, config):
        captured.update(vars(config))
        return np.zeros(len(u_arg), dtype=np.float64)

    monkeypatch.setattr(
        "pyscarcopula._native.scar_ou.predictive_mean",
        fake_predictive_mean,
    )
    monkeypatch.setattr(
        "pyscarcopula._native.scar_ou.prepare_objective",
        lambda *args, **kwargs: (
            (_ for _ in ()).throw(
                _cpp_scar_ou.NativeUnsupported("test fallback"))),
    )

    from pyscarcopula import api

    out = api.predictive_mean(copula, u, result)

    np.testing.assert_allclose(out, 0.0)
    assert captured["K"] == 37
    assert captured["grid_range"] == 2.5
    assert captured["grid_method"] == "sparse"
    assert captured["adaptive"] is False
    assert captured["pts_per_sigma"] == 6
    assert captured["transition_method"] == "auto"
    assert captured["max_K"] is None
    assert captured["r_gh"] == 2.25
    assert captured["gh_order"] == 7


def test_scar_tm_posterior_methods_use_prepared_object(monkeypatch):
    rng = np.random.default_rng(8)
    u = rng.uniform(0.05, 0.95, size=(5, 2))
    copula = BivariateGaussianCopula()
    result = LatentResult(
        log_likelihood=0.0,
        method="SCAR-TM-OU",
        copula_name=copula.name,
        success=True,
        params=ou_params(0.8, 0.1, 1.0),
    )
    calls = []

    class FakePrepared:
        def update_copula(self, copula_arg):
            calls.append(("update_copula", copula_arg is copula))

        def predictive_mean(self, kappa, mu, nu):
            calls.append(("predictive_mean", kappa, mu, nu))
            return np.full(len(u), 0.25)

        def mixture_h(self, kappa, mu, nu):
            calls.append(("mixture_h", kappa, mu, nu))
            return np.full(len(u), 0.5)

        def state_distribution(self, kappa, mu, nu, horizon="current"):
            calls.append(("state_distribution", horizon, kappa, mu, nu))
            return np.array([-1.0, 1.0]), np.array([0.4, 0.6])

    def fake_prepare(u_arg, copula_arg, config):
        calls.append((
            "prepare",
            u_arg is u,
            copula_arg is copula,
            config.transition_method,
            config.K,
        ))
        return FakePrepared()

    monkeypatch.setattr(_cpp_scar_ou, "prepare_objective", fake_prepare)

    strategy = SCARTMStrategy(
        transition_method="matrix",
        K=17,
        max_K=17,
        adaptive=False,
    )
    state_cache = {}

    predictive = strategy.predictive_mean(copula, u, result)
    rosenblatt = strategy.rosenblatt_e2(copula, u, result)
    mixed = strategy.mixture_h(
        copula,
        u,
        result,
        state_cache=state_cache,
        current_cache_key="current",
        next_cache_key="next",
    )
    state = strategy.predictive_state(
        copula, u, result, horizon="current")

    np.testing.assert_allclose(predictive, 0.25)
    np.testing.assert_allclose(rosenblatt, 0.5)
    np.testing.assert_allclose(mixed, 0.5)
    np.testing.assert_allclose(state.z_grid, [-1.0, 1.0])
    np.testing.assert_allclose(state.prob, [0.4, 0.6])
    np.testing.assert_allclose(state_cache["current"][1], [0.4, 0.6])
    np.testing.assert_allclose(state_cache["next"][1], [0.4, 0.6])
    assert calls.count(("prepare", True, True, "matrix", 17)) == 4
    assert calls.count(("update_copula", True)) == 4
    assert ("predictive_mean", 0.8, 0.1, 1.0) in calls
    assert calls.count(("mixture_h", 0.8, 0.1, 1.0)) == 2
    assert ("state_distribution", "current", 0.8, 0.1, 1.0) in calls
    assert ("state_distribution", "next", 0.8, 0.1, 1.0) in calls


def test_scar_tm_posterior_cache_reuses_prepared_object(monkeypatch):
    rng = np.random.default_rng(9)
    u = rng.uniform(0.05, 0.95, size=(5, 2))
    copula = BivariateGaussianCopula()
    result = LatentResult(
        log_likelihood=0.0,
        method="SCAR-TM-OU",
        copula_name=copula.name,
        success=True,
        params=ou_params(0.8, 0.1, 1.0),
    )
    calls = []

    class FakePrepared:
        def update_copula(self, copula_arg):
            calls.append(("update_copula", copula_arg is copula))

        def predictive_mean(self, kappa, mu, nu):
            calls.append(("predictive_mean", kappa, mu, nu))
            return np.full(len(u), 0.25)

        def mixture_h(self, kappa, mu, nu):
            calls.append(("mixture_h", kappa, mu, nu))
            return np.full(len(u), 0.5)

        def state_distribution(self, kappa, mu, nu, horizon="current"):
            calls.append(("state_distribution", horizon, kappa, mu, nu))
            return np.array([-1.0, 1.0]), np.array([0.4, 0.6])

    def fake_prepare(u_arg, copula_arg, config):
        calls.append((
            "prepare",
            u_arg is u,
            copula_arg is copula,
            config.transition_method,
            config.K,
        ))
        return FakePrepared()

    monkeypatch.setattr(_cpp_scar_ou, "prepare_objective", fake_prepare)

    strategy = SCARTMStrategy(
        transition_method="matrix",
        K=17,
        max_K=17,
        adaptive=False,
    )
    posterior_cache = {}
    state_cache = {}

    strategy.predictive_mean(
        copula, u, result, posterior_cache=posterior_cache)
    strategy.rosenblatt_e2(
        copula, u, result, posterior_cache=posterior_cache)
    strategy.mixture_h(
        copula,
        u,
        result,
        state_cache=state_cache,
        current_cache_key="current",
        next_cache_key="next",
        posterior_cache=posterior_cache,
    )
    strategy.predictive_state(
        copula,
        u,
        result,
        horizon="current",
        posterior_cache=posterior_cache,
    )

    assert calls.count(("prepare", True, True, "matrix", 17)) == 1
    assert calls.count(("update_copula", True)) == 6
    assert set(state_cache) == {"current", "next"}
    np.testing.assert_allclose(state_cache["current"][1], [0.4, 0.6])
    np.testing.assert_allclose(state_cache["next"][1], [0.4, 0.6])
    assert len(posterior_cache) == 1


def test_native_loglik_matches_negative_objective_on_notebook_dataset(
        crypto_data):
    """Use the BTC/ETH dataset from examples/02_bivariate.ipynb."""
    copula = GumbelCopula(rotate=180)

    params = {
        "kappa": 59.02,
        "mu": 2.17,
        "nu": 15.80,
        "K": 60,
        "grid_range": 5.0,
        "grid_method": "auto",
        "adaptive": True,
        "pts_per_sigma": 4,
    }

    backward_loglik = -tm_loglik(**params, u=crypto_data, copula=copula)
    config = AutoTMConfig(
        transition_method="matrix",
        K=params["K"],
        grid_range=params["grid_range"],
        grid_method=params["grid_method"],
        adaptive=params["adaptive"],
        pts_per_sigma=params["pts_per_sigma"],
    )
    native_loglik, _ = _cpp_scar_ou.loglik(
        params["kappa"],
        params["mu"],
        params["nu"],
        crypto_data,
        copula,
        config,
    )

    np.testing.assert_allclose(
        native_loglik,
        backward_loglik,
        rtol=1e-10,
        atol=1e-8,
    )
