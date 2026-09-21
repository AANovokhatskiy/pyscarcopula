"""Numerical and resource regressions for the Hermite engineering backport."""

import numpy as np
import pytest

from pyscarcopula import BivariateGaussianCopula, FrankCopula, StochasticStudentCopula
from pyscarcopula._native import NativeError, scar_ou
from pyscarcopula.numerical._scar_ou_config import AutoTMConfig


def prepared(u, model, **options):
    return scar_ou.prepare_objective(u, model, AutoTMConfig(
        transition_method="spectral", basis_order=32, quad_order=80, **options))


@pytest.mark.parametrize("kappa", [1e-8, 2.0, 1e6])
@pytest.mark.parametrize("family", [BivariateGaussianCopula, FrankCopula])
def test_physical_gradient_matches_independent_eta_finite_difference(kappa, family):
    u = np.random.default_rng(128).uniform(.15, .85, (19, 2))
    obj = prepared(u, family())
    eta = np.array([np.log(kappa), .4, np.log(.6)])

    def physical(x):
        k = np.exp(x[0])
        return k, x[1], np.exp(x[2]) * np.sqrt(2 * k)

    value, g, _ = obj.neg_loglik_with_grad_info(*physical(eta))
    k, _, nu = physical(eta)
    mapped = np.array([k*g[0] + .5*nu*g[2], g[1], nu*g[2]])
    fd = []
    for i in range(3):
        shift = np.eye(3)[i] * 1e-5
        hi = obj.neg_loglik_info(*physical(eta+shift))[0]
        lo = obj.neg_loglik_info(*physical(eta-shift))[0]
        fd.append((hi-lo)/2e-5)
    assert np.isfinite(value)
    np.testing.assert_allclose(mapped, fd, atol=2e-7, rtol=2e-5)


def test_gaussian_fast_path_survives_entire_density_row_underflow():
    # Almost perfectly positive correlation, but discordant observations:
    # every unscaled density is below float64's range.
    u = np.tile([.01, .99], (8, 1))
    obj = prepared(u, BivariateGaussianCopula())
    params = np.array([2., 20., .02])
    value, g, _ = obj.neg_loglik_with_grad_info(*params)
    reference = obj.neg_loglik_info(*params)[0]
    assert value > 1000 and np.all(np.isfinite(g))
    np.testing.assert_allclose(value, reference, rtol=2e-12, atol=1e-8)
    step = 1e-4
    hi, lo = params.copy(), params.copy()
    hi[1] += step
    lo[1] -= step
    fd = (obj.neg_loglik_info(*hi)[0] - obj.neg_loglik_info(*lo)[0])/(2*step)
    np.testing.assert_allclose(g[1], fd, rtol=2e-5, atol=1e-4)


def test_auto_keeps_successful_scaled_two_node_gaussian_likelihood():
    # Before scaled Gaussian emissions, this case failed solely from underflow
    # and fell back to matrix. A finite spectral evaluation must now be retained.
    u = np.random.default_rng(1).uniform(.001, .999, (50, 2))
    config = AutoTMConfig(
        transition_method="auto", small_kdt=1e-9, basis_order=2, quad_order=2,
        K=30, adaptive=False, max_K=None)
    model = BivariateGaussianCopula()
    params = (.1, 0., 10.)
    value, scalar_info = scar_ou.neg_loglik_info(*params, u, model, config)
    grad_value, gradient, grad_info = scar_ou.neg_loglik_with_grad_info(
        *params, u, model, config)
    assert value > 1000 and np.isfinite(value)
    assert np.all(np.isfinite(gradient))
    np.testing.assert_allclose(value, grad_value, rtol=2e-12)
    for info in (scalar_info, grad_info):
        assert info["selected_backend"] == info["backend"] == "spectral"
        assert not info.get("fallback_chain")


def test_underflowed_ou_modes_do_not_overflow_log_kappa_derivative():
    obj = prepared(np.array([[.3, .7], [.4, .6]]), BivariateGaussianCopula())
    kappa = 1e307
    nu = .6 * np.sqrt(2 * kappa)
    value, g, _ = obj.neg_loglik_with_grad_info(kappa, .4, nu)
    assert np.isfinite(value) and np.all(np.isfinite(g))
    # Independent-transition limit at fixed stationary sigma.
    reference = obj.neg_loglik_with_grad_info(1e4, .4, .6*np.sqrt(2e4))
    np.testing.assert_allclose(value, reference[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(nu*g[2], .6*np.sqrt(2e4)*reference[1][2], atol=1e-12)


@pytest.mark.parametrize("mu", [-4., .7, 4.])
def test_scaled_gaussian_agrees_in_prepared_and_stateless_paths(mu):
    u = np.random.default_rng(43).uniform(.01, .99, (17, 2))
    model = BivariateGaussianCopula()
    config = AutoTMConfig(transition_method="spectral", basis_order=32, quad_order=80)
    obj = scar_ou.prepare_objective(u, model, config)
    params = (4., mu, .8)
    actual = obj.neg_loglik_with_grad_info(*params)
    expected = scar_ou.neg_loglik_with_grad_info(*params, u, model, config)
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_allclose(actual[0], obj.neg_loglik_info(*params)[0], atol=1e-12)


@pytest.mark.parametrize("dimension", [3, 4, 5])
def test_correlation_blocks_and_remainders_match_density_finite_difference(dimension):
    rng = np.random.default_rng(784)
    u = rng.uniform(.15, .85, (9, dimension))
    r = np.full((dimension, dimension), .2)
    np.fill_diagonal(r, 1.)
    obj = prepared(u, StochasticStudentCopula(d=dimension, R=r))
    params = (3., .5, .7)
    value, ou, corr, _ = obj.neg_loglik_with_grad_and_corr_info(*params)
    ordinary = obj.neg_loglik_with_grad_info(*params)
    np.testing.assert_allclose(value, ordinary[0], atol=1e-12)
    np.testing.assert_allclose(ou, ordinary[1], atol=1e-12)
    direction = rng.normal(size=len(corr))
    directed = obj.neg_loglik_with_grad_and_corr_directional_info(*params, direction)
    np.testing.assert_allclose(directed[2], [corr @ direction], atol=1e-10, rtol=1e-10)
    # Full call following a directional call must not retain stale derivatives.
    again = obj.neg_loglik_with_grad_and_corr_info(*params)
    np.testing.assert_array_equal(corr, again[2])
    fd = []
    for i in range(1, dimension):
        for j in range(i):
            vals = []
            for sign in [1, -1]:
                trial = r.copy()
                trial[i,j] += sign * 1e-5
                trial[j,i] = trial[i,j]
                trial_obj = prepared(u, StochasticStudentCopula(d=dimension, R=trial))
                vals.append(trial_obj.neg_loglik_info(*params)[0])
            fd.append((vals[0]-vals[1])/2e-5)
    np.testing.assert_allclose(corr, fd, atol=2e-6, rtol=2e-5)


def test_spectral_correlation_budget_is_checked_and_directional_remains_available():
    u = np.random.default_rng(13).uniform(.15, .85, (9, 5))
    model = StochasticStudentCopula(d=5, R=np.eye(5))
    # Full 10-tangent workspace exceeds this budget, one tangent fits.
    obj = prepared(u, model, corr_gradient_block_bytes=4096)
    with pytest.raises(NativeError, match="invalid_size"):
        obj.neg_loglik_with_grad_and_corr_info(3., .5, .7)
    value, ou, corr, _ = obj.neg_loglik_with_grad_and_corr_directional_info(
        3., .5, .7, np.ones(10))
    assert np.isfinite(value) and np.all(np.isfinite(ou)) and np.all(np.isfinite(corr))
    # The same setting must not limit an ordinary three-tangent OU call.
    assert np.isfinite(obj.neg_loglik_with_grad_info(3., .5, .7)[0])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_correlation_direction_is_rejected(bad):
    u = np.full((3, 3), .4)
    obj = prepared(u, StochasticStudentCopula(d=3, R=np.eye(3)))
    with pytest.raises((NativeError, ValueError)):
        obj.neg_loglik_with_grad_and_corr_directional_info(2., .4, .8, [bad, 0., 0.])
