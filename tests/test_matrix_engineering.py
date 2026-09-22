"""Independent continuous-OU quadrature and matrix backport regressions."""
import numpy as np
import pytest
from scipy.special import ndtr, ndtri, roots_legendre

from pyscarcopula import BivariateGaussianCopula, FrankCopula
from pyscarcopula._native import scar_ou
from pyscarcopula.numerical._scar_ou_config import AutoTMConfig


def objective(u, k, storage="sparse", radius=8, family=None):
    return scar_ou.prepare_objective(u, family or BivariateGaussianCopula(), AutoTMConfig(
        transition_method="matrix", grid_method=storage, adaptive=False,
        K=k, max_K=k, grid_range=radius, n_threads=1))


def coordinates(obj, t, a, mu=.2, sigma=.9):
    kappa = a*(t-1)
    nu = sigma*np.sqrt(2*kappa)
    value, g, _ = obj.neg_loglik_with_grad_info(kappa, mu, nu)
    return np.array([value, kappa*g[0]+.5*nu*g[2], g[1], nu*g[2]])


def legendre_reference(u, a, order, radius=8, mu=.2, sigma=.9):
    """Gaussian OU density integrated with GL; tangent in (log a, mu, log sigma).

    No native transition, copula density, derivative or support selector is used.
    """
    z, w = roots_legendre(order)
    z, w = radius*z, radius*w
    rho, variance = np.exp(-a), -np.expm1(-2*a)
    delta = z[None, :]-rho*z[:, None]
    p = np.exp(-delta**2/(2*variance))*w[None, :]/np.sqrt(2*np.pi*variance)
    dp = -a*rho*p*(rho/variance + delta*z[:, None]/variance
                            - rho*delta**2/variance**2)
    x = mu+sigma*z
    tanh = np.tanh(x/4)
    r = .9999*tanh
    dr = .9999*(1-tanh*tanh)/4
    v = 1-r*r
    normal = ndtri(u)
    beta = np.zeros((order, 4))
    beta[:, 0] = 1
    log_scale = 0.
    for t in range(len(u)-1, -1, -1):
        z1, z2 = normal[t]
        squares, cross = z1*z1+z2*z2, z1*z2
        logf = -.5*np.log(v)-(r*r*squares-2*r*cross)/(2*v)
        shift = logf.max()
        f = np.exp(logf-shift)
        dlog = (r/v + (cross*(1+r*r)-r*squares)/(v*v))*dr
        target = beta*f[:, None]
        target[:, 2] += beta[:, 0]*f*dlog
        target[:, 3] += beta[:, 0]*f*dlog*sigma*z
        log_scale += shift
        if t:
            # Direct contractions avoid BLAS thread overhead for four columns
            # without changing the process-wide thread configuration.
            beta = np.einsum("ij,jk->ik", p, target, optimize=False)
            beta[:, 1] += np.einsum("ij,j->i", dp, target[:, 0], optimize=False)
            scale = beta[:, 0].max()
            beta /= scale
            log_scale += np.log(scale)
        else:
            total = np.einsum(
                "i,ij->j", w*np.exp(-z*z/2)/np.sqrt(2*np.pi), target,
                optimize=False)
    return -np.r_[np.log(total[0])+log_scale, total[1:]/total[0]]


@pytest.mark.parametrize("historical", [False, True])
@pytest.mark.parametrize("t,a,k", [(40, .01, 1025), (200, .01, 513),
                                  (200, .01, 1025), (40, .7, 257),
                                  (1000, .003, 1025)])
def test_sparse_and_auto_against_independent_legendre(t, a, k, historical):
    u = np.random.default_rng(400000+t).uniform(.03, .97, (t, 2))
    if historical:
        rng = np.random.default_rng(20260913+t+400000)
        u = np.clip(ndtr(rng.normal(size=(t, 2))*1.5), 1e-5, 1-1e-5)
    reference = legendre_reference(u, a, 600)
    refined = legendre_reference(u, a, 900, radius=10)
    tolerance = np.array([1e-5, 1e-4, 1e-4, 1e-4])
    assert np.all(np.abs(reference-refined) < tolerance*.01)
    for storage in ("dense", "sparse", "auto"):
        actual = coordinates(objective(u, k, storage), t, a)
        assert np.all(np.abs(actual-refined) < tolerance)
    np.testing.assert_allclose(
        coordinates(objective(u, k), t, a),
        coordinates(objective(u, k, "dense"), t, a), atol=2e-10, rtol=2e-10)


@pytest.mark.parametrize("family", [BivariateGaussianCopula, FrankCopula])
@pytest.mark.parametrize("storage", ["dense", "sparse"])
def test_fused_physical_scores_and_scalar_likelihood(family, storage):
    u = np.random.default_rng(172).uniform(.1, .9, (23, 2))
    obj = objective(u, 257, storage, family=family())
    eta = np.array([np.log(.02), .2, np.log(.9)])
    def physical(e):
        kappa = np.exp(e[0])*(len(u)-1)
        return kappa, e[1], np.exp(e[2])*np.sqrt(2*kappa)
    actual = coordinates(obj, len(u), .02)
    assert actual[0] == pytest.approx(obj.neg_loglik_info(*physical(eta))[0], abs=2e-11)
    fd = []
    for i in range(3):
        step = np.eye(3)[i]*1e-5
        fd.append((obj.neg_loglik_info(*physical(eta+step))[0]
                   - obj.neg_loglik_info(*physical(eta-step))[0])/2e-5)
    np.testing.assert_allclose(actual[1:], fd, atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize("a", [1e-12, 1e-18])
def test_small_a_retains_conditional_variance(a):
    # Deliberately unresolved spatial grid: tests arithmetic agreement only,
    # not continuous-OU accuracy in this regime (rho rounds to one at 1e-18).
    u = np.random.default_rng(44).uniform(.2, .8, (5, 2))
    obj = objective(u, 33)
    params = (a*4, .2, .9*np.sqrt(8*a))
    value, g, _ = obj.neg_loglik_with_grad_info(*params)
    assert np.isfinite(value) and np.all(np.isfinite(g))
    np.testing.assert_allclose(value, obj.neg_loglik_info(*params)[0], atol=2e-11)
