"""Opt-in prepared full-emission reconstruction; no default path change."""
import gc

import numpy as np
import pytest
from scipy.special import gammaln
from scipy.stats import t

from pyscarcopula import StochasticStudentCopula
from pyscarcopula._native import scar_ou
from pyscarcopula.numerical._scar_ou_config import AutoTMConfig


def prepared(backend="spectral", u=None):
    if u is None:
        u = np.random.default_rng(440).uniform(.015, .985, (12, 3))
    R = np.full((3, 3), .3)
    np.fill_diagonal(R, 1)
    model = StochasticStudentCopula(d=3, R=R, corr_mode="fixed")
    cfg = AutoTMConfig(transition_method=backend, K=81, max_K=81,
                       adaptive=False, basis_order=24, quad_order=64)
    obj = scar_ou.prepare_objective(u, model, cfg)
    return obj, model, u, R


@pytest.mark.parametrize("backend", ["spectral", "matrix", "local"])
def test_cache_ou_gradient_matches_same_objective_finite_difference(backend):
    obj, _, _, _ = prepared(backend)
    obj._native.configure_student_emission_cache()
    params = np.array([3., 2., 1.5])
    value, gradient, info = obj.neg_loglik_with_grad_info(*params)
    assert np.isfinite(value)
    assert info["backend"] == backend
    numerical = []
    for index in range(3):
        step = 2e-5 * max(1, abs(params[index]))
        delta = np.zeros(3)
        delta[index] = step
        numerical.append((obj.neg_loglik_info(*(params + delta))[0]
                          - obj.neg_loglik_info(*(params - delta))[0]) / (2 * step))
    np.testing.assert_allclose(gradient, numerical, atol=2e-5, rtol=2e-4)
    info = obj._native.student_emission_cache_info()
    assert info["active"] and info["interpolation_hits"] > 0
    assert not info["integration_certified"]
    assert info["max_value_residual"] <= info["value_tolerance"]
    assert info["max_score_residual"] <= info["score_tolerance"]


@pytest.mark.parametrize("mu", [-50., -5., .35, 8., 1200.])
def test_cache_static_limit_matches_independent_scipy_student(mu):
    u = np.repeat(np.array([[.02, .47, .91]]), 3, axis=0)
    obj, model, _, R = prepared(u=u)
    obj._native.configure_student_emission_cache()
    value = obj.neg_loglik_info(10., mu, 1e-8)[0]
    df = 2.000001 + np.logaddexp(0, mu)
    z = t.ppf(u, df)
    q = np.einsum("ti,ij,tj->t", z, np.linalg.inv(R), z)
    d = 3
    exact = (gammaln((df+d)/2) + (d-1)*gammaln(df/2)
             - d*gammaln((df+1)/2) - .5*np.linalg.slogdet(R)[1]
             - .5*(df+d)*np.log1p(q/df)
             + .5*(df+1)*np.log1p(z*z/df).sum(axis=1)).sum()
    np.testing.assert_allclose(value, -exact, atol=2e-6, rtol=0)
    info = obj._native.student_emission_cache_info()
    if mu == -50:
        assert info["exact_endpoint_hits"] > 0
    if mu == 1200:
        assert info["exact_fallbacks"] > 0


def test_cache_update_clear_ownership_and_atomic_configuration():
    obj, model, u, _ = prepared()
    baseline = obj.neg_loglik_with_grad_info(3., 2., 1.5)[:2]
    obj._native.configure_student_emission_cache()
    cached = obj.neg_loglik_with_grad_info(3., 2., 1.5)[:2]
    u[:] = .5
    gc.collect()
    again = obj.neg_loglik_with_grad_info(3., 2., 1.5)[:2]
    assert cached[0] == again[0]
    np.testing.assert_array_equal(cached[1], again[1])
    with pytest.raises(ValueError, match="correlation gradients"):
        obj.neg_loglik_with_grad_and_corr_info(3., 2., 1.5)
    with pytest.raises(ValueError, match="memory budget"):
        obj._native.configure_student_emission_cache(max_bytes=1)
    assert obj._native.student_emission_cache_info()["active"]
    with pytest.raises(ValueError):
        obj._native.update_student_factor(np.full(9, np.nan), 0.)
    assert obj._native.student_emission_cache_info()["active"]
    obj._native.update_student_factor(model._L_inv.reshape(-1), model._log_det)
    assert not obj._native.student_emission_cache_info()["active"]
    restored = obj.neg_loglik_with_grad_info(3., 2., 1.5)[:2]
    assert restored[0] == baseline[0]
    np.testing.assert_array_equal(restored[1], baseline[1])
    obj._native.clear_student_emission_cache()


@pytest.mark.parametrize("kwargs", [
    {"min_coordinate": float("nan")}, {"max_coordinate": float("inf")},
    {"min_coordinate": -37.}, {"max_coordinate": 14.},
    {"value_tolerance": 0.}, {"score_tolerance": -1.},
    {"max_knots": 2}, {"initial_intervals": 0}, {"max_depth": 17},
])
def test_cache_rejects_invalid_configuration(kwargs):
    obj, _, _, _ = prepared()
    with pytest.raises(ValueError):
        obj._native.configure_student_emission_cache(**kwargs)
    assert not obj._native.student_emission_cache_info()["active"]


def test_cache_refinement_budget_exhaustion_is_explicit_and_atomic():
    obj, _, _, _ = prepared()
    with pytest.raises(RuntimeError, match="exhausted"):
        obj._native.configure_student_emission_cache(
            initial_intervals=1, max_knots=2, max_depth=0,
            value_tolerance=1e-14, score_tolerance=1e-14)
    assert not obj._native.student_emission_cache_info()["active"]


def test_optional_lower_boundary_handles_rounded_degrees_of_freedom():
    obj, _, _, _ = prepared()
    obj._native.configure_student_emission_cache(min_coordinate=-36.)
    for mu in (-50., -36., -35.5, -30., -24.):
        value, gradient, _ = obj.neg_loglik_with_grad_info(10., mu, 1e-8)
        assert np.isfinite(value)
        assert np.all(np.isfinite(gradient))
        assert abs(gradient[1]) < 1e-10


@pytest.mark.parametrize("upper", [6.2, float(np.log(1000. - 2.000001))])
def test_refined_cache_and_direct_fallback_near_central_quantile_regression(upper):
    # The legacy unprepared quantile at the upper df point can fail to finish
    # Newton's bracket when CDF subtraction near p=.5 stalls. In the reported
    # us6 row it returned approximately half the correct quantile.
    u = np.repeat(np.array([[.33942766, 1261 / 2516, .95389507]]), 3, axis=0)
    obj, _, _, R = prepared(u=u)
    obj._native.configure_student_emission_cache(
        min_coordinate=-36., max_coordinate=upper)
    for df in (523.11649323681615, 523.2017926352811, 523.287105996051):
        mu = df - 2.000001
        actual, gradient, _ = obj.neg_loglik_with_grad_info(10., mu, 1e-8)
        z = t.ppf(u, df)
        q = np.einsum("ti,ij,tj->t", z, np.linalg.inv(R), z)
        exact = (gammaln((df+3)/2) + 2*gammaln(df/2)
                 - 3*gammaln((df+1)/2) - .5*np.linalg.slogdet(R)[1]
                 - .5*(df+3)*np.log1p(q/df)
                 + .5*(df+1)*np.log1p(z*z/df).sum(axis=1)).sum()
        np.testing.assert_allclose(actual, -exact, atol=2e-6, rtol=0)
        step = .01
        finite_difference = (obj.neg_loglik_info(10., mu+step, 1e-8)[0]
                             - obj.neg_loglik_info(10., mu-step, 1e-8)[0]) / (2*step)
        np.testing.assert_allclose(gradient[1], finite_difference, atol=2e-7, rtol=0)


@pytest.mark.parametrize("adjacent", [False, True])
def test_probability_deduplication_uses_exact_equality(adjacent):
    ranks = np.arange(1, 13, dtype=float) / 13
    u = np.ascontiguousarray(np.column_stack((ranks, ranks[::-1], np.roll(ranks, 3))))
    if adjacent:
        u[:, 1] = np.nextafter(u[:, 1], 1.)
    obj, _, _, _ = prepared(u=u)
    obj._native.configure_student_emission_cache()
    info = obj._native.student_emission_cache_info()
    assert info["observation_entries"] == u.size
    assert info["unique_probabilities"] == len(np.unique(u))
    assert info["unique_probabilities"] == (24 if adjacent else 12)


def test_probability_deduplication_scratch_is_budgeted_before_build():
    obj, _, _, _ = prepared()
    with pytest.raises(ValueError, match="deduplication exceeds memory budget"):
        obj._native.configure_student_emission_cache(
            max_knots=33, max_depth=0, max_bytes=9000)
    assert not obj._native.student_emission_cache_info()["active"]
