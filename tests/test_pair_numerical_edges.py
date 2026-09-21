"""Regressions against independent limits and high-precision reference values."""
from decimal import Decimal, localcontext

import numpy as np
import pytest
from scipy.special import ndtri

from pyscarcopula import ClaytonCopula, FrankCopula, BivariateGaussianCopula, GumbelCopula, JoeCopula


@pytest.mark.parametrize("parameter,expected", [
    (1e-10, 0.08000000000048668),
    (1e-8, 0.08000000004866667),
])
def test_frank_score_near_independence(parameter, expected):
    # References evaluated from the defining density at 90 decimal digits.
    actual = FrankCopula().dlog_pdf_dr_unrotated(.3, .3, parameter)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-14)


@pytest.mark.parametrize("cls,parameter,u,expected", [
    (FrankCopula, 1000., .9, .001),
    (JoeCopula, 30., 1 - 1e-12, .03371259508673416),
    (JoeCopula, 1000., .9, .0010003078538204417),
    (JoeCopula, 1000., 1., .0010003078538204417),
])
def test_score_survives_underflow(cls, parameter, u, expected):
    actual = cls().dlog_pdf_dr_unrotated(u, u, parameter)
    np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-12)


@pytest.mark.parametrize("cls", [FrankCopula, JoeCopula])
def test_density_grid_score_survives_underflow(cls):
    copula = cls()
    u = np.array([.9, .9])
    x = np.array([1000.])
    _, gradient = copula.pdf_and_grad_on_grid(u, x)
    step = .01
    reference = (copula.pdf_on_grid(u, x + step)
                 - copula.pdf_on_grid(u, x - step)) / (2 * step)
    np.testing.assert_allclose(gradient, reference, rtol=2e-7)


@pytest.mark.parametrize("parameter,expected_log,expected_score", [
    (1e-16, 4.160490490458664e-18, .04160490490458670),
    (1e-8, 4.1604911443402805e-10, .04160491798221893),
])
def test_clayton_independence_limit(parameter, expected_log, expected_score):
    copula = ClaytonCopula()
    np.testing.assert_allclose(copula.log_pdf(.3, .3, parameter),
                               expected_log, rtol=1e-12, atol=1e-28)
    np.testing.assert_allclose(copula.dlog_pdf_dr_unrotated(.3, .3, parameter),
                               expected_score, rtol=1e-12)


def test_finite_boundary_density_and_score():
    assert GumbelCopula().log_pdf(1., 1., 1.)[0] == 0.
    np.testing.assert_allclose(
        ClaytonCopula().dlog_pdf_dr_unrotated(1., 1., 2.), 1 / 3)
    # For r > 1, exactly one upper endpoint has zero density; introducing
    # a positive log-argument floor here would change Bayes conditioning.
    np.testing.assert_array_equal(GumbelCopula().pdf([0., .3], [1., 1.], 2.),
                                  [0., 0.])


@pytest.mark.parametrize("rho", [np.nextafter(1., 0.), np.nextafter(-1., 0.)])
def test_gaussian_nearly_singular_diagonal(rho):
    u, v = (.9, .9) if rho > 0 else (.9, .1)
    x, y = ndtri(u), ndtri(v)
    # Evaluate the defining quadratic form at high precision, independently
    # of the native kernel's rearrangement near singular correlations.
    with localcontext() as context:
        context.prec = 80
        dx, dy, dr = map(Decimal.from_float, (float(x), float(y), float(rho)))
        variance = 1 - dr * dr
        squares, product = dx * dx + dy * dy, dx * dy
        numerator = dr * dr * squares - 2 * dr * product
        reference = float(-variance.ln() / 2 - numerator / (2 * variance))
        reference_score = float(dr / variance - (
            (2 * dr * squares - 2 * product) * variance + 2 * dr * numerator
        ) / (2 * variance * variance))
    copula = BivariateGaussianCopula()
    # The existing unrefined normal quantile at .1/.9 differs from SciPy
    # by 1.405e-9. On this diagonal its propagated log-density error is
    # bounded by |z|*delta + delta^2/2; do not demand refined-quantile accuracy.
    quantile_error = 1.5e-9
    density_error = abs(x) * quantile_error + quantile_error**2 / 2
    np.testing.assert_allclose(copula.log_pdf(u, v, rho), reference,
                               rtol=1e-13, atol=density_error)
    np.testing.assert_allclose(copula.dlog_pdf_dr_unrotated(u, v, rho),
                               reference_score, rtol=1e-13)


@pytest.mark.parametrize("parameter", [1e-8, 1000.])
def test_frank_score_at_upper_boundary(parameter):
    # At (1,1), log c = log(theta) - log(1-exp(-theta)).
    expected = (.5 - parameter / 12 if parameter < 1e-5
                else 1 / parameter - 1 / np.expm1(min(parameter, 700.)))
    np.testing.assert_allclose(
        FrankCopula().dlog_pdf_dr_unrotated(1., 1., parameter), expected,
        rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("parameter", [1e8, 1e10, 1e12, 1e16])
@pytest.mark.parametrize("offset", [0.0, -2.0, 0.5, 2.0])
def test_frank_large_parameter_score_retains_diagonal_and_nearby_limits(parameter, offset):
    u, v = .9, .9 + offset / parameter
    # On this interior strip the omitted exp(-theta*u), exp(-theta*v)
    # and exp(-theta*(1-v)) terms are smaller than exp(-1e6).
    # Differentiate theta/(4*cosh(theta*(u-v)/2)**2) in Decimal,
    # preserving the actual float inputs rather than the requested offset.
    with localcontext() as context:
        context.prec = 80
        theta, first, second = map(Decimal.from_float, (parameter, u, v))
        difference = first - second
        exponential = (theta * difference).exp()
        reference = float(1 / theta + difference * (1 - exponential) / (1 + exponential))
    copula = FrankCopula()
    score = copula.dlog_pdf_dr_unrotated(u, v, parameter)
    np.testing.assert_allclose(score, reference, rtol=2e-13, atol=1e-28)
    state = copula.inv_transform(np.array([parameter]))
    np.testing.assert_array_equal(copula.transform(state), [parameter])
    density, gradient = copula.pdf_and_grad_on_grid(np.array([u, v]), state)
    assert np.all(np.isfinite(density)) and np.all(density > 0)
    # Shifted softplus has unit derivative here, so both scores agree.
    np.testing.assert_allclose(gradient / density, reference, rtol=2e-13, atol=1e-28)
