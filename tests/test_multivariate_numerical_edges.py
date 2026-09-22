"""Independent oracles at representable correlation boundaries."""

import numpy as np
import pytest
from scipy.special import ndtri

from pyscarcopula import (
    EquicorrGaussianCopula,
    FactorCorrelation,
    GaussianCopula,
)
from pyscarcopula.copula.multivariate.equicorr_prepared import EquicorrPreparedData


@pytest.mark.parametrize("dimension", [2, 3, 10])
def test_equicorr_transform_stays_strictly_inside_spd_domain(dimension):
    model = EquicorrGaussianCopula(dimension)
    states = np.array([-1000., -20., -19., 19., 20., 1000.])
    rho = model.transform(states)
    assert np.all(rho > -1.0 / (dimension - 1))
    assert np.all(rho < 1.0)
    assert np.all(np.isfinite(model.dtransform(states)))
    assert np.all(model.dtransform(states) >= 0.0)
    pdf, gradient = model.pdf_and_grad_on_grid(
        np.full(dimension, .9), np.array([19., 20.]))
    assert np.all(np.isfinite(pdf))
    assert np.all(pdf > 0.)
    assert np.all(np.isfinite(gradient))
    assert np.all(gradient >= 0.)
    # Zero scores isolate the determinant at the negative SPD boundary.
    lower_pdf, lower_gradient = model.pdf_and_grad_on_grid(
        np.full(dimension, .5), np.array([-19., -20.]))
    assert np.all(np.isfinite(lower_pdf))
    assert np.all(lower_pdf > 0.)
    assert np.all(np.isfinite(lower_gradient))


@pytest.mark.parametrize("dimension", [2, 3, 10])
def test_equicorr_inverse_preserves_near_boundary_parameters(dimension):
    model = EquicorrGaussianCopula(dimension)
    lower = -1.0 / (dimension - 1)
    rho = np.array([
        np.nextafter(lower, 1.), lower + 1e-8,
        1. - 1e-8, np.nextafter(1., 0.),
    ])
    recovered = model.transform(model.inv_transform(rho))
    np.testing.assert_array_max_ulp(recovered, rho, maxulp=1)


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("rho", [.999, 1.-1e-12, np.nextafter(1., 0.)])
def test_equicorr_diagonal_log_density_and_score(prepared, rho):
    model = EquicorrGaussianCopula(3)
    data = np.full((1, 3), .9)
    if prepared:
        data = model.prepare_sufficient_statistics(data)
    value, score = model.log_pdf_and_dlog_dr_rows(data, rho)
    z = ndtri(.9)
    eigenvalue = 1. + 2. * rho
    expected = (-np.log1p(-rho) - .5 * np.log(eigenvalue)
                + 1.5 * z * z * (1. - 1. / eigenvalue))
    expected_score = (1. / (1. - rho) - 1. / eigenvalue
                      + 3. * z * z / eigenvalue**2)
    np.testing.assert_allclose(value, expected, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(score, expected_score, rtol=2e-13)
    np.testing.assert_allclose(model.log_likelihood(data, rho), expected,
                               rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("prepared", [False, True])
def test_equicorr_keeps_small_real_orthogonal_variance(prepared):
    model = EquicorrGaussianCopula(3)
    data = np.array([[.9 - 1e-8, .9, .9 + 1e-8]])
    scores = ndtri(data[0])
    mean = scores.mean()
    centered = np.sum((scores - mean)**2)
    rho = 1. - 1e-12
    eigenvalue = 1. + 2. * rho
    expected = (-np.log1p(-rho) - .5 * np.log(eigenvalue)
                - .5 * (rho / (1. - rho) * centered
                          - 2. * rho / eigenvalue * (3. * mean * mean)))
    if prepared:
        data = model.prepare_sufficient_statistics(data)
    np.testing.assert_allclose(model.log_pdf_rows(data, rho), expected,
                               rtol=0., atol=2e-8)


@pytest.mark.parametrize("rho", [.3, 1. - 1e-12])
def test_legacy_equicorr_stats_allow_resolved_variance(rho):
    model = EquicorrGaussianCopula(3)
    data = np.array([[.2, .5, .8]])
    scores = ndtri(data)
    legacy = EquicorrPreparedData(
        sum_z=scores.sum(axis=1), sum_z2=np.square(scores).sum(axis=1),
        n_obs=1, dimension=3)
    np.testing.assert_allclose(model.log_pdf_rows(legacy, rho),
                               model.log_pdf_rows(data, rho), rtol=1e-12)


def test_legacy_equicorr_stats_reject_unresolved_singular_variance():
    z = ndtri(.9)
    legacy = EquicorrPreparedData(sum_z=np.array([3. * z]),
                                 sum_z2=np.array([3. * z * z]),
                                 n_obs=1, dimension=3)
    with pytest.raises((FloatingPointError, RuntimeError)):
        EquicorrGaussianCopula(3).log_pdf_rows(legacy, np.nextafter(1., 0.))


@pytest.mark.parametrize("dimension", [3, 5, 10])
@pytest.mark.parametrize("epsilon", [1e-8, 1e-14, 1e-16])
def test_factor_quadratic_retains_common_component(dimension, epsilon):
    loadings = np.full((dimension, 1), np.sqrt(1. - epsilon))
    factor = FactorCorrelation(loadings, uniqueness_min=epsilon / 2)
    prepared = factor.prepare()
    rho = loadings[0, 0] ** 2
    expected = dimension / (1. + (dimension - 1) * rho)
    np.testing.assert_allclose(prepared.quadratic_form(np.ones(dimension)),
                               expected, rtol=1e-13)
    model = GaussianCopula(
        d=dimension, corr_mode="factor", factor_rank=1,
        factor_loadings=loadings, factor_uniqueness_min=epsilon / 2)
    z = ndtri(.9)
    expected_log = (-.5 * ((dimension - 1) * np.log1p(-rho)
                           + np.log1p((dimension - 1) * rho))
                    - .5 * z * z * (expected - dimension))
    # Static Gaussian scores use the unrefined Acklam approximation (about
    # 1e-9 relative quantile error). Along this common direction its log-density
    # error scales as (dimension-1)*z*delta_z, without singular amplification.
    # The quadratic itself above retains the much tighter algebraic tolerance.
    np.testing.assert_allclose(model.log_pdf_rows(np.full((1, dimension), .9)),
                               expected_log, rtol=0., atol=3e-9 * dimension)
