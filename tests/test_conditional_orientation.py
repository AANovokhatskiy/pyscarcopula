"""Conditional directions must retain the fitted rotation and variable order."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import roots_legendre
from scipy.stats import norm

from pyscarcopula import (
    BivariateGaussianCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    IndependentCopula,
    JoeCopula,
    api,
)
from pyscarcopula._types import (
    GASResult,
    IndependentResult,
    LatentResult,
    MLEResult,
    gas_params,
    jacobi_params,
    ou_params,
)
from pyscarcopula.numerical import (
    tm_forward_mixture_h,
    tm_forward_rosenblatt,
)
from pyscarcopula.stattests import (
    _bivariate_rosenblatt_from_result,
    rosenblatt_transform_mle,
    rvine_rosenblatt_transform,
)
from pyscarcopula.strategy._base import get_strategy_for_result
from pyscarcopula.vine._pair_copula import PairCopula
from pyscarcopula.vine._rvine_edges import (
    _edge_h,
    _edge_h_inverse,
    _edge_h_inverse_for_variables,
)
from rvine_runtime_cases import configured_static_dvine


OBSERVATIONS = np.array([
    [0.25, 0.63], [0.41, 0.72], [0.66, 0.37],
    [0.18, 0.81], [0.73, 0.22], [0.58, 0.47],
])
FAMILIES = [
    (ClaytonCopula, 1.4), (GumbelCopula, 1.8), (JoeCopula, 1.8),
    (FrankCopula, 2.3), (BivariateGaussianCopula, 0.4),
    (IndependentCopula, 0.0),
]
ROTATABLE = {ClaytonCopula, GumbelCopula, JoeCopula}
METHODS = ['MLE', 'GAS', 'SCAR-TM-OU', 'SCAR-TM-JACOBI']
OU_PARAMETERS = (1.7, 0.35, 0.65)
JACOBI_PARAMETERS = (1.8, 0.43, 0.3)
GRID_OPTIONS = dict(
    K=201, grid_range=6.0, adaptive=False, grid_method='dense', gh_order=17,
)
CASES = [
    pytest.param((family, parameter, rotation, method),
                 id=f'{family.__name__}-{rotation}-{method}')
    for family, parameter in FAMILIES
    for rotation in ([0, 90, 180, 270] if family in ROTATABLE else [0])
    for method in (['MLE'] if family is IndependentCopula else METHODS)
]


def _result(copula, parameter, method):
    common = dict(
        method=method, copula_name=copula.name, log_likelihood=0.0,
        success=True,
    )
    if method == 'MLE':
        if isinstance(copula, IndependentCopula):
            return IndependentResult(**common)
        return MLEResult(copula_param=parameter, **common)
    if method == 'GAS':
        return GASResult(
            params=gas_params(0.07, 0.08, 0.65), scaling='unit',
            score_eps=1e-4, **common)
    if method == 'SCAR-TM-OU':
        return LatentResult(
            params=ou_params(*OU_PARAMETERS), transition_method='matrix',
            max_K=None, **GRID_OPTIONS, **common)
    return LatentResult(
        params=jacobi_params(*JACOBI_PARAMETERS),
        spectral_basis_order=16, spectral_quad_order=48,
        transition_method='local_fixed', theta_cap=10.0, **common)


def _ou_reference_filter(copula):
    """Independent normal-transition OU filter on Gauss-Legendre nodes.

    Production uses a uniform grid. This reference shares its density/link,
    but neither its h-functions nor its filtering implementation.
    """
    kappa, mu, nu = OU_PARAMETERS
    sigma = nu / np.sqrt(2.0 * kappa)
    nodes, weights = roots_legendre(401)
    nodes = mu + 7.0 * sigma * nodes
    weights = 7.0 * sigma * weights
    probability = norm.pdf(nodes, loc=mu, scale=sigma) * weights
    probability /= probability.sum()
    rho = np.exp(-kappa / (len(OBSERVATIONS) - 1))
    transition = norm.pdf(
        nodes[None, :], loc=(mu + rho * (nodes - mu))[:, None],
        scale=sigma * np.sqrt(1.0 - rho * rho)) * weights[None, :]
    theta = copula.transform(nodes)
    predicted = []
    for u1, u2 in OBSERVATIONS:
        predicted.append(probability.copy())
        posterior = probability * copula.pdf(u1, u2, theta)
        posterior /= posterior.sum()
        probability = posterior @ transition
        probability /= probability.sum()
    return np.broadcast_to(theta, (len(OBSERVATIONS), len(theta))), np.array(predicted)


def _density_integrals(copula, theta, weights):
    """Return second|first, first|second using density quadrature only."""
    expected = []
    for (u1, u2), parameters, probability in zip(
            OBSERVATIONS, theta, weights):
        second_given_first = quad(
            lambda value: float(np.dot(
                probability, copula.pdf(u1, value, parameters))),
            0.0, u2, epsabs=2e-10, epsrel=2e-10, limit=150)[0]
        first_given_second = quad(
            lambda value: float(np.dot(
                probability, copula.pdf(value, u2, parameters))),
            0.0, u1, epsabs=2e-10, epsrel=2e-10, limit=150)[0]
        expected.append((second_given_first, first_given_second))
    return np.array(expected)


@pytest.fixture(scope='module', params=CASES)
def conditional_case(request):
    family, parameter, rotation, method = request.param
    copula = family(rotate=rotation)
    result = _result(copula, parameter, method)
    strategy = get_strategy_for_result(result)
    if method == 'MLE':
        theta = np.full((len(OBSERVATIONS), 1), parameter)
        weights = np.ones_like(theta)
    elif method == 'GAS':
        # Test the directed conditional at the actual score-driven path;
        # independent correctness of the score recursion is a separate test.
        theta = strategy.predictive_mean(copula, OBSERVATIONS, result)[:, None]
        weights = np.ones_like(theta)
    elif method == 'SCAR-TM-OU':
        theta, weights = _ou_reference_filter(copula)
    else:
        filtered = strategy._prepared_evaluator(
            OBSERVATIONS, copula).filter(*JACOBI_PARAMETERS)
        weights = filtered['predicted']
        theta = np.broadcast_to(filtered['theta'], weights.shape)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=3e-12)
    expected = _density_integrals(copula, theta, weights)
    return copula, result, strategy, expected


def test_bivariate_conditional_routes_match_density_integrals(conditional_case):
    copula, result, strategy, expected = conditional_case
    pair = np.column_stack(strategy.mixture_h_pair(
        copula, OBSERVATIONS, result))
    np.testing.assert_allclose(pair, expected, rtol=0.0, atol=2e-8)
    scalar = api.mixture_h(copula, OBSERVATIONS, result)
    residuals = _bivariate_rosenblatt_from_result(
        copula, OBSERVATIONS, result, K=201, grid_range=6.0)
    np.testing.assert_allclose(scalar, expected[:, 0], rtol=0.0, atol=2e-8)
    np.testing.assert_array_equal(residuals[:, 0], OBSERVATIONS[:, 0])
    np.testing.assert_allclose(
        residuals[:, 1], expected[:, 0], rtol=0.0, atol=2e-8)
    edge = PairCopula(copula=copula, fit_result=result)
    np.testing.assert_allclose(
        edge.h(OBSERVATIONS[:, 1], OBSERVATIONS[:, 0]),
        expected[:, 0], rtol=0.0, atol=2e-8)
    if result.method == 'MLE':
        direct = rosenblatt_transform_mle(
            copula, OBSERVATIONS, result.copula_param)
        np.testing.assert_allclose(
            direct[:, 1], expected[:, 0], rtol=0.0, atol=2e-8)


def test_two_dimensional_vine_residuals_preserve_pair_orientation(conditional_case):
    copula, result, _, expected = conditional_case
    vine = configured_static_dvine(2, order=(0, 1))
    key = next(iter(vine.pair_copulas))
    vine.pair_copulas[key] = PairCopula(
        copula=copula, fit_result=result,
        param=result.copula_param if result.method == 'MLE' else None)
    vine.method = result.method
    actual = rvine_rosenblatt_transform(
        vine, OBSERVATIONS, K=201, grid_range=6.0)
    np.testing.assert_array_equal(actual[:, 0], OBSERVATIONS[:, 0])
    np.testing.assert_allclose(
        actual[:, 1], expected[:, 0], rtol=0.0, atol=2e-8)
    nodes = vine._compute_pseudo_obs(OBSERVATIONS)
    np.testing.assert_allclose(
        nodes[(1, frozenset({0}))], expected[:, 0], rtol=0.0, atol=2e-8)
    np.testing.assert_allclose(
        nodes[(0, frozenset({1}))], expected[:, 1], rtol=0.0, atol=2e-8)


@pytest.mark.parametrize('family', [ClaytonCopula, GumbelCopula, JoeCopula])
@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
@pytest.mark.parametrize('method,r_gh,forced', [
    ('matrix', 3.0, None), ('local', 3.0, None),
    ('auto', 0.01, 'matrix'), ('auto', 1000.0, 'local'),
    ('spectral', 3.0, 'matrix'),
])
def test_ou_forward_rosenblatt_uses_directed_mixture(
        family, rotation, method, r_gh, forced):
    copula = family(rotate=rotation)
    theta, weights = _ou_reference_filter(copula)
    expected = _density_integrals(copula, theta, weights)[:, 0]
    options = dict(GRID_OPTIONS, transition_method=method, r_gh=r_gh)
    actual = tm_forward_rosenblatt(
        *OU_PARAMETERS, OBSERVATIONS, copula, **options)
    paired_route = tm_forward_mixture_h(
        *OU_PARAMETERS, OBSERVATIONS, copula, **options)
    np.testing.assert_array_equal(actual[:, 0], OBSERVATIONS[:, 0])
    # The first row uses the stationary prior, independent of the selected
    # transition approximation. Local GH has small later discretization error.
    np.testing.assert_allclose(actual[0, 1], expected[0], rtol=0.0, atol=2e-8)
    tolerance = 1e-5 if method == 'local' or forced == 'local' else 2e-8
    np.testing.assert_allclose(actual[:, 1], expected, rtol=0.0, atol=tolerance)
    np.testing.assert_allclose(actual[:, 1], paired_route, rtol=0.0, atol=2e-12)
    if forced is not None:
        forced_values = tm_forward_rosenblatt(
            *OU_PARAMETERS, OBSERVATIONS, copula,
            **dict(options, transition_method=forced))
        np.testing.assert_array_equal(actual, forced_values)


@pytest.mark.parametrize('family,parameter', FAMILIES[:3])
@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
@pytest.mark.parametrize('mode', ['fitted', 'explicit', 'unfitted'])
def test_scalar_edge_parameter_routes_and_inverse_keep_canonical_direction(
        family, parameter, rotation, mode):
    copula = family(rotate=rotation)
    result = _result(copula, parameter, 'MLE') if mode != 'unfitted' else None
    edge = PairCopula(copula=copula, param=parameter, fit_result=result)
    config = {'r': np.full(len(OBSERVATIONS), parameter)} if mode == 'explicit' else None
    theta = np.full((len(OBSERVATIONS), 1), parameter)
    expected = _density_integrals(copula, theta, np.ones_like(theta))
    actual = _edge_h(
        edge, OBSERVATIONS[:, 1], OBSERVATIONS[:, 0], config=config)
    np.testing.assert_allclose(actual, expected[:, 0], rtol=0.0, atol=2e-8)
    restored = _edge_h_inverse(
        edge, expected[:, 0], OBSERVATIONS[:, 0], config=config)
    np.testing.assert_allclose(restored, OBSERVATIONS[:, 1], rtol=0.0, atol=1e-5)
    # Explicit orientation from the variable-aware helper must remain intact;
    # applying the default transpose again would break either direction.
    for target, given, column in [(1, 0, 0), (0, 1, 1)]:
        restored = _edge_h_inverse_for_variables(
            edge, target, expected[:, column], given,
            OBSERVATIONS[:, given], config=config)
        np.testing.assert_allclose(
            restored, OBSERVATIONS[:, target], rtol=0.0, atol=1e-5)
