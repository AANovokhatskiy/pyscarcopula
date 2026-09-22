"""OU spectral positivity failures must route value and gradient consistently."""

from dataclasses import replace

import numpy as np
import pytest

from pyscarcopula import BivariateGaussianCopula, EquicorrGaussianCopula
from pyscarcopula._native import NativeError
from pyscarcopula._native import scar_ou
from pyscarcopula.numerical._scar_ou_config import AutoTMConfig


# These fixed physical points previously returned finite but spurious spectral
# likelihoods near 865/867. Independent positive OU quadrature converged to the
# matrix values below. The increased order remains an unsafe spectral case.
_CRYPTO_POINTS = [
    pytest.param(
        (5.279269882457205, 1.5439744056953264, 4.371199306462524),
        96, 208, 793.153279687182, id="old-fit-point"),
    pytest.param(
        (7.831206838527489, 1.1039779691398892, 5.407858661120532),
        64, 144, 795.491794789468, id="dev-fit-point"),
    pytest.param(
        (7.831206838527489, 1.1039779691398892, 5.407858661120532),
        128, 288, 795.491794789468, id="dev-fit-point-double-order"),
]


def _evaluators(copula, u, config, prepared):
    if prepared:
        objective = scar_ou.prepare_objective(u, copula, config)
        return objective.neg_loglik_info, objective.neg_loglik_with_grad_info
    return (
        lambda *params: scar_ou.neg_loglik_info(
            *params, u, copula, config),
        lambda *params: scar_ou.neg_loglik_with_grad_info(
            *params, u, copula, config),
    )


@pytest.mark.data
@pytest.mark.parametrize("params,basis_order,quad_order,expected_loglik", _CRYPTO_POINTS)
@pytest.mark.parametrize("prepared", [False, True], ids=["direct", "prepared"])
def test_crypto_spectral_positivity_failure_routes_to_matrix(
        crypto_data_6d, params, basis_order, quad_order, expected_loglik, prepared):
    copula = EquicorrGaussianCopula(d=6)
    u = crypto_data_6d
    # The exactly equal ranks amplify the emission in the rho -> 1 tail.
    np.testing.assert_array_equal(u[70], np.full(6, 1.0 / 251.0))
    config = AutoTMConfig(
        transition_method="spectral", basis_order=basis_order,
        quad_order=quad_order, K=600, max_K=10000, grid_range=7.0)
    value, gradient = _evaluators(copula, u, config, prepared)
    for evaluate in (value, gradient):
        with pytest.raises(NativeError, match="numerical_failure"):
            evaluate(*params)

    auto_value, auto_gradient = _evaluators(
        copula, u, replace(config, transition_method="auto"), prepared)
    matrix_value, matrix_gradient = _evaluators(
        copula, u, replace(config, transition_method="matrix"), prepared)
    objective, info = auto_value(*params)
    gradient_objective, derivative, gradient_info = auto_gradient(*params)
    expected_objective, _ = matrix_value(*params)
    expected_gradient_objective, expected_derivative, _ = matrix_gradient(*params)

    for diagnostics in (info, gradient_info):
        assert diagnostics["backend"] == "matrix"
        assert diagnostics["selected_backend"] == "spectral"
        assert diagnostics["fallback_chain"] == ["spectral"]
    assert objective == pytest.approx(-expected_loglik, abs=2e-7)
    assert objective == pytest.approx(expected_objective, rel=0, abs=1e-10)
    assert gradient_objective == pytest.approx(
        expected_gradient_objective, rel=0, abs=1e-10)
    assert gradient_objective == pytest.approx(objective, rel=0, abs=2e-9)
    assert np.all(np.isfinite(derivative))
    np.testing.assert_allclose(derivative, expected_derivative, rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize("dimension", [2, 6])
@pytest.mark.parametrize("prepared", [False, True], ids=["direct", "prepared"])
def test_well_resolved_synthetic_spectral_remains_available(dimension, prepared):
    copula = (BivariateGaussianCopula() if dimension == 2
              else EquicorrGaussianCopula(d=dimension))
    u = np.random.default_rng(71283).uniform(0.15, 0.85, (40, dimension))
    params = np.array([8.0, 0.25, 0.35])
    config = AutoTMConfig(
        transition_method="auto", basis_order=32, quad_order=80,
        K=300, max_K=10000, grid_range=7.0)
    value, gradient = _evaluators(copula, u, config, prepared)
    objective, info = value(*params)
    gradient_objective, derivative, gradient_info = gradient(*params)
    for diagnostics in (info, gradient_info):
        assert diagnostics["backend"] == "spectral"
        assert not diagnostics.get("fallback_chain")
    assert np.isfinite(objective)
    assert gradient_objective == pytest.approx(objective, rel=0, abs=1e-10)

    matrix, _ = _evaluators(
        copula, u, replace(config, transition_method="matrix"), prepared)
    assert objective == pytest.approx(matrix(*params)[0], rel=0, abs=2e-8)
    finite_difference = []
    for index in range(3):
        plus = params.copy()
        minus = params.copy()
        step = 1e-5 * max(1.0, abs(params[index]))
        plus[index] += step
        minus[index] -= step
        finite_difference.append((value(*plus)[0] - value(*minus)[0]) / (2 * step))
    np.testing.assert_allclose(derivative, finite_difference, rtol=3e-5, atol=2e-7)
