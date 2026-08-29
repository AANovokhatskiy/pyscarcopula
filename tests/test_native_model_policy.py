import numpy as np
import pytest

from pyscarcopula import (
    ClaytonCopula,
    BivariateGaussianCopula,
    FrankCopula,
    GumbelCopula,
    IndependentCopula,
    JoeCopula,
)
from pyscarcopula._native import jacobi, model_policy, scar_ou, validation
from pyscarcopula._types import NumericalConfig


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (IndependentCopula, []),
        (ClaytonCopula, [(0.0001, np.inf)]),
        (FrankCopula, [(0.0001, np.inf)]),
        (GumbelCopula, [(1.0001, np.inf)]),
        (JoeCopula, [(1.0001, np.inf)]),
        (BivariateGaussianCopula, [(-0.9999, 0.9999)]),
    ],
)
def test_pair_public_bounds_are_native_owned(factory, expected):
    assert model_policy.public_bounds(factory()) == expected


def test_latent_and_fit_policy_contracts():
    config = NumericalConfig()
    assert model_policy.latent_bounds("ou") == (
        (0.001, -np.inf, 0.001),
        (np.inf, np.inf, np.inf),
    )
    assert model_policy.latent_bounds("jacobi") == (
        (0.001, 1e-6, 0.001),
        (np.inf, 1.0 - 1e-6, np.inf),
    )
    assert model_policy.latent_bounds(
        "gas",
        gamma_bound=config.gas_gamma_bound,
        beta_bound=config.gas_beta_bound,
    ) == (
        (-np.inf, -config.gas_gamma_bound, -config.gas_beta_bound),
        (np.inf, config.gas_gamma_bound, config.gas_beta_bound),
    )
    assert model_policy.student_fit_policy(3, stochastic=False) == (
        5.0, (2.001, 10000.0))
    assert model_policy.student_fit_policy(3, stochastic=True) == (
        5.0, (2.000001, None))
    assert model_policy.equicorr_fit_policy() == (0.5, (-8.0, 8.0))


def test_native_default_initial_point_contracts():
    np.testing.assert_array_equal(
        model_policy.gas_default_initial_point(2.0),
        [0.1, 0.05, 0.95],
    )
    np.testing.assert_array_equal(
        scar_ou.default_initial_point(-0.4)[0],
        [1.0, -0.4, 1.0],
    )
    np.testing.assert_array_equal(
        jacobi.initial_point(None, 1e-6),
        [1.0, 0.5, 0.2],
    )
    np.testing.assert_array_equal(
        jacobi.initial_point(1.0, 1e-6),
        [1.0, 1.0 - 1e-6, 0.2],
    )


def test_native_optimizer_failure_policy_owns_value_and_gradient():
    value, directional = model_policy.optimizer_failure_evaluation(
        [2.0, 1.0], [1.0, 1.0], 400.0,
        directional_gradient=True)
    assert value == 400.0
    np.testing.assert_array_equal(directional, [20.0, 0.0])

    value, zero = model_policy.optimizer_failure_evaluation(
        [2.0, 1.0], [1.0, 1.0], 400.0,
        directional_gradient=False)
    assert value == 400.0
    np.testing.assert_array_equal(zero, [0.0, 0.0])

    with pytest.raises(ValueError, match="optimizer failure evaluation"):
        model_policy.optimizer_failure_evaluation(
            [1.0], [1.0], -1.0, directional_gradient=True)


def test_native_optimizer_failure_policy_owns_shape_only_gradient():
    error = FloatingPointError("native numerical failure")
    error.status = 7

    value, gradient = (
        model_policy.optimizer_numerical_failure_evaluation_for_size(
            error, 3, 123.0))

    assert value == 123.0
    np.testing.assert_array_equal(gradient, [0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, False), (1e9, True), (np.inf, True), (np.nan, True)],
)
def test_native_validation_owns_invalid_objective_threshold(value, expected):
    assert validation.objective_is_invalid(value) is expected


def test_native_generic_optimizer_scale_and_projection_policy():
    np.testing.assert_array_equal(
        model_policy.optimizer_unit_scale([-4.0, 0.25, 2.0]),
        [4.0, 1.0, 2.0],
    )
    np.testing.assert_array_equal(
        model_policy.project_optimizer_point(
            [-4.0, 0.25, 2.0],
            [-3.0, -np.inf, -1.0],
            [3.0, np.inf, 1.5],
        ),
        [-3.0, 0.25, 1.5],
    )
    with pytest.raises(ValueError, match="optimizer unit scale"):
        model_policy.optimizer_unit_scale([np.nan])
    with pytest.raises(ValueError, match="initial-point projection"):
        model_policy.project_optimizer_point([0.0], [1.0], [-1.0])


def test_native_ou_backend_basis_and_quadrature_policies():
    assert model_policy.ou_kappa_dt(2.0, 5) == 0.5
    assert model_policy.ou_auto_backend(0.01, 11, 0.01) == "local"
    assert model_policy.ou_auto_backend(1.0, 11, 0.01) == "spectral"

    expected_basis_orders = (
        (0.014999, 128),
        (0.015, 96),
        (0.025, 64),
        (0.06, 32),
    )
    for kappa_dt, expected in expected_basis_orders:
        assert model_policy.ou_adaptive_spectral_basis_order(
            kappa_dt, 2) == expected

    assert model_policy.ou_resolve_quad_order(16) == 48
    assert model_policy.ou_resolve_quad_order(32) == 80
    assert model_policy.ou_resolve_quad_order(32, 64) == 64
    with pytest.raises(ValueError, match="quadrature-order policy"):
        model_policy.ou_resolve_quad_order(64, 32)


def test_native_ou_optimizer_parameterization_and_pullbacks():
    physical = np.array([2.0, -0.5, 1.5, 0.3])
    gradient = np.array([0.7, -0.2, 1.1, -0.4])
    stationary_scale = physical[2] / np.sqrt(2.0 * physical[0])
    expected_optimizer = np.array([
        np.log(physical[0]),
        physical[1],
        np.log(stationary_scale),
        physical[3],
    ])
    optimizer = scar_ou.to_log_stationary(physical)
    np.testing.assert_allclose(
        optimizer, expected_optimizer, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(
        scar_ou.from_log_stationary(optimizer),
        physical,
        rtol=2e-15,
        atol=2e-15,
    )

    expected_pullback = gradient.copy()
    expected_pullback[0] = (
        gradient[0] * physical[0]
        + 0.5 * gradient[2] * physical[2])
    expected_pullback[2] = gradient[2] * physical[2]
    pullback = scar_ou.gradient_to_log_stationary(physical, gradient)
    np.testing.assert_allclose(
        pullback, expected_pullback, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(
        scar_ou.gradient_from_log_stationary(physical, pullback),
        gradient,
        rtol=2e-15,
        atol=2e-15,
    )


def test_native_ou_projection_scaling_and_scaled_bounds():
    values = np.array([-20.0, 2.0, 20.0, 4.0])
    lower = np.array([-3.0, -np.inf, -2.0])
    upper = np.array([3.0, np.inf, 2.0])
    np.testing.assert_array_equal(
        scar_ou.project_optimizer_block(values, lower, upper),
        [-3.0, 2.0, 2.0, 4.0],
    )
    physical = np.array([2.0, -0.5, 1.5])
    scale = np.array([2.0, 1.0, 3.0])
    scaled = scar_ou.physical_to_scaled(physical, scale)
    np.testing.assert_array_equal(scaled, [1.0, -0.5, 0.5])
    np.testing.assert_array_equal(
        scar_ou.scaled_to_physical(scaled, scale), physical)
    bounds = model_policy.ou_scaled_optimizer_bounds(scale)
    np.testing.assert_array_equal(bounds[0], [0.0005, -np.inf, 1.0 / 3000.0])
    np.testing.assert_array_equal(bounds[1], [np.inf, np.inf, np.inf])


def test_pair_constructor_propagates_missing_native_policy(monkeypatch):
    def fail(_copula):
        raise RuntimeError("native policy sentinel")

    monkeypatch.setattr(model_policy, "public_bounds", fail)
    with pytest.raises(RuntimeError, match="native policy sentinel"):
        GumbelCopula()


def test_pair_subclass_inherits_policy_without_native_compute_registration():
    class ClaytonSubclass(ClaytonCopula):
        pass

    assert ClaytonSubclass().bounds == ClaytonCopula().bounds
