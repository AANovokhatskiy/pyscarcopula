import numpy as np
import pytest

from pyscarcopula import GumbelCopula
from pyscarcopula.numerical.jacobi_tm import (
    jacobi_forward_mixture_h,
    jacobi_forward_predictive_mean,
    jacobi_local_transition_matrix,
    jacobi_matrix_neg_loglik,
    jacobi_matrix_neg_loglik_with_grad,
    jacobi_loglik,
    jacobi_matrix_forward_mixture_h,
    jacobi_matrix_forward_predictive_mean,
    jacobi_matrix_loglik,
    jacobi_matrix_state_distribution,
    jacobi_rule,
    jacobi_spectral_transition_matrix,
    jacobi_state_distribution,
    jacobi_transition_matrix,
)


class UnitEmissionCopula:
    def tau_to_param(self, tau):
        return np.asarray(tau, dtype=np.float64)

    def pdf(self, u1, u2, r):
        return np.ones_like(np.asarray(r, dtype=np.float64))


def test_jacobi_rule_returns_orthonormal_basis():
    tau, weights, basis = jacobi_rule(
        alpha=2.5,
        beta=3.5,
        quad_order=24,
        basis_order=8,
    )

    assert np.all((tau > 0.0) & (tau < 1.0))
    np.testing.assert_allclose(np.sum(weights), 1.0, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(
        basis.T @ (weights[:, np.newaxis] * basis),
        np.eye(8),
        rtol=1e-11,
        atol=1e-11,
    )


def test_jacobi_spectral_transition_matrix_is_row_stochastic():
    tau, weights, transition, diagnostics = jacobi_spectral_transition_matrix(
        kappa=1.5,
        m=0.4,
        xi=0.35,
        dt=0.25,
        basis_order=8,
        quad_order=32,
        return_diagnostics=True,
    )

    assert transition.shape == (len(tau), len(tau))
    np.testing.assert_allclose(
        np.sum(transition, axis=1), 1.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        weights @ transition, weights, rtol=1e-10, atol=1e-10)
    assert diagnostics["stationary_error"] < 1e-10


def test_jacobi_spectral_transition_order_one_is_stationary_kernel():
    _, weights, transition = jacobi_spectral_transition_matrix(
        kappa=1.5,
        m=0.4,
        xi=0.35,
        dt=0.25,
        basis_order=1,
        quad_order=24,
    )

    expected = np.tile(weights, (len(weights), 1))
    np.testing.assert_allclose(transition, expected, rtol=1e-12, atol=1e-12)


def test_jacobi_local_transition_matrix_is_nonnegative_and_stochastic():
    tau, weights, transition, diagnostics = jacobi_local_transition_matrix(
        kappa=1.5,
        m=0.4,
        xi=0.35,
        dt=1e-3,
        quad_order=40,
        gh_order=5,
        return_diagnostics=True,
    )

    assert transition.shape == (len(tau), len(tau))
    assert diagnostics["min_entry"] >= 0.0
    assert diagnostics["max_row_sum_error"] < 1e-14
    assert np.all(transition >= 0.0)
    np.testing.assert_allclose(
        np.sum(transition, axis=1), 1.0, rtol=1e-12, atol=1e-12)
    assert weights.shape == tau.shape


def test_jacobi_local_transition_is_local_for_small_dt():
    tau, _, transition = jacobi_local_transition_matrix(
        kappa=1.5,
        m=0.4,
        xi=0.35,
        dt=1e-6,
        quad_order=40,
        gh_order=5,
    )

    expected_tau_next = transition @ tau
    assert np.max(np.abs(expected_tau_next - tau)) < 1e-3


def test_jacobi_transition_matrix_auto_falls_back_on_truncated_negativity():
    _, _, _, diagnostics = jacobi_transition_matrix(
        kappa=1.5,
        m=0.4,
        xi=0.35,
        dt=1e-6,
        basis_order=6,
        quad_order=40,
        transition_method="auto",
        return_diagnostics=True,
    )

    assert diagnostics["transition_method_requested"] == "auto"
    assert diagnostics["transition_method"] == "local"


def test_jacobi_transition_matrix_auto_falls_back_on_spectral_exception():
    kwargs = {
        "kappa": 1.0,
        "m": 0.6468628643818045,
        "xi": 0.2,
        "n_obs": 1460,
        "basis_order": 100,
        "quad_order": 216,
    }

    with pytest.raises(FloatingPointError):
        jacobi_transition_matrix(
            **kwargs,
            transition_method="spectral_matrix",
            return_diagnostics=True,
        )

    _, _, _, diagnostics = jacobi_transition_matrix(
        **kwargs,
        transition_method="auto",
        return_diagnostics=True,
    )

    assert diagnostics["transition_method_requested"] == "auto"
    assert diagnostics["transition_method"] == "local"
    assert "FloatingPointError" in diagnostics["spectral_error"]


def test_jacobi_transition_matrix_respects_soft_negative_mass_tol():
    kwargs = {
        "kappa": 1.0,
        "m": 0.8683,
        "xi": 7.9,
        "n_obs": 1460,
        "basis_order": 32,
        "quad_order": 80,
    }

    _, _, _, strict_diagnostics = jacobi_transition_matrix(
        **kwargs,
        transition_method="auto",
        negative_mass_tol=1e-10,
        return_diagnostics=True,
    )
    _, _, _, soft_diagnostics = jacobi_transition_matrix(
        **kwargs,
        transition_method="auto",
        negative_mass_tol=1e-5,
        return_diagnostics=True,
    )

    assert strict_diagnostics["transition_method"] == "local"
    assert soft_diagnostics["transition_method"] == "spectral_matrix"
    assert 1e-10 < soft_diagnostics["raw_negative_mass"] < 1e-5


def test_jacobi_loglik_unit_emission_is_zero():
    u = np.array([[0.2, 0.3], [0.4, 0.7], [0.8, 0.6]])

    ll = jacobi_loglik(
        kappa=1.5,
        m=0.4,
        xi=0.35,
        u=u,
        copula=UnitEmissionCopula(),
        basis_order=6,
        quad_order=32,
    )

    np.testing.assert_allclose(ll, 0.0, rtol=0.0, atol=1e-12)


def test_jacobi_matrix_loglik_unit_emission_is_zero():
    u = np.array([[0.2, 0.3], [0.4, 0.7], [0.8, 0.6]])

    ll = jacobi_matrix_loglik(
        kappa=1.5,
        m=0.4,
        xi=0.35,
        u=u,
        copula=UnitEmissionCopula(),
        basis_order=6,
        quad_order=32,
        transition_method="local",
    )

    np.testing.assert_allclose(ll, 0.0, rtol=0.0, atol=1e-12)


def test_jacobi_fixed_grid_gradient_matches_finite_difference():
    u = np.array([
        [0.18, 0.31],
        [0.34, 0.42],
        [0.58, 0.66],
        [0.76, 0.81],
    ], dtype=np.float64)
    copula = GumbelCopula()
    alpha = np.array([1.2, 0.42, 0.7], dtype=np.float64)
    kwargs = {
        "basis_order": 3,
        "quad_order": 24,
        "transition_method": "local_fixed",
        "gh_order": 5,
    }

    value, grad = jacobi_matrix_neg_loglik_with_grad(
        *alpha, u, copula, **kwargs)

    eps = 1e-6
    grad_num = np.empty(3, dtype=np.float64)
    for p in range(3):
        plus = alpha.copy()
        minus = alpha.copy()
        plus[p] += eps
        minus[p] -= eps
        grad_num[p] = (
            jacobi_matrix_neg_loglik(*plus, u, copula, **kwargs)
            - jacobi_matrix_neg_loglik(*minus, u, copula, **kwargs)
        ) / (2.0 * eps)

    assert np.isfinite(value)
    np.testing.assert_allclose(grad, grad_num, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize(
    "transition_method",
    ["local", "spectral_matrix"],
)
def test_jacobi_moving_grid_gradient_matches_finite_difference(
        transition_method):
    u = np.array([
        [0.18, 0.31],
        [0.34, 0.42],
        [0.58, 0.66],
        [0.76, 0.81],
    ], dtype=np.float64)
    copula = GumbelCopula()
    alpha = np.array([1.2, 0.42, 0.7], dtype=np.float64)
    kwargs = {
        "basis_order": 3,
        "quad_order": 24,
        "transition_method": transition_method,
    }

    value, grad = jacobi_matrix_neg_loglik_with_grad(
        *alpha, u, copula, **kwargs)

    eps = 1e-5
    grad_num = np.empty(3, dtype=np.float64)
    for p in range(3):
        plus = alpha.copy()
        minus = alpha.copy()
        plus[p] += eps
        minus[p] -= eps
        grad_num[p] = (
            jacobi_matrix_neg_loglik(*plus, u, copula, **kwargs)
            - jacobi_matrix_neg_loglik(*minus, u, copula, **kwargs)
        ) / (2.0 * eps)

    assert np.isfinite(value)
    np.testing.assert_allclose(grad, grad_num, rtol=2e-3, atol=2e-4)


def test_jacobi_basis_order_one_matches_stationary_mixture():
    u = np.array([[0.2, 0.3], [0.4, 0.7], [0.8, 0.6]])
    copula = GumbelCopula()
    kappa = 1.2
    m = 0.45
    xi = 0.4
    quad_order = 40

    ll = jacobi_loglik(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=1,
        quad_order=quad_order,
    )

    alpha = 2.0 * kappa * m / (xi * xi)
    beta = 2.0 * kappa * (1.0 - m) / (xi * xi)
    tau, weights, _ = jacobi_rule(alpha, beta, quad_order, basis_order=1)
    theta = copula.tau_to_param(tau)
    expected = 0.0
    for row in u:
        expected += np.log(np.sum(
            weights
            * copula.pdf(
                np.full(len(tau), row[0]),
                np.full(len(tau), row[1]),
                theta,
            )
        ))

    np.testing.assert_allclose(ll, expected, rtol=1e-12, atol=1e-12)


def test_jacobi_matrix_spectral_order_one_matches_stationary_mixture():
    u = np.array([[0.2, 0.3], [0.4, 0.7], [0.8, 0.6]])
    copula = GumbelCopula()
    kappa = 1.2
    m = 0.45
    xi = 0.4
    quad_order = 40

    ll = jacobi_matrix_loglik(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=1,
        quad_order=quad_order,
        transition_method="spectral_matrix",
    )

    alpha = 2.0 * kappa * m / (xi * xi)
    beta = 2.0 * kappa * (1.0 - m) / (xi * xi)
    tau, weights, _ = jacobi_rule(alpha, beta, quad_order, basis_order=1)
    theta = copula.tau_to_param(tau)
    expected = 0.0
    for row in u:
        expected += np.log(np.sum(
            weights
            * copula.pdf(
                np.full(len(tau), row[0]),
                np.full(len(tau), row[1]),
                theta,
            )
        ))

    np.testing.assert_allclose(ll, expected, rtol=1e-12, atol=1e-12)


def test_jacobi_forward_outputs_are_in_valid_ranges():
    u = np.array([[0.2, 0.3], [0.4, 0.7], [0.8, 0.6]])
    copula = GumbelCopula()

    mean = jacobi_forward_predictive_mean(
        1.5, 0.4, 0.35, u, copula, basis_order=6, quad_order=32)
    h_mix = jacobi_forward_mixture_h(
        1.5, 0.4, 0.35, u, copula, basis_order=6, quad_order=32)
    tau, prob = jacobi_state_distribution(
        1.5, 0.4, 0.35, u, copula, basis_order=6, quad_order=32)

    assert mean.shape == (len(u),)
    assert np.all(mean >= 1.0)
    assert h_mix.shape == (len(u),)
    assert np.all((h_mix > 0.0) & (h_mix < 1.0))
    assert tau.shape == prob.shape
    assert np.all((tau > 0.0) & (tau < 1.0))
    np.testing.assert_allclose(np.sum(prob), 1.0, rtol=1e-12, atol=1e-12)


def test_jacobi_matrix_forward_outputs_are_in_valid_ranges():
    u = np.array([[0.2, 0.3], [0.4, 0.7], [0.8, 0.6]])
    copula = GumbelCopula()

    mean = jacobi_matrix_forward_predictive_mean(
        1.5, 0.4, 0.35, u, copula,
        basis_order=6,
        quad_order=32,
        transition_method="local",
    )
    h_mix = jacobi_matrix_forward_mixture_h(
        1.5, 0.4, 0.35, u, copula,
        basis_order=6,
        quad_order=32,
        transition_method="local",
    )
    tau, prob = jacobi_matrix_state_distribution(
        1.5, 0.4, 0.35, u, copula,
        basis_order=6,
        quad_order=32,
        transition_method="local",
    )

    assert mean.shape == (len(u),)
    assert np.all(mean >= 1.0)
    assert h_mix.shape == (len(u),)
    assert np.all((h_mix > 0.0) & (h_mix < 1.0))
    assert tau.shape == prob.shape
    assert np.all((tau > 0.0) & (tau < 1.0))
    np.testing.assert_allclose(np.sum(prob), 1.0, rtol=1e-12, atol=1e-12)
