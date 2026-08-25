import numpy as np
import pytest

from pyscarcopula import GumbelCopula
from pyscarcopula.numerical import jacobi_tm
from pyscarcopula.numerical.jacobi_tm import (
    DEFAULT_JACOBI_MEMORY_BUDGET_BYTES,
    MAX_JACOBI_ORDER,
    _validate_jacobi_workspace,
    jacobi_forward_mixture_h,
    jacobi_forward_mixture_h_pair,
    jacobi_forward_predictive_mean,
    jacobi_local_transition_matrix,
    jacobi_matrix_neg_loglik,
    jacobi_matrix_neg_loglik_with_grad,
    jacobi_loglik,
    jacobi_matrix_forward_mixture_h,
    jacobi_matrix_forward_mixture_h_pair,
    jacobi_matrix_forward_predictive_mean,
    jacobi_matrix_loglik,
    jacobi_matrix_state_distribution,
    jacobi_rule,
    jacobi_spectral_transition_matrix,
    jacobi_state_distribution,
    jacobi_transition_matrix,
    sample_jacobi_grid_trajectory,
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


def test_jacobi_rule_rejects_hard_cap_before_native_rule(monkeypatch):
    monkeypatch.setattr(
        jacobi_tm.jacobi_native,
        "jacobi_rule",
        lambda *args, **kwargs: pytest.fail("native rule must not be called"),
    )
    with pytest.raises(ValueError, match=f"<= {MAX_JACOBI_ORDER}"):
        jacobi_rule(
            alpha=2.5,
            beta=3.5,
            quad_order=MAX_JACOBI_ORDER + 1,
            basis_order=8,
        )


def test_jacobi_rule_checks_memory_budget_before_native_rule(monkeypatch):
    monkeypatch.setattr(
        jacobi_tm.jacobi_native,
        "jacobi_rule",
        lambda *args, **kwargs: pytest.fail("native rule must not be called"),
    )
    with pytest.raises(MemoryError, match="memory_budget_bytes"):
        jacobi_rule(
            alpha=2.5,
            beta=3.5,
            quad_order=24,
            basis_order=8,
            memory_budget_bytes=1,
        )


def test_jacobi_default_budget_rejects_multi_gigabyte_workspace():
    n_obs = DEFAULT_JACOBI_MEMORY_BUDGET_BYTES // (5 * 64 * 8) + 1
    with pytest.raises(MemoryError, match="memory_budget_bytes"):
        _validate_jacobi_workspace(
            quad_order=64,
            basis_order=16,
            n_obs=n_obs,
            matrix=True,
        )


@pytest.mark.parametrize("n", [True, 1.5])
def test_jacobi_grid_sampler_rejects_non_integer_n(n):
    with pytest.raises(TypeError, match="n"):
        sample_jacobi_grid_trajectory(1.2, 0.4, 0.25, n)


def test_jacobi_grid_sampler_rejects_negative_n():
    with pytest.raises(ValueError, match="non-negative"):
        sample_jacobi_grid_trajectory(1.2, 0.4, 0.25, -1)


def test_jacobi_grid_sampler_zero_does_not_advance_rng():
    rng = np.random.default_rng(20260728)
    path = sample_jacobi_grid_trajectory(
        1.2, 0.4, 0.25, 0, rng=rng)

    assert path.shape == (0,)
    np.testing.assert_array_equal(
        rng.random(8),
        np.random.default_rng(20260728).random(8),
    )


def test_jacobi_grid_sampler_one_draw_is_stationary_grid_atom():
    path, diagnostics = sample_jacobi_grid_trajectory(
        1.2,
        0.4,
        0.25,
        1,
        rng=np.random.default_rng(11),
        basis_order=4,
        quad_order=24,
        transition_method="local_fixed",
        return_diagnostics=True,
    )
    tau, _, _ = jacobi_transition_matrix(
        1.2,
        0.4,
        0.25,
        n_obs=1,
        basis_order=4,
        quad_order=24,
        transition_method="local_fixed",
    )

    assert path.shape == (1,)
    assert path[0] in tau
    assert diagnostics["transition_method"] == "stationary_only"


def test_jacobi_grid_sampler_stationary_mean_and_variance():
    kappa, m, xi = 1.2, 0.4, 0.25
    rng = np.random.default_rng(111)
    draws = np.array([
        sample_jacobi_grid_trajectory(
            kappa,
            m,
            xi,
            1,
            rng=rng,
            basis_order=1,
            quad_order=32,
            transition_method="local_fixed",
        )[0]
        for _ in range(5_000)
    ])
    shape_sum = 2.0 * kappa / (xi * xi)
    expected_variance = m * (1.0 - m) / (shape_sum + 1.0)

    assert np.mean(draws) == pytest.approx(m, abs=0.005)
    assert np.var(draws) == pytest.approx(expected_variance, abs=0.001)


def test_jacobi_grid_sampler_is_seed_reproducible_and_uses_grid_atoms():
    kwargs = {
        "basis_order": 6,
        "quad_order": 32,
        "transition_method": "local_fixed",
    }
    first = sample_jacobi_grid_trajectory(
        1.2, 0.4, 0.25, 200, rng=np.random.default_rng(12), **kwargs)
    second = sample_jacobi_grid_trajectory(
        1.2, 0.4, 0.25, 200, rng=np.random.default_rng(12), **kwargs)
    different = sample_jacobi_grid_trajectory(
        1.2, 0.4, 0.25, 200, rng=np.random.default_rng(13), **kwargs)
    tau, _, _ = jacobi_transition_matrix(
        1.2, 0.4, 0.25, n_obs=200, **kwargs)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)
    assert np.all(np.isin(first, tau))
    assert np.all((first > 0.0) & (first < 1.0))


def test_jacobi_grid_sampler_matches_row_probabilities(monkeypatch):
    tau = np.array([0.2, 0.8], dtype=np.float64)
    stationary = np.array([1.0, 0.0], dtype=np.float64)
    target = np.array([0.25, 0.75], dtype=np.float64)
    transition = np.vstack([target, target])

    def fixed_transition(*args, **kwargs):
        diagnostics = {"transition_method": "local_fixed"}
        return tau.copy(), stationary.copy(), transition.copy(), diagnostics

    monkeypatch.setattr(
        jacobi_tm, "jacobi_transition_matrix", fixed_transition)
    path = sample_jacobi_grid_trajectory(
        1.2,
        0.4,
        0.25,
        40_000,
        rng=np.random.default_rng(14),
        basis_order=1,
        quad_order=2,
        transition_method="local_fixed",
    )
    observed = np.array([
        np.mean(path[1:] == tau[0]),
        np.mean(path[1:] == tau[1]),
    ])

    np.testing.assert_allclose(observed, target, atol=0.01)


def test_jacobi_grid_sampler_spectral_coeff_uses_safe_auto_matrix():
    path, diagnostics = sample_jacobi_grid_trajectory(
        1.2,
        0.4,
        0.25,
        12,
        rng=np.random.default_rng(15),
        basis_order=4,
        quad_order=24,
        transition_method="spectral_coeff",
        return_diagnostics=True,
    )

    assert path.shape == (12,)
    assert diagnostics["model_transition_method_requested"] == "spectral_coeff"
    assert diagnostics["sampling_transition_method_requested"] == "auto"
    assert diagnostics["transition_method"] in {"local", "spectral_matrix"}


def test_jacobi_grid_sampler_memory_guard_precedes_rng_draw():
    rng = np.random.default_rng(16)
    with pytest.raises(MemoryError, match="memory_budget_bytes"):
        sample_jacobi_grid_trajectory(
            1.2,
            0.4,
            0.25,
            10,
            rng=rng,
            basis_order=4,
            quad_order=24,
            memory_budget_bytes=1,
        )
    np.testing.assert_array_equal(
        rng.random(8),
        np.random.default_rng(16).random(8),
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


def test_jacobi_spectral_transition_preserves_conditional_first_moment():
    kappa, m, dt = 1.2, 0.4, 1.0
    tau, _, transition = jacobi_transition_matrix(
        kappa=kappa,
        m=m,
        xi=0.25,
        dt=dt,
        basis_order=24,
        quad_order=64,
        transition_method="spectral_matrix",
    )
    expected = m + (tau - m) * np.exp(-kappa * dt)

    np.testing.assert_allclose(
        transition @ tau, expected, rtol=1e-12, atol=1e-12)


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
    _, _, soft_transition, soft_diagnostics = jacobi_transition_matrix(
        **kwargs,
        transition_method="auto",
        negative_mass_tol=1e-5,
        return_diagnostics=True,
    )

    assert strict_diagnostics["transition_method"] == "local"
    assert soft_diagnostics["transition_method"] == "spectral_matrix"
    assert 1e-10 < soft_diagnostics["raw_negative_mass"] < 1e-5
    assert soft_diagnostics["probability_cleanup_applied"]
    assert np.all(soft_transition >= 0.0)
    np.testing.assert_allclose(
        np.sum(soft_transition, axis=1), 1.0, rtol=1e-12, atol=1e-12)


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
    assert value == pytest.approx(
        jacobi_matrix_neg_loglik(*alpha, u, copula, **kwargs),
        rel=1e-12,
        abs=1e-12,
    )

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
    assert value == pytest.approx(
        jacobi_matrix_neg_loglik(*alpha, u, copula, **kwargs),
        rel=1e-12,
        abs=1e-12,
    )

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


def test_jacobi_spectral_gradient_rejects_signed_transition_like_value_path():
    u = np.array([
        [0.18, 0.31],
        [0.34, 0.42],
        [0.58, 0.66],
        [0.76, 0.81],
        [0.22, 0.89],
    ], dtype=np.float64)
    kwargs = {
        "basis_order": 8,
        "quad_order": 48,
        "transition_method": "spectral_matrix",
        "clip_negative": False,
    }

    value = jacobi_matrix_neg_loglik(
        0.08, 0.15, 0.3, u, GumbelCopula(), **kwargs)
    gradient_value, gradient = jacobi_matrix_neg_loglik_with_grad(
        0.08, 0.15, 0.3, u, GumbelCopula(), **kwargs)

    assert value == pytest.approx(1e10)
    assert gradient_value == pytest.approx(value)
    np.testing.assert_array_equal(gradient, np.zeros(3))


def test_jacobi_explicit_spectral_rejects_material_negative_mass():
    with pytest.raises(
            FloatingPointError, match="negative probability mass"):
        jacobi_transition_matrix(
            0.08,
            0.15,
            0.3,
            n_obs=5,
            basis_order=8,
            quad_order=48,
            transition_method="spectral_matrix",
            clip_negative=False,
        )


def test_jacobi_auto_gradient_freezes_selected_backend(monkeypatch):
    from pyscarcopula.numerical import jacobi_tm

    original = jacobi_tm.jacobi_transition_matrix
    requested_methods = []

    def recorded(*args, **kwargs):
        requested_methods.append(kwargs.get("transition_method"))
        return original(*args, **kwargs)

    monkeypatch.setattr(jacobi_tm, "jacobi_transition_matrix", recorded)
    u = np.array([
        [0.18, 0.31],
        [0.34, 0.42],
        [0.58, 0.66],
        [0.76, 0.81],
        [0.22, 0.89],
    ], dtype=np.float64)

    value, gradient = jacobi_matrix_neg_loglik_with_grad(
        0.08,
        0.15,
        0.3,
        u,
        GumbelCopula(),
        basis_order=8,
        quad_order=48,
        transition_method="auto",
    )

    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))
    assert requested_methods[0] == "auto"
    assert set(requested_methods[1:]) == {"local"}


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


def test_jacobi_mixture_h_pair_matches_directional_calls():
    u = np.array([[0.2, 0.3], [0.4, 0.7], [0.8, 0.6]])
    copula = GumbelCopula()
    kwargs = dict(basis_order=6, quad_order=32)

    first, second = jacobi_forward_mixture_h_pair(
        1.5, 0.4, 0.35, u, copula, **kwargs)
    np.testing.assert_allclose(
        first,
        jacobi_forward_mixture_h(1.5, 0.4, 0.35, u, copula, **kwargs),
    )
    np.testing.assert_allclose(
        second,
        jacobi_forward_mixture_h(
            1.5, 0.4, 0.35, u[:, ::-1], copula, **kwargs),
    )

    first_matrix, second_matrix = jacobi_matrix_forward_mixture_h_pair(
        1.5, 0.4, 0.35, u, copula,
        transition_method="local", **kwargs)
    np.testing.assert_allclose(
        first_matrix,
        jacobi_matrix_forward_mixture_h(
            1.5, 0.4, 0.35, u, copula,
            transition_method="local", **kwargs),
    )
    np.testing.assert_allclose(
        second_matrix,
        jacobi_matrix_forward_mixture_h(
            1.5, 0.4, 0.35, u[:, ::-1], copula,
            transition_method="local", **kwargs),
    )


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
