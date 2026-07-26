"""Numerical kernel regression tests."""
import numpy as np
import pytest

from pyscarcopula._utils import pobs
from pyscarcopula.copula.gumbel import GumbelCopula
from pyscarcopula.numerical.hermite_tm import standard_normal_hermite_rule
from pyscarcopula.numerical.jacobi_tm import jacobi_rule
from pyscarcopula.numerical.mc_samplers import p_sampler_loglik
from pyscarcopula.numerical._arrays import (
    as_float64_array,
    as_pseudo_observation_array,
)
from pyscarcopula.numerical.ou_kernels import (
    calculate_dwt,
    ou_init_state,
    ou_sample_paths,
    ou_sample_paths_exact,
    ou_stationary_state_from_dwt,
    sample_ou_trajectory,
)


@pytest.mark.parametrize("value", [3.9, "4", True])
def test_spectral_integer_options_reject_non_integer_types(value):
    with pytest.raises(TypeError, match="quad_order"):
        standard_normal_hermite_rule(value, 2)
    with pytest.raises(TypeError, match="quad_order"):
        jacobi_rule(2.0, 3.0, value, 2)


def test_array_normalization_rejects_complex_values_without_lossy_cast():
    values = np.array([0.2 + 7.0j, 0.8 - 3.0j])

    with pytest.raises(TypeError, match="real values"):
        as_float64_array(values, name="observations")


def test_pseudo_observation_validation_is_reusable_and_boundary_aware():
    boundary = np.array([[0.0, 1.0]])
    assert as_pseudo_observation_array(boundary) is boundary

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        as_pseudo_observation_array(np.array([[-0.01, 0.5]]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        as_pseudo_observation_array(np.array([[0.5, 1.01]]))
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        as_pseudo_observation_array(boundary, allow_boundary=False)


def test_ou_sample_paths_zero_aux_matches_exact_kernel():
    T, n_tr = 30, 5
    kappa, mu, nu = 1.4, 0.2, 0.9
    dwt = calculate_dwt(T, n_tr, seed=7)
    x0 = ou_init_state(mu, n_tr)
    zeros = np.zeros(T)

    exact = ou_sample_paths_exact(kappa, mu, nu, dwt, x0)
    via_eis = ou_sample_paths(kappa, mu, nu, zeros, zeros, dwt, x0)

    np.testing.assert_allclose(via_eis, exact, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("n", [1, 2, 17, 1000, 10001])
def test_sample_ou_trajectory_preserves_scalar_rng_contract(n):
    kappa, mu, nu = 1.4, 0.2, 0.9
    expected_rng = np.random.default_rng(20260803)
    actual_rng = np.random.default_rng(20260803)

    dt = 1.0 / (n - 1) if n > 1 else 1.0
    rho = np.exp(-kappa * dt)
    sigma_cond = np.sqrt(
        nu ** 2 / (2.0 * kappa) * (1.0 - rho ** 2)
    )
    expected = np.empty(n, dtype=np.float64)
    expected[0] = expected_rng.normal(mu, nu / np.sqrt(2.0 * kappa))
    for t in range(1, n):
        expected[t] = (
            mu + rho * (expected[t - 1] - mu)
            + sigma_cond * expected_rng.standard_normal()
        )

    actual = sample_ou_trajectory(kappa, mu, nu, n, actual_rng)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_rng.random(8), expected_rng.random(8))


def test_sample_ou_trajectory_zero_is_empty_and_does_not_advance_rng():
    actual_rng = np.random.default_rng(20260724)
    expected_rng = np.random.default_rng(20260724)

    actual = sample_ou_trajectory(1.4, 0.2, 0.9, 0, actual_rng)

    assert actual.shape == (0,)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual_rng.random(8), expected_rng.random(8))


@pytest.mark.parametrize(
    ("n", "error"),
    [(-1, ValueError), (True, TypeError), (1.5, TypeError)],
)
def test_sample_ou_trajectory_rejects_invalid_size(n, error):
    with pytest.raises(error, match="n must"):
        sample_ou_trajectory(
            1.4, 0.2, 0.9, n, np.random.default_rng(20260724))


def test_stationary_state_is_deterministic_from_dwt():
    dwt = calculate_dwt(20, 10, seed=123)

    x0_a = ou_stationary_state_from_dwt(1.2, 0.5, 0.7, dwt)
    x0_b = ou_stationary_state_from_dwt(1.2, 0.5, 0.7, dwt)

    np.testing.assert_allclose(x0_a, x0_b, rtol=0.0, atol=0.0)


def test_p_sampler_loglik_is_deterministic_for_fixed_dwt():
    u = pobs(np.random.default_rng(1).standard_normal((40, 2)))
    dwt = calculate_dwt(40, 300, seed=123)
    cop = GumbelCopula(rotate=180)

    vals = [
        p_sampler_loglik(1.2, 0.5, 0.7, u, dwt, cop, True)
        for _ in range(3)
    ]

    np.testing.assert_allclose(vals, vals[0], rtol=0.0, atol=0.0)
