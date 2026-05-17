"""Numerical kernel regression tests."""
import numpy as np

from pyscarcopula._utils import pobs
from pyscarcopula.copula.gumbel import GumbelCopula
from pyscarcopula.numerical.mc_samplers import p_sampler_loglik
from pyscarcopula.numerical.ou_kernels import (
    calculate_dwt,
    ou_init_state,
    ou_sample_paths,
    ou_sample_paths_exact,
    ou_stationary_state_from_dwt,
)


def test_ou_sample_paths_zero_aux_matches_exact_kernel():
    T, n_tr = 30, 5
    kappa, mu, nu = 1.4, 0.2, 0.9
    dwt = calculate_dwt(T, n_tr, seed=7)
    x0 = ou_init_state(mu, n_tr)
    zeros = np.zeros(T)

    exact = ou_sample_paths_exact(kappa, mu, nu, dwt, x0)
    via_eis = ou_sample_paths(kappa, mu, nu, zeros, zeros, dwt, x0)

    np.testing.assert_allclose(via_eis, exact, rtol=0.0, atol=0.0)


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
