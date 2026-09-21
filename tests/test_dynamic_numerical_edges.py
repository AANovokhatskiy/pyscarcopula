"""Regression oracles for extreme but finite OU and Jacobi parameters."""

import math

import numpy as np
import pytest

from pyscarcopula._native import scar_ou
from pyscarcopula.numerical import jacobi_rule
from pyscarcopula.numerical.ou_kernels import sample_ou_trajectory


@pytest.mark.parametrize(
    "kappa, nu",
    [(1e-20, 1.0), (1e-16, 1.0), (1.0, 1e155), (1e308, 1e154)],
)
def test_ou_conditional_scale_keeps_finite_noise(kappa, nu):
    # A zero initial draw isolates the conditional standard deviation. The
    # closed-form oracle avoids squaring nu or forming 2*kappa.
    stationary = nu / math.sqrt(kappa) / math.sqrt(2.0)
    expected = stationary * math.sqrt(-math.expm1(-2.0 * kappa))
    actual = scar_ou.sample_trajectory(kappa, 0.0, nu, np.array([0.0, 1.0]))
    np.testing.assert_allclose(actual, [0.0, expected], rtol=2e-15, atol=0.0)


def test_ou_stationary_scale_does_not_overflow_twice_kappa():
    actual = scar_ou.sample_trajectory(1e308, 0.0, 1e154, np.array([1.0]))
    np.testing.assert_allclose(actual, [1.0 / math.sqrt(2.0)], rtol=2e-15)


def test_public_ou_sampler_accepts_representable_large_scale_and_preserves_rng():
    actual_rng = np.random.default_rng(51)
    reference_rng = np.random.default_rng(51)
    actual = sample_ou_trajectory(1.4, 0.0, 1e300, 17, actual_rng)
    reference = sample_ou_trajectory(1.4, 0.0, 1.0, 17, reference_rng)
    assert np.all(np.isfinite(actual))
    # The zero-mean OU process is homogeneous in its diffusion scale.
    np.testing.assert_allclose(actual / 1e300, reference, rtol=1e-13, atol=2e-15)
    np.testing.assert_array_equal(actual_rng.random(8), reference_rng.random(8))


@pytest.mark.parametrize(
    "alpha, beta", [(1e-12, 3e-12), (2e-12, 3e-12), (1e-17, 1.0),
                    (1e-17, 2e-17)],
)
def test_jacobi_small_shapes_preserve_support_and_beta_moments(alpha, beta):
    nodes, weights, basis = jacobi_rule(alpha, beta, 8, 4)
    assert np.all(np.isfinite(nodes))
    assert np.all((nodes >= 0.0) & (nodes <= 1.0))
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= 0.0)
    assert np.all(np.isfinite(basis))
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=0.0, atol=1e-14)
    # Gauss quadrature integrates these polynomials exactly; the independent
    # beta moment identity remains well-conditioned for these small shapes.
    for power in (1, 2, 3):
        expected = alpha / (alpha + beta)
        for index in range(1, power):
            expected *= (alpha + index) / (alpha + beta + index)
        np.testing.assert_allclose(
            weights @ nodes**power, expected, rtol=0.0, atol=2e-14)
