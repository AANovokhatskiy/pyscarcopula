"""Numerical kernel regression tests."""
import warnings

import numpy as np
import pytest

from pyscarcopula.numerical.hermite_tm import standard_normal_hermite_rule
from pyscarcopula.numerical.jacobi_tm import jacobi_rule
from pyscarcopula.numerical._arrays import (
    as_float64_array,
    as_pseudo_observation_array,
    validate_integer,
    validate_sampling_memory_budget,
    validate_sampling_n_threads,
)
from pyscarcopula.numerical.ou_kernels import (
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


@pytest.mark.parametrize("scalar", [complex, np.complex64, np.complex128])
@pytest.mark.parametrize("imaginary", [0.0, 1.0])
def test_array_normalization_rejects_complex_object_scalars(scalar, imaginary):
    values = np.array([scalar(0.5 + imaginary * 1j)], dtype=object)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(TypeError, match="real values"):
            as_float64_array(values, name="observations")
    assert not caught


def test_array_normalization_preserves_real_object_coercion():
    values = np.array([np.float32(0.5), 1, "0.25"], dtype=object)
    np.testing.assert_array_equal(as_float64_array(values), [0.5, 1.0, 0.25])


def test_pseudo_observation_validation_is_reusable_and_boundary_aware():
    boundary = np.array([[0.0, 1.0]])
    assert as_pseudo_observation_array(boundary) is boundary

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        as_pseudo_observation_array(np.array([[-0.01, 0.5]]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        as_pseudo_observation_array(np.array([[0.5, 1.01]]))
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        as_pseudo_observation_array(boundary, allow_boundary=False)


@pytest.mark.parametrize("value", [False, np.bool_(True), 1.5, "1"])
def test_integer_validation_rejects_non_integer_types(value):
    with pytest.raises(TypeError, match="count must be an integer"):
        validate_integer(value, "count")


def test_integer_validation_preserves_inclusive_minimum_contract():
    assert validate_integer(np.int64(0), "count") == 0
    assert validate_integer(1, "count", minimum=1) == 1
    with pytest.raises(ValueError, match="count must be non-negative"):
        validate_integer(-1, "count")
    with pytest.raises(ValueError, match="count must be positive"):
        validate_integer(0, "count", minimum=1)


@pytest.mark.parametrize("value", [0, 257, True, 1.5])
def test_sampling_thread_validation_preserves_public_contract(value):
    with pytest.raises((TypeError, ValueError), match="n_threads"):
        validate_sampling_n_threads(value)


def test_sampling_memory_budget_preserves_error_message_and_boundaries():
    assert validate_sampling_memory_budget(None, 64, "reduce n") is None
    assert validate_sampling_memory_budget(64, 64, "reduce n") is None
    with pytest.raises(MemoryError, match=r"approximately 64 bytes; reduce n"):
        validate_sampling_memory_budget(63, 64, "reduce n")


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


@pytest.mark.parametrize("n", [1, 17, 10001])
def test_ou_sampling_uses_only_raw_normal_draws(monkeypatch, n):
    class RawNormalRng:
        def __init__(self):
            self.calls = []

        def standard_normal(self, size):
            self.calls.append(size)
            return np.zeros(size)

    def forbidden(*args, **kwargs):
        raise AssertionError("Python OU parameter arithmetic was called")

    monkeypatch.setattr(np, "exp", forbidden)
    monkeypatch.setattr(np, "sqrt", forbidden)
    rng = RawNormalRng()
    result = sample_ou_trajectory(1.4, 0.2, 0.9, n, rng)
    np.testing.assert_array_equal(result, np.full(n, 0.2))
    assert rng.calls == [n]


def test_ou_sampling_does_not_swallow_native_failure(monkeypatch):
    from pyscarcopula._native import _extension
    from pyscarcopula._native.errors import NativeUnsupported

    def unsupported(*args, **kwargs):
        raise NativeUnsupported("sentinel OU sampling failure")

    monkeypatch.setattr(_extension.load(), "ou_sample_trajectory", unsupported)
    with pytest.raises(NativeUnsupported, match="sentinel OU sampling failure"):
        sample_ou_trajectory(1.4, 0.2, 0.9, 3, np.random.default_rng(1))


@pytest.mark.parametrize("params", [(-1., .2, .9), (0., .2, .9),
                                    (1.4, np.nan, .9), (1.4, .2, -1.),
                                    (1.4, .2, 1e300)])
def test_invalid_ou_parameters_do_not_advance_rng(params):
    rng, reference = np.random.default_rng(51), np.random.default_rng(51)
    with pytest.raises(ValueError):
        sample_ou_trajectory(*params, 17, rng)
    np.testing.assert_array_equal(rng.random(8), reference.random(8))


@pytest.mark.parametrize("quad_order,basis_order", [(8, 4), (48, 16), (80, 32)])
def test_hermite_rule_matches_independent_scipy_rule(quad_order, basis_order):
    from scipy.special import roots_hermitenorm

    nodes, weights, basis = standard_normal_hermite_rule(quad_order, basis_order)
    expected_nodes, expected_weights = roots_hermitenorm(quad_order)
    np.testing.assert_allclose(nodes, expected_nodes, rtol=5e-13, atol=5e-14)
    np.testing.assert_allclose(weights, expected_weights / np.sqrt(2 * np.pi),
                               rtol=5e-12, atol=5e-15)
    np.testing.assert_allclose(basis.T @ (weights[:, None] * basis),
                               np.eye(basis_order), rtol=5e-12, atol=5e-12)


def test_hermite_utilities_dispatch_to_production_native_owner(monkeypatch):
    from pyscarcopula._native import _extension
    from pyscarcopula._native.errors import NativeUnsupported
    from pyscarcopula.numerical.hermite_tm import default_quad_order

    standard_normal_hermite_rule.cache_clear()
    module = _extension.load()

    def unsupported(*args, **kwargs):
        raise NativeUnsupported("sentinel Hermite failure")

    monkeypatch.setattr(module, "ou_hermite_rule", unsupported)
    monkeypatch.setattr(module, "ou_default_quad_order", unsupported)
    with pytest.raises(NativeUnsupported, match="sentinel Hermite failure"):
        standard_normal_hermite_rule(24, 8)
    with pytest.raises(NativeUnsupported, match="sentinel Hermite failure"):
        default_quad_order(8)
