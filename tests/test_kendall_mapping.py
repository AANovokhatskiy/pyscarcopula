import numpy as np
import pytest
from scipy.special import spence

from pyscarcopula import (
    BivariateGaussianCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    JoeCopula,
)


@pytest.mark.parametrize(
    "copula",
    [
        ClaytonCopula(),
        FrankCopula(),
        GumbelCopula(),
        JoeCopula(),
        BivariateGaussianCopula(),
    ],
)
def test_tau_parameter_roundtrip_supported_copulas(copula):
    tau = np.array([0.1, 0.35, 0.75])

    param = copula.tau_to_param(tau)
    recovered = copula.param_to_tau(param)

    np.testing.assert_allclose(recovered, tau, rtol=1e-12, atol=1e-12)


def test_gaussian_tau_mapping_accepts_signed_tau():
    copula = BivariateGaussianCopula()
    tau = np.array([-0.5, 0.0, 0.5])

    rho = copula.tau_to_param(tau)
    recovered = copula.param_to_tau(rho)

    np.testing.assert_allclose(recovered, tau, rtol=1e-12, atol=1e-12)


def test_frank_tau_mapping_matches_independence_limit():
    copula = FrankCopula()
    theta = np.array([1e-5, 1e-4, 1e-3])

    tau = copula.param_to_tau(theta)

    np.testing.assert_allclose(tau, theta / 9.0, rtol=1e-4, atol=1e-12)


def test_frank_inverse_mapping_preserves_tiny_interior_tau():
    copula = FrankCopula()
    tau = np.array([1e-16, 1e-14, 1e-12, 1e-8, 0.009])

    recovered = copula.param_to_tau(copula.tau_to_param(tau))

    np.testing.assert_allclose(recovered, tau, rtol=1e-12, atol=1e-28)


def test_frank_tau_mapping_matches_exact_debye_identity_at_two():
    copula = FrankCopula()
    theta = 2.0
    integral = (
        np.pi ** 2 / 6.0
        + theta * np.log1p(-np.exp(-theta))
        - spence(1.0 - np.exp(-theta))
    )
    expected = 1.0 - 4.0 / theta + 4.0 * integral / theta ** 2

    actual = float(copula.param_to_tau([theta])[0])

    assert actual == pytest.approx(expected, rel=0.0, abs=2e-15)


def test_joe_tau_mapping_known_value_at_two():
    copula = JoeCopula()

    tau = copula.param_to_tau(np.array([2.0]))

    np.testing.assert_allclose(tau, [2.0 - np.pi ** 2 / 6.0], rtol=1e-12)


def test_joe_tau_mapping_is_stable_around_two():
    copula = JoeCopula()
    theta = np.array([1.999999999, 2.0, 2.000000001])
    expected = np.array([
        0.3550659329303347,
        0.35506593315177337,
        0.3550659333732120,
    ])

    actual = copula.param_to_tau(theta)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-13)
