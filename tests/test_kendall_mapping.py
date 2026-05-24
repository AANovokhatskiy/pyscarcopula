import numpy as np
import pytest

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


def test_joe_tau_mapping_known_value_at_two():
    copula = JoeCopula()

    tau = copula.param_to_tau(np.array([2.0]))

    np.testing.assert_allclose(tau, [2.0 - np.pi ** 2 / 6.0], rtol=1e-12)
