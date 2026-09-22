"""Prepared GAS must retain the small orthogonal component near singularity."""

import numpy as np
import pytest
from scipy.special import ndtri

from pyscarcopula import EquicorrGaussianCopula
from pyscarcopula._native import gas


@pytest.mark.parametrize("spread", [0.0, 1e-8])
def test_prepared_gas_matches_constant_state_eigenvalue_oracle(spread):
    model = EquicorrGaussianCopula(3)
    data = np.tile([.9 - spread, .9, .9 + spread], (4, 1))
    prepared = model.prepare_sufficient_statistics(data, dimension_tile=2)
    parameters = (14.0, 0.0, 0.0)
    rho = model.transform(np.array([parameters[0]]))[0]
    scores = ndtri(data[0])
    centered = np.sum((scores - scores.mean())**2)
    common = scores.sum()**2 / 3
    eigenvalue = 1 + 2 * rho
    expected = len(data) * (
        -np.log1p(-rho) - .5 * np.log(eigenvalue)
        - .5 * (rho / (1 - rho) * centered
                  - 2 * rho / eigenvalue * common))
    result = gas.filter_result(*parameters, prepared, model)
    np.testing.assert_allclose(result.log_likelihood, expected, atol=2e-7, rtol=0)
    np.testing.assert_allclose(
        gas.log_likelihood(*parameters, prepared, model), expected,
        atol=2e-7, rtol=0)
    np.testing.assert_allclose(
        gas.negative_log_likelihood(*parameters, prepared, model), -expected,
        atol=2e-7, rtol=0)
    np.testing.assert_allclose(
        gas.predict_parameter(*parameters, prepared, model), rho,
        atol=0, rtol=0)
    objective, gradient = gas.negative_log_likelihood_and_gradient(
        *parameters, prepared, model)
    np.testing.assert_allclose(objective, -expected, atol=2e-7, rtol=0)
    assert np.all(np.isfinite(gradient))
