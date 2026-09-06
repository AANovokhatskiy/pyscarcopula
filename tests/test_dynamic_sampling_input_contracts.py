"""Public sample/predict size contracts after a real dynamic fit."""

from copy import deepcopy

import numpy as np
import pytest

from pyscarcopula import GumbelCopula, api


@pytest.fixture(scope="module", params=["GAS", "SCAR-TM-OU", "SCAR-TM-JACOBI"])
def fitted_model(request):
    model = GumbelCopula()
    observations = model.sample_at_parameter(120, 2., rng=np.random.default_rng(5))
    result = model.fit(observations, method=request.param)
    assert result.success, result.message
    return model, observations, result


@pytest.mark.parametrize("entry", ["sample", "predict", "api.sample", "api.predict"])
@pytest.mark.parametrize("n", [-1, True, np.bool_(True), 1.5, "3", 0, np.int64(1)])
def test_dynamic_public_size_contract(fitted_model, entry, n):
    model, observations, result = fitted_model
    rng = np.random.default_rng(34)
    before = deepcopy(rng.bit_generator.state)

    def draw():
        if entry.startswith("api."):
            return getattr(api, entry[4:])(model, observations, result, n, rng=rng)
        return getattr(model, entry)(n, rng=rng)

    if isinstance(n, (bool, np.bool_, float, str)):
        expected_error = TypeError
    elif n < 0 or (n == 0 and result.method == "GAS"):
        expected_error = ValueError
    else:
        expected_error = None

    if expected_error:
        with pytest.raises(expected_error):
            draw()
        assert rng.bit_generator.state == before
    else:
        samples = draw()
        assert samples.shape == (n, 2)
        assert samples.dtype == np.float64
        assert np.all(np.isfinite(samples))
        assert np.all((samples > 0.) & (samples < 1.))
        if n == 0:
            assert rng.bit_generator.state == before
