"""Numerical failure penalties follow the public objective configuration."""
import numpy as np
import pytest

from pyscarcopula import (
    BivariateGaussianCopula, EquicorrGaussianCopula, NumericalConfig,
    StochasticStudentCopula,
)
from pyscarcopula._native import gas as native_gas
from pyscarcopula._native.errors import raise_for_status
from pyscarcopula.strategy.gas import GASStrategy


@pytest.mark.parametrize("scaling", ["unit", "fisher"])
@pytest.mark.parametrize("prepared", [False, True])
def test_natural_equicorr_failure_honors_config(scaling, prepared):
    model = EquicorrGaussianCopula(4)
    observations = np.random.default_rng(91).uniform(0.1, 0.9, (8, 4))
    data = model.prepare_sufficient_statistics(observations) if prepared else observations
    parameters = [100.0, 0.1, 0.5]
    with pytest.raises(FloatingPointError) as error:
        native_gas.negative_log_likelihood(*parameters, data, model, scaling, 1e-4)
    assert error.value.status == 7
    # The model helper accepts dense observations; the strategy additionally
    # accepts the prepared representation used by Equicorr fitting.
    def objective(config):
        if prepared:
            return GASStrategy(config=config, scaling=scaling).objective(model, data, parameters)
        return model.mlog_likelihood(
            parameters, data, method="GAS", scaling=scaling, config=config)

    assert objective(NumericalConfig(fail_value=4321.0)) == 4321.0
    assert objective(NumericalConfig()) == 1e10


@pytest.mark.parametrize("family", ["bivariate", "equicorr", "student", "factor"])
@pytest.mark.parametrize("scaling", ["unit", "fisher"])
def test_objective_failure_override_after_original_native_call(family, scaling, monkeypatch):
    if family == "bivariate":
        model = BivariateGaussianCopula()
        dimension = 2
    elif family == "equicorr":
        model = EquicorrGaussianCopula(4)
        dimension = 4
    elif family == "student":
        model = StochasticStudentCopula(4, R=np.eye(4) * 0.72 + 0.28)
        dimension = 4
    else:
        model = StochasticStudentCopula(
            4, corr_mode="factor", factor_rank=1,
            factor_loadings=np.array([[0.2], [0.4], [-0.3], [0.25]]))
        dimension = 4
    observations = np.random.default_rng(93).uniform(0.1, 0.9, (8, dimension))
    parameters = [0.02, 0.04, 0.72]
    expected = model.mlog_likelihood(parameters, observations, method="GAS", scaling=scaling)
    assert model.mlog_likelihood(
        parameters, observations, method="GAS", scaling=scaling,
        config=NumericalConfig(fail_value=8765.0)) == expected
    original = native_gas.negative_log_likelihood
    completed = []

    def fail_after_native(*args, **kwargs):
        value = original(*args, **kwargs)
        assert np.isfinite(value)
        completed.append(value)
        raise_for_status({"status": 7, "failure_index": 0}, "controlled numerical failure")

    monkeypatch.setattr(native_gas, "negative_log_likelihood", fail_after_native)
    assert model.mlog_likelihood(
        parameters, observations, method="GAS", scaling=scaling,
        config=NumericalConfig(fail_value=8765.0)) == 8765.0
    assert completed == [expected]


def test_failure_override_does_not_hide_unstructured_errors(monkeypatch):
    def fail(*args, **kwargs):
        raise FloatingPointError("unstructured error")

    monkeypatch.setattr(native_gas, "negative_log_likelihood", fail)
    with pytest.raises(FloatingPointError, match="unstructured error"):
        BivariateGaussianCopula().mlog_likelihood(
            [0.02, 0.04, 0.72], np.full((3, 2), 0.55), method="GAS",
            config=NumericalConfig(fail_value=4321.0))
