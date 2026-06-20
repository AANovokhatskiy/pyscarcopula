"""Acceptance tests for the native-only SCAR-TM-OU strategy."""

import inspect

import numpy as np
import pytest

from pyscarcopula.copula.clayton import ClaytonCopula
from pyscarcopula.copula.frank import FrankCopula
from pyscarcopula.copula.gumbel import GumbelCopula
from pyscarcopula.copula.independent import IndependentCopula
from pyscarcopula.copula.joe import JoeCopula
from pyscarcopula.copula.multivariate.equicorr import (
    EquicorrGaussianCopula,
)
from pyscarcopula.numerical import _cpp_scar_ou
from pyscarcopula.numerical._scar_ou_config import AutoTMConfig
from pyscarcopula import stattests
from pyscarcopula.strategy import scar_tm
from pyscarcopula.strategy.scar_tm import SCARTMStrategy


_MATRIX_CONFIG = AutoTMConfig(
    transition_method="matrix",
    K=24,
    adaptive=False,
    max_K=24,
)


@pytest.mark.parametrize(
    "copula",
    [
        ClaytonCopula(transform_type="xtanh"),
        FrankCopula(transform_type="xtanh"),
        GumbelCopula(transform_type="xtanh"),
        JoeCopula(transform_type="xtanh"),
        IndependentCopula(),
    ],
)
def test_native_ou_support_covers_bivariate_family_and_transform_combinations(copula):
    u = np.random.default_rng(20260618).uniform(0.05, 0.95, size=(12, 2))

    value, gradient, info = _cpp_scar_ou.neg_loglik_with_grad_info(
        1.2, 0.1, 0.7, u, copula, _MATRIX_CONFIG)

    assert _cpp_scar_ou.supported(copula)
    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))
    assert info["engine"] == "cpp"


def test_native_ou_supports_equicorr_forward_and_state():
    copula = EquicorrGaussianCopula(4)
    u = np.random.default_rng(20260619).uniform(0.05, 0.95, size=(12, 4))

    value, gradient, _ = _cpp_scar_ou.neg_loglik_with_grad_info(
        1.2, 0.1, 0.7, u, copula, _MATRIX_CONFIG)
    predictive = _cpp_scar_ou.predictive_mean(
        1.2, 0.1, 0.7, u, copula, _MATRIX_CONFIG)
    z_grid, probability = _cpp_scar_ou.state_distribution(
        1.2, 0.1, 0.7, u, copula, _MATRIX_CONFIG, horizon="next")

    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))
    assert predictive.shape == (len(u),)
    assert z_grid.shape == probability.shape == (24,)
    assert np.sum(probability) == pytest.approx(1.0)


def test_spectral_forward_and_state_use_native_grid_reconstruction():
    copula = ClaytonCopula(transform_type="xtanh")
    u = np.random.default_rng(20260620).uniform(0.05, 0.95, size=(10, 2))
    config = AutoTMConfig(
        transition_method="spectral",
        basis_order=16,
        K=20,
        adaptive=False,
        max_K=20,
    )

    predictive = _cpp_scar_ou.predictive_mean(
        1.0, 0.0, 0.8, u, copula, config)
    z_grid, probability = _cpp_scar_ou.state_distribution(
        1.0, 0.0, 0.8, u, copula, config)

    assert predictive.shape == (len(u),)
    assert z_grid.shape == probability.shape == (20,)
    assert np.sum(probability) == pytest.approx(1.0)


@pytest.mark.parametrize("backend", ["python", "auto", "cpp"])
def test_public_backend_selection_is_removed(backend):
    with pytest.raises(TypeError, match="backend selection was removed"):
        SCARTMStrategy(backend=backend)


def test_strategy_has_no_python_ou_production_imports():
    source = inspect.getsource(scar_tm) + inspect.getsource(stattests)
    for symbol in (
        "tm_loglik",
        "tm_loglik_with_grad",
        "tm_forward_predictive_mean",
        "tm_forward_rosenblatt",
        "tm_forward_mixture_h",
        "auto_neg_loglik",
        "predictive_tm.tm_state_distribution",
    ):
        assert symbol not in source


def test_missing_native_extension_fails_before_optimizer(monkeypatch):
    copula = ClaytonCopula()
    u = np.random.default_rng(20260621).uniform(0.05, 0.95, size=(8, 2))
    optimizer_called = False

    def unavailable():
        raise RuntimeError("native extension unavailable")

    def forbidden_optimizer(*args, **kwargs):
        nonlocal optimizer_called
        optimizer_called = True
        raise AssertionError("optimizer must not run")

    monkeypatch.setattr(_cpp_scar_ou, "require_available", unavailable)
    monkeypatch.setattr(scar_tm, "minimize", forbidden_optimizer)

    with pytest.raises(RuntimeError, match="native extension unavailable"):
        SCARTMStrategy(smart_init=False).fit(
            copula,
            u,
            alpha0=np.array([1.0, 0.0, 1.0]),
        )
    assert optimizer_called is False
