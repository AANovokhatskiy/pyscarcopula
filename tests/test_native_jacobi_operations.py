"""Contracts for native copula operations used by Jacobi filtering."""

import numpy as np
import pytest

from pyscarcopula import (
    BivariateGaussianCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    JoeCopula,
)
from pyscarcopula._types import PredictiveState
from pyscarcopula.numerical import _cpp_extension, copula_native
from pyscarcopula.numerical.jacobi_tm import (
    _emission_grid,
    _h_grid_on_theta,
    jacobi_forward_mixture_h,
    jacobi_loglik,
    jacobi_matrix_forward_mixture_h,
    jacobi_matrix_loglik,
)
from pyscarcopula.strategy.scar_jacobi import SCARJacobiStrategy


_U = np.array(
    [
        [0.12, 0.83],
        [0.71, 0.28],
        [0.44, 0.62],
        [0.91, 0.17],
        [0.33, 0.76],
        [0.58, 0.39],
    ],
    dtype=np.float64,
)


def test_pybind_exports_jacobi_copula_operations():
    module = _cpp_extension.load()
    expected = {
        "copula_tau_to_param",
        "copula_param_to_tau",
        "copula_pdf_parameter_grid",
        "copula_h_parameter_grid",
    }
    assert expected <= set(dir(module))


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
def test_native_kendall_mapping_roundtrip(copula):
    tau = np.array([1e-4, 0.01, 0.1, 0.3, 0.6, 0.9, 0.99])
    parameter = copula_native.tau_to_param(copula, tau)
    recovered = copula_native.param_to_tau(copula, parameter)
    np.testing.assert_allclose(recovered, tau, rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize(
    "copula",
    [
        ClaytonCopula(rotate=90),
        FrankCopula(),
        GumbelCopula(rotate=180),
        JoeCopula(rotate=270),
        BivariateGaussianCopula(),
    ],
)
def test_native_parameter_grids_match_point_operations(copula):
    tau = np.array([0.05, 0.2, 0.45, 0.75])
    theta = copula.tau_to_param(tau)
    expected_pdf = np.vstack([
        copula.pdf(
            np.full(len(theta), row[0]),
            np.full(len(theta), row[1]),
            theta,
        )
        for row in _U
    ])
    expected_h = np.vstack([
        copula.h(
            np.full(len(theta), row[1]),
            np.full(len(theta), row[0]),
            theta,
        )
        for row in _U
    ])

    np.testing.assert_allclose(
        copula_native.pdf_parameter_grid(copula, _U, theta),
        expected_pdf,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        copula_native.h_parameter_grid(copula, _U, theta),
        expected_h,
        rtol=1e-13,
        atol=1e-13,
    )


def test_builtin_jacobi_grid_does_not_call_python_family_methods(monkeypatch):
    copula = GumbelCopula(rotate=180)
    tau = np.linspace(0.05, 0.9, 12)

    def fail(*args, **kwargs):
        raise AssertionError("Python family method was called")

    monkeypatch.setattr(copula, "tau_to_param", fail)
    monkeypatch.setattr(copula, "pdf", fail)
    monkeypatch.setattr(copula, "h", fail)

    emissions, theta = _emission_grid(_U, copula, tau)
    h_grid = _h_grid_on_theta(copula, _U, theta)

    assert emissions.shape == h_grid.shape == (len(_U), len(tau))
    assert np.all(np.isfinite(emissions))
    assert np.all((h_grid > 0.0) & (h_grid < 1.0))


@pytest.mark.parametrize("n_obs", [2, 9])
def test_emission_grid_uses_one_pybind_call_independent_of_t(
        monkeypatch, n_obs):
    module = _cpp_extension.load()
    original = module.copula_pdf_parameter_grid
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "copula_pdf_parameter_grid", counted)
    tau = np.linspace(0.05, 0.9, 17)
    _emission_grid(np.resize(_U, (n_obs, 2)), GumbelCopula(), tau)

    assert calls == 1


@pytest.mark.parametrize("n_obs", [2, 9])
def test_h_grid_uses_one_pybind_call_independent_of_t(monkeypatch, n_obs):
    module = _cpp_extension.load()
    original = module.copula_h_parameter_grid
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "copula_h_parameter_grid", counted)
    tau = np.linspace(0.05, 0.9, 17)
    copula = GumbelCopula()
    theta = copula_native.tau_to_param(copula, tau)
    _h_grid_on_theta(copula, np.resize(_U, (n_obs, 2)), theta)

    assert calls == 1


def test_synthetic_jacobi_copula_retains_python_fallback():
    class SyntheticCopula:
        def tau_to_param(self, tau):
            return np.asarray(tau, dtype=np.float64)

        def pdf(self, u1, u2, r):
            return np.ones_like(np.asarray(r, dtype=np.float64))

        def h(self, u_conditioned, u_given, r):
            return np.full_like(np.asarray(r, dtype=np.float64), 0.25)

    copula = SyntheticCopula()
    tau = np.linspace(0.1, 0.9, 5)
    emissions, theta = _emission_grid(_U[:2], copula, tau)
    h_grid = _h_grid_on_theta(copula, _U[:2], theta)

    np.testing.assert_array_equal(emissions, np.ones((2, 5)))
    np.testing.assert_array_equal(h_grid, np.full((2, 5), 0.25))


def test_condition_state_uses_native_log_density(monkeypatch):
    copula = GumbelCopula()
    state = PredictiveState(
        method="SCAR-TM-JACOBI",
        horizon="next",
        kind="grid",
        z_grid=np.array([0.1, 0.3, 0.6]),
        prob=np.array([0.2, 0.5, 0.3]),
    )

    def fail(*args, **kwargs):
        raise AssertionError("copula.log_pdf was called")

    monkeypatch.setattr(copula, "log_pdf", fail)
    updated = SCARJacobiStrategy().condition_state(
        copula, state, _U[:1], result=None)

    np.testing.assert_allclose(np.sum(updated.prob), 1.0)
    assert not np.allclose(updated.prob, state.prob)


def test_jacobi_native_copula_integration_matches_regression_values():
    copula = GumbelCopula(rotate=180)
    args = (1.35, 0.42, 0.31, _U, copula)

    assert jacobi_loglik(
        *args, basis_order=6, quad_order=36
    ) == pytest.approx(-2.8730087908266886, rel=2e-12, abs=2e-12)
    np.testing.assert_allclose(
        jacobi_forward_mixture_h(
            *args, basis_order=6, quad_order=36),
        [
            0.9680738818230168,
            0.1214908042149461,
            0.6667306780331765,
            0.02982599015934154,
            0.8574200318173748,
            0.2823748151418195,
        ],
        rtol=2e-12,
        atol=2e-12,
    )
    assert jacobi_matrix_loglik(
        *args,
        basis_order=6,
        quad_order=36,
        transition_method="local",
        gh_order=5,
    ) == pytest.approx(-2.887284577703751, rel=2e-12, abs=2e-12)
    np.testing.assert_allclose(
        jacobi_matrix_forward_mixture_h(
            *args,
            basis_order=6,
            quad_order=36,
            transition_method="local",
            gh_order=5,
        ),
        [
            0.968073881823017,
            0.12084768248216446,
            0.6679776999068185,
            0.029773818830135974,
            0.8575679132435231,
            0.28178186278442124,
        ],
        rtol=2e-12,
        atol=2e-12,
    )
