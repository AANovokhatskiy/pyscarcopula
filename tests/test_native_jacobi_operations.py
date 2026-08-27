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
from pyscarcopula._native import jacobi as jacobi_native
from pyscarcopula._native import _extension as _cpp_extension, pair as copula_native
from pyscarcopula.numerical.jacobi_tm import (
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
        copula.h_pair(
            np.full(len(theta), row[0]),
            np.full(len(theta), row[1]),
            theta,
        )[1]
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

    def fail(*args, **kwargs):
        raise AssertionError("Python family method was called")

    monkeypatch.setattr(copula, "tau_to_param", fail)
    monkeypatch.setattr(copula, "pdf", fail)
    monkeypatch.setattr(copula, "h", fail)

    evaluator = jacobi_native.PreparedScarJacobiEvaluator(
        _U,
        copula,
        basis_order=4,
        quad_order=16,
        transition_method="local",
        gh_order=3,
    )
    state = evaluator.filter(1.2, 0.4, 0.25)
    h_grid = evaluator.mixture_h(1.2, 0.4, 0.25)

    assert state["emissions"].shape == (len(_U), 16)
    assert np.all(np.isfinite(state["emissions"]))
    assert np.all((h_grid > 0.0) & (h_grid < 1.0))
    assert evaluator.preparation_count == 1


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
