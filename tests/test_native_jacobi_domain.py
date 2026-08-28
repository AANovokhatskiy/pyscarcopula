"""Contracts for the native Jacobi domain core."""

from pathlib import Path

import numpy as np
import pytest
from scipy.special import eval_jacobi, roots_jacobi

from pyscarcopula._native import jacobi as jacobi_native
from pyscarcopula.numerical import jacobi_tm
from pyscarcopula.strategy import scar_jacobi


def test_pybind_exports_typed_jacobi_domain_contract():
    module = jacobi_native.load()
    expected = {
        "JacobiBoundaryPolicy",
        "JacobiNumericalConfig",
        "JacobiParameterBounds",
        "JacobiParams",
        "jacobi_build_fixed_rule",
        "jacobi_build_rule",
        "jacobi_estimate_workspace",
        "jacobi_gauss_hermite_rule",
        "jacobi_gauss_jacobi_rule",
        "jacobi_inverse_lamperti_values",
        "jacobi_lamperti_drift_values",
        "jacobi_lamperti_values",
        "jacobi_physical_to_raw",
        "jacobi_raw_to_physical",
        "jacobi_stationary_shape",
    }
    assert expected <= set(dir(module))


def test_native_parameterization_matches_public_contract_values():
    np.testing.assert_allclose(
        jacobi_native.raw_to_physical([-3.0, -1.25, 0.8]),
        [0.049787068367863944, 0.22270013882530884, 2.225540928492468],
        rtol=5e-11,
        atol=5e-12,
    )
    np.testing.assert_allclose(
        jacobi_native.raw_to_physical([-60.0, -60.0, -60.0]),
        [1.9287498479639178e-22] * 3,
        rtol=5e-11,
        atol=5e-12,
    )
    np.testing.assert_allclose(
        jacobi_native.raw_to_physical([60.0, 60.0, 60.0]),
        [5.184705528587072e21, 1.0, 5.184705528587072e21],
        rtol=5e-11,
        atol=5e-12,
    )
    raw = jacobi_native.physical_to_raw([1.2, 0.4, 0.25], 1e-6)
    np.testing.assert_allclose(
        raw,
        [0.1823215567939546, -0.4054651081081643, -1.3862943611198906],
        rtol=5e-11,
        atol=5e-12,
    )
    assert jacobi_native.stationary_shape(1.2, 0.4, 0.25) == pytest.approx(
        (15.36, 23.04), rel=5e-12, abs=5e-12)


@pytest.mark.parametrize(
    ("alpha", "beta", "quad_order", "basis_order"),
    [(2.5, 3.5, 24, 8), (15.36, 23.04, 16, 4), (0.7, 1.3, 32, 6)],
)
def test_native_gauss_jacobi_rule_matches_scipy_reference(
        alpha, beta, quad_order, basis_order):
    tau, weights, basis = jacobi_tm.jacobi_rule(
        alpha, beta, quad_order, basis_order)

    x, raw_weights = roots_jacobi(quad_order, beta - 1.0, alpha - 1.0)
    expected_tau = 0.5 * (x + 1.0)
    expected_weights = raw_weights / np.sum(raw_weights)
    poly = np.column_stack([
        eval_jacobi(degree, beta - 1.0, alpha - 1.0, x)
        for degree in range(basis_order)
    ])
    gram = poly.T @ (expected_weights[:, None] * poly)
    expected_basis = poly @ np.linalg.inv(np.linalg.cholesky(gram).T)

    np.testing.assert_allclose(tau, expected_tau, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(
        weights, expected_weights, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(
        basis, expected_basis, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(
        basis.T @ (weights[:, None] * basis),
        np.eye(basis_order),
        rtol=2e-11,
        atol=2e-11,
    )


def test_native_gauss_hermite_rule_has_probability_moments():
    nodes, weights = jacobi_native.gauss_hermite_rule(9)
    np.testing.assert_allclose(
        [np.sum(weights), weights @ nodes, weights @ nodes**2,
         weights @ nodes**4],
        [1.0, 0.0, 0.5, 0.75],
        rtol=2e-13,
        atol=2e-13,
    )


def test_native_fixed_weight_derivatives_match_physical_finite_difference():
    params = np.array([1.2, 0.4, 0.25])
    tau, weights, derivatives = jacobi_native.fixed_tau_rule(
        *params, 24, 1024**3)
    assert tau.shape == weights.shape == (24,)
    assert derivatives.shape == (3, 24)
    np.testing.assert_allclose(derivatives.sum(axis=1), 0.0, atol=2e-13)

    step = 1e-6
    numerical = np.empty_like(derivatives)
    for parameter in range(3):
        plus = params.copy()
        minus = params.copy()
        plus[parameter] += step
        minus[parameter] -= step
        plus_weights = jacobi_native.fixed_tau_rule(
            *plus, 24, 1024**3)[1]
        minus_weights = jacobi_native.fixed_tau_rule(
            *minus, 24, 1024**3)[1]
        numerical[parameter] = (plus_weights - minus_weights) / (2.0 * step)
    np.testing.assert_allclose(
        derivatives, numerical, rtol=3e-6, atol=2e-9)


def test_native_lamperti_roundtrip_drift_and_boundary_rules():
    tau = np.array([0.0, 1e-6, 0.2, 0.6, 0.999999, 1.0])
    values = jacobi_native.lamperti(tau, 0.25)
    np.testing.assert_allclose(
        jacobi_native.inverse_lamperti(values, 0.25),
        tau,
        rtol=2e-13,
        atol=2e-15,
    )
    drift = jacobi_native.lamperti_drift(
        tau, 1.2, 0.4, 0.25, interior_eps=1e-10)
    assert np.all(np.isfinite(drift))

    module = jacobi_native.load()
    reflected = module.jacobi_apply_boundary(
        -0.25, 1.0, module.JacobiBoundaryPolicy.Reflect)
    clipped = module.jacobi_apply_boundary(
        1.25, 1.0, module.JacobiBoundaryPolicy.Clip)
    assert reflected["status"] == clipped["status"] == 0
    assert reflected["intervened"] and reflected["value"] == pytest.approx(0.25)
    assert clipped["intervened"] and clipped["value"] == pytest.approx(1.0)


def test_native_checked_memory_contract_accounts_for_eigensolvers():
    assert jacobi_native.estimate_workspace(
        quad_order=16,
        basis_order=4,
        gh_order=5,
        n_obs=8,
        matrix=True,
        gradient=True,
        memory_budget_bytes=1024**3,
    ) == 26368
    with pytest.raises(MemoryError, match="memory_budget_bytes=1024"):
        jacobi_native.estimate_workspace(
            quad_order=16,
            basis_order=4,
            gh_order=5,
            n_obs=8,
            matrix=True,
            gradient=True,
            memory_budget_bytes=1024,
        )
    assert jacobi_native.estimate_workspace(
        quad_order=16,
        basis_order=4,
        gh_order=128,
        n_obs=8,
        matrix=True,
        gradient=True,
        memory_budget_bytes=1024**3,
    ) > 26368


def test_native_quadrature_budget_covers_eigenvector_matrices():
    quad_order = 128
    budget = 100_000
    with pytest.raises(MemoryError, match="memory_budget_bytes=100000"):
        jacobi_native.estimate_workspace(
            quad_order=quad_order,
            basis_order=1,
            gh_order=1,
            matrix=False,
            memory_budget_bytes=budget,
        )
    with pytest.raises(MemoryError, match="memory_budget_bytes=100000"):
        jacobi_native.jacobi_rule(
            2.5, 3.5, quad_order, 1, budget)
    with pytest.raises(MemoryError, match="memory_budget_bytes=100000"):
        jacobi_native.gauss_hermite_rule(quad_order, budget)


def test_production_python_delegates_domain_formulas_to_native():
    root = Path(__file__).resolve().parents[1]
    numerical = (root / "pyscarcopula/numerical/jacobi_tm.py").read_text(
        encoding="utf-8")
    strategy = (root / "pyscarcopula/strategy/scar_jacobi.py").read_text(
        encoding="utf-8")
    for marker in (
            "scipy.special", "roots_jacobi", "eval_jacobi", "hermgauss",
            "betaln", "digamma"):
        assert marker not in numerical
    assert "expit" not in strategy
    assert "def _logit" not in strategy
    assert "jacobi_native.raw_to_physical" in strategy
    assert "jacobi_native.physical_to_raw" in strategy


@pytest.mark.parametrize(
    "call",
    [
        lambda: jacobi_native.raw_to_physical([np.nan, 0.0, 0.0]),
        lambda: jacobi_native.jacobi_rule(-1.0, 2.0, 8, 4, 1024**3),
        lambda: jacobi_native.lamperti([0.5], 0.0),
    ],
)
def test_native_invalid_domains_fail_without_python_fallback(call):
    with pytest.raises(ValueError):
        call()
