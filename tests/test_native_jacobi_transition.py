"""Stage 8.3.3 native Jacobi transition ownership contracts."""

from pathlib import Path

import numpy as np
import pytest

from pyscarcopula._native import jacobi as jacobi_native
from pyscarcopula.numerical import jacobi_sparse, jacobi_tm


def test_pybind_exports_typed_jacobi_transition_contract():
    module = jacobi_native.load()
    expected = {
        "JacobiAdaptiveThresholds",
        "JacobiStationarityCorrection",
        "JacobiTransitionConfig",
        "JacobiTransitionMethod",
        "JacobiTransitionStorage",
        "jacobi_build_dense_transition",
        "jacobi_build_coefficient_transition",
        "jacobi_build_fixed_transition",
        "jacobi_build_local_transition",
        "jacobi_build_sparse_transition",
        "jacobi_build_spectral_transition",
        "jacobi_default_quad_order",
        "jacobi_apply_coefficient_transition",
        "jacobi_estimate_sparse_storage",
        "jacobi_estimate_sparse_workspace",
        "jacobi_select_sparse_order",
        "jacobi_sparse_full_horizon_diagnostics",
        "jacobi_sparse_left_multiply",
        "jacobi_sparse_to_dense",
        "jacobi_transition_powers",
    }
    assert expected <= set(dir(module))


def test_sparse_workspace_compatibility_facade_is_native():
    assert jacobi_sparse._validate_sparse_workspace(
        quad_order=16,
        gh_order=3,
        correction="ipfp",
    ) == 4864


def test_sparse_workspace_estimator_matches_enforced_preflight():
    assert jacobi_native.estimate_sparse_workspace(
        quad_order=16,
        gh_order=3,
        correction="ipfp",
    ) == 8744
    with pytest.raises(MemoryError, match="memory_budget_bytes=5000"):
        jacobi_native.estimate_sparse_workspace(
            quad_order=16,
            gh_order=3,
            correction="ipfp",
            memory_budget_bytes=5000,
        )


def test_native_spectral_coefficient_transition_and_propagation():
    tau, weights, basis, powers, diagnostics = (
        jacobi_native.coefficient_transition(
            1.2,
            0.4,
            0.25,
            n_obs=5,
            quad_order=16,
            basis_order=4,
        )
    )
    coefficients = np.array([1.0, 2.0, 3.0, 4.0])
    propagated = jacobi_native.apply_coefficient_transition(
        powers, coefficients)

    assert tau.shape == weights.shape == (16,)
    assert basis.shape == (16, 4)
    assert diagnostics["transition_method"] == "spectral_coeff"
    assert diagnostics["transition_method_requested"] == "spectral_coeff"
    np.testing.assert_array_equal(propagated, powers * coefficients)


def test_transition_time_grid_is_derived_only_from_observation_count():
    module = jacobi_native.load()
    assert not hasattr(module.JacobiTransitionConfig(), "dt")

    _, _, _, diagnostics = jacobi_tm.jacobi_spectral_transition_matrix(
        1.5,
        0.4,
        0.35,
        n_obs=5,
        basis_order=8,
        quad_order=32,
        return_diagnostics=True,
    )
    assert diagnostics["dt"] == pytest.approx(0.25)

    with pytest.raises(ValueError, match="n_obs must be at least 2"):
        jacobi_tm.jacobi_spectral_transition_matrix(
            1.2, 0.4, 0.25, n_obs=1, basis_order=4, quad_order=24)
    with pytest.raises(TypeError):
        jacobi_tm.jacobi_spectral_transition_matrix(
            1.2, 0.4, 0.25, dt=0.25, n_obs=5)


def test_dense_auto_selection_is_owned_by_single_native_call(monkeypatch):
    module = jacobi_native.load()
    original = module.jacobi_build_dense_transition
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "jacobi_build_dense_transition", counted)
    _, _, transition, diagnostics = jacobi_tm.jacobi_transition_matrix(
        1.2,
        0.4,
        0.25,
        n_obs=1_000_001,
        basis_order=6,
        quad_order=40,
        transition_method="auto",
        return_diagnostics=True,
    )

    assert calls == 1
    assert diagnostics["transition_method"] == "local"
    np.testing.assert_allclose(transition.sum(axis=1), 1.0, atol=2e-15)


@pytest.mark.parametrize("correction", ["none", "mh", "ipfp"])
def test_sparse_corrections_and_matvec_execute_through_native(correction):
    tau, weights, transition, diagnostics = (
        jacobi_sparse.jacobi_sparse_local_transition(
            1.2,
            0.4,
            0.25,
            n_obs=33,
            quad_order=48,
            gh_order=5,
            correction=correction,
            return_diagnostics=True,
        )
    )
    propagated = transition.left_multiply(weights)

    assert tau.shape == weights.shape == propagated.shape == (48,)
    assert diagnostics["correction"] == correction
    np.testing.assert_allclose(propagated.sum(), 1.0, atol=2e-15)
    if correction in {"mh", "ipfp"}:
        np.testing.assert_allclose(propagated, weights, atol=2e-13)


def test_native_sparse_to_dense_rejects_invalid_rows_safely():
    with pytest.raises(ValueError, match="sparse transition dense"):
        jacobi_native.sparse_to_dense(
            np.array([[0]], dtype=np.int64),
            np.array([[1.0]], dtype=np.float64),
            np.array([2], dtype=np.int64),
        )


def test_native_fixed_sparse_derivatives_match_dense():
    dense = jacobi_tm.jacobi_fixed_grid_transition_matrix(
        1.2,
        0.4,
        0.25,
        n_obs=17,
        quad_order=24,
        gh_order=5,
        return_grad=True,
    )
    sparse = jacobi_sparse.jacobi_sparse_fixed_grid_transition(
        1.2,
        0.4,
        0.25,
        n_obs=17,
        quad_order=24,
        gh_order=5,
        return_grad=True,
    )
    _, _, dense_transition, dense_derivatives = dense
    _, _, sparse_transition, sparse_derivatives = sparse
    reconstructed = np.zeros_like(dense_derivatives)
    for row in range(24):
        count = int(sparse_transition.counts[row])
        reconstructed[:, row, sparse_transition.indices[row, :count]] = (
            sparse_derivatives[:, row, :count])

    np.testing.assert_allclose(
        sparse_transition.to_dense(), dense_transition, atol=4e-15)
    np.testing.assert_allclose(reconstructed, dense_derivatives, atol=2e-14)


def test_production_python_has_no_stage833_transition_kernels():
    root = Path(__file__).resolve().parents[1]
    dense = (root / "pyscarcopula/numerical/jacobi_tm.py").read_text(
        encoding="utf-8")
    sparse = (root / "pyscarcopula/numerical/jacobi_sparse.py").read_text(
        encoding="utf-8")
    for marker in (
            "def _add_interpolated_mass",
            "def _probability_transition_matrix",
            "kernel = (basis * powers",
            "np.exp(-eig * dt)",
            "predicted_coeff = powers * coeff"):
        assert marker not in dense
    for marker in (
            "def _build_sparse_local_kernel",
            "def _build_sparse_fixed_kernel",
            "def _mh_correct_sparse_transition",
            "def _ipfp_correct_sparse_transition",
            "def _sparse_left_multiply"):
        assert marker not in sparse
    assert "jacobi_native.dense_transition" in dense
    assert "jacobi_native.PreparedScarJacobiEvaluator" in dense
    assert "jacobi_native.sparse_transition" in sparse
    assert "jacobi_native.select_sparse_order" in sparse
    assert "numba" not in sparse
    assert "def _sparse_to_dense" not in sparse
    assert "jacobi_native.sparse_to_dense" in sparse


def test_native_transition_memory_budget_fails_before_result_allocation():
    with pytest.raises(MemoryError, match="memory_budget_bytes=1"):
        jacobi_tm.jacobi_transition_matrix(
            1.2,
            0.4,
            0.25,
            n_obs=8,
            basis_order=8,
            quad_order=24,
            memory_budget_bytes=1,
        )


def test_native_memory_failure_diagnostics_preserve_requested_policy():
    module = jacobi_native.load()
    config = jacobi_native._transition_config(
        n_obs=8,
        quad_order=24,
        basis_order=8,
        gh_order=5,
        method="local",
        storage="sparse",
        correction="ipfp",
        memory_budget_bytes=1,
    )
    result = module.jacobi_build_sparse_transition(
        jacobi_native._params(1.2, 0.4, 0.25), config)
    diagnostics = result["diagnostics"]

    assert result["status"] == 2
    assert diagnostics["method_requested"] == 2
    assert diagnostics["method_used"] == 2
    assert diagnostics["storage"] == 1
    assert diagnostics["correction"] == 2


def test_dense_fixed_diagnostics_include_derivative_payload_bytes():
    result = jacobi_native.dense_transition(
        1.2,
        0.4,
        0.25,
        n_obs=17,
        quad_order=24,
        basis_order=1,
        gh_order=5,
        method="local_fixed",
        raw_backend="local_fixed",
        return_grad=True,
    )
    probabilities = result[2]
    derivatives = result[3]
    diagnostics = result[-1]

    assert diagnostics["dense_bytes"] == (
        probabilities.nbytes + derivatives.nbytes)
