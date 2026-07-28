"""Sparse local Jacobi transition contracts."""

import numpy as np
import pytest

from pyscarcopula import GumbelCopula
from pyscarcopula._types import LatentResult, jacobi_params
from pyscarcopula.numerical.jacobi_sparse import (
    SparseJacobiTransition,
    jacobi_sparse_matrix_forward_mixture_h,
    jacobi_sparse_matrix_forward_mixture_h_pair,
    jacobi_sparse_matrix_forward_predictive_mean,
    jacobi_sparse_matrix_loglik,
    jacobi_sparse_matrix_state_distribution,
    jacobi_sparse_local_transition,
    sample_sparse_jacobi_trajectory,
    sparse_jacobi_full_horizon_diagnostics,
)
from pyscarcopula.numerical.jacobi_tm import (
    jacobi_local_transition_matrix,
    jacobi_matrix_forward_mixture_h,
    jacobi_matrix_forward_mixture_h_pair,
    jacobi_matrix_forward_predictive_mean,
    jacobi_matrix_loglik,
    jacobi_matrix_state_distribution,
    sample_jacobi_grid_trajectory,
)
from pyscarcopula.strategy._base import get_strategy_for_result
from pyscarcopula.vine._edge_adapter import sample_r_path


@pytest.mark.parametrize(
    ("quad_order", "gh_order"),
    [(32, 5), (48, 5), (48, 9)],
)
def test_sparse_local_transition_reconstructs_dense_reference(
        quad_order, gh_order):
    dense_tau, dense_weights, dense = jacobi_local_transition_matrix(
        1.2,
        0.4,
        0.25,
        n_obs=33,
        quad_order=quad_order,
        gh_order=gh_order,
    )
    tau, weights, sparse, diagnostics = jacobi_sparse_local_transition(
        1.2,
        0.4,
        0.25,
        n_obs=33,
        quad_order=quad_order,
        gh_order=gh_order,
        return_diagnostics=True,
    )

    np.testing.assert_array_equal(tau, dense_tau)
    np.testing.assert_array_equal(weights, dense_weights)
    np.testing.assert_allclose(
        sparse.to_dense(), dense, rtol=0.0, atol=2e-16)
    assert sparse.nnz <= 2 * gh_order * quad_order
    assert diagnostics["max_width"] == 2 * gh_order


def test_sparse_left_multiply_matches_dense_operator():
    _, _, sparse = jacobi_sparse_local_transition(
        1.2, 0.4, 0.25, n_obs=80, quad_order=48, gh_order=5)
    dense = sparse.to_dense()
    vector = np.random.default_rng(1).random(48)
    vector /= vector.sum()

    actual = sparse.left_multiply(vector)
    expected = vector @ dense

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=3e-17)


@pytest.fixture
def filter_case():
    u = np.random.default_rng(11).uniform(0.05, 0.95, size=(37, 2))
    return (1.2, 0.4, 0.25, u, GumbelCopula())


def _dense_filter_kwargs():
    return {
        "basis_order": 16,
        "quad_order": 48,
        "transition_method": "local",
        "gh_order": 5,
    }


def _sparse_filter_kwargs():
    return {
        "basis_order": 16,
        "quad_order": 48,
        "gh_order": 5,
    }


def test_sparse_loglik_matches_dense_local_filter(filter_case):
    dense = jacobi_matrix_loglik(
        *filter_case, **_dense_filter_kwargs())
    sparse = jacobi_sparse_matrix_loglik(
        *filter_case, **_sparse_filter_kwargs())

    assert sparse == pytest.approx(dense, rel=0.0, abs=2e-13)


def test_sparse_predictive_mean_matches_dense_local_filter(filter_case):
    dense = jacobi_matrix_forward_predictive_mean(
        *filter_case, **_dense_filter_kwargs())
    sparse = jacobi_sparse_matrix_forward_predictive_mean(
        *filter_case, **_sparse_filter_kwargs())

    np.testing.assert_allclose(sparse, dense, rtol=0.0, atol=3e-14)


def test_sparse_mixture_h_matches_dense_local_filter(filter_case):
    dense = jacobi_matrix_forward_mixture_h(
        *filter_case, **_dense_filter_kwargs())
    sparse = jacobi_sparse_matrix_forward_mixture_h(
        *filter_case, **_sparse_filter_kwargs())

    np.testing.assert_allclose(sparse, dense, rtol=0.0, atol=3e-14)


def test_sparse_mixture_h_pair_matches_dense_local_filter(filter_case):
    dense = jacobi_matrix_forward_mixture_h_pair(
        *filter_case, **_dense_filter_kwargs())
    sparse = jacobi_sparse_matrix_forward_mixture_h_pair(
        *filter_case, **_sparse_filter_kwargs())

    np.testing.assert_allclose(
        sparse[0], dense[0], rtol=0.0, atol=3e-14)
    np.testing.assert_allclose(
        sparse[1], dense[1], rtol=0.0, atol=3e-14)


@pytest.mark.parametrize("horizon", ["current", "next"])
def test_sparse_state_distribution_matches_dense_local_filter(
        filter_case, horizon):
    dense_tau, dense_probability = jacobi_matrix_state_distribution(
        *filter_case, **_dense_filter_kwargs(), horizon=horizon)
    sparse_tau, sparse_probability = jacobi_sparse_matrix_state_distribution(
        *filter_case, **_sparse_filter_kwargs(), horizon=horizon)

    np.testing.assert_array_equal(sparse_tau, dense_tau)
    np.testing.assert_allclose(
        sparse_probability, dense_probability, rtol=0.0, atol=3e-14)


def test_explicit_local_sampling_uses_sparse_storage_and_is_exact():
    args = (1.2, 0.4, 0.25)
    tau, weights, sparse = jacobi_sparse_local_transition(
        *args, n_obs=100, quad_order=48, gh_order=5)
    direct = sample_sparse_jacobi_trajectory(
        tau, weights, sparse, 100, rng=np.random.default_rng(2))
    integrated, diagnostics = sample_jacobi_grid_trajectory(
        *args,
        100,
        rng=np.random.default_rng(2),
        basis_order=16,
        quad_order=48,
        transition_method="local",
        gh_order=5,
        return_diagnostics=True,
    )

    np.testing.assert_array_equal(integrated, direct)
    assert diagnostics["transition_storage"] == "sparse"
    assert diagnostics["transition_method"] == "local"


def test_sparse_sampling_matches_legacy_dense_cdf_path():
    args = (1.2, 0.4, 0.25)
    tau, weights, sparse = jacobi_sparse_local_transition(
        *args, n_obs=100, quad_order=48, gh_order=5)
    sparse_path = sample_sparse_jacobi_trajectory(
        tau, weights, sparse, 100, rng=np.random.default_rng(3))

    dense_tau, dense_weights, dense = jacobi_local_transition_matrix(
        *args, n_obs=100, quad_order=48, gh_order=5)
    stationary_cdf = np.cumsum(dense_weights)
    stationary_cdf[-1] = 1.0
    dense_cdf = np.cumsum(dense, axis=1)
    dense_cdf[:, -1] = 1.0
    uniforms = np.random.default_rng(3).random(100)
    index = int(np.searchsorted(
        stationary_cdf, uniforms[0], side="right"))
    dense_path = np.empty(100)
    dense_path[0] = dense_tau[index]
    for observation in range(1, 100):
        index = int(np.searchsorted(
            dense_cdf[index], uniforms[observation], side="right"))
        dense_path[observation] = dense_tau[index]

    np.testing.assert_array_equal(sparse_path, dense_path)


def test_sparse_transition_reduces_retained_memory_at_production_order():
    _, _, sparse, diagnostics = jacobi_sparse_local_transition(
        1.2,
        0.4,
        0.25,
        n_obs=400,
        quad_order=384,
        gh_order=5,
        return_diagnostics=True,
    )

    assert diagnostics["dense_bytes"] / sparse.retained_bytes >= 10.0
    assert sparse.nnz <= 3840


def test_sparse_filter_checks_combined_workspace_before_emissions():
    u = np.full((100, 2), 0.5)
    with pytest.raises(MemoryError, match="memory_budget_bytes"):
        jacobi_sparse_matrix_loglik(
            1.2,
            0.4,
            0.25,
            u,
            GumbelCopula(),
            basis_order=16,
            quad_order=48,
            gh_order=5,
            memory_budget_bytes=10_000,
        )


def test_full_horizon_diagnostics_match_direct_dense_propagation():
    tau, weights, sparse = jacobi_sparse_local_transition(
        1.2, 0.4, 0.25, n_obs=33, quad_order=48, gh_order=5)
    diagnostics = sparse_jacobi_full_horizon_diagnostics(
        tau, weights, sparse, steps=32, kappa=1.2, m=0.4)
    dense = sparse.to_dense()
    propagated = weights.copy()
    for _ in range(32):
        propagated = propagated @ dense
    expected_tv = 0.5 * np.sum(np.abs(propagated - weights))
    expected_mean = np.sum(propagated * tau)
    expected_variance = np.sum(
        propagated * (tau - expected_mean) ** 2)

    assert diagnostics["full_horizon_stationary_tv"] == pytest.approx(
        expected_tv, abs=2e-15)
    assert diagnostics["propagated_mean"] == pytest.approx(
        expected_mean, abs=2e-15)
    assert diagnostics["propagated_variance"] == pytest.approx(
        expected_variance, abs=2e-15)


def test_mh_correction_preserves_stationary_weights_and_detailed_balance():
    tau, weights, corrected, diagnostics = (
        jacobi_sparse_local_transition(
            1.2,
            0.4,
            0.25,
            n_obs=33,
            quad_order=48,
            gh_order=5,
            correction="mh",
            return_diagnostics=True,
        )
    )
    horizon = sparse_jacobi_full_horizon_diagnostics(
        tau, weights, corrected, steps=32, kappa=1.2, m=0.4)

    assert diagnostics["stationary_error"] <= 1e-15
    assert diagnostics["detailed_balance_error"] <= 1e-15
    assert horizon["full_horizon_stationary_tv"] <= 1e-14
    assert 0.0 < diagnostics["acceptance_mass_ratio"] <= 1.0
    assert diagnostics["reverse_missing_edge_fraction"] > 0.0
    assert 0.0 <= diagnostics["min_row_acceptance_ratio"] <= 1.0
    assert 0.0 <= diagnostics["mean_stay_probability"] <= 1.0
    assert 0.0 <= diagnostics["max_stay_probability"] <= 1.0
    assert horizon["conditional_mean_rmse"] > 0.0
    assert np.isfinite(horizon["lag_one_correlation_error"])


def test_mh_correction_is_used_by_integrated_grid_sampling():
    path, diagnostics = sample_jacobi_grid_trajectory(
        1.2,
        0.4,
        0.25,
        50,
        rng=np.random.default_rng(8),
        basis_order=16,
        quad_order=48,
        transition_method="local",
        gh_order=5,
        stationarity_correction="mh",
        return_diagnostics=True,
    )

    assert path.shape == (50,)
    assert diagnostics["correction"] == "mh"
    assert diagnostics["stationary_error"] <= 1e-15


def test_mh_correction_rejects_nonlocal_grid_sampling():
    with pytest.raises(ValueError, match="transition_method='local'"):
        sample_jacobi_grid_trajectory(
            1.2,
            0.4,
            0.25,
            10,
            transition_method="auto",
            stationarity_correction="mh",
        )


def test_vine_edge_adapter_restores_sparse_mh_sampling_semantics():
    result = LatentResult(
        log_likelihood=0.0,
        method="SCAR-TM-JACOBI",
        copula_name="Gumbel copula",
        success=True,
        params=jacobi_params(1.2, 0.4, 0.25),
        transition_method="local",
        transition_storage="sparse",
        stationarity_correction="mh",
        gh_order=5,
        spectral_basis_order=8,
        spectral_quad_order=32,
    )
    copula = GumbelCopula()
    adapter_path = sample_r_path(
        copula, result, 30, rng=np.random.default_rng(9))
    direct_path = get_strategy_for_result(result).model_sample_params(
        copula, result, 30, rng=np.random.default_rng(9))

    np.testing.assert_array_equal(adapter_path, direct_path)


@pytest.mark.parametrize("correction", ["bad", None, 1])
def test_sparse_transition_rejects_unknown_correction(correction):
    with pytest.raises((TypeError, ValueError), match="correction"):
        jacobi_sparse_local_transition(
            1.2,
            0.4,
            0.25,
            n_obs=10,
            quad_order=32,
            correction=correction,
        )


def test_sparse_sampling_memory_failure_precedes_rng_draw():
    tau, weights, sparse = jacobi_sparse_local_transition(
        1.2, 0.4, 0.25, n_obs=10, quad_order=32)
    rng = np.random.default_rng(4)
    with pytest.raises(MemoryError, match="memory_budget_bytes"):
        sample_sparse_jacobi_trajectory(
            tau,
            weights,
            sparse,
            10,
            rng=rng,
            memory_budget_bytes=79,
        )
    np.testing.assert_array_equal(
        rng.random(8), np.random.default_rng(4).random(8))


def test_sparse_transition_type_rejects_unsorted_active_indices():
    with pytest.raises(ValueError, match="strictly increasing"):
        SparseJacobiTransition(
            indices=np.array([[1, 0], [0, 1]]),
            probabilities=np.array([[0.5, 0.5], [0.5, 0.5]]),
            counts=np.array([2, 2]),
        )
