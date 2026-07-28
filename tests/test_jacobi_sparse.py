"""Sparse local Jacobi transition contracts."""

import numpy as np
import pytest

from scipy.stats import norm

from pyscarcopula import GumbelCopula, VineCopula
from pyscarcopula._types import LatentResult, jacobi_params
from pyscarcopula.numerical.jacobi_sparse import (
    SparseJacobiTransition,
    compare_sparse_jacobi_corrections,
    jacobi_sparse_fixed_grid_transition,
    jacobi_sparse_matrix_forward_mixture_h,
    jacobi_sparse_matrix_forward_mixture_h_pair,
    jacobi_sparse_matrix_forward_predictive_mean,
    jacobi_sparse_matrix_loglik,
    jacobi_sparse_matrix_neg_loglik_with_grad,
    jacobi_sparse_matrix_state_distribution,
    jacobi_sparse_local_transition,
    sample_sparse_jacobi_trajectory,
    select_sparse_jacobi_order,
    sparse_jacobi_full_horizon_diagnostics,
)
from pyscarcopula.numerical.jacobi_tm import (
    jacobi_fixed_grid_transition_matrix,
    jacobi_local_transition_matrix,
    jacobi_matrix_forward_mixture_h,
    jacobi_matrix_forward_mixture_h_pair,
    jacobi_matrix_forward_predictive_mean,
    jacobi_matrix_loglik,
    jacobi_matrix_neg_loglik_with_grad,
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


def _dense_sparse_derivatives(transition, dprobabilities):
    output = np.zeros((3,) + transition.shape, dtype=np.float64)
    for row in range(transition.shape[0]):
        count = int(transition.counts[row])
        output[
            :,
            row,
            transition.indices[row, :count],
        ] = dprobabilities[:, row, :count]
    return output


@pytest.mark.parametrize("quad_order", [32, 48])
def test_sparse_fixed_transition_and_derivatives_match_dense(quad_order):
    dense = jacobi_fixed_grid_transition_matrix(
        1.2,
        0.4,
        0.25,
        n_obs=37,
        quad_order=quad_order,
        gh_order=5,
        return_grad=True,
    )
    sparse = jacobi_sparse_fixed_grid_transition(
        1.2,
        0.4,
        0.25,
        n_obs=37,
        quad_order=quad_order,
        gh_order=5,
        return_grad=True,
    )
    dense_tau, dense_weights, dense_transition, dense_derivatives = dense
    tau, weights, transition, derivatives = sparse

    np.testing.assert_array_equal(tau, dense_tau)
    np.testing.assert_array_equal(weights, dense_weights)
    np.testing.assert_allclose(
        transition.to_dense(), dense_transition, rtol=0.0, atol=4e-15)
    np.testing.assert_allclose(
        _dense_sparse_derivatives(transition, derivatives),
        dense_derivatives,
        rtol=0.0,
        atol=2e-14,
    )


def test_sparse_fixed_gradient_matches_dense_filter(filter_case):
    args = filter_case
    dense = jacobi_matrix_neg_loglik_with_grad(
        *args,
        basis_order=16,
        quad_order=48,
        transition_method="local_fixed",
        gh_order=5,
    )
    sparse = jacobi_sparse_matrix_neg_loglik_with_grad(
        *args,
        basis_order=16,
        quad_order=48,
        transition_method="local_fixed",
        gh_order=5,
    )

    assert sparse[0] == pytest.approx(dense[0], rel=0.0, abs=2e-13)
    np.testing.assert_allclose(
        sparse[1], dense[1], rtol=0.0, atol=2e-11)


def test_sparse_fixed_forward_operations_match_dense(filter_case):
    dense_kwargs = {
        "basis_order": 16,
        "quad_order": 48,
        "transition_method": "local_fixed",
        "gh_order": 5,
    }
    sparse_kwargs = {
        "basis_order": 16,
        "quad_order": 48,
        "transition_method": "local_fixed",
        "gh_order": 5,
    }
    dense_loglik = jacobi_matrix_loglik(
        *filter_case, **dense_kwargs)
    sparse_loglik = jacobi_sparse_matrix_loglik(
        *filter_case, **sparse_kwargs)
    assert sparse_loglik == pytest.approx(
        dense_loglik, rel=0.0, abs=2e-13)

    dense_mean = jacobi_matrix_forward_predictive_mean(
        *filter_case, **dense_kwargs)
    sparse_mean = jacobi_sparse_matrix_forward_predictive_mean(
        *filter_case, **sparse_kwargs)
    np.testing.assert_allclose(
        sparse_mean, dense_mean, rtol=0.0, atol=3e-14)

    dense_pair = jacobi_matrix_forward_mixture_h_pair(
        *filter_case, **dense_kwargs)
    sparse_pair = jacobi_sparse_matrix_forward_mixture_h_pair(
        *filter_case, **sparse_kwargs)
    np.testing.assert_allclose(
        sparse_pair[0], dense_pair[0], rtol=0.0, atol=3e-14)
    np.testing.assert_allclose(
        sparse_pair[1], dense_pair[1], rtol=0.0, atol=3e-14)


def test_sparse_fixed_sampling_matches_dense_fixed_seed():
    common = {
        "basis_order": 16,
        "quad_order": 48,
        "transition_method": "local_fixed",
        "gh_order": 5,
    }
    dense = sample_jacobi_grid_trajectory(
        1.2,
        0.4,
        0.25,
        100,
        rng=np.random.default_rng(12),
        transition_storage="dense",
        **common,
    )
    sparse, diagnostics = sample_jacobi_grid_trajectory(
        1.2,
        0.4,
        0.25,
        100,
        rng=np.random.default_rng(12),
        transition_storage="sparse",
        return_diagnostics=True,
        **common,
    )

    np.testing.assert_array_equal(sparse, dense)
    assert diagnostics["transition_storage"] == "sparse"
    assert diagnostics["transition_method"] == "local_fixed"


def test_sparse_fixed_gradient_storage_reduction():
    _, _, transition, derivatives, diagnostics = (
        jacobi_sparse_fixed_grid_transition(
            1.2,
            0.4,
            0.25,
            n_obs=400,
            quad_order=384,
            gh_order=5,
            return_grad=True,
            return_diagnostics=True,
        )
    )

    assert derivatives.shape == (3, 384, 10)
    assert diagnostics["dense_bytes"] / diagnostics["retained_bytes"] >= 20
    assert transition.nnz <= 3840


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


def test_adaptive_sparse_order_uses_full_horizon_gates():
    tau, weights, transition, report = select_sparse_jacobi_order(
        1.2,
        0.4,
        0.25,
        n_obs=17,
        quad_orders=(48, 128),
    )

    assert report["selected_quad_order"] == 128
    assert report["passed"]
    assert not report["exhausted"]
    assert [item["passed"] for item in report["candidates"]] == [
        False, True]
    assert transition.shape == (128, 128)
    assert tau.shape == weights.shape == (128,)


def test_adaptive_sparse_order_can_return_explicit_exhausted_candidate():
    _, _, transition, report = select_sparse_jacobi_order(
        1.2,
        0.4,
        0.25,
        n_obs=17,
        quad_orders=(48, 80),
        max_full_horizon_tv=0.0,
        max_relative_variance_error=0.0,
        max_conditional_mean_rmse=0.0,
        max_lag_one_correlation_error=0.0,
    )

    assert transition.shape == (80, 80)
    assert report["selected_quad_order"] == 80
    assert not report["passed"]
    assert report["exhausted"]


def test_adaptive_sparse_order_require_pass_fails_explicitly():
    with pytest.raises(RuntimeError, match="full-horizon gates"):
        select_sparse_jacobi_order(
            1.2,
            0.4,
            0.25,
            n_obs=17,
            quad_orders=(48,),
            max_full_horizon_tv=0.0,
            require_pass=True,
        )


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


def test_ipfp_correction_preserves_stationarity_on_feasible_support():
    tau, weights, corrected, diagnostics = (
        jacobi_sparse_local_transition(
            1.2,
            0.4,
            0.25,
            n_obs=33,
            quad_order=48,
            gh_order=5,
            correction="ipfp",
            return_diagnostics=True,
        )
    )
    horizon = sparse_jacobi_full_horizon_diagnostics(
        tau, weights, corrected, steps=32, kappa=1.2, m=0.4)

    assert diagnostics["ipfp_iterations"] > 0
    assert diagnostics["ipfp_stationary_residual"] <= 1e-15
    assert diagnostics["ipfp_kl_divergence"] > 0.0
    assert diagnostics["ipfp_max_probability_change"] > 0.0
    assert horizon["full_horizon_stationary_tv"] <= 1e-12
    assert horizon["relative_variance_error"] <= 1e-11
    assert np.isfinite(horizon["conditional_mean_rmse"])
    assert np.isfinite(horizon["lag_one_correlation_error"])


def test_ipfp_reports_infeasible_sparse_support():
    with pytest.raises(
            FloatingPointError, match="cannot reach every stationary node"):
        jacobi_sparse_local_transition(
            1.2,
            0.4,
            0.25,
            n_obs=17,
            quad_order=128,
            correction="ipfp",
        )


def test_correction_comparison_records_support_failure_and_higher_k():
    records = compare_sparse_jacobi_corrections(
        1.2,
        0.4,
        0.25,
        n_obs=17,
        quad_orders=(48, 128),
    )
    keyed = {
        (record["quad_order"], record["correction"]): record
        for record in records
    }

    assert keyed[(48, "none")]["status"] == "ok"
    assert keyed[(48, "mh")]["status"] == "ok"
    assert keyed[(48, "ipfp")]["status"] == "ok"
    assert keyed[(128, "none")]["status"] == "ok"
    assert keyed[(128, "ipfp")]["status"] == "unsupported"
    assert (
        keyed[(128, "none")]["full_horizon_stationary_tv"]
        < keyed[(48, "none")]["full_horizon_stationary_tv"])


def test_ipfp_is_used_end_to_end_by_sparse_strategy():
    u = np.random.default_rng(25).uniform(0.05, 0.95, size=(33, 2))
    copula = GumbelCopula()
    strategy = get_strategy_for_result(LatentResult(
        log_likelihood=0.0,
        method="SCAR-TM-JACOBI",
        copula_name="Gumbel copula",
        success=True,
        params=jacobi_params(1.2, 0.4, 0.25),
        transition_method="local",
        transition_storage="sparse",
        stationarity_correction="ipfp",
        gh_order=5,
        spectral_basis_order=8,
        spectral_quad_order=48,
    ))

    assert np.isfinite(strategy._loglik(1.2, 0.4, 0.25, u, copula))
    diagnostics = {}
    sampled = strategy.model_sample_params(
        copula,
        LatentResult(
            log_likelihood=0.0,
            method="SCAR-TM-JACOBI",
            copula_name="Gumbel copula",
            success=True,
            params=jacobi_params(1.2, 0.4, 0.25),
            transition_method="local",
            transition_storage="sparse",
            stationarity_correction="ipfp",
            gh_order=5,
            spectral_basis_order=8,
            spectral_quad_order=48,
        ),
        33,
        rng=np.random.default_rng(26),
        sampling_diagnostics=diagnostics,
    )

    assert sampled.shape == (33,)
    assert diagnostics["correction"] == "ipfp"
    assert diagnostics["ipfp_stationary_residual"] <= 1e-15


def test_ipfp_integrated_sampling_matches_prepared_sparse_transition():
    tau, weights, transition = jacobi_sparse_local_transition(
        1.2,
        0.4,
        0.25,
        n_obs=33,
        quad_order=48,
        gh_order=5,
        correction="ipfp",
    )
    direct = sample_sparse_jacobi_trajectory(
        tau,
        weights,
        transition,
        33,
        rng=np.random.default_rng(27),
    )
    integrated = sample_jacobi_grid_trajectory(
        1.2,
        0.4,
        0.25,
        33,
        rng=np.random.default_rng(27),
        basis_order=8,
        quad_order=48,
        transition_method="local",
        transition_storage="sparse",
        stationarity_correction="ipfp",
        gh_order=5,
    )

    np.testing.assert_array_equal(integrated, direct)


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


@pytest.mark.parametrize(
    "factory",
    [
        lambda: VineCopula.cvine(3, candidates=[GumbelCopula]),
        lambda: VineCopula.rvine(candidates=[GumbelCopula]),
    ],
    ids=["cvine", "rvine"],
)
def test_generic_vines_fit_and_sample_sparse_jacobi_edges(factory):
    rng = np.random.default_rng(23)
    latent = 0.8 * rng.normal(size=(30, 1))
    data = norm.cdf(latent + 0.6 * rng.normal(size=(30, 3)))
    vine = factory().fit(
        data,
        method="scar-tm-jacobi",
        alpha0=np.array([1.2, 0.4, 0.25]),
        basis_order=3,
        quad_order=16,
        transition_method="local",
        transition_storage="sparse",
        smart_init=False,
        maxiter=1,
        maxfun=8,
    )
    sampled = vine.sample(12, rng=np.random.default_rng(24))

    assert sampled.shape == (12, 3)
    assert np.all(np.isfinite(sampled))
    assert np.all((sampled >= 0.0) & (sampled <= 1.0))


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
