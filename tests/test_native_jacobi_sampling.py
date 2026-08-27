"""Stage 8.3.5 fixed-draw native Jacobi sampling contracts."""

from __future__ import annotations

import inspect

import numpy as np

from pyscarcopula import GumbelCopula, IndependentCopula
from pyscarcopula._native import jacobi as jacobi_native
from pyscarcopula._native._extension import load
from pyscarcopula._types import LatentResult, PredictiveState, jacobi_params
from pyscarcopula.api import sample
from pyscarcopula._native import pair as copula_native
from pyscarcopula.numerical.jacobi_sampling import (
    sample_jacobi_lamperti_trajectory,
)
from pyscarcopula.numerical.jacobi_tm import sample_jacobi_grid_trajectory
from pyscarcopula.strategy.scar_jacobi import SCARJacobiStrategy


def test_native_sampling_symbols_are_exposed():
    module = load()
    for name in (
            "jacobi_sample_grid_trajectory",
            "jacobi_sample_prepared_sparse_trajectory",
            "jacobi_sample_lamperti_chunk",
            "jacobi_sample_state_distribution",
            "jacobi_state_histogram_cells",
            "copula_conditional_sample_from_uniforms"):
        assert hasattr(module, name)


def test_dense_grid_public_rng_contract_matches_fixed_draw_native_call():
    n = 31
    public_rng = np.random.default_rng(8101)
    fixed_rng = np.random.default_rng(8101)
    public, public_diagnostics = sample_jacobi_grid_trajectory(
        1.2,
        0.4,
        0.25,
        n,
        rng=public_rng,
        basis_order=4,
        quad_order=20,
        transition_method="local_fixed",
        return_diagnostics=True,
    )
    fixed, native_diagnostics = (
        jacobi_native.sample_grid_trajectory_fixed_draws(
            1.2,
            0.4,
            0.25,
            fixed_rng.random(n),
            basis_order=4,
            quad_order=20,
            gh_order=5,
            method="local_fixed",
            storage="dense",
        )
    )

    np.testing.assert_array_equal(public, fixed)
    assert "draws_used" not in public_diagnostics
    assert native_diagnostics["draws_used"] == n
    np.testing.assert_array_equal(public_rng.random(8), fixed_rng.random(8))


def test_sparse_grid_fixed_draw_contract_reports_exact_consumption():
    uniforms = np.random.default_rng(8102).random(27)
    path, diagnostics = jacobi_native.sample_grid_trajectory_fixed_draws(
        1.2,
        0.4,
        0.25,
        uniforms,
        basis_order=4,
        quad_order=24,
        gh_order=5,
        method="local",
        storage="sparse",
    )

    assert path.shape == (27,)
    assert diagnostics["draws_used"] == 27
    assert diagnostics["transition_storage"] == "sparse"
    assert np.all((path > 0.0) & (path < 1.0))


def test_lamperti_chunks_report_exact_normal_consumption_and_rng_tail():
    small_rng = np.random.default_rng(8103)
    large_rng = np.random.default_rng(8103)
    common = dict(
        kappa=0.3,
        m=0.4,
        xi=0.6,
        n=29,
        substeps=3,
        boundary="reflect",
        return_diagnostics=True,
    )
    small, small_diagnostics = sample_jacobi_lamperti_trajectory(
        rng=small_rng, chunk_observations=1, **common)
    large, large_diagnostics = sample_jacobi_lamperti_trajectory(
        rng=large_rng, chunk_observations=1000, **common)

    np.testing.assert_array_equal(small, large)
    assert small_diagnostics["euler_steps"] == 28 * 3
    assert large_diagnostics["euler_steps"] == 28 * 3
    assert small_diagnostics["sampling_engine"] == "native"
    np.testing.assert_array_equal(small_rng.random(8), large_rng.random(8))

    initial_y = float(jacobi_native.lamperti(
        np.array([0.4], dtype=np.float64), 0.6)[0])
    fixed = jacobi_native.sample_lamperti_chunk_fixed_draws(
        0.3,
        0.4,
        0.6,
        initial_y,
        np.zeros((2, 3), dtype=np.float64),
        n_obs=29,
        substeps=3,
        boundary="reflect",
        interior_eps=1e-10,
    )
    assert fixed["normal_draws_used"] == 6


def test_predictive_grid_sampling_and_tau_mapping_are_native():
    copula = GumbelCopula()
    strategy = SCARJacobiStrategy(theta_cap=2.0)
    state = PredictiveState(
        method="SCAR-TM-JACOBI",
        horizon="next",
        kind="grid",
        z_grid=np.array([0.2, 0.6], dtype=np.float64),
        prob=np.array([0.25, 0.75], dtype=np.float64),
    )
    actual_rng = np.random.default_rng(8104)
    expected_rng = np.random.default_rng(8104)
    actual = strategy.sample_params(
        copula, state, 20, rng=actual_rng, predictive_r_mode="grid")
    indices = expected_rng.choice(2, size=20, p=state.prob)
    expected = np.minimum(
        copula_native.tau_to_param(copula, state.z_grid[indices]), 2.0)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_rng.random(8), expected_rng.random(8))


def test_predictive_histogram_preserves_rng_calls_with_native_cells():
    copula = GumbelCopula()
    strategy = SCARJacobiStrategy()
    state = PredictiveState(
        method="SCAR-TM-JACOBI",
        horizon="next",
        kind="grid",
        z_grid=np.array([0.2, 0.6], dtype=np.float64),
        prob=np.array([0.25, 0.75], dtype=np.float64),
    )
    actual_rng = np.random.default_rng(8106)
    expected_rng = np.random.default_rng(8106)
    actual = strategy.sample_params(
        copula, state, 20, rng=actual_rng, predictive_r_mode="histogram")
    indices = expected_rng.choice(2, size=20, p=state.prob)
    left, right = jacobi_native.state_histogram_cells(
        state.z_grid, indices)
    tau = expected_rng.uniform(left, right)
    expected = copula_native.tau_to_param(copula, tau)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_rng.random(8), expected_rng.random(8))


def test_jacobi_conditional_sample_uses_native_inverse_h(monkeypatch):
    copula = GumbelCopula()
    result = LatentResult(
        log_likelihood=0.0,
        method="SCAR-TM-JACOBI",
        copula_name=copula.name,
        success=True,
        params=jacobi_params(1.2, 0.4, 0.25),
        transition_method="local_fixed",
        spectral_basis_order=4,
        spectral_quad_order=20,
    )

    def fail(*args, **kwargs):
        raise AssertionError("Python copula inverse-h method was called")

    monkeypatch.setattr(copula, "h_inverse", fail)
    sampled = sample(
        copula,
        np.array([[0.2, 0.7], [0.4, 0.6]], dtype=np.float64),
        result,
        16,
        rng=np.random.default_rng(8105),
        given={0: 0.65},
    )

    np.testing.assert_array_equal(sampled[:, 0], np.full(16, 0.65))
    assert np.all((sampled[:, 1] > 0.0) & (sampled[:, 1] < 1.0))


def test_native_conditional_sample_accepts_numpy_zero_endpoint():
    sampled = copula_native.conditional_sample_from_uniforms(
        IndependentCopula(),
        np.array([0.0, 0.5], dtype=np.float64),
        np.array([0.0, 0.0], dtype=np.float64),
        given_coordinate=0,
        given_value=0.6,
    )

    np.testing.assert_array_equal(
        sampled,
        np.array([[0.6, 0.0], [0.6, 0.5]], dtype=np.float64),
    )


def test_production_python_contains_no_jacobi_trajectory_math_kernels():
    import pyscarcopula.numerical.jacobi_sampling as lamperti_module
    import pyscarcopula.numerical.jacobi_sparse as sparse_module

    lamperti_source = inspect.getsource(lamperti_module)
    sparse_source = inspect.getsource(sparse_module)
    grid_source = inspect.getsource(sample_jacobi_grid_trajectory)
    assert "@njit" not in lamperti_source
    assert "_lamperti_chunk_kernel" not in lamperti_source
    assert "_sample_sparse_path_kernel" not in sparse_source
    assert "np.cumsum(transition" not in grid_source
    assert "jacobi_native.sample_grid_trajectory_fixed_draws" in grid_source
