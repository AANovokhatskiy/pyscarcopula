from dataclasses import replace

import numpy as np
import pytest

from pyscarcopula import GumbelCopula
from pyscarcopula._types import LatentResult, ou_params
from pyscarcopula.api import fit
from pyscarcopula.numerical.tm_functions import (
    _forward_loglik,
    tm_forward_mixture_h,
    tm_forward_predictive_mean,
    tm_forward_rosenblatt,
    tm_loglik,
)
from pyscarcopula.numerical.hermite_tm import (
    hermite_loglik,
    hermite_loglik_with_grad,
)
from pyscarcopula.numerical.tm_gradient import tm_loglik_with_grad
from pyscarcopula.numerical.tm_grid import TMGrid
from pyscarcopula.numerical.gof_blocks import (
    _forward_block_size,
    forward_block_size,
    iter_forward_weight_blocks,
)
from pyscarcopula.copula.elliptical import BivariateGaussianCopula
from pyscarcopula.copula.experimental import (
    StochasticStudentCopula,
    StochasticStudentDCCCopula,
)
from pyscarcopula.strategy._base import get_strategy_for_result
from pyscarcopula.strategy.scar_tm import SCARTMStrategy
from pyscarcopula.strategy import scar_tm
from pyscarcopula import stattests


def _direct_forward_density(grid, phi):
    """Direct quadrature for phi_next[i] = sum_j p(i|j) phi[j] w[j]."""
    z = grid.z
    means = grid.rho * z
    diff = z[np.newaxis, :] - means[:, np.newaxis]
    p = (
        np.exp(-0.5 * (diff / grid.sigma_cond) ** 2)
        / (grid.sigma_cond * np.sqrt(2.0 * np.pi))
    )
    return p.T @ (phi * grid.trap_w)


def _density_to_mass(grid, phi):
    mass = phi * grid.trap_w
    return mass / np.sum(mass)


def _materialized_forward_predictive_mean(kappa, mu, nu, u, copula, **kwargs):
    grid = TMGrid(kappa, mu, nu, len(u), **kwargs)
    fi_grid = grid.copula_grid(u, copula)
    weights = grid.forward_weights(fi_grid)
    g_grid = copula.transform(grid.z + grid.mu)
    return weights @ g_grid


def _materialized_forward_rosenblatt(kappa, mu, nu, u, copula, **kwargs):
    grid = TMGrid(kappa, mu, nu, len(u), **kwargs)
    fi_grid = grid.copula_grid(u, copula)
    weights = grid.forward_weights(fi_grid)
    r_grid = copula.transform(grid.z + grid.mu)

    e = np.empty((len(u), 2))
    e[:, 0] = u[:, 0]
    for k, row in enumerate(u):
        h_vals = copula.h(
            np.full(grid.K, row[1]),
            np.full(grid.K, row[0]),
            r_grid,
        )
        e[k, 1] = np.sum(h_vals * weights[k])
    return np.clip(e, 1e-6, 1.0 - 1e-6)


def _materialized_forward_mixture_h(kappa, mu, nu, u, copula, **kwargs):
    grid = TMGrid(kappa, mu, nu, len(u), **kwargs)
    fi_grid = grid.copula_grid(u, copula)
    weights = grid.forward_weights(fi_grid)
    r_grid = copula.transform(grid.z + grid.mu)

    h_mix = np.empty(len(u))
    for k, row in enumerate(u):
        h_vals = copula.h(
            np.full(grid.K, row[1]),
            np.full(grid.K, row[0]),
            r_grid,
        )
        h_mix[k] = np.sum(h_vals * weights[k])
    return np.clip(h_mix, 1e-6, 1.0 - 1e-6)


def test_forward_block_size_bounds_memory_and_keeps_compat_alias():
    assert forward_block_size(300) == 512
    assert forward_block_size(10_000) == 200
    assert forward_block_size(10_000_000) == 1
    assert forward_block_size(300, max_elements=1_000, max_rows=10) == 3
    assert forward_block_size(
        100, max_elements=1_000, max_rows=10, element_width=5) == 2
    assert _forward_block_size(10_000) == forward_block_size(10_000)


def test_iter_forward_weight_blocks_matches_materialized_weights():
    rng = np.random.default_rng(20260520)
    u = rng.uniform(0.05, 0.95, size=(7, 2))
    copula = BivariateGaussianCopula()
    grid = TMGrid(
        kappa=0.8,
        mu=0.0,
        nu=0.7,
        n=len(u),
        K=25,
        grid_range=3.0,
        adaptive=False,
    )
    fi_grid = grid.copula_grid(u, copula)
    weights_ref = grid.forward_weights(fi_grid)

    seen_weights = []
    seen_emissions = []
    n_blocks = 0
    for k, local, weights, fi_block in iter_forward_weight_blocks(
            grid, u, copula, block_size=3):
        assert fi_block.shape[0] <= 3
        assert fi_block.shape[1] == grid.K
        if local == 0:
            n_blocks += 1
        seen_weights.append(weights)
        seen_emissions.append(fi_block[local].copy())

    np.testing.assert_allclose(np.asarray(seen_weights), weights_ref)
    np.testing.assert_allclose(np.asarray(seen_emissions), fi_grid)
    assert n_blocks == 3


def test_local_gh_transition_rows_sum_to_one():
    grid = TMGrid(
        kappa=0.5,
        mu=0.0,
        nu=1.0,
        n=100,
        K=80,
        adaptive=False,
        transition_method="gh",
        gh_order=5,
    )

    row_sums = np.asarray(grid._T_op.sum(axis=1)).ravel()

    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-12, atol=1e-12)
    assert grid.diagnostics()["grid_method"] == "local"
    assert grid.diagnostics()["transition_method"] == "gh"


def test_local_transition_predict_matvec_preserves_mass():
    grid = TMGrid(
        kappa=0.5,
        mu=0.0,
        nu=1.0,
        n=100,
        K=80,
        adaptive=False,
        transition_method="gh",
        gh_order=5,
    )
    phi = np.ones(grid.K)
    phi /= np.sum(phi * grid.trap_w)

    next_phi = grid.predict_matvec(phi * grid.trap_w)

    np.testing.assert_allclose(
        np.sum(next_phi * grid.trap_w),
        1.0,
        rtol=1e-12,
        atol=1e-12,
    )


def test_max_k_caps_adaptive_grid_and_auto_selects_local_transition():
    grid = TMGrid(
        kappa=0.05,
        mu=0.0,
        nu=1.0,
        n=12000,
        K=300,
        max_K=500,
        transition_method="auto",
    )
    diag = grid.diagnostics()

    assert diag["K"] == 500
    assert diag["K_adaptive"] > diag["K"]
    assert diag["adaptive_was_capped"] is True
    assert diag["grid_method"] == "local"
    assert diag["transition_method"] == "gh"


def test_auto_transition_uses_gh_for_very_narrow_kernel():
    grid = TMGrid(
        kappa=0.001,
        mu=0.0,
        nu=1.0,
        n=12000,
        K=300,
        max_K=500,
        transition_method="auto",
        r_gh=3.0,
    )
    diag = grid.diagnostics()

    assert diag["r_kernel_grid"] < diag["r_gh"]
    assert diag["transition_method"] == "gh"


def test_matrix_transition_keeps_sparse_dense_backend_selection():
    sparse_grid = TMGrid(
        kappa=1.0,
        mu=0.0,
        nu=1.0,
        n=20,
        K=20,
        grid_method="sparse",
        transition_method="matrix",
    )
    dense_grid = TMGrid(
        kappa=1.0,
        mu=0.0,
        nu=1.0,
        n=20,
        K=20,
        grid_method="dense",
        transition_method="matrix",
    )

    assert sparse_grid.diagnostics()["grid_method"] == "sparse"
    assert dense_grid.diagnostics()["grid_method"] == "dense"
    assert sparse_grid.diagnostics()["transition_method"] == "matrix"
    assert dense_grid.diagnostics()["transition_method"] == "matrix"


def test_local_transition_loglik_is_finite():
    rng = np.random.default_rng(0)
    u = rng.uniform(0.05, 0.95, size=(40, 2))
    copula = BivariateGaussianCopula()

    value = tm_loglik(
        1.0,
        0.0,
        1.0,
        u,
        copula,
        K=40,
        adaptive=False,
        transition_method="gh",
    )

    assert np.isfinite(value)


def test_gaussian_copula_batch_grid_matches_scalar_path():
    rng = np.random.default_rng(3)
    u = rng.uniform(0.05, 0.95, size=(7, 2))
    x_grid = np.linspace(-3.0, 3.0, 11)
    copula = BivariateGaussianCopula()

    fi_batch, dfi_batch = copula.pdf_and_grad_on_grid_batch(u, x_grid)
    fi_grid = copula.copula_grid_batch(u, x_grid)

    fi_expected = np.empty_like(fi_batch)
    dfi_expected = np.empty_like(dfi_batch)
    for idx, row in enumerate(u):
        fi_expected[idx], dfi_expected[idx] = copula.pdf_and_grad_on_grid(
            row, x_grid)

    np.testing.assert_allclose(fi_batch, fi_expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dfi_batch, dfi_expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fi_grid, fi_expected, rtol=1e-12, atol=1e-12)


def test_scar_tm_local_transition_keeps_analytical_gradient_enabled():
    assert SCARTMStrategy().analytical_grad is True
    assert SCARTMStrategy(transition_method="auto").analytical_grad is True
    assert SCARTMStrategy(max_K=100).analytical_grad is True


def test_scar_tm_strategy_defaults_to_auto_likelihood_with_auto_grid():
    strategy = SCARTMStrategy()
    assert strategy.transition_method == "auto"
    assert strategy.max_K == 1000
    assert strategy._tm_kwargs()["transition_method"] == "auto"
    assert strategy._tm_kwargs()["max_K"] == 1000
    assert strategy._grid_tm_kwargs()["transition_method"] == "auto"
    assert strategy._grid_tm_kwargs()["max_K"] == 1000

    matrix_strategy = SCARTMStrategy(transition_method="matrix", max_K=None)
    assert matrix_strategy._tm_kwargs() == {}
    assert matrix_strategy._grid_tm_kwargs() == {}

    spectral_strategy = SCARTMStrategy(transition_method="spectral")
    assert spectral_strategy.transition_method == "spectral"
    assert spectral_strategy._grid_tm_kwargs()["transition_method"] == "auto"


def test_scar_tm_strategy_recovered_from_legacy_result_uses_matrix_path():
    result = LatentResult(
        log_likelihood=0.0,
        method="SCAR-TM-OU",
        copula_name="BivariateGaussian",
        success=True,
        params=ou_params(0.8, 0.1, 1.0),
        K=41,
        grid_range=3.0,
        pts_per_sigma=2,
    )

    strategy = get_strategy_for_result(result)

    assert strategy.K == 41
    assert strategy.grid_range == 3.0
    assert strategy.pts_per_sigma == 2
    assert strategy.transition_method == "matrix"
    assert strategy.max_K is None
    assert strategy._tm_kwargs() == {}


def test_scar_tm_strategy_recovered_from_result_preserves_explicit_no_cap():
    result = LatentResult(
        log_likelihood=0.0,
        method="SCAR-TM-OU",
        copula_name="BivariateGaussian",
        success=True,
        params=ou_params(0.8, 0.1, 1.0),
        K=41,
        grid_range=3.0,
        pts_per_sigma=2,
        transition_method="auto",
        max_K=None,
    )

    strategy = get_strategy_for_result(result)
    overridden = get_strategy_for_result(
        replace(result, max_K=500),
        max_K=None,
    )

    assert strategy.transition_method == "auto"
    assert strategy.max_K is None
    assert strategy._tm_kwargs()["max_K"] is None
    assert overridden.transition_method == "auto"
    assert overridden.max_K is None


def test_local_transition_analytical_gradient_matches_finite_difference():
    rng = np.random.default_rng(1)
    u = rng.uniform(0.1, 0.9, size=(20, 2))
    copula = BivariateGaussianCopula()
    alpha = np.array([0.8, 0.1, 1.2])
    kwargs = {
        "K": 45,
        "grid_range": 4.0,
        "adaptive": False,
        "transition_method": "gh",
        "gh_order": 5,
    }

    value, grad = tm_loglik_with_grad(*alpha, u, copula, **kwargs)
    direct = tm_loglik(*alpha, u, copula, **kwargs)

    np.testing.assert_allclose(value, direct, rtol=1e-11, atol=1e-11)

    fd = np.empty(3)
    steps = np.array([1e-5, 1e-5, 1e-5])
    for idx, step in enumerate(steps):
        plus = alpha.copy()
        minus = alpha.copy()
        plus[idx] += step
        minus[idx] -= step
        fd[idx] = (
            tm_loglik(*plus, u, copula, **kwargs)
            - tm_loglik(*minus, u, copula, **kwargs)
        ) / (2.0 * step)

    np.testing.assert_allclose(grad, fd, rtol=2e-4, atol=2e-4)


def test_spectral_analytical_gradient_matches_finite_difference_on_weak_data():
    rng = np.random.default_rng(42)
    u = rng.uniform(0.05, 0.95, size=(30, 2))
    copula = BivariateGaussianCopula()
    alpha = np.array([5.0, 0.0, 0.6])
    kwargs = {"basis_order": 16}

    value, grad = hermite_loglik_with_grad(*alpha, u, copula, **kwargs)
    direct = -hermite_loglik(*alpha, u, copula, **kwargs)

    np.testing.assert_allclose(value, direct, rtol=1e-11, atol=1e-11)

    fd = np.empty(3)
    steps = np.array([1e-5, 1e-5, 1e-5])
    for idx, step in enumerate(steps):
        plus = alpha.copy()
        minus = alpha.copy()
        plus[idx] += step
        minus[idx] -= step
        fd[idx] = (
            -hermite_loglik(*plus, u, copula, **kwargs)
            + hermite_loglik(*minus, u, copula, **kwargs)
        ) / (2.0 * step)

    np.testing.assert_allclose(grad, fd, rtol=2e-5, atol=2e-5)


def test_spectral_batch_calls_preserve_time_index():
    class IndexedCopula:
        name = "IndexedCopula"

        def __init__(self):
            self.value_calls = []
            self.grad_calls = []

        def copula_grid_batch(self, u, x_grid, t_index=0):
            self.value_calls.append((int(t_index), len(u)))
            return np.ones((len(u), len(x_grid)))

        def pdf_and_grad_on_grid_batch(self, u, x_grid, t_index=0):
            self.grad_calls.append((int(t_index), len(u)))
            fi = np.ones((len(u), len(x_grid)))
            return fi, np.zeros_like(fi)

    u = np.full((7, 2), 0.5)
    copula = IndexedCopula()
    kwargs = {"basis_order": 4, "quad_order": 24, "block_size": 2}

    assert hermite_loglik(2.0, 0.0, 0.5, u, copula, **kwargs) == pytest.approx(0.0)
    value, grad = hermite_loglik_with_grad(2.0, 0.0, 0.5, u, copula, **kwargs)

    assert value == pytest.approx(-0.0)
    np.testing.assert_allclose(grad, np.zeros(3), atol=1e-14)
    assert copula.value_calls == [(5, 2), (3, 2), (1, 2), (0, 1)]
    assert copula.grad_calls == [(5, 2), (3, 2), (1, 2), (0, 1)]


def test_scar_tm_failed_dispatch_objective_is_not_success(monkeypatch):
    def fail_objective(*args, **kwargs):
        return 1e10, np.zeros(3)

    monkeypatch.setattr(scar_tm, "auto_neg_loglik_with_grad", fail_objective)
    u = np.random.default_rng(17).uniform(0.05, 0.95, size=(10, 2))
    copula = BivariateGaussianCopula()

    result = SCARTMStrategy(smart_init=False).fit(
        copula,
        u,
        alpha0=np.array([1.0, 0.0, 1.0]),
        maxiter=2,
        maxfun=5,
    )

    assert not result.success
    assert result.log_likelihood == pytest.approx(-1e10)
    assert "invalid objective value" in result.message


def test_experimental_ppf_cache_does_not_trust_reused_object_id():
    for model in (StochasticStudentCopula(d=2), StochasticStudentDCCCopula(d=2)):
        first = np.full((2, 2), 0.25)
        second = np.full((2, 2), 0.75)
        sentinel = object()

        model._ppf_table = sentinel
        model._ppf_table_u = first
        model._ppf_table_u_id = id(second)

        table = model._get_ppf_table(second)

        assert table is not sentinel
        assert model._ppf_table_u is second
        assert model._ppf_table_u_id == id(second)


def test_scar_tm_accepts_flat_boundary_line_search_convergence():
    u = np.random.default_rng(16).uniform(0.001, 0.999, size=(60, 2))
    copula = BivariateGaussianCopula()

    result = fit(
        copula,
        u,
        method="scar-tm-ou",
        spectral_basis_order=16,
        maxiter=80,
        maxfun=160,
        gtol=1e-5,
    )

    assert result.success
    assert "accepted as boundary convergence" in result.message
    assert result.params.nu == pytest.approx(0.001)


def test_predict_matvec_matches_direct_forward_quadrature():
    grid = TMGrid(
        kappa=3.0,
        mu=0.4,
        nu=1.2,
        n=25,
        K=15,
        grid_range=2.0,
        grid_method="dense",
        adaptive=False,
    )
    phi = grid.p0 * (1.0 + 0.25 * grid.z / grid.sigma)

    direct = _direct_forward_density(grid, phi)
    via_grid = grid.predict_matvec(phi * grid.trap_w)

    np.testing.assert_allclose(via_grid, direct, rtol=1e-12, atol=1e-12)


def test_forward_weights_are_predictive_before_current_observation():
    grid = TMGrid(
        kappa=2.0,
        mu=0.0,
        nu=1.0,
        n=5,
        K=11,
        grid_range=1.8,
        grid_method="dense",
        adaptive=False,
    )
    fi = np.ones((5, grid.K))

    weights = grid.forward_weights(fi)
    expected_0 = _density_to_mass(grid, grid.p0)
    expected_1 = _density_to_mass(grid, _direct_forward_density(grid, grid.p0))

    np.testing.assert_allclose(weights[0], expected_0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(weights[1], expected_1, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("transition_method", ["matrix", "gh"])
def test_streaming_forward_gof_matches_materialized_path(transition_method):
    rng = np.random.default_rng(4)
    u = rng.uniform(0.05, 0.95, size=(12, 2))
    copula = BivariateGaussianCopula()
    kwargs = {
        "K": 25,
        "grid_range": 3.0,
        "grid_method": "dense",
        "adaptive": False,
        "transition_method": transition_method,
        "gh_order": 5,
    }
    params = (0.7, 0.2, 1.1, u, copula)

    np.testing.assert_allclose(
        tm_forward_predictive_mean(*params, **kwargs),
        _materialized_forward_predictive_mean(*params, **kwargs),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        tm_forward_rosenblatt(*params, **kwargs),
        _materialized_forward_rosenblatt(*params, **kwargs),
        rtol=1e-9,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        tm_forward_mixture_h(*params, **kwargs),
        _materialized_forward_mixture_h(*params, **kwargs),
        rtol=1e-9,
        atol=1e-9,
    )


def test_streaming_forward_gof_does_not_materialize_forward_weights(monkeypatch):
    rng = np.random.default_rng(5)
    u = rng.uniform(0.05, 0.95, size=(8, 2))
    copula = BivariateGaussianCopula()
    kwargs = {
        "K": 20,
        "grid_range": 3.0,
        "grid_method": "dense",
        "adaptive": False,
        "transition_method": "matrix",
    }

    def fail_forward_weights(self, fi_grid):
        raise AssertionError("forward_weights should not be called")

    monkeypatch.setattr(TMGrid, "forward_weights", fail_forward_weights)

    tm_forward_predictive_mean(0.8, 0.1, 1.0, u, copula, **kwargs)
    tm_forward_rosenblatt(0.8, 0.1, 1.0, u, copula, **kwargs)
    tm_forward_mixture_h(0.8, 0.1, 1.0, u, copula, **kwargs)


def test_bivariate_gof_passes_stored_transition_options(monkeypatch):
    rng = np.random.default_rng(6)
    u = rng.uniform(0.05, 0.95, size=(6, 2))
    copula = BivariateGaussianCopula()
    result = LatentResult(
        log_likelihood=0.0,
        method="SCAR-TM-OU",
        copula_name=copula.name,
        success=True,
        params=ou_params(0.8, 0.1, 1.0),
        transition_method="auto",
        max_K=123,
        r_gh=2.5,
        gh_order=7,
        pts_per_sigma=6,
    )
    captured = {}

    def fake_rosenblatt(kappa, mu, nu, u_arg, copula_arg, K=300,
                        grid_range=5.0, grid_method="auto", adaptive=True,
                        pts_per_sigma=4, **kwargs):
        captured.update(kwargs)
        captured["pts_per_sigma"] = pts_per_sigma
        return np.column_stack((u_arg[:, 0], u_arg[:, 1]))

    monkeypatch.setattr(
        "pyscarcopula.numerical.tm_functions.tm_forward_rosenblatt",
        fake_rosenblatt,
    )

    stattests._bivariate_rosenblatt_from_result(copula, u, result)

    assert captured["transition_method"] == "auto"
    assert captured["max_K"] == 123
    assert captured["r_gh"] == 2.5
    assert captured["gh_order"] == 7
    assert captured["pts_per_sigma"] == 6


def test_top_level_api_uses_stored_scar_tm_options(monkeypatch):
    rng = np.random.default_rng(7)
    u = rng.uniform(0.05, 0.95, size=(6, 2))
    copula = BivariateGaussianCopula()
    result = LatentResult(
        log_likelihood=0.0,
        method="SCAR-TM-OU",
        copula_name=copula.name,
        success=True,
        params=ou_params(0.8, 0.1, 1.0),
        K=37,
        grid_range=2.5,
        pts_per_sigma=6,
        transition_method="auto",
        max_K=None,
        r_gh=2.25,
        gh_order=7,
    )
    captured = {}

    def fake_predictive_mean(kappa, mu, nu, u_arg, copula_arg, K=300,
                             grid_range=5.0, grid_method="auto",
                             adaptive=True, pts_per_sigma=4, **kwargs):
        captured["K"] = K
        captured["grid_range"] = grid_range
        captured["pts_per_sigma"] = pts_per_sigma
        captured.update(kwargs)
        return np.zeros(len(u_arg), dtype=np.float64)

    monkeypatch.setattr(
        "pyscarcopula.strategy.scar_tm.tm_forward_predictive_mean",
        fake_predictive_mean,
    )

    from pyscarcopula import api

    out = api.predictive_mean(copula, u, result)

    np.testing.assert_allclose(out, 0.0)
    assert captured["K"] == 37
    assert captured["grid_range"] == 2.5
    assert captured["pts_per_sigma"] == 6
    assert captured["transition_method"] == "auto"
    assert captured["max_K"] is None
    assert captured["r_gh"] == 2.25
    assert captured["gh_order"] == 7


def test_forward_loglik_matches_backward_pass_on_notebook_dataset(crypto_data):
    """Use the BTC/ETH dataset from examples/02_bivariate.ipynb."""
    copula = GumbelCopula(rotate=180)

    params = {
        "kappa": 59.02,
        "mu": 2.17,
        "nu": 15.80,
        "K": 60,
        "grid_range": 5.0,
        "grid_method": "auto",
        "adaptive": True,
        "pts_per_sigma": 4,
    }

    backward_loglik = -tm_loglik(**params, u=crypto_data, copula=copula)
    forward_loglik = _forward_loglik(**params, u=crypto_data, copula=copula)

    np.testing.assert_allclose(
        forward_loglik,
        backward_loglik,
        rtol=1e-10,
        atol=1e-8,
    )


def test_fit_path_matches_with_forward_and_backward_objectives(
        crypto_data, monkeypatch):
    """Run two independent SCAR-TM fits on examples/02_bivariate.ipynb data."""
    copula = GumbelCopula(rotate=180)
    original_tm_loglik = scar_tm.tm_loglik

    fit_kwargs = {
        "alpha0": np.array([59.02, 2.0, 15.0]),
        "smart_init": False,
        "analytical_grad": False,
        "K": 60,
        "grid_range": 5.0,
        "grid_method": "dense",
        "adaptive": False,
        "pts_per_sigma": 4,
        "transition_method": "matrix",
        "max_K": None,
        "maxiter": 5,
        "maxfun": 40,
        "gtol": 1e-3,
        "eps": 1e-4,
    }

    backward_calls = []
    forward_calls = []

    def backward_objective(kappa, mu, nu, u, cop, K=300, grid_range=5.0,
                           grid_method="auto", adaptive=True,
                           pts_per_sigma=4):
        alpha = np.array([kappa, mu, nu], dtype=np.float64)
        value = original_tm_loglik(
            kappa, mu, nu, u, cop, K, grid_range,
            grid_method, adaptive, pts_per_sigma)
        backward_calls.append((alpha, value))
        return value

    def forward_objective(kappa, mu, nu, u, cop, K=300, grid_range=5.0,
                          grid_method="auto", adaptive=True,
                          pts_per_sigma=4):
        alpha = np.array([kappa, mu, nu], dtype=np.float64)
        value = -_forward_loglik(
            kappa, mu, nu, u, cop, K, grid_range,
            grid_method, adaptive, pts_per_sigma)
        forward_calls.append((alpha, value))
        return value

    monkeypatch.setattr(scar_tm, "tm_loglik", backward_objective)
    backward_fit = fit(copula, crypto_data, method="scar-tm-ou", **fit_kwargs)

    monkeypatch.setattr(scar_tm, "tm_loglik", forward_objective)
    forward_fit = fit(copula, crypto_data, method="scar-tm-ou", **fit_kwargs)

    assert backward_fit.success
    assert forward_fit.success
    assert len(backward_calls) == len(forward_calls)

    backward_path = np.array([alpha for alpha, _ in backward_calls])
    forward_path = np.array([alpha for alpha, _ in forward_calls])
    backward_values = np.array([value for _, value in backward_calls])
    forward_values = np.array([value for _, value in forward_calls])

    np.testing.assert_allclose(
        forward_path,
        backward_path,
        rtol=1e-7,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        forward_values,
        backward_values,
        rtol=1e-9,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        forward_fit.params.values,
        backward_fit.params.values,
        rtol=1e-7,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        forward_fit.log_likelihood,
        backward_fit.log_likelihood,
        rtol=1e-9,
        atol=1e-6,
    )
