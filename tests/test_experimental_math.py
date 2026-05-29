"""Mathematical contracts for experimental multivariate copulas."""

import numpy as np
import pytest
from pathlib import Path
from scipy.stats import multivariate_normal, multivariate_t, norm
from scipy.stats import t as t_dist

from pyscarcopula.io import load_model, save_model
from pyscarcopula._utils import pobs
from pyscarcopula._types import LatentResult, ou_params
from pyscarcopula.api import (
    fit,
    log_likelihood,
    mixture_h,
    predict,
    predictive_mean,
    sample,
)
from pyscarcopula.copula.experimental.equicorr import (
    EquicorrGaussianCopula,
    _equicorr_dlog_pdf_drho,
    _equicorr_log_pdf,
)
from pyscarcopula.copula.experimental.stochastic_student import (
    StochasticStudentCopula,
    _log_copula_inlined,
    _student_copula_logpdf as student_logpdf,
)
from pyscarcopula.copula.experimental.stochastic_student_dcc import (
    StochasticStudentDCCCopula,
    _dcc_fast_loglik,
    _dcc_fast_recursion_path_loglik,
    _dcc_loglik_gaussian,
    _dcc_recursion,
    _garch11_filter,
    _student_copula_logpdf as dcc_student_logpdf,
)
from pyscarcopula.numerical.tm_grid import TMGrid
from pyscarcopula.numerical.tm_gradient import tm_loglik_with_grad
from pyscarcopula.numerical.gas_filter import (
    _gas_filter_multivariate,
    _gas_score_multivariate,
    gas_filter,
)
from pyscarcopula.numerical.hermite_tm import hermite_loglik_with_grad
from pyscarcopula.numerical.tm_functions import (
    tm_forward_predictive_mean,
    tm_loglik,
)
from pyscarcopula.numerical.student_emission import (
    student_interp_ppf_rows,
    student_logpdf_and_dlog_rows_fd_numba,
    student_logpdf_from_x_rows,
    student_pdf_and_grad_grid_fd_numba,
    student_pdf_grid_numba,
)
from pyscarcopula.stattests import (
    equicorr_rosenblatt_transform,
    gof_test,
    stochastic_student_dcc_rosenblatt_transform,
    stochastic_student_rosenblatt_transform,
)


def _scar_result(K=12, grid_range=3.0):
    return LatentResult(
        log_likelihood=0.0,
        method="SCAR-TM-OU",
        copula_name="test",
        success=True,
        params=ou_params(0.8, 0.0, 1.0),
        K=K,
        grid_range=grid_range,
    )


def _u():
    rng = np.random.default_rng(20260518)
    return pobs(rng.standard_normal((45, 4)))


def _R():
    return np.array(
        [
            [1.0, 0.35, -0.15, 0.22],
            [0.35, 1.0, 0.18, -0.10],
            [-0.15, 0.18, 1.0, 0.28],
            [0.22, -0.10, 0.28, 1.0],
        ],
        dtype=np.float64,
    )


def _materialized_equicorr_scar_rosenblatt(copula, u, fit_result, K, grid_range):
    eps = 1e-10
    u_c = np.clip(u, eps, 1.0 - eps)
    x_norm = norm.ppf(u_c)
    T, d = u.shape
    kappa, mu, nu = fit_result.params.values
    grid = TMGrid(kappa, mu, nu, T, K, grid_range)
    x_grid = grid.z + grid.mu
    rho_grid = copula.transform(x_grid)
    fi_grid = grid.copula_grid(u, copula)
    weights = grid.forward_weights(fi_grid)

    e = np.empty((T, d))
    e[:, 0] = u[:, 0]
    for k in range(T):
        for i in range(1, d):
            sx = np.sum(x_norm[k, :i])
            denom = 1.0 + (i - 1) * rho_grid
            cond_mean = rho_grid * sx / denom
            cond_var = np.maximum(1.0 - i * rho_grid ** 2 / denom, 1e-10)
            z_i = (x_norm[k, i] - cond_mean) / np.sqrt(cond_var)
            e[k, i] = np.sum(weights[k] * norm.cdf(z_i))
    return np.clip(e, eps, 1.0 - eps)


def _student_scar_static_terms(R, d):
    R_inv_sub = []
    beta_sub = []
    sigma_cond_sub = []
    for i in range(1, d):
        R_11 = R[:i, :i]
        R_21 = R[i, :i]
        R_22 = R[i, i]
        R_11_inv = np.linalg.inv(R_11)
        beta_sub.append(R_21 @ R_11_inv)
        sigma2 = R_22 - R_21 @ R_11_inv @ R_21
        sigma_cond_sub.append(np.sqrt(max(sigma2, 1e-12)))
        R_inv_sub.append(R_11_inv)
    return beta_sub, sigma_cond_sub, R_inv_sub


def _materialized_student_scar_rosenblatt(copula, u, fit_result, K, grid_range):
    eps = 1e-10
    T, d = u.shape
    kappa, mu, nu = fit_result.params.values
    grid = TMGrid(kappa, mu, nu, T, K, grid_range)
    x_grid = grid.z + grid.mu
    df_grid = copula.transform(x_grid)
    fi_grid = grid.copula_grid(u, copula)
    weights = grid.forward_weights(fi_grid)
    beta_sub, sigma_cond_sub, R_inv_sub = _student_scar_static_terms(copula.R, d)
    u_c = np.clip(u, eps, 1.0 - eps)

    e = np.empty((T, d))
    e[:, 0] = u[:, 0]
    for k in range(T):
        x_all = np.empty((grid.K, d), dtype=np.float64)
        for dim in range(d):
            x_all[:, dim] = t_dist.ppf(u_c[k, dim], df=df_grid)
        for i in range(1, d):
            x_prev = x_all[:, :i]
            mu_cond = x_prev @ beta_sub[i - 1]
            quad = np.sum(x_prev @ R_inv_sub[i - 1] * x_prev, axis=1)
            df_cond = df_grid + i
            scale = (df_grid + quad) / df_cond
            z_i = (x_all[:, i] - mu_cond) / (
                sigma_cond_sub[i - 1] * np.sqrt(np.maximum(scale, 1e-12)))
            e[k, i] = np.sum(weights[k] * t_dist.cdf(z_i, df=df_cond))
    return np.clip(e, eps, 1.0 - eps)


def _materialized_dcc_scar_rosenblatt(copula, u, fit_result, K, grid_range):
    eps = 1e-10
    T, d = u.shape
    kappa, mu, nu = fit_result.params.values
    grid = TMGrid(kappa, mu, nu, T, K, grid_range)
    x_grid = grid.z + grid.mu
    df_grid = copula.transform(x_grid)
    fi_grid = np.vstack([
        copula.pdf_on_grid(u[k], x_grid, t_index=k)
        for k in range(T)
    ])
    weights = grid.forward_weights(fi_grid)
    u_c = np.clip(u, eps, 1.0 - eps)

    e = np.empty((T, d))
    e[:, 0] = u[:, 0]
    for k in range(T):
        x_all = np.empty((grid.K, d), dtype=np.float64)
        for dim in range(d):
            x_all[:, dim] = t_dist.ppf(u_c[k, dim], df=df_grid)
        R_t = copula.R_path[k]
        for i in range(1, d):
            R_11 = R_t[:i, :i]
            R_21 = R_t[i, :i]
            R_22 = R_t[i, i]
            R_11_inv = np.linalg.inv(R_11)
            beta = R_21 @ R_11_inv
            sigma2 = R_22 - R_21 @ R_11_inv @ R_21
            sigma_c = np.sqrt(max(sigma2, 1e-12))
            x_prev = x_all[:, :i]
            mu_cond = x_prev @ beta
            quad = np.sum(x_prev @ R_11_inv * x_prev, axis=1)
            df_cond = df_grid + i
            scale = (df_grid + quad) / df_cond
            z_i = (x_all[:, i] - mu_cond) / (
                sigma_c * np.sqrt(np.maximum(scale, 1e-12)))
            e[k, i] = np.sum(weights[k] * t_dist.cdf(z_i, df=df_cond))
    return np.clip(e, eps, 1.0 - eps)


def test_equicorr_log_pdf_matches_gaussian_copula_formula():
    u = _u()
    z = norm.ppf(np.clip(u, 1e-10, 1.0 - 1e-10))

    for rho in (-0.2, 0.0, 0.35, 0.8):
        R_eq = (1.0 - rho) * np.eye(4) + rho * np.ones((4, 4))
        ref = multivariate_normal.logpdf(
            z, mean=np.zeros(4), cov=R_eq
        ) - np.sum(norm.logpdf(z), axis=1)
        got = _equicorr_log_pdf(z, np.full(len(u), rho), 4)
        np.testing.assert_allclose(got, ref, atol=2e-12, rtol=1e-12)


def test_equicorr_rho_derivative_matches_finite_difference():
    u = _u()
    z = norm.ppf(np.clip(u, 1e-10, 1.0 - 1e-10))
    rho = 0.27
    eps = 1e-6

    analytic = _equicorr_dlog_pdf_drho(z, np.full(len(u), rho), 4)
    finite_diff = (
        _equicorr_log_pdf(z, np.full(len(u), rho + eps), 4)
        - _equicorr_log_pdf(z, np.full(len(u), rho - eps), 4)
    ) / (2.0 * eps)

    np.testing.assert_allclose(analytic, finite_diff, atol=1e-7, rtol=1e-7)


def test_equicorr_gas_score_uses_analytical_fast_path():
    class CountingEquicorr(EquicorrGaussianCopula):
        def __init__(self, d):
            super().__init__(d)
            self.calls = 0

        def dlog_pdf_dr_rows(self, u, r, t_index=None):
            self.calls += 1
            return super().dlog_pdf_dr_rows(u, r, t_index=t_index)

    u = _u()[:1]
    copula = CountingEquicorr(d=4)
    g_t = 0.2
    r_t = float(copula.transform(np.array([g_t]))[0])
    ll_t = float(copula.log_pdf_rows(u, np.array([r_t]))[0])

    score = _gas_score_multivariate(
        u, 0, g_t, ll_t, copula, scaling="unit", score_eps=1e-4)

    assert copula.calls == 1
    z = norm.ppf(np.clip(u, 1e-10, 1.0 - 1e-10))
    expected = (
        _equicorr_dlog_pdf_drho(z, np.array([r_t]), 4)[0]
        * copula.dtransform(np.array([g_t]))[0]
    )
    np.testing.assert_allclose(score, expected, rtol=1e-12, atol=1e-12)


def test_student_log_pdf_matches_scipy_t_copula_formula():
    u = _u()
    R = _R()

    for df in (2.2, 5.0, 30.0):
        x = t_dist.ppf(np.clip(u, 1e-10, 1.0 - 1e-10), df=df)
        ref = multivariate_t.logpdf(
            x, loc=np.zeros(4), shape=R, df=df
        ) - np.sum(t_dist.logpdf(x, df=df), axis=1)
        got = student_logpdf(u, R, df)
        np.testing.assert_allclose(got, ref, atol=6e-12, rtol=1e-12)


def test_stochastic_student_emission_cache_materializes_full_dataset_ppf():
    u = _u()
    R = _R()
    copula = StochasticStudentCopula(d=4, R=R)

    cache = copula.prepare_emission_cache(u)

    assert copula.prepare_emission_cache(u) is cache
    assert cache.u_shape == u.shape
    assert cache.d == 4
    assert cache.ppf_table.shape[1:] == u.shape
    assert cache.ppf_nodes.ndim == 1
    np.testing.assert_allclose(cache.L_inv, copula._L_inv)
    assert cache.log_det == pytest.approx(copula._log_det)

    start, stop = 3, 9
    slice_cache = copula.prepare_emission_cache(u[start:stop])
    np.testing.assert_allclose(
        cache.ppf(5.0)[start:stop],
        slice_cache.ppf(5.0),
        atol=0.0,
        rtol=0.0,
    )


def test_stochastic_student_emission_cache_lifecycle_and_persistence():
    u = _u()
    R = _R()
    copula = StochasticStudentCopula(d=4, R=R)

    cache = copula.prepare_emission_cache(u)
    assert copula.prepare_emission_cache(u) is cache

    u2 = np.ascontiguousarray(u.copy())
    cache2 = copula.prepare_emission_cache(u2)
    assert cache2 is not cache

    copula._set_R(0.5 * (R + np.eye(4)))
    assert copula._emission_cache is None
    cache3 = copula.prepare_emission_cache(u2)
    assert cache3 is not cache2

    path = Path("tmp-stochastic-student-cache-persistence.json")
    try:
        save_model(copula, path, include_data=True)
        text = path.read_text(encoding="utf-8")
        assert "StochasticStudentEmissionCache" not in text
        assert "_emission_cache" in text
        assert "\"_emission_cache\":null" in text
        assert "\"_ppf_table\":null" in text

        loaded = load_model(path, expected_type=StochasticStudentCopula)
        assert loaded._emission_cache is None
        assert loaded._ppf_table is None
        rebuilt = loaded.prepare_emission_cache(u2)
        assert rebuilt.u_shape == u2.shape
    finally:
        path.unlink(missing_ok=True)


def test_stochastic_student_batch_emissions_accept_cache_and_t_index():
    u = _u()
    R = _R()
    copula = StochasticStudentCopula(d=4, R=R)
    cache = copula.prepare_emission_cache(u)
    x_grid = np.linspace(-2.0, 2.0, 7)
    start, stop = 5, 17
    u_block = u[start:stop]

    grid_ref = copula.copula_grid_batch(u_block, x_grid)
    grid_cached = copula.copula_grid_batch(
        u_block, x_grid, t_index=start, cache=cache)
    np.testing.assert_allclose(grid_cached, grid_ref, atol=2e-14, rtol=2e-14)

    fi_ref, dfi_ref = copula.pdf_and_grad_on_grid_batch(u_block, x_grid)
    fi_cached, dfi_cached = copula.pdf_and_grad_on_grid_batch(
        u_block, x_grid, t_index=start, cache=cache)
    np.testing.assert_allclose(fi_cached, fi_ref, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(dfi_cached, dfi_ref, atol=1e-9, rtol=1e-9)


def test_stochastic_student_dcc_emission_cache_materializes_full_dataset_ppf():
    u = _u()
    R = _R()
    R_path = np.stack([R, 0.5 * (R + np.eye(4))] * 23)[: len(u)]
    copula = StochasticStudentDCCCopula(d=4)
    copula._set_R_path(R_path)

    cache = copula.prepare_emission_cache(u)

    assert copula.prepare_emission_cache(u) is cache
    assert cache.u_shape == u.shape
    assert cache.d == 4
    assert cache.ppf_table.shape[1:] == u.shape
    assert cache.ppf_nodes.ndim == 1
    assert cache.L_inv_path is copula._L_inv_path
    assert cache.log_det_path is copula._log_det_path

    start, stop = 4, 13
    slice_cache = copula.prepare_emission_cache(u[start:stop])
    np.testing.assert_allclose(
        cache.ppf(5.0)[start:stop],
        slice_cache.ppf(5.0),
        atol=0.0,
        rtol=0.0,
    )

    copula._emission_cache = cache
    copula._set_R_path(R_path.copy())
    assert copula._emission_cache is None
    with pytest.raises(ValueError, match="another R_path"):
        copula.copula_grid_batch(
            u[start:stop], np.linspace(-1.0, 1.0, 3),
            t_index=start, cache=cache)

    path = Path("tmp-stochastic-student-dcc-cache-persistence.json")
    try:
        save_model(copula, path, include_data=True)
        text = path.read_text(encoding="utf-8")
        assert "StochasticStudentDCCEmissionCache" not in text
        assert "_emission_cache" in text
        assert "\"_emission_cache\":null" in text
        assert "\"_ppf_table\":null" in text

        loaded = load_model(path, expected_type=StochasticStudentDCCCopula)
        assert loaded._emission_cache is None
        assert loaded._ppf_table is None
        rebuilt = loaded.prepare_emission_cache(u)
        assert rebuilt.u_shape == u.shape
    finally:
        path.unlink(missing_ok=True)


def test_stochastic_student_dcc_batch_emissions_accept_cache_and_t_index():
    u = _u()
    R = _R()
    R_path = np.stack([R, 0.5 * (R + np.eye(4))] * 23)[: len(u)]
    copula = StochasticStudentDCCCopula(d=4)
    copula._set_R_path(R_path)
    cache = copula.prepare_emission_cache(u)
    x_grid = np.linspace(-2.0, 2.0, 7)
    start, stop = 5, 17
    u_block = u[start:stop]

    grid_ref = copula.copula_grid_batch(
        u_block, x_grid, t_index=start)
    grid_cached = copula.copula_grid_batch(
        u_block, x_grid, t_index=start, cache=cache)
    np.testing.assert_allclose(grid_cached, grid_ref, atol=2e-14, rtol=2e-14)

    fi_ref, dfi_ref = copula.pdf_and_grad_on_grid_batch(
        u_block, x_grid, t_index=start)
    fi_cached, dfi_cached = copula.pdf_and_grad_on_grid_batch(
        u_block, x_grid, t_index=start, cache=cache)
    np.testing.assert_allclose(fi_cached, fi_ref, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(dfi_cached, dfi_ref, atol=1e-9, rtol=1e-9)


def test_student_emission_numba_kernels_match_cache_python_path():
    u = _u()
    R = _R()
    copula = StochasticStudentCopula(d=4, R=R)
    cache = copula.prepare_emission_cache(u)
    x_grid = np.linspace(-2.0, 2.0, 6)
    df_grid = copula.transform(x_grid)
    dpsi = copula.dtransform(x_grid)
    start, stop = 4, 15
    n_rows = stop - start
    eps = 1e-5

    x_rows = student_interp_ppf_rows(
        cache.ppf_nodes, cache.ppf_table, 5.0, start, stop)
    np.testing.assert_allclose(
        x_rows,
        cache.ppf_rows(5.0, start, stop),
        atol=0.0,
        rtol=0.0,
    )

    ll_numba = student_logpdf_from_x_rows(
        x_rows, 5.0, cache.L_inv, cache.log_det)
    ll_ref = _log_copula_inlined(
        x_rows, 5.0, cache.d, cache.L_inv, cache.log_det)
    np.testing.assert_allclose(ll_numba, ll_ref, atol=2e-14, rtol=2e-14)

    fi_numba = student_pdf_grid_numba(
        cache.ppf_nodes, cache.ppf_table, start, n_rows,
        df_grid, cache.L_inv, cache.log_det)
    fi_ref = copula.copula_grid_batch(
        u[start:stop], x_grid, t_index=start, cache=cache)
    np.testing.assert_allclose(fi_numba, fi_ref, atol=2e-14, rtol=2e-14)

    fi_grad_numba, dfi_numba = student_pdf_and_grad_grid_fd_numba(
        cache.ppf_nodes, cache.ppf_table, start, n_rows,
        df_grid, dpsi, cache.L_inv, cache.log_det, eps)
    fi_grad_ref, dfi_ref = copula.pdf_and_grad_on_grid_batch(
        u[start:stop], x_grid, t_index=start, cache=cache)
    np.testing.assert_allclose(fi_grad_numba, fi_grad_ref, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(dfi_numba, dfi_ref, atol=2e-10, rtol=2e-10)

    ll_rows, dlog_rows = student_logpdf_and_dlog_rows_fd_numba(
        cache.ppf_nodes,
        cache.ppf_table,
        start,
        n_rows,
        np.array([5.0], dtype=np.float64),
        cache.L_inv,
        cache.log_det,
        eps,
    )
    ll_rows_ref = _log_copula_inlined(
        cache.ppf_rows(5.0, start, stop),
        5.0,
        cache.d,
        cache.L_inv,
        cache.log_det,
    )
    lp_ref = _log_copula_inlined(
        cache.ppf_rows(5.0 + eps, start, stop),
        5.0 + eps,
        cache.d,
        cache.L_inv,
        cache.log_det,
    )
    lm_ref = _log_copula_inlined(
        cache.ppf_rows(5.0 - eps, start, stop),
        5.0 - eps,
        cache.d,
        cache.L_inv,
        cache.log_det,
    )
    np.testing.assert_allclose(ll_rows, ll_rows_ref, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(
        dlog_rows, (lp_ref - lm_ref) / (2.0 * eps),
        atol=2e-10,
        rtol=2e-10,
    )


def test_stochastic_student_hermite_uses_full_dataset_emission_cache():
    class CountingStudent(StochasticStudentCopula):
        def __init__(self, d, R):
            super().__init__(d=d, R=R)
            self.prepare_calls = 0
            self.batch_calls = []

        def prepare_emission_cache(self, u):
            self.prepare_calls += 1
            return super().prepare_emission_cache(u)

        def pdf_and_grad_on_grid_batch(
                self, u, x_grid, t_index=0, cache=None):
            self.batch_calls.append((len(u), int(t_index), cache is not None))
            return super().pdf_and_grad_on_grid_batch(
                u, x_grid, t_index=t_index, cache=cache)

    u = _u()
    copula = CountingStudent(d=4, R=_R())

    value, grad = hermite_loglik_with_grad(
        1.0,
        1.0,
        0.8,
        u,
        copula,
        basis_order=8,
        quad_order=20,
        block_size=7,
    )

    assert np.isfinite(value)
    assert np.all(np.isfinite(grad))
    assert copula.prepare_calls == 1
    assert copula.batch_calls
    assert all(has_cache for _, _, has_cache in copula.batch_calls)
    assert 0 in {t_index for _, t_index, _ in copula.batch_calls}
    assert any(t_index > 0 for _, t_index, _ in copula.batch_calls)


def test_stochastic_student_dcc_hermite_uses_full_dataset_emission_cache():
    class CountingDCC(StochasticStudentDCCCopula):
        def __init__(self, d):
            super().__init__(d)
            self.prepare_calls = 0
            self.batch_calls = []

        def prepare_emission_cache(self, u):
            self.prepare_calls += 1
            return super().prepare_emission_cache(u)

        def pdf_and_grad_on_grid_batch(
                self, u, x_grid, t_index=0, cache=None):
            self.batch_calls.append((len(u), int(t_index), cache is not None))
            return super().pdf_and_grad_on_grid_batch(
                u, x_grid, t_index=t_index, cache=cache)

    u = _u()
    R = _R()
    copula = CountingDCC(d=4)
    copula._set_R_path(np.stack([R, 0.5 * (R + np.eye(4))] * 23)[: len(u)])

    value, grad = hermite_loglik_with_grad(
        1.0,
        1.0,
        0.8,
        u,
        copula,
        basis_order=8,
        quad_order=20,
        block_size=7,
    )

    assert np.isfinite(value)
    assert np.all(np.isfinite(grad))
    assert copula.prepare_calls == 1
    assert copula.batch_calls
    assert all(has_cache for _, _, has_cache in copula.batch_calls)
    assert 0 in {t_index for _, t_index, _ in copula.batch_calls}
    assert any(t_index > 0 for _, t_index, _ in copula.batch_calls)


def test_stochastic_student_tm_gradient_uses_full_dataset_emission_cache():
    class CountingStudent(StochasticStudentCopula):
        def __init__(self, d, R):
            super().__init__(d=d, R=R)
            self.prepare_calls = 0
            self.batch_calls = []

        def prepare_emission_cache(self, u):
            self.prepare_calls += 1
            return super().prepare_emission_cache(u)

        def pdf_and_grad_on_grid_batch(
                self, u, x_grid, t_index=0, cache=None):
            self.batch_calls.append((len(u), int(t_index), cache is not None))
            return super().pdf_and_grad_on_grid_batch(
                u, x_grid, t_index=t_index, cache=cache)

    u = _u()
    copula = CountingStudent(d=4, R=_R())

    value, grad = tm_loglik_with_grad(
        1.0,
        1.0,
        0.8,
        u,
        copula,
        K=25,
        grid_range=3.0,
        adaptive=False,
        transition_method="local",
    )

    assert np.isfinite(value)
    assert np.all(np.isfinite(grad))
    assert copula.prepare_calls == 1
    assert copula.batch_calls == [(len(u), 0, True)]


def test_stochastic_student_tm_value_and_forward_use_emission_cache_blocks(
        monkeypatch):
    from pyscarcopula.numerical import tm_functions

    class CountingStudent(StochasticStudentCopula):
        def __init__(self, d, R):
            super().__init__(d=d, R=R)
            self.prepare_calls = 0
            self.grid_calls = []

        def prepare_emission_cache(self, u):
            self.prepare_calls += 1
            return super().prepare_emission_cache(u)

        def copula_grid_batch(self, u, x_grid, t_index=0, cache=None):
            self.grid_calls.append((len(u), int(t_index), cache is not None))
            return super().copula_grid_batch(
                u, x_grid, t_index=t_index, cache=cache)

    u = _u()
    copula = CountingStudent(d=4, R=_R())

    value = tm_loglik(
        1.0,
        1.0,
        0.8,
        u,
        copula,
        K=25,
        grid_range=3.0,
        adaptive=False,
        transition_method="local",
    )
    assert np.isfinite(value)
    assert copula.prepare_calls == 1
    assert copula.grid_calls == [(len(u), 0, True)]

    copula.prepare_calls = 0
    copula.grid_calls = []
    monkeypatch.setattr(tm_functions, "forward_block_size", lambda K: 11)
    out = tm_forward_predictive_mean(
        1.0,
        1.0,
        0.8,
        u,
        copula,
        K=25,
        grid_range=3.0,
        adaptive=False,
        transition_method="local",
    )

    assert out.shape == (len(u),)
    assert np.all(np.isfinite(out))
    assert copula.prepare_calls == 1
    assert copula.grid_calls
    assert all(has_cache for _, _, has_cache in copula.grid_calls)
    assert 0 in {t_index for _, t_index, _ in copula.grid_calls}
    assert any(t_index > 0 for _, t_index, _ in copula.grid_calls)


def test_stochastic_student_gas_score_fast_path_matches_g_space_fd():
    u = _u()[:1]
    R = _R()
    copula = StochasticStudentCopula(d=4, R=R)
    g_t = 1.4
    score_eps = 1e-5
    r_t = float(copula.transform(np.array([g_t]))[0])
    ll_t = float(copula.log_pdf_rows(u, np.array([r_t]))[0])

    score = _gas_score_multivariate(
        u, 0, g_t, ll_t, copula, scaling="unit", score_eps=score_eps)

    r_plus = float(copula.transform(np.array([g_t + score_eps]))[0])
    r_minus = float(copula.transform(np.array([g_t - score_eps]))[0])
    fd = (
        float(copula.log_pdf_rows(u, np.array([r_plus]))[0])
        - float(copula.log_pdf_rows(u, np.array([r_minus]))[0])
    ) / (2.0 * score_eps)
    np.testing.assert_allclose(score, fd, rtol=2e-4, atol=2e-4)


def test_stochastic_student_gas_score_near_df_lower_bound_matches_fd():
    u = _u()[:1]
    R = _R()
    copula = StochasticStudentCopula(d=4, R=R)
    g_t = -8.0
    score_eps = 1e-5
    r_t = float(copula.transform(np.array([g_t]))[0])
    ll_t = float(copula.log_pdf_rows(u, np.array([r_t]))[0])

    score = _gas_score_multivariate(
        u, 0, g_t, ll_t, copula, scaling="unit", score_eps=score_eps)

    r_plus = float(copula.transform(np.array([g_t + score_eps]))[0])
    r_minus = float(copula.transform(np.array([g_t - score_eps]))[0])
    fd = (
        float(copula.log_pdf_rows(u, np.array([r_plus]))[0])
        - float(copula.log_pdf_rows(u, np.array([r_minus]))[0])
    ) / (2.0 * score_eps)
    np.testing.assert_allclose(score, fd, rtol=5e-4, atol=5e-4)


def test_stochastic_student_combined_log_pdf_score_matches_separate_paths():
    u = _u()
    R = _R()
    copula = StochasticStudentCopula(d=4, R=R)
    df_path = np.linspace(2.4, 12.0, len(u))

    ll, dlog = copula.log_pdf_and_dlog_dr_rows(u, df_path)

    np.testing.assert_allclose(
        ll, copula.log_pdf_rows(u, df_path), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        dlog, copula.dlog_pdf_dr_rows(u, df_path), atol=1e-12, rtol=1e-12)


def test_stochastic_student_gas_filter_fast_path_matches_generic_path():
    u = _u()
    R = _R()
    copula = StochasticStudentCopula(d=4, R=R)
    params = (0.08, 0.04, 0.92)

    g_fast, r_fast, ll_fast = gas_filter(*params, u, copula, scaling="unit")
    g_ref, r_ref, ll_ref = _gas_filter_multivariate(
        *params, u, copula, scaling="unit")

    np.testing.assert_allclose(g_fast, g_ref, atol=5e-4, rtol=5e-4)
    np.testing.assert_allclose(r_fast, r_ref, atol=5e-4, rtol=5e-4)
    np.testing.assert_allclose(ll_fast, ll_ref, atol=1e-3, rtol=5e-4)


def test_dcc_log_pdf_rows_matches_static_t_copula_per_row():
    u = _u()
    R = _R()
    R_path = np.stack([R, 0.5 * (R + np.eye(4))] * 23)[: len(u)]
    df_path = np.linspace(3.0, 12.0, len(u))
    cop = StochasticStudentDCCCopula(d=4)
    cop._set_R_path(R_path)

    got = cop.log_pdf_rows(u, df_path)
    ref = np.array(
        [
            dcc_student_logpdf(u[i : i + 1], R_path[i], df_path[i])[0]
            for i in range(len(u))
        ]
    )

    np.testing.assert_allclose(got, ref, atol=6e-12, rtol=1e-12)


def test_dcc_fast_recursion_matches_reference_path_and_loglik():
    rng = np.random.default_rng(20260526)
    z = rng.standard_normal((20, 4))
    qbar = _R()
    a, b = 0.04, 0.91

    q_ref, r_ref = _dcc_recursion(z, a, b, qbar)
    ll_ref = _dcc_loglik_gaussian(z, r_ref)

    q_fast, r_fast, ll_fast = _dcc_fast_recursion_path_loglik(z, a, b, qbar)
    ll_direct = _dcc_fast_loglik(z, a, b, qbar)

    np.testing.assert_allclose(q_fast, q_ref, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(r_fast, r_ref, atol=2e-12, rtol=2e-12)
    assert ll_fast == pytest.approx(ll_ref, abs=2e-12, rel=2e-12)
    assert ll_direct == pytest.approx(ll_ref, abs=2e-12, rel=2e-12)


def test_dcc_garch_filter_matches_reference_recursion():
    rng = np.random.default_rng(20260526)
    r = rng.standard_normal(30)
    omega, alpha, beta = 0.02, 0.07, 0.88

    got = _garch11_filter(r, omega, alpha, beta)

    expected = np.empty_like(r, dtype=np.float64)
    expected[0] = omega / max(1.0 - alpha - beta, 1e-6)
    for t in range(1, len(r)):
        expected[t] = omega + alpha * r[t - 1] ** 2 + beta * expected[t - 1]
    expected = np.maximum(expected, 1e-12)

    np.testing.assert_allclose(got, expected, atol=2e-16, rtol=2e-16)


def test_dcc_combined_log_pdf_score_matches_separate_paths():
    u = _u()
    R = _R()
    R_path = np.stack([R, 0.5 * (R + np.eye(4))] * 23)[: len(u)]
    df_path = np.linspace(2.4, 12.0, len(u))
    cop = StochasticStudentDCCCopula(d=4)
    cop._set_R_path(R_path)

    ll, dlog = cop.log_pdf_and_dlog_dr_rows(u, df_path)

    np.testing.assert_allclose(
        ll, cop.log_pdf_rows(u, df_path), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        dlog, cop.dlog_pdf_dr_rows(u, df_path), atol=1e-12, rtol=1e-12)


def test_dcc_gas_score_uses_row_derivative_path():
    class CountingDCC(StochasticStudentDCCCopula):
        def __init__(self, d):
            super().__init__(d)
            self.calls = 0

        def dlog_pdf_dr_rows(self, u, r, t_index=None):
            self.calls += 1
            return super().dlog_pdf_dr_rows(u, r, t_index=t_index)

    u = _u()[:1]
    R = _R()
    copula = CountingDCC(d=4)
    copula._set_R_path(np.stack([R, 0.5 * (R + np.eye(4))]))
    g_t = 1.4
    score_eps = 1e-5
    r_t = float(copula.transform(np.array([g_t]))[0])
    ll_t = float(copula.log_pdf_rows(u, np.array([r_t]))[0])

    score = _gas_score_multivariate(
        u, 0, g_t, ll_t, copula, scaling="unit", score_eps=score_eps)

    assert copula.calls == 1
    r_plus = float(copula.transform(np.array([g_t + score_eps]))[0])
    r_minus = float(copula.transform(np.array([g_t - score_eps]))[0])
    fd = (
        float(copula.log_pdf_rows(u, np.array([r_plus]))[0])
        - float(copula.log_pdf_rows(u, np.array([r_minus]))[0])
    ) / (2.0 * score_eps)
    np.testing.assert_allclose(score, fd, rtol=3e-4, atol=3e-4)


def test_dcc_gas_score_near_df_lower_bound_matches_fd():
    u = _u()[:1]
    R = _R()
    copula = StochasticStudentDCCCopula(d=4)
    copula._set_R_path(np.stack([R, 0.5 * (R + np.eye(4))]))
    g_t = -8.0
    score_eps = 1e-5
    r_t = float(copula.transform(np.array([g_t]))[0])
    ll_t = float(copula.log_pdf_rows(u, np.array([r_t]))[0])

    score = _gas_score_multivariate(
        u, 0, g_t, ll_t, copula, scaling="unit", score_eps=score_eps)

    r_plus = float(copula.transform(np.array([g_t + score_eps]))[0])
    r_minus = float(copula.transform(np.array([g_t - score_eps]))[0])
    fd = (
        float(copula.log_pdf_rows(u, np.array([r_plus]))[0])
        - float(copula.log_pdf_rows(u, np.array([r_minus]))[0])
    ) / (2.0 * score_eps)
    np.testing.assert_allclose(score, fd, rtol=5e-4, atol=5e-4)


def test_multivariate_gas_filter_uses_combined_score_path():
    class CountingDCC(StochasticStudentDCCCopula):
        def __init__(self, d):
            super().__init__(d)
            self.combined_calls = 0

        def log_pdf_and_dlog_dr_rows(self, u, r, t_index=None):
            self.combined_calls += 1
            return super().log_pdf_and_dlog_dr_rows(
                u, r, t_index=t_index)

    u = _u()[:6]
    R = _R()
    copula = CountingDCC(d=4)
    copula._set_R_path(np.stack([R] * len(u)))

    gas_filter(0.1, 0.2, 0.5, u, copula)

    assert copula.combined_calls == len(u) - 1


def test_multivariate_gas_filter_combined_path_matches_separate_path():
    u = _u()
    R = _R()
    R_path = np.stack([R, 0.5 * (R + np.eye(4))] * 23)[: len(u)]
    params = (0.1, 0.2, 0.7)

    fast = StochasticStudentDCCCopula(d=4)
    fast._set_R_path(R_path)
    slow = StochasticStudentDCCCopula(d=4)
    slow._set_R_path(R_path)
    slow.log_pdf_and_dlog_dr_rows = None

    fast_g, fast_r, fast_ll = gas_filter(*params, u, fast)
    slow_g, slow_r, slow_ll = gas_filter(*params, u, slow)

    np.testing.assert_allclose(fast_g, slow_g, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(fast_r, slow_r, atol=0.0, rtol=0.0)
    assert fast_ll == pytest.approx(slow_ll, abs=0.0)


def test_multivariate_gas_rejects_wrong_input_dimension():
    u = np.full((5, 3), 0.5)
    copula = EquicorrGaussianCopula(d=4)

    with pytest.raises(ValueError, match=r"shape \(T, 4\)"):
        gas_filter(0.0, 0.0, 0.0, u, copula)


def test_experimental_grid_batches_match_reference_density():
    u = _u()
    R = _R()
    R_path = np.stack([R, 0.5 * (R + np.eye(4))] * 23)[: len(u)]
    x_grid = np.linspace(-2.5, 2.5, 17)

    eq = EquicorrGaussianCopula(d=4)
    eq_grid = eq.copula_grid_batch(u, x_grid)
    eq_ref = np.exp(
        np.column_stack([eq.log_pdf_rows(u, rho) for rho in eq.transform(x_grid)])
    )
    np.testing.assert_allclose(eq_grid, eq_ref, rtol=1e-6, atol=1e-10)

    student = StochasticStudentCopula(d=4, R=R)
    student._set_R(R)
    st_grid = student.copula_grid_batch(u, x_grid)
    st_ref = np.exp(
        np.column_stack(
            [student.log_pdf_rows(u, df) for df in student.transform(x_grid)]
        )
    )
    np.testing.assert_allclose(st_grid, st_ref, rtol=5e-4, atol=1e-10)

    dcc = StochasticStudentDCCCopula(d=4)
    dcc._set_R_path(R_path)
    dcc_grid = dcc.copula_grid_batch(u, x_grid)
    dcc_ref = np.exp(
        np.column_stack([dcc.log_pdf_rows(u, df) for df in dcc.transform(x_grid)])
    )
    np.testing.assert_allclose(dcc_grid, dcc_ref, rtol=5e-4, atol=1e-10)

    start = 5
    stop = 17
    dcc_block = dcc.copula_grid_batch(u[start:stop], x_grid, t_index=start)
    dcc_block_ref = np.exp(
        np.column_stack([
            dcc.log_pdf_rows(u[start:stop], df, t_index=start)
            for df in dcc.transform(x_grid)
        ])
    )
    np.testing.assert_allclose(
        dcc_block, dcc_block_ref, rtol=5e-4, atol=1e-10)


def test_experimental_scar_grid_batches_do_not_call_row_pdf(monkeypatch):
    u = _u()
    R = _R()
    x_grid = np.linspace(-2.0, 2.0, 9)

    for copula in (EquicorrGaussianCopula(d=4),
                   StochasticStudentCopula(d=4, R=R)):
        def fail_pdf_on_grid(*args, **kwargs):
            raise AssertionError("pdf_on_grid should not be called")

        monkeypatch.setattr(copula, "pdf_on_grid", fail_pdf_on_grid)
        fi, dfi = copula.pdf_and_grad_on_grid_batch(u, x_grid)

        assert fi.shape == (len(u), len(x_grid))
        assert dfi.shape == fi.shape
        assert np.all(np.isfinite(fi))
        assert np.all(np.isfinite(dfi))


def test_experimental_mle_loglik_and_gof_contracts():
    u = _u()
    R = _R()
    models = [
        EquicorrGaussianCopula(d=4),
        StochasticStudentCopula(d=4, R=R),
    ]
    dcc = StochasticStudentDCCCopula(d=4)
    dcc._set_R_path(np.stack([R] * len(u)))
    models.append(dcc)

    for copula in models:
        if hasattr(copula, "_set_R") and getattr(copula, "R", None) is not None:
            copula._set_R(getattr(copula, "R"))
        result = copula.fit(u, method="mle", to_pobs=False)
        ll = copula.log_likelihood(u, result.copula_param)
        gof = gof_test(copula, u, to_pobs=False, fit_result=result)

        np.testing.assert_allclose(ll, result.log_likelihood, atol=1e-8, rtol=1e-10)
        assert np.isfinite(gof.statistic)
        assert 0.0 <= gof.pvalue <= 1.0


def test_experimental_scar_gof_does_not_materialize_forward_weights(monkeypatch):
    u = pobs(np.random.default_rng(20260519).standard_normal((20, 3)))
    R = np.array(
        [
            [1.0, 0.25, -0.10],
            [0.25, 1.0, 0.15],
            [-0.10, 0.15, 1.0],
        ],
        dtype=np.float64,
    )
    dcc = StochasticStudentDCCCopula(d=3)
    dcc._set_R_path(np.stack([R] * len(u)))
    models = [
        EquicorrGaussianCopula(d=3),
        StochasticStudentCopula(d=3, R=R),
        dcc,
    ]

    def fail_forward_weights(self, fi_grid):
        raise AssertionError("forward_weights should not be called")

    monkeypatch.setattr(TMGrid, "forward_weights", fail_forward_weights)

    for copula in models:
        result = _scar_result(K=15, grid_range=3.0)
        gof = gof_test(copula, u, to_pobs=False, fit_result=result, K=15,
                       grid_range=3.0)
        assert np.isfinite(gof.statistic)
        assert 0.0 <= gof.pvalue <= 1.0


def test_experimental_scar_gof_matches_materialized_reference():
    u = pobs(np.random.default_rng(20260521).standard_normal((14, 3)))
    R0 = np.array(
        [
            [1.0, 0.25, -0.10],
            [0.25, 1.0, 0.15],
            [-0.10, 0.15, 1.0],
        ],
        dtype=np.float64,
    )
    R1 = np.array(
        [
            [1.0, 0.10, 0.20],
            [0.10, 1.0, -0.15],
            [0.20, -0.15, 1.0],
        ],
        dtype=np.float64,
    )
    result = _scar_result(K=13, grid_range=3.0)

    equicorr = EquicorrGaussianCopula(d=3)
    eq_ref = _materialized_equicorr_scar_rosenblatt(
        equicorr, u, result, K=13, grid_range=3.0)
    eq_got = equicorr_rosenblatt_transform(
        equicorr, u, result, K=13, grid_range=3.0)
    np.testing.assert_allclose(eq_got, eq_ref, atol=1e-12, rtol=1e-12)

    student = StochasticStudentCopula(d=3, R=R0)
    st_ref = _materialized_student_scar_rosenblatt(
        student, u, result, K=13, grid_range=3.0)
    st_got = stochastic_student_rosenblatt_transform(
        student, u, result, K=13, grid_range=3.0)
    np.testing.assert_allclose(st_got, st_ref, atol=8e-4, rtol=8e-4)

    dcc = StochasticStudentDCCCopula(d=3)
    dcc._set_R_path(np.stack([R0 if k % 2 == 0 else R1 for k in range(len(u))]))
    dcc_ref = _materialized_dcc_scar_rosenblatt(
        dcc, u, result, K=13, grid_range=3.0)
    dcc_got = stochastic_student_dcc_rosenblatt_transform(
        dcc, u, result, K=13, grid_range=3.0)
    np.testing.assert_allclose(dcc_got, dcc_ref, atol=8e-4, rtol=8e-4)


def test_equicorr_scar_gof_uses_block_batch_emissions(monkeypatch):
    u = pobs(np.random.default_rng(20260520).standard_normal((18, 3)))
    copula = EquicorrGaussianCopula(d=3)
    result = _scar_result(K=12, grid_range=3.0)

    def fail_pdf_on_grid(*args, **kwargs):
        raise AssertionError("pdf_on_grid should not be called")

    monkeypatch.setattr(copula, "pdf_on_grid", fail_pdf_on_grid)
    e = equicorr_rosenblatt_transform(
        copula, u, result, K=12, grid_range=3.0)

    assert e.shape == u.shape
    assert np.all(np.isfinite(e))
    assert np.all((e > 0.0) & (e < 1.0))


def test_stochastic_student_scar_gof_uses_block_batch_emissions(monkeypatch):
    u = pobs(np.random.default_rng(20260520).standard_normal((18, 3)))
    R = np.array(
        [
            [1.0, 0.25, -0.10],
            [0.25, 1.0, 0.15],
            [-0.10, 0.15, 1.0],
        ],
        dtype=np.float64,
    )
    copula = StochasticStudentCopula(d=3, R=R)
    result = _scar_result(K=12, grid_range=3.0)

    def fail_pdf_on_grid(*args, **kwargs):
        raise AssertionError("pdf_on_grid should not be called")

    monkeypatch.setattr(copula, "pdf_on_grid", fail_pdf_on_grid)
    e = stochastic_student_rosenblatt_transform(
        copula, u, result, K=12, grid_range=3.0)

    assert e.shape == u.shape
    assert np.all(np.isfinite(e))
    assert np.all((e > 0.0) & (e < 1.0))


def test_stochastic_student_dcc_scar_gof_uses_block_batch_emissions(monkeypatch):
    u = pobs(np.random.default_rng(20260520).standard_normal((18, 3)))
    R0 = np.array(
        [
            [1.0, 0.25, -0.10],
            [0.25, 1.0, 0.15],
            [-0.10, 0.15, 1.0],
        ],
        dtype=np.float64,
    )
    R1 = np.array(
        [
            [1.0, 0.10, 0.20],
            [0.10, 1.0, -0.15],
            [0.20, -0.15, 1.0],
        ],
        dtype=np.float64,
    )
    R_path = np.stack([R0 if k % 2 == 0 else R1 for k in range(len(u))])
    copula = StochasticStudentDCCCopula(d=3)
    copula._set_R_path(R_path)
    result = _scar_result(K=12, grid_range=3.0)

    def fail_pdf_on_grid(*args, **kwargs):
        raise AssertionError("pdf_on_grid should not be called")

    monkeypatch.setattr(copula, "pdf_on_grid", fail_pdf_on_grid)
    e = stochastic_student_dcc_rosenblatt_transform(
        copula, u, result, K=12, grid_range=3.0)

    assert e.shape == u.shape
    assert np.all(np.isfinite(e))
    assert np.all((e > 0.0) & (e < 1.0))


def test_forward_weight_block_arrays_match_row_iterator():
    from pyscarcopula.numerical.gof_blocks import (
        iter_forward_weight_block_arrays,
        iter_forward_weight_blocks,
    )

    u = pobs(np.random.default_rng(20260525).standard_normal((17, 3)))
    copula = EquicorrGaussianCopula(d=3)
    grid = TMGrid(0.8, 0.0, 1.0, len(u), 9, 3.0)
    x_grid = grid.z + grid.mu

    row_weights = []
    row_fi = []
    for _k, _local, weights, fi_block in iter_forward_weight_blocks(
            grid, u, copula, x_grid=x_grid, block_size=4):
        row_weights.append(weights.copy())
        row_fi.append(fi_block[_local].copy())

    block_weights = []
    block_fi = []
    for _start, _stop, weights_block, fi_block, _u_block in (
            iter_forward_weight_block_arrays(
                grid, u, copula, x_grid=x_grid, block_size=4)):
        block_weights.extend(weights_block)
        block_fi.extend(fi_block)

    np.testing.assert_allclose(block_weights, row_weights, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(block_fi, row_fi, atol=0.0, rtol=0.0)


def test_experimental_scar_gof_block_size_accounts_for_dimension(monkeypatch):
    from pyscarcopula.numerical import gof_blocks

    rng = np.random.default_rng(20260521)
    u = pobs(rng.standard_normal((10, 4)))
    R = _R()
    result = _scar_result(K=8, grid_range=3.0)
    calls = []

    def capture_block_size(K, max_elements=2_000_000, max_rows=512,
                           element_width=1):
        calls.append(element_width)
        return 3

    monkeypatch.setattr(gof_blocks, "forward_block_size", capture_block_size)

    equicorr = EquicorrGaussianCopula(d=4)
    equicorr_rosenblatt_transform(equicorr, u, result, K=8, grid_range=3.0)

    student = StochasticStudentCopula(d=4, R=R)
    stochastic_student_rosenblatt_transform(
        student, u, result, K=8, grid_range=3.0)

    dcc = StochasticStudentDCCCopula(d=4)
    dcc._set_R_path(np.stack([R] * len(u)))
    stochastic_student_dcc_rosenblatt_transform(
        dcc, u, result, K=8, grid_range=3.0)

    assert calls == [4, 8, 8]


def test_experimental_models_support_top_level_api_except_pair_h():
    u = _u()
    R = _R()

    def make_model(name):
        if name == "equicorr":
            return EquicorrGaussianCopula(d=4)
        if name == "student":
            return StochasticStudentCopula(d=4, R=R)
        dcc = StochasticStudentDCCCopula(d=4)
        dcc._set_R_path(np.stack([R] * len(u)))
        return dcc

    for name in ("equicorr", "student", "dcc"):
        for method in ("mle", "gas", "scar-tm-ou"):
            copula = make_model(name)
            fit_kwargs = {"gtol": 0.5, "maxiter": 10, "maxfun": 10}
            api_kwargs = {}
            if method == "scar-tm-ou":
                fit_kwargs.update({"K": 25, "grid_range": 4.0})
                api_kwargs.update({"K": 25, "grid_range": 4.0})

            result = fit(copula, u, method=method, **fit_kwargs)

            assert np.isfinite(log_likelihood(copula, u, result, **api_kwargs))
            assert predictive_mean(copula, u, result, **api_kwargs).shape == (len(u),)
            assert sample(copula, u, result, 5, **api_kwargs).shape == (5, 4)
            assert predict(copula, u, result, 5, **api_kwargs).shape == (5, 4)

            with pytest.raises(NotImplementedError):
                mixture_h(copula, u, result, **api_kwargs)


def test_experimental_conditional_predict_honors_given_coordinates():
    u = _u()
    R = _R()

    models = [
        EquicorrGaussianCopula(d=4),
        StochasticStudentCopula(d=4, R=R),
    ]
    dcc = StochasticStudentDCCCopula(d=4)
    dcc._set_R_path(np.stack([R] * len(u)))
    models.append(dcc)

    for copula in models:
        result = fit(copula, u, method="mle")
        samples = predict(
            copula,
            u,
            result,
            32,
            given={0: 0.25, 2: 0.75},
            rng=np.random.default_rng(20260522),
        )

        assert samples.shape == (32, 4)
        assert np.allclose(samples[:, 0], 0.25)
        assert np.allclose(samples[:, 2], 0.75)
        assert np.all((samples > 0.0) & (samples < 1.0))


def test_experimental_direct_predict_honors_all_given_coordinates():
    u = _u()
    R = _R()
    given = {0: 0.2, 1: 0.4, 2: 0.6, 3: 0.8}

    models = [
        EquicorrGaussianCopula(d=4),
        StochasticStudentCopula(d=4, R=R),
    ]
    dcc = StochasticStudentDCCCopula(d=4)
    dcc._set_R_path(np.stack([R] * len(u)))
    models.append(dcc)

    for copula in models:
        copula.fit(u, method="mle")
        kwargs = {}
        if isinstance(copula, StochasticStudentDCCCopula):
            kwargs = {"mode": "last_R", "df_mode": "fitted"}
        samples = copula.predict(
            7, u=u, given=given, rng=np.random.default_rng(1), **kwargs)

        expected = np.array([given[i] for i in range(4)], dtype=np.float64)
        np.testing.assert_allclose(samples, np.tile(expected, (7, 1)))


def test_equicorr_conditional_mean_matches_gaussian_oracle():
    copula = EquicorrGaussianCopula(d=3)
    rho = 0.45
    given_u = 0.8
    samples = copula.sample_conditional(
        40000,
        r=rho,
        given={0: given_u},
        rng=np.random.default_rng(9),
    )

    z_given = norm.ppf(given_u)
    mean_z = rho * z_given
    var_z = 1.0 - rho * rho
    expected_u_mean = norm.cdf(mean_z / np.sqrt(1.0 + var_z))

    assert np.allclose(samples[:, 0], given_u)
    assert np.mean(samples[:, 1]) == pytest.approx(expected_u_mean, abs=0.01)
    assert np.mean(samples[:, 2]) == pytest.approx(expected_u_mean, abs=0.01)
