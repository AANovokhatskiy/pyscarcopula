"""Mathematical contracts for multivariate copulas."""

import math
import numpy as np
import pytest
from pathlib import Path
from scipy.special import stdtrit
from scipy.stats import multivariate_normal, multivariate_t, norm
from scipy.stats import t as t_dist

from pyscarcopula._constants import PSEUDO_OBS_EPS
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
from pyscarcopula.copula.multivariate.equicorr import (
    EquicorrGaussianCopula,
)
from pyscarcopula.copula.multivariate.conditional import (
    sample_gaussian_conditional,
    sample_student_conditional,
    validate_multivariate_given,
)
from pyscarcopula.copula.multivariate.student_ppf_cache import (
    StudentPPFCache,
    StudentPPFTable,
)
from pyscarcopula.copula.multivariate.stochastic_student import (
    StochasticStudentCopula,
)
from pyscarcopula.numerical.tm_grid import TMGrid
from pyscarcopula.numerical import _cpp_gas
from pyscarcopula.numerical.gas_filter import gas_filter
from pyscarcopula.stattests import (
    equicorr_rosenblatt_transform,
    gof_test,
    stochastic_student_rosenblatt_transform,
)


def student_logpdf(u, R, df):
    copula = StochasticStudentCopula(d=u.shape[1], R=R)
    return copula.log_pdf_rows(u, df)


def _skip_removed_dcc(*args, **kwargs):
    pytest.skip("StochasticStudentDCCCopula has been removed")


class StochasticStudentDCCCopula:
    def __init__(self, *args, **kwargs):
        _skip_removed_dcc()


_dcc_fast_loglik = _skip_removed_dcc
_dcc_fast_recursion_path_loglik = _skip_removed_dcc
_dcc_loglik_gaussian = _skip_removed_dcc
_dcc_recursion = _skip_removed_dcc
_garch11_filter = _skip_removed_dcc
dcc_student_logpdf = _skip_removed_dcc
stochastic_student_dcc_rosenblatt_transform = _skip_removed_dcc


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
            prefix_density = _equicorr_leading_density(
                x_norm[k, :i], rho_grid)
            state_weights = _prefix_reweighted(weights[k], prefix_density)
            e[k, i] = np.sum(state_weights * norm.cdf(z_i))
    return np.clip(e, eps, 1.0 - eps)


def _prefix_reweighted(weights, prefix_density):
    raw = weights * prefix_density
    total = np.sum(raw)
    if total > 0.0 and np.isfinite(total):
        return raw / total
    return weights


def _equicorr_leading_density(x_prefix, rho_grid):
    m = len(x_prefix)
    if m <= 1:
        return np.ones_like(rho_grid)
    rho = rho_grid
    a = 1.0 - rho
    b = 1.0 + (m - 1) * rho
    s2 = np.sum(x_prefix * x_prefix)
    s1 = np.sum(x_prefix) ** 2
    log_density = (
        -0.5 * ((m - 1) * np.log(a) + np.log(b))
        -0.5 * ((rho / a) * s2 - (rho / (a * b)) * s1)
    )
    return np.exp(log_density)


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


def _student_leading_density(x_prefix_grid, df_grid, R_inv, log_det):
    x_prefix_grid = np.asarray(x_prefix_grid, dtype=np.float64)
    if x_prefix_grid.ndim == 1:
        x_prefix_grid = np.repeat(
            x_prefix_grid[np.newaxis, :], len(df_grid), axis=0)
    m = x_prefix_grid.shape[1]
    if m <= 1:
        return np.ones_like(df_grid)
    out = np.empty_like(df_grid)
    for idx, df in enumerate(df_grid):
        x_prefix = x_prefix_grid[idx]
        q = float(x_prefix @ R_inv @ x_prefix)
        joint = (
            math.lgamma(0.5 * (df + m))
            - math.lgamma(0.5 * df)
            - 0.5 * m * math.log(df * math.pi)
            - 0.5 * log_det
            - 0.5 * (df + m) * math.log1p(q / df)
        )
        marginal = 0.0
        for x in x_prefix:
            marginal += (
                math.lgamma(0.5 * (df + 1.0))
                - math.lgamma(0.5 * df)
                - 0.5 * math.log(df * math.pi)
                - 0.5 * (df + 1.0) * math.log1p((x * x) / df)
            )
        out[idx] = math.exp(joint - marginal)
    return out


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
    log_det_sub = [0.0]
    for i in range(2, d):
        sign, log_det = np.linalg.slogdet(copula.R[:i, :i])
        assert sign > 0
        log_det_sub.append(float(log_det))
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
            prefix_density = _student_leading_density(
                x_all[:, :i], df_grid, R_inv_sub[i - 1], log_det_sub[i - 1])
            state_weights = _prefix_reweighted(weights[k], prefix_density)
            e[k, i] = np.sum(state_weights * t_dist.cdf(z_i, df=df_cond))
    return np.clip(e, eps, 1.0 - eps)


def _equicorr_scar_rosenblatt_predictive_only(copula, u, fit_result, K, grid_range):
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


def _student_scar_rosenblatt_predictive_only(copula, u, fit_result, K, grid_range):
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
    copula = EquicorrGaussianCopula(d=4)

    for rho in (-0.2, 0.0, 0.35, 0.8):
        R_eq = (1.0 - rho) * np.eye(4) + rho * np.ones((4, 4))
        ref = multivariate_normal.logpdf(
            z, mean=np.zeros(4), cov=R_eq
        ) - np.sum(norm.logpdf(z), axis=1)
        got = copula.log_pdf_rows(u, np.full(len(u), rho))
        np.testing.assert_allclose(got, ref, atol=2e-12, rtol=1e-12)


def test_equicorr_rho_derivative_matches_finite_difference():
    u = _u()
    copula = EquicorrGaussianCopula(d=4)
    rho = 0.27
    eps = 1e-6

    analytic = copula.dlog_pdf_dr_rows(u, np.full(len(u), rho))
    finite_diff = (
        copula.log_pdf_rows(u, np.full(len(u), rho + eps))
        - copula.log_pdf_rows(u, np.full(len(u), rho - eps))
    ) / (2.0 * eps)

    np.testing.assert_allclose(analytic, finite_diff, atol=1e-7, rtol=1e-7)


def test_equicorr_gas_score_does_not_call_python_derivative():
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

    score = _cpp_gas.update_one(
        0.0, 0.0, 0.0, g_t, u, copula, "unit", 1e-4).score

    assert copula.calls == 0
    expected = (
        copula.dlog_pdf_dr_rows(u, np.array([r_t]))[0]
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

    assert isinstance(cache, StudentPPFCache)
    assert copula.prepare_emission_cache(u) is cache
    assert cache.u_shape == u.shape
    assert cache.d == 4
    assert cache.ppf_table.shape[1:] == u.shape
    assert cache.ppf_nodes.ndim == 1
    assert cache.version > 0
    assert cache.u_snapshot.flags.c_contiguous
    assert not cache.u_snapshot.flags.writeable
    np.testing.assert_array_equal(cache.u_snapshot, u)
    assert not hasattr(cache, "L_inv")
    assert not hasattr(cache, "log_det")

    start, stop = 3, 9
    slice_cache = copula.prepare_emission_cache(u[start:stop])
    np.testing.assert_allclose(
        cache.ppf(5.0)[start:stop],
        slice_cache.ppf(5.0),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize("df", [2.001, 500.0, 1000.0, 1_000_000.0])
def test_stochastic_student_ppf_cache_uses_exact_quantiles_outside_table(df):
    u = _u()
    copula = StochasticStudentCopula(d=4, R=_R())
    cache = copula.prepare_emission_cache(u)
    expected = stdtrit(
        df,
        np.clip(u, PSEUDO_OBS_EPS, 1.0 - PSEUDO_OBS_EPS),
    )

    np.testing.assert_allclose(
        cache.ppf(df),
        expected,
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        cache.ppf_rows(df, 3, 11),
        expected[3:11],
        rtol=2e-13,
        atol=2e-13,
    )


def test_student_ppf_table_uses_common_pseudo_observation_boundaries():
    u = np.array(
        [
            [0.0, PSEUDO_OBS_EPS / 10.0, PSEUDO_OBS_EPS],
            [
                1.0,
                1.0 - PSEUDO_OBS_EPS / 10.0,
                1.0 - PSEUDO_OBS_EPS,
            ],
        ],
        dtype=np.float64,
    )
    clipped = np.clip(
        u, PSEUDO_OBS_EPS, 1.0 - PSEUDO_OBS_EPS)
    table = StudentPPFTable(u)

    np.testing.assert_array_equal(table.u, clipped)
    for df in (table.nodes[0], 500.0):
        np.testing.assert_allclose(
            table(df),
            stdtrit(df, clipped),
            rtol=2e-13,
            atol=2e-13,
        )


def test_stochastic_student_emission_cache_lifecycle_and_persistence():
    u = _u()
    R = _R()
    copula = StochasticStudentCopula(d=4, R=R)

    cache = copula.prepare_emission_cache(u)
    assert copula.prepare_emission_cache(u) is cache

    u2 = np.ascontiguousarray(u.copy())
    cache2 = copula.prepare_emission_cache(u2)
    assert cache2 is cache

    u2[0, 0] = 0.5 * u2[0, 0]
    cache2 = copula.prepare_emission_cache(u2)
    assert cache2 is not cache
    assert cache2.version > cache.version
    np.testing.assert_array_equal(cache.u_snapshot, u)

    ppf_nodes = cache2.ppf_nodes
    ppf_table = cache2.ppf_table
    corr_version = copula._corr_cache_version
    copula._set_R(0.5 * (R + np.eye(4)))
    assert copula._corr_cache_version == corr_version + 1
    assert copula._ppf_cache is cache2
    cache3 = copula.prepare_emission_cache(u2)
    assert cache3 is cache2
    assert cache3.ppf_nodes is ppf_nodes
    assert cache3.ppf_table is ppf_table

    path = Path("tmp-stochastic-student-cache-persistence.json")
    try:
        save_model(copula, path, include_data=True)
        text = path.read_text(encoding="utf-8")
        assert "StochasticStudentEmissionCache" not in text
        assert "\"_ppf_cache\":null" in text
        assert "\"_emission_cache\"" not in text
        assert "\"_ppf_table\"" not in text

        loaded = load_model(path, expected_type=StochasticStudentCopula)
        assert loaded._ppf_cache is None
        assert not hasattr(loaded, "_ppf_table")
        rebuilt = loaded.prepare_emission_cache(u2)
        assert rebuilt.u_shape == u2.shape
    finally:
        path.unlink(missing_ok=True)


def test_stochastic_student_cache_refreshes_after_view_source_mutation():
    base = np.ascontiguousarray(_u())
    view = base[:, ::-1]
    copula = StochasticStudentCopula(d=4, R=_R())

    cache = copula.prepare_emission_cache(view)
    snapshot = cache.u_snapshot.copy()
    base[0, -1] = 0.123456
    refreshed = copula.prepare_emission_cache(view)

    assert refreshed is not cache
    assert refreshed.version > cache.version
    np.testing.assert_array_equal(cache.u_snapshot, snapshot)
    np.testing.assert_array_equal(refreshed.u_snapshot, view)
    assert not refreshed.u_snapshot.flags.writeable


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

    assert isinstance(cache, StudentPPFCache)
    assert copula.prepare_emission_cache(u) is cache
    assert cache.u_shape == u.shape
    assert cache.d == 4
    assert cache.ppf_table.shape[1:] == u.shape
    assert cache.ppf_nodes.ndim == 1
    assert not hasattr(cache, "L_inv_path")
    assert not hasattr(cache, "log_det_path")

    start, stop = 4, 13
    u_slice = u[start:stop]
    slice_cache = copula.prepare_emission_cache(u_slice)
    np.testing.assert_allclose(
        cache.ppf(5.0)[start:stop],
        slice_cache.ppf(5.0),
        atol=0.0,
        rtol=0.0,
    )

    new_R_path = np.repeat(
        np.eye(4, dtype=np.float64)[None, :, :], len(u), axis=0)
    copula._set_R_path(new_R_path)
    assert copula._ppf_cache is slice_cache
    assert copula.prepare_emission_cache(u_slice) is slice_cache
    cached_grid = copula.copula_grid_batch(
        u_slice, np.linspace(-1.0, 1.0, 3),
        t_index=0, cache=slice_cache)
    fresh = StochasticStudentDCCCopula(d=4)
    fresh._set_R_path(new_R_path)
    fresh_grid = fresh.copula_grid_batch(
        u_slice, np.linspace(-1.0, 1.0, 3), t_index=0)
    np.testing.assert_allclose(
        cached_grid, fresh_grid, atol=0.0, rtol=0.0)

    path = Path("tmp-stochastic-student-dcc-cache-persistence.json")
    try:
        save_model(copula, path, include_data=True)
        text = path.read_text(encoding="utf-8")
        assert "StochasticStudentDCCEmissionCache" not in text
        assert "\"_ppf_cache\":null" in text
        assert "\"_emission_cache\"" not in text
        assert "\"_ppf_table\"" not in text

        loaded = load_model(path, expected_type=StochasticStudentDCCCopula)
        assert loaded._ppf_cache is None
        assert not hasattr(loaded, "_ppf_table")
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


def test_stochastic_student_gas_score_fast_path_matches_g_space_fd():
    u = _u()[:1]
    R = _R()
    copula = StochasticStudentCopula(d=4, R=R)
    g_t = 1.4
    score_eps = 1e-5
    r_t = float(copula.transform(np.array([g_t]))[0])
    ll_t = float(copula.log_pdf_rows(u, np.array([r_t]))[0])

    score = _cpp_gas.update_one(
        0.0, 0.0, 0.0, g_t, u, copula, "unit", score_eps).score

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

    score = _cpp_gas.update_one(
        0.0, 0.0, 0.0, g_t, u, copula, "unit", score_eps).score

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


def test_stochastic_student_gas_filter_returns_native_paths():
    u = _u()
    R = _R()
    copula = StochasticStudentCopula(d=4, R=R)
    params = (0.08, 0.04, 0.92)

    g_path, r_path, log_likelihood = gas_filter(
        *params, u, copula, scaling="unit")

    assert g_path.shape == (len(u),)
    assert r_path.shape == (len(u),)
    assert np.all(np.isfinite(g_path))
    assert np.all(np.isfinite(r_path))
    assert np.isfinite(log_likelihood)


def test_stochastic_student_gas_cache_uses_current_correlation_state():
    u = _u()
    copula = StochasticStudentCopula(d=4, R=np.eye(4))
    cache = copula.prepare_emission_cache(u)
    params = (0.08, 0.04, 0.92)

    _, _, identity_log_likelihood = gas_filter(
        *params, u, copula, scaling="unit")
    copula._set_R(_R())
    g_path, r_path, log_likelihood = gas_filter(
        *params, u, copula, scaling="unit")

    assert copula.prepare_emission_cache(u) is cache
    assert np.all(np.isfinite(g_path))
    assert np.all(np.isfinite(r_path))
    assert np.isfinite(log_likelihood)
    assert log_likelihood != pytest.approx(identity_log_likelihood)


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


def test_multivariate_gas_rejects_wrong_input_dimension():
    u = np.full((5, 3), 0.5)
    copula = EquicorrGaussianCopula(d=4)

    with pytest.raises(ValueError, match=r"shape \(T, 4\)"):
        gas_filter(0.0, 0.0, 0.0, u, copula)


def test_multivariate_grid_batches_match_reference_density():
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


def test_multivariate_scar_grid_batches_do_not_call_row_pdf(monkeypatch):
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


def test_multivariate_mle_loglik_and_gof_contracts():
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


def test_multivariate_scar_gof_does_not_materialize_forward_weights(monkeypatch):
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


def test_multivariate_scar_gof_matches_materialized_reference():
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


def test_multivariate_scar_gof_reweights_state_by_observed_prefix():
    u = pobs(np.random.default_rng(20260625).standard_normal((12, 3)))
    R = np.array(
        [
            [1.0, 0.42, -0.18],
            [0.42, 1.0, 0.27],
            [-0.18, 0.27, 1.0],
        ],
        dtype=np.float64,
    )
    result = _scar_result(K=15, grid_range=3.25)

    equicorr = EquicorrGaussianCopula(d=3)
    eq_expected = _materialized_equicorr_scar_rosenblatt(
        equicorr, u, result, K=15, grid_range=3.25)
    eq_predictive_only = _equicorr_scar_rosenblatt_predictive_only(
        equicorr, u, result, K=15, grid_range=3.25)
    eq_got = equicorr_rosenblatt_transform(
        equicorr, u, result, K=15, grid_range=3.25)

    np.testing.assert_allclose(eq_got, eq_expected, atol=1e-12, rtol=1e-12)
    assert not np.allclose(
        eq_expected[:, 2], eq_predictive_only[:, 2], atol=1e-5, rtol=1e-5)

    student = StochasticStudentCopula(d=3, R=R)
    st_expected = _materialized_student_scar_rosenblatt(
        student, u, result, K=15, grid_range=3.25)
    st_predictive_only = _student_scar_rosenblatt_predictive_only(
        student, u, result, K=15, grid_range=3.25)
    st_got = stochastic_student_rosenblatt_transform(
        student, u, result, K=15, grid_range=3.25)

    np.testing.assert_allclose(st_got, st_expected, atol=8e-4, rtol=8e-4)
    assert not np.allclose(
        st_expected[:, 2], st_predictive_only[:, 2], atol=1e-5, rtol=1e-5)


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


def test_multivariate_scar_gof_block_size_accounts_for_dimension(monkeypatch):
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


def test_multivariate_models_support_top_level_api_except_pair_h():
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


def test_multivariate_conditional_predict_honors_given_coordinates():
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


def test_multivariate_direct_predict_honors_all_given_coordinates():
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


@pytest.mark.parametrize(
    ("given_value", "quantile_value"),
    [
        (np.nextafter(0.0, 1.0), PSEUDO_OBS_EPS),
        (np.nextafter(1.0, 0.0), 1.0 - PSEUDO_OBS_EPS),
    ],
)
@pytest.mark.parametrize("family", ["gaussian", "student"])
def test_conditional_sampling_uses_common_quantile_boundary(
        family, given_value, quantile_value):
    n = 24
    given = {0: given_value}
    clipped_given = {0: quantile_value}
    if family == "gaussian":
        sample_boundary = sample_gaussian_conditional(
            n, 4, 0.35, given, rng=np.random.default_rng(20260620))
        sample_clipped = sample_gaussian_conditional(
            n, 4, 0.35, clipped_given,
            rng=np.random.default_rng(20260620))
    else:
        sample_boundary = sample_student_conditional(
            n, _R(), 6.0, given, rng=np.random.default_rng(20260620))
        sample_clipped = sample_student_conditional(
            n, _R(), 6.0, clipped_given,
            rng=np.random.default_rng(20260620))

    np.testing.assert_array_equal(
        sample_boundary[:, 1:], sample_clipped[:, 1:])
    np.testing.assert_array_equal(
        sample_boundary[:, 0], np.full(n, given_value))
    assert np.all(np.isfinite(sample_boundary))
    assert np.all((sample_boundary > 0.0) & (sample_boundary < 1.0))


@pytest.mark.parametrize("family", ["gaussian", "student"])
def test_conditional_sampling_multiple_given_and_parameter_paths(family):
    n = 18
    low = PSEUDO_OBS_EPS / 10.0
    high = 1.0 - PSEUDO_OBS_EPS / 10.0
    given = {0: low, 2: high}
    clipped_given = {
        0: PSEUDO_OBS_EPS,
        2: 1.0 - PSEUDO_OBS_EPS,
    }
    if family == "gaussian":
        parameters = np.linspace(0.15, 0.45, n)
        boundary = sample_gaussian_conditional(
            n, 4, parameters, given, rng=np.random.default_rng(20260621))
        clipped = sample_gaussian_conditional(
            n, 4, parameters, clipped_given,
            rng=np.random.default_rng(20260621))
    else:
        R_path = np.repeat(_R()[None, :, :], n, axis=0)
        parameters = np.linspace(4.0, 12.0, n)
        boundary = sample_student_conditional(
            n, R_path, parameters, given,
            rng=np.random.default_rng(20260621))
        clipped = sample_student_conditional(
            n, R_path, parameters, clipped_given,
            rng=np.random.default_rng(20260621))

    np.testing.assert_array_equal(boundary[:, [1, 3]], clipped[:, [1, 3]])
    np.testing.assert_array_equal(boundary[:, 0], np.full(n, low))
    np.testing.assert_array_equal(boundary[:, 2], np.full(n, high))
    assert np.all(np.isfinite(boundary))


@pytest.mark.parametrize("family", ["gaussian", "student"])
def test_native_conditional_sampling_preserves_public_seed_reproducibility(
        family):
    n = 19
    given = {0: 0.2, 2: 0.8}
    seed = 20260622
    if family == "gaussian":
        parameters = np.linspace(0.1, 0.4, n)
        first = sample_gaussian_conditional(
            n, 4, parameters, given, rng=np.random.default_rng(seed))
        second = sample_gaussian_conditional(
            n, 4, parameters, given, rng=np.random.default_rng(seed))
    else:
        R_path = np.repeat(_R()[None, :, :], n, axis=0)
        parameters = np.linspace(4.0, 10.0, n)
        first = sample_student_conditional(
            n, R_path, parameters, given, rng=np.random.default_rng(seed))
        second = sample_student_conditional(
            n, R_path, parameters, given, rng=np.random.default_rng(seed))

    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("family", ["gaussian", "student"])
def test_conditional_sampling_all_fixed_preserves_extreme_valid_values(family):
    given = {
        0: np.nextafter(0.0, 1.0),
        1: 0.5,
        2: np.nextafter(1.0, 0.0),
    }
    if family == "gaussian":
        samples = sample_gaussian_conditional(
            5, 3, 0.3, given, rng=np.random.default_rng(1))
    else:
        samples = sample_student_conditional(
            5, _R()[:3, :3], 7.0, given,
            rng=np.random.default_rng(1))

    expected = np.array([given[i] for i in range(3)])
    np.testing.assert_array_equal(samples, np.tile(expected, (5, 1)))


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_multivariate_given_still_rejects_closed_interval_boundaries(value):
    with pytest.raises(ValueError, match=r"pseudo-observation space \(0, 1\)"):
        validate_multivariate_given({0: value}, 3)


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
