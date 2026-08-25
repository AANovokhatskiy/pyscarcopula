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
from pyscarcopula._types import (
    GASResult,
    LatentResult,
    gas_params,
    ou_params,
)
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
from pyscarcopula.numerical import _cpp_gas, _cpp_scar_ou
from pyscarcopula.numerical._scar_ou_config import AutoTMConfig
from pyscarcopula.numerical.gas_filter import gas_filter
from pyscarcopula.stattests import (
    equicorr_rosenblatt_transform,
    gof_test,
    stochastic_student_rosenblatt_transform,
)


def student_logpdf(u, R, df):
    copula = StochasticStudentCopula(d=u.shape[1], R=R)
    return copula.log_pdf_rows(u, df)


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


def _materialized_equicorr_scar_rosenblatt(
        copula, u, fit_result, K, grid_range, config=None):
    eps = 1e-10
    u_c = np.clip(u, eps, 1.0 - eps)
    x_norm = norm.ppf(u_c)
    T, d = u.shape
    kappa, mu, nu = fit_result.params.values
    grid_kwargs = {}
    if config is not None:
        grid_kwargs = {
            "grid_method": config.grid_method,
            "adaptive": config.adaptive,
            "pts_per_sigma": config.pts_per_sigma,
            "transition_method": config.transition_method,
            "max_K": config.max_K,
            "r_gh": config.r_gh,
            "gh_order": config.gh_order,
        }
    grid = TMGrid(
        kappa, mu, nu, T, K, grid_range, **grid_kwargs)
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


def _materialized_student_scar_rosenblatt(
        copula, u, fit_result, K, grid_range, config=None):
    eps = 1e-10
    T, d = u.shape
    kappa, mu, nu = fit_result.params.values
    grid_kwargs = {}
    if config is not None:
        grid_kwargs = {
            "grid_method": config.grid_method,
            "adaptive": config.adaptive,
            "pts_per_sigma": config.pts_per_sigma,
            "transition_method": config.transition_method,
            "max_K": config.max_K,
            "r_gh": config.r_gh,
            "gh_order": config.gh_order,
        }
    grid = TMGrid(
        kappa, mu, nu, T, K, grid_range, **grid_kwargs)
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
        np.testing.assert_allclose(got, ref, atol=1e-10, rtol=1e-12)


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
def test_stochastic_student_ppf_cache_is_accurate_across_df_range(df):
    u = _u()
    copula = StochasticStudentCopula(d=4, R=_R())
    cache = copula.prepare_emission_cache(u)
    expected = stdtrit(
        df,
        np.clip(u, PSEUDO_OBS_EPS, 1.0 - PSEUDO_OBS_EPS),
    )

    tolerance = 2e-13 if df > cache.ppf_nodes[-1] else 1e-8
    np.testing.assert_allclose(
        cache.ppf(df),
        expected,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        cache.ppf_rows(df, 3, 11),
        expected[3:11],
        rtol=tolerance,
        atol=tolerance,
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
    assert table.nodes[0] == pytest.approx(2.0 + 1e-6)
    assert table.nodes[-1] == pytest.approx(1000.0)
    for df in (table.nodes[0], table.nodes[-1]):
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


def test_multivariate_gas_rejects_wrong_input_dimension():
    u = np.full((5, 3), 0.5)
    copula = EquicorrGaussianCopula(d=4)

    with pytest.raises(ValueError, match=r"shape \(T, 4\)"):
        gas_filter(0.0, 0.0, 0.0, u, copula)


def test_multivariate_grid_batches_match_reference_density():
    u = _u()
    R = _R()
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
    models = [
        EquicorrGaussianCopula(d=3),
        StochasticStudentCopula(d=3, R=R),
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


@pytest.mark.parametrize(
    ("transition_method", "grid_method", "atol"),
    [
        ("matrix", "dense", 2e-13),
        ("matrix", "sparse", 2e-8),
        ("local", "auto", 2e-13),
    ],
)
def test_native_equicorr_gaussian_rosenblatt_matches_tmgrid_oracle(
        transition_method, grid_method, atol):
    u = pobs(np.random.default_rng(20260730).standard_normal((19, 4)))
    copula = EquicorrGaussianCopula(d=4)
    result = _scar_result(K=35, grid_range=3.25)
    config = AutoTMConfig(
        K=35,
        grid_range=3.25,
        grid_method=grid_method,
        adaptive=False,
        transition_method=transition_method,
        max_K=None,
        gh_order=7,
    )

    expected = _materialized_equicorr_scar_rosenblatt(
        copula,
        u,
        result,
        K=35,
        grid_range=3.25,
        config=config,
    )
    actual = _cpp_scar_ou.gaussian_rosenblatt(
        *result.params.values, u, copula, config)

    np.testing.assert_allclose(
        actual, expected, rtol=0.0, atol=atol)


@pytest.mark.parametrize(
    ("transition_method", "grid_method", "atol"),
    [
        ("matrix", "dense", 5e-8),
        ("matrix", "sparse", 5e-8),
        ("local", "auto", 5e-8),
    ],
)
def test_native_student_rosenblatt_matches_tmgrid_oracle(
        transition_method, grid_method, atol):
    u = pobs(np.random.default_rng(20260730).standard_normal((19, 4)))
    copula = StochasticStudentCopula(d=4, R=_R())
    result = _scar_result(K=35, grid_range=3.25)
    config = AutoTMConfig(
        K=35,
        grid_range=3.25,
        grid_method=grid_method,
        adaptive=False,
        transition_method=transition_method,
        max_K=None,
        gh_order=7,
    )

    expected = _materialized_student_scar_rosenblatt(
        copula,
        u,
        result,
        K=35,
        grid_range=3.25,
        config=config,
    )
    actual = _cpp_scar_ou.student_rosenblatt(
        *result.params.values, u, copula, config)

    np.testing.assert_allclose(
        actual, expected, rtol=0.0, atol=atol)


@pytest.mark.parametrize(
    ("copula", "native_name", "transform"),
    [
        (
            EquicorrGaussianCopula(d=4),
            "gaussian_rosenblatt",
            equicorr_rosenblatt_transform,
        ),
        (
            StochasticStudentCopula(d=4, R=_R()),
            "student_rosenblatt",
            stochastic_student_rosenblatt_transform,
        ),
    ],
)
def test_multivariate_scar_gof_preserves_stored_grid_options(
        monkeypatch, copula, native_name, transform):
    u = _u()[:8]
    result = LatentResult(
        log_likelihood=0.0,
        method="SCAR-TM-OU",
        copula_name=copula.name,
        success=True,
        params=ou_params(0.8, 0.0, 1.0),
        K=43,
        grid_range=2.75,
        grid_method="sparse",
        adaptive=False,
        pts_per_sigma=6,
        transition_method="matrix",
        max_K=None,
        r_gh=2.25,
        gh_order=7,
    )
    captured = {}

    def fake_native(kappa, mu, nu, u_arg, copula_arg, config):
        captured.update(vars(config))
        return np.asarray(u_arg, dtype=np.float64)

    monkeypatch.setattr(_cpp_scar_ou, native_name, fake_native)
    transformed = transform(
        copula, u, result, K=result.K, grid_range=result.grid_range)

    np.testing.assert_array_equal(transformed, u)
    assert captured == {
        "transition_method": "matrix",
        "small_kdt": 0.01,
        "basis_order": 32,
        "quad_order": None,
        "K": 43,
        "grid_range": 2.75,
        "grid_method": "sparse",
        "adaptive": False,
        "pts_per_sigma": 6,
        "max_K": None,
        "gh_order": 7,
        "r_gh": 2.25,
        "n_threads": 1,
    }


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


def test_equicorr_scar_gof_native_path_does_not_construct_tmgrid(
        monkeypatch):
    u = pobs(np.random.default_rng(20260731).standard_normal((18, 4)))
    copula = EquicorrGaussianCopula(d=4)
    result = _scar_result(K=31, grid_range=3.0)

    def fail_tmgrid(*args, **kwargs):
        raise AssertionError(
            "equicorrelation Gaussian GoF must not construct TMGrid")

    monkeypatch.setattr(TMGrid, "__init__", fail_tmgrid)
    transformed = equicorr_rosenblatt_transform(
        copula, u, result, K=31, grid_range=3.0)

    assert transformed.shape == u.shape
    assert np.all(np.isfinite(transformed))


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


def test_student_scar_gof_native_path_does_not_construct_tmgrid(
        monkeypatch):
    u = pobs(np.random.default_rng(20260801).standard_normal((18, 4)))
    copula = StochasticStudentCopula(d=4, R=_R())
    result = _scar_result(K=31, grid_range=3.0)

    def fail_tmgrid(*args, **kwargs):
        raise AssertionError("Student GoF must not construct TMGrid")

    monkeypatch.setattr(TMGrid, "__init__", fail_tmgrid)
    transformed = stochastic_student_rosenblatt_transform(
        copula, u, result, K=31, grid_range=3.0)

    assert transformed.shape == u.shape
    assert np.all(np.isfinite(transformed))


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


def test_student_scar_gof_bypasses_python_block_sizing(monkeypatch):
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

    student = StochasticStudentCopula(d=4, R=R)
    stochastic_student_rosenblatt_transform(
        student, u, result, K=8, grid_range=3.0)

    assert calls == []


def test_multivariate_models_support_top_level_api_except_pair_h():
    u = _u()
    R = _R()

    def make_model(name):
        if name == "equicorr":
            return EquicorrGaussianCopula(d=4)
        return StochasticStudentCopula(d=4, R=R)

    for name in ("equicorr", "student"):
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

    for copula in models:
        copula.fit(u, method="mle")
        samples = copula.predict(
            7, u=u, given=given, rng=np.random.default_rng(1))

        expected = np.array([given[i] for i in range(4)], dtype=np.float64)
        np.testing.assert_allclose(samples, np.tile(expected, (7, 1)))


@pytest.mark.parametrize("horizon", ["current", "next"])
@pytest.mark.parametrize(
    "method",
    ["scar-tm-ou", "gas"],
)
@pytest.mark.parametrize("corr_mode", ["fixed", "factor"])
def test_stochastic_student_direct_dynamic_predict_matches_api(
        corr_mode, method, horizon):
    u = _u()
    if corr_mode == "fixed":
        copula = StochasticStudentCopula(d=4, R=_R())
    else:
        copula = StochasticStudentCopula(
            d=4, corr_mode="factor", factor_rank=2)
        copula.initialize_factor(u)
    if method == "scar-tm-ou":
        result = _scar_result(K=12, grid_range=3.0)
    elif method == "gas":
        result = GASResult(
            log_likelihood=0.0,
            method="GAS",
            copula_name=copula.name,
            success=True,
            params=gas_params(0.1, 0.2, 0.7),
            scaling="unit",
            r_last=5.0,
        )
    copula.fit_result = result
    copula._last_u = u
    given = {0: 0.25, 2: 0.75}
    seed = 20260728

    direct = copula.predict(
        16,
        horizon=horizon,
        given=given,
        rng=np.random.default_rng(seed),
    )
    through_api = predict(
        copula,
        u,
        result,
        16,
        horizon=horizon,
        given=given,
        rng=np.random.default_rng(seed),
    )

    np.testing.assert_array_equal(direct, through_api)


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


def test_fallback_predict_draws_independent_latent_parameter_per_sample():
    """M1 regression: fallback predict (no conditioning data) must draw n
    independent latent OU states, not a single shared one."""
    u = _u()
    R = _R()
    n = 64

    for copula in (EquicorrGaussianCopula(d=4),
                   StochasticStudentCopula(d=4, R=R)):
        result = fit(copula, u, method="scar-tm-ou", K=25, grid_range=4.0)
        copula.fit_result = result  # ensure SCAR result, not a stored MLE one
        copula._last_u = None  # force the stationary-OU fallback branch

        captured = {}
        original = copula.sample_conditional

        def spy(n_, r=None, given=None, rng=None, _orig=original):
            captured["r"] = np.atleast_1d(np.asarray(r, dtype=np.float64))
            return _orig(n_, r=r, given=given, rng=rng)

        copula.sample_conditional = spy
        try:
            copula.predict(n, rng=np.random.default_rng(0))
        finally:
            copula.sample_conditional = original

        r = captured["r"]
        assert r.shape == (n,)
        assert np.unique(r).size > 1


def test_mle_optimizer_failure_does_not_fake_convergence(monkeypatch):
    """M2 regression: failed likelihood evaluations must not yield a zero
    gradient that L-BFGS-B would misread as convergence."""
    from pyscarcopula.numerical import static_likelihood

    u = _u()
    copula = StochasticStudentCopula(d=4, R=_R())

    def failing_objective(self, parameter, *, fail_value=1e10):
        return float(fail_value), np.array([0.0])

    monkeypatch.setattr(
        static_likelihood.StaticLikelihoodEvaluator,
        "objective_and_gradient",
        failing_objective,
    )
    result = copula.fit(u, method="mle")
    assert not result.success


def test_mle_optimizer_unexpected_error_propagates(monkeypatch):
    """M2 regression: only expected numerical exceptions may be swallowed
    by the objective wrapper; real bugs must surface."""
    from pyscarcopula.numerical import static_likelihood

    u = _u()
    copula = StochasticStudentCopula(d=4, R=_R())

    def buggy_objective(self, parameter, *, fail_value=1e10):
        raise RuntimeError("simulated bug")

    monkeypatch.setattr(
        static_likelihood.StaticLikelihoodEvaluator,
        "objective_and_gradient",
        buggy_objective,
    )
    with pytest.raises(RuntimeError, match="simulated bug"):
        copula.fit(u, method="mle")


def test_student_ppf_table_memory_limit_falls_back_to_exact():
    """M4 regression: over the memory budget the table is skipped and
    evaluations use the exact stdtrit quantile."""
    u = _u()
    table = StudentPPFTable(u, max_table_bytes=1)
    assert table.table is None
    for df in (3.0, 7.5, 300.0):
        np.testing.assert_allclose(
            table(df), stdtrit(df, table.u), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(
        table.rows(4.5, 2, 9), stdtrit(4.5, table.u[2:9]),
        rtol=1e-12, atol=0.0)


def test_scar_log_likelihood_with_degraded_ppf_cache(monkeypatch):
    """M4 regression: a cache without a precomputed table still works
    end-to-end (C++ exact-quantile fallback) and matches the cached path."""
    import pyscarcopula.copula.multivariate.stochastic_student as mod

    u = _u()
    R = _R()
    ref_copula = StochasticStudentCopula(d=4, R=R)
    result = fit(ref_copula, u, method="scar-tm-ou", K=25, grid_range=4.0)
    reference = log_likelihood(
        ref_copula, u, result, K=25, grid_range=4.0)

    monkeypatch.setattr(
        mod, "_PPFTable",
        lambda values: StudentPPFTable(values, max_table_bytes=1))
    degraded_copula = StochasticStudentCopula(d=4, R=R)
    degraded = log_likelihood(
        degraded_copula, u, result, K=25, grid_range=4.0)

    assert np.isfinite(degraded)
    assert degraded == pytest.approx(reference, rel=1e-6, abs=1e-6)


def test_static_gaussian_student_fit_rejects_non_mle_method():
    """M8 regression: static models declare method='mle' explicitly and
    reject dynamic strategies with a clear error."""
    from pyscarcopula.copula.multivariate.gaussian import GaussianCopula
    from pyscarcopula.copula.multivariate.student import StudentCopula

    u = _u()
    for copula in (GaussianCopula(), StudentCopula()):
        copula.fit(u)  # default method='mle' works
        with pytest.raises(ValueError, match="only method='mle'"):
            copula.fit(u, method="scar-tm-ou")
        with pytest.raises(ValueError, match="only method='mle'"):
            copula.fit(u, method="gas")


def test_static_gaussian_student_conditional_predict_honors_given():
    """M8 regression: GaussianCopula/StudentCopula support given= in
    predict, matching the Stochastic/Equicorr contract."""
    from pyscarcopula.copula.multivariate.gaussian import GaussianCopula
    from pyscarcopula.copula.multivariate.student import StudentCopula

    u = _u()
    given = {0: 0.25, 2: 0.75}
    for copula in (GaussianCopula(), StudentCopula()):
        copula.fit(u)
        samples = copula.predict(
            64, given=given, rng=np.random.default_rng(7))
        assert samples.shape == (64, 4)
        assert np.allclose(samples[:, 0], 0.25)
        assert np.allclose(samples[:, 2], 0.75)
        assert np.all((samples > 0.0) & (samples < 1.0))


def test_static_conditional_sampling_through_public_api_honors_given():
    """Static multivariate models must not receive the dynamic ``r`` arg."""
    from pyscarcopula.copula.multivariate.gaussian import GaussianCopula
    from pyscarcopula.copula.multivariate.student import StudentCopula

    u = _u()
    given = {0: 0.25, 2: 0.75}
    for copula in (GaussianCopula(), StudentCopula()):
        result = fit(copula, u, method="mle")
        for public_sampler in (predict, sample):
            samples = public_sampler(
                copula,
                u,
                result,
                64,
                given=given,
                rng=np.random.default_rng(7),
            )
            assert samples.shape == (64, 4)
            assert np.allclose(samples[:, 0], 0.25)
            assert np.allclose(samples[:, 2], 0.75)
            assert np.all((samples > 0.0) & (samples < 1.0))


def test_gaussian_copula_conditional_mean_matches_oracle():
    """M8: conditional mean of free coordinates equals
    R_fg R_gg^{-1} z_g for a general (non-equicorr) correlation."""
    from pyscarcopula.copula.multivariate.gaussian import GaussianCopula

    copula = GaussianCopula()
    copula.fit(_u())
    R = copula.corr
    given_u = {0: 0.8, 1: 0.3}
    z_g = norm.ppf(np.array([0.8, 0.3]))
    R_gg = R[:2, :2]
    R_fg = R[2:, :2]
    expected_z = R_fg @ np.linalg.solve(R_gg, z_g)
    var_z = 1.0 - np.sum((R_fg @ np.linalg.inv(R_gg)) * R_fg, axis=1)
    expected_u_mean = norm.cdf(expected_z / np.sqrt(1.0 + var_z))

    samples = copula.predict(
        40000, given=given_u, rng=np.random.default_rng(11))
    assert np.allclose(samples[:, 0], 0.8)
    assert np.allclose(samples[:, 1], 0.3)
    np.testing.assert_allclose(
        samples[:, 2:].mean(axis=0), expected_u_mean, atol=0.01)


def test_student_conditional_moments_match_analytic_oracle():
    """M9: the conditional Student kernel must apply the scale
    (nu + z_g' R_gg^{-1} z_g) / (nu + k) to the Schur complement.

    For a multivariate t with df nu and correlation R, the conditional
    distribution of the free coordinates given k fixed ones is t with
    df nu + k, mean R_fg R_gg^{-1} z_g, and covariance
    (nu + delta) / (nu + k - 2) * Schur, delta = z_g' R_gg^{-1} z_g.
    """
    nu = 8.0
    R = np.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.4],
        [0.3, 0.4, 1.0],
    ])
    given = {0: 0.85, 1: 0.20}
    n = 60000

    samples = sample_student_conditional(
        n, R, nu, given=given, rng=np.random.default_rng(123))

    # Latent recovery inverts the cdf applied in sample_student_conditional.
    x_free = t_dist.ppf(samples[:, 2], df=nu)

    z_g = t_dist.ppf(np.array([given[0], given[1]]), df=nu)
    R_gg = R[:2, :2]
    R_fg = R[2:, :2]
    solved = np.linalg.solve(R_gg, z_g)
    delta = float(z_g @ solved)
    expected_mean = float((R_fg @ solved)[0])
    schur = 1.0 - float((R_fg @ np.linalg.solve(R_gg, R_fg.T))[0, 0])
    k = len(given)
    expected_var = (nu + delta) / (nu + k - 2.0) * schur

    assert np.allclose(samples[:, 0], given[0])
    assert np.allclose(samples[:, 1], given[1])
    assert np.mean(x_free) == pytest.approx(
        expected_mean, abs=4.0 * np.sqrt(expected_var / n))
    assert np.var(x_free) == pytest.approx(expected_var, rel=0.05)


def test_api_fit_stores_strategy_result_on_copula():
    """M10 regression: top-level api.fit must leave the strategy result in
    copula.fit_result (not a stale intermediate MLE), mirroring copula.fit()."""
    u = _u()
    R = _R()

    for copula in (EquicorrGaussianCopula(d=4),
                   StochasticStudentCopula(d=4, R=R)):
        result = fit(copula, u, method="scar-tm-ou", K=25, grid_range=4.0)
        assert copula.fit_result is result
        if "_last_latent_result" in copula.__dict__:
            assert copula._last_latent_result is result
        # Convenience path now reaches the SCAR branch without explicit args.
        samples = copula.predict(8, rng=np.random.default_rng(3))
        assert samples.shape == (8, 4)

        mle = fit(copula, u, method="mle", gtol=0.5, maxiter=10, maxfun=10)
        assert copula.fit_result is mle
