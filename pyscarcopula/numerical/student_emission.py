"""Fast emission kernels for scalar-latent Student-t copulas."""

from math import lgamma

import numpy as np
from numba import njit


@njit(cache=True)
def _find_ppf_interval(nodes, df):
    idx = np.searchsorted(nodes, df) - 1
    if idx < 0:
        idx = 0
    hi = len(nodes) - 2
    if idx > hi:
        idx = hi
    return idx


@njit(cache=True)
def _interp_ppf_value(nodes, table, df, row, col):
    idx = _find_ppf_interval(nodes, df)
    denom = nodes[idx + 1] - nodes[idx]
    alpha = (df - nodes[idx]) / denom
    return (1.0 - alpha) * table[idx, row, col] + alpha * table[idx + 1, row, col]


@njit(cache=True)
def student_interp_ppf_rows(nodes, table, df, start, stop):
    """Interpolate t-quantiles for rows [start:stop] from a PPF table."""
    n = stop - start
    d = table.shape[2]
    out = np.empty((n, d), dtype=np.float64)
    for t in range(n):
        row = start + t
        for j in range(d):
            out[t, j] = _interp_ppf_value(nodes, table, df, row, j)
    return out


@njit(cache=True)
def student_logpdf_from_x_rows(x, df, L_inv, log_det):
    """Log Student copula density from precomputed t-quantiles."""
    T = x.shape[0]
    d = x.shape[1]
    out = np.empty(T, dtype=np.float64)

    log_norm_joint = (
        lgamma(0.5 * (df + d))
        - lgamma(0.5 * df)
        - 0.5 * d * np.log(df * np.pi)
        - 0.5 * log_det
    )
    log_norm_marg = (
        lgamma(0.5 * (df + 1.0))
        - lgamma(0.5 * df)
        - 0.5 * np.log(df * np.pi)
    )

    for t in range(T):
        quad = 0.0
        for i in range(d):
            y_i = 0.0
            for j in range(d):
                y_i += L_inv[i, j] * x[t, j]
            quad += y_i * y_i

        log_marg = 0.0
        for j in range(d):
            x_j = x[t, j]
            log_marg += log_norm_marg - 0.5 * (df + 1.0) * np.log1p(x_j * x_j / df)

        log_joint = log_norm_joint - 0.5 * (df + d) * np.log1p(quad / df)
        out[t] = log_joint - log_marg

    return out


@njit(cache=True)
def _student_logpdf_table_row(nodes, table, row, df, d, L_inv, log_det):
    log_norm_joint = (
        lgamma(0.5 * (df + d))
        - lgamma(0.5 * df)
        - 0.5 * d * np.log(df * np.pi)
        - 0.5 * log_det
    )
    log_norm_marg = (
        lgamma(0.5 * (df + 1.0))
        - lgamma(0.5 * df)
        - 0.5 * np.log(df * np.pi)
    )

    quad = 0.0
    log_marg = 0.0
    x = np.empty(d, dtype=np.float64)
    for j in range(d):
        x_j = _interp_ppf_value(nodes, table, df, row, j)
        x[j] = x_j
        log_marg += log_norm_marg - 0.5 * (df + 1.0) * np.log1p(x_j * x_j / df)

    for i in range(d):
        y_i = 0.0
        for j in range(d):
            y_i += L_inv[i, j] * x[j]
        quad += y_i * y_i

    log_joint = log_norm_joint - 0.5 * (df + d) * np.log1p(quad / df)
    return log_joint - log_marg


@njit(cache=True)
def student_pdf_grid_numba(nodes, table, start, n_rows, df_grid, L_inv, log_det):
    """Evaluate Student copula densities for a row block and df grid."""
    K = len(df_grid)
    d = table.shape[2]
    fi = np.empty((n_rows, K), dtype=np.float64)
    for k in range(K):
        df = df_grid[k]
        for t in range(n_rows):
            row = start + t
            fi[t, k] = np.exp(
                _student_logpdf_table_row(nodes, table, row, df, d, L_inv, log_det)
            )
    return fi


@njit(cache=True)
def student_pdf_and_grad_grid_fd_numba(
        nodes, table, start, n_rows, df_grid, dpsi, L_inv, log_det, eps):
    """Evaluate density and d density / d latent x using df finite diff."""
    K = len(df_grid)
    d = table.shape[2]
    fi = np.empty((n_rows, K), dtype=np.float64)
    dfi = np.empty((n_rows, K), dtype=np.float64)
    for k in range(K):
        df_c = df_grid[k]
        df_p = df_c + eps
        df_m = df_c - eps
        if df_m < 2.001:
            df_m = 2.001
        denom = df_p - df_m
        for t in range(n_rows):
            row = start + t
            lc = _student_logpdf_table_row(nodes, table, row, df_c, d, L_inv, log_det)
            lp = _student_logpdf_table_row(nodes, table, row, df_p, d, L_inv, log_det)
            lm = _student_logpdf_table_row(nodes, table, row, df_m, d, L_inv, log_det)
            val = np.exp(lc)
            fi[t, k] = val
            dfi[t, k] = val * (lp - lm) / denom * dpsi[k]
    return fi, dfi


@njit(cache=True)
def student_logpdf_and_dlog_rows_fd_numba(
        nodes, table, start, n_rows, df_arr, L_inv, log_det, eps):
    """Evaluate row-wise log-density and d log-density / d df."""
    d = table.shape[2]
    ll = np.empty(n_rows, dtype=np.float64)
    dlog = np.empty(n_rows, dtype=np.float64)
    scalar_df = len(df_arr) == 1
    for t in range(n_rows):
        row = start + t
        df_c = df_arr[0] if scalar_df else df_arr[t]
        df_p = df_c + eps
        df_m = df_c - eps
        if df_m < 2.0 + 1e-8:
            df_m = 2.0 + 1e-8
        lc = _student_logpdf_table_row(nodes, table, row, df_c, d, L_inv, log_det)
        lp = _student_logpdf_table_row(nodes, table, row, df_p, d, L_inv, log_det)
        lm = _student_logpdf_table_row(nodes, table, row, df_m, d, L_inv, log_det)
        ll[t] = lc
        dlog[t] = (lp - lm) / (df_p - df_m)
    return ll, dlog


@njit(cache=True)
def _student_softplus_scalar(x):
    if x > 30.0:
        return x
    if x < -500.0:
        x = -500.0
    elif x > 30.0:
        x = 30.0
    return np.log1p(np.exp(x))


@njit(cache=True)
def _student_softplus_deriv_scalar(x):
    if x < -500.0:
        x = -500.0
    elif x > 500.0:
        x = 500.0
    return 1.0 / (1.0 + np.exp(-x))


@njit(cache=True)
def student_gas_filter_numba(
        omega, gamma, beta, nodes, table, L_inv, log_det, score_eps,
        g_clip, s_clip):
    """Fast GAS(unit) filter for fixed-R StochasticStudentCopula."""
    T = table.shape[1]
    d = table.shape[2]
    g_path = np.empty(T, dtype=np.float64)
    r_path = np.empty(T, dtype=np.float64)
    total_logL = 0.0

    if abs(beta) < 1.0 - 1e-8:
        g_t = omega / (1.0 - beta)
    else:
        g_t = omega

    for t in range(T):
        g_path[t] = g_t
        df_t = 2.0 + _student_softplus_scalar(g_t)
        r_path[t] = df_t

        ll_t = _student_logpdf_table_row(
            nodes, table, t, df_t, d, L_inv, log_det)
        if not np.isfinite(ll_t):
            return g_path, r_path, -1e10
        total_logL += ll_t

        if t < T - 1:
            df_p = df_t + score_eps
            df_m = df_t - score_eps
            if df_m < 2.0 + 1e-8:
                df_m = 2.0 + 1e-8
            lp = _student_logpdf_table_row(
                nodes, table, t, df_p, d, L_inv, log_det)
            lm = _student_logpdf_table_row(
                nodes, table, t, df_m, d, L_inv, log_det)
            dlog_dr = (lp - lm) / (df_p - df_m)
            s_t = dlog_dr * _student_softplus_deriv_scalar(g_t)
            if not np.isfinite(s_t):
                return g_path, r_path, -1e10
            if s_t > s_clip:
                s_t = s_clip
            elif s_t < -s_clip:
                s_t = -s_clip

            g_t = omega + beta * g_t + gamma * s_t
            if g_t > g_clip:
                g_t = g_clip
            elif g_t < -g_clip:
                g_t = -g_clip

    return g_path, r_path, total_logL


__all__ = [
    "student_interp_ppf_rows",
    "student_logpdf_from_x_rows",
    "student_pdf_grid_numba",
    "student_pdf_and_grad_grid_fd_numba",
    "student_logpdf_and_dlog_rows_fd_numba",
    "student_gas_filter_numba",
]
