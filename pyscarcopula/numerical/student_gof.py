"""Numba helpers for Student-t copula goodness-of-fit transforms."""

import math

import numpy as np
from numba import njit


@njit(cache=True)
def student_conditional_z_block(x_all, df_grid, beta, r_inv, sigma):
    """Compute conditional Student residuals for a fixed-R block.

    Parameters
    ----------
    x_all : ndarray, shape (B, K, d)
        Student quantiles for one row block and all grid df values.
    df_grid : ndarray, shape (K,)
        Degrees of freedom on the latent grid.
    beta : ndarray, shape (d - 1, d)
        Padded conditional regression coefficients. Row ``i - 1`` uses the
        first ``i`` entries.
    r_inv : ndarray, shape (d - 1, d, d)
        Padded inverse leading correlation submatrices.
    sigma : ndarray, shape (d - 1,)
        Conditional standard deviations.

    Returns
    -------
    ndarray, shape (d - 1, B, K)
        Standardized residuals for conditional dimensions 1..d-1.
    """
    B = x_all.shape[0]
    K = x_all.shape[1]
    d = x_all.shape[2]
    out = np.empty((d - 1, B, K), dtype=np.float64)

    for dim in range(1, d):
        cond_idx = dim - 1
        for b in range(B):
            for k in range(K):
                mu_cond = 0.0
                for p in range(dim):
                    mu_cond += x_all[b, k, p] * beta[cond_idx, p]

                quad = 0.0
                for p in range(dim):
                    acc = 0.0
                    for q in range(dim):
                        acc += r_inv[cond_idx, p, q] * x_all[b, k, q]
                    quad += x_all[b, k, p] * acc

                df_cond = df_grid[k] + dim
                scale = (df_grid[k] + quad) / df_cond
                if scale < 1e-12:
                    scale = 1e-12
                out[cond_idx, b, k] = (
                    (x_all[b, k, dim] - mu_cond)
                    / (sigma[cond_idx] * np.sqrt(scale))
                )

    return out


@njit(cache=True)
def _student_leading_log_copula_density(x_all, df, r_inv, log_det, dim, b, k):
    if dim <= 1:
        return 0.0

    quad = 0.0
    for p in range(dim):
        acc = 0.0
        for q in range(dim):
            acc += r_inv[dim - 1, p, q] * x_all[b, k, q]
        quad += x_all[b, k, p] * acc

    dim_float = float(dim)
    joint_log = (
        math.lgamma(0.5 * (df + dim_float))
        - math.lgamma(0.5 * df)
        - 0.5 * dim_float * math.log(df * math.pi)
        - 0.5 * log_det[dim - 1]
        - 0.5 * (df + dim_float) * math.log1p(quad / df)
    )

    marginal_log = 0.0
    for p in range(dim):
        x = x_all[b, k, p]
        marginal_log += (
            math.lgamma(0.5 * (df + 1.0))
            - math.lgamma(0.5 * df)
            - 0.5 * math.log(df * math.pi)
            - 0.5 * (df + 1.0) * math.log1p((x * x) / df)
        )
    return joint_log - marginal_log


@njit(cache=True)
def student_weighted_cdf_block(
        cdf_blocks, weights_block, x_all, df_grid, r_inv, log_det):
    """Mix conditional CDF values by state weights reconditioned on prefixes."""
    n_cond = cdf_blocks.shape[0]
    B = cdf_blocks.shape[1]
    K = cdf_blocks.shape[2]
    out = np.empty((B, n_cond), dtype=np.float64)
    for b in range(B):
        for dim in range(n_cond):
            prefix_dim = dim + 1
            max_log_weight = -np.inf
            for k in range(K):
                weight = weights_block[b, k]
                if weight > 0.0:
                    log_weight = math.log(weight)
                    log_weight += _student_leading_log_copula_density(
                        x_all, df_grid[k], r_inv, log_det, prefix_dim, b, k)
                    if log_weight > max_log_weight:
                        max_log_weight = log_weight

            total = 0.0
            norm = 0.0
            for k in range(K):
                weight = weights_block[b, k]
                if weight > 0.0 and np.isfinite(max_log_weight):
                    log_weight = math.log(weight)
                    log_weight += _student_leading_log_copula_density(
                        x_all, df_grid[k], r_inv, log_det, prefix_dim, b, k)
                    reweighted = math.exp(log_weight - max_log_weight)
                    norm += reweighted
                    total += reweighted * cdf_blocks[dim, b, k]
            if norm > 0.0:
                out[b, dim] = total / norm
            else:
                fallback = 0.0
                for k in range(K):
                    fallback += weights_block[b, k] * cdf_blocks[dim, b, k]
                out[b, dim] = fallback
    return out


__all__ = [
    "student_conditional_z_block",
    "student_weighted_cdf_block",
]
