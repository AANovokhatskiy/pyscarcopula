"""Native SCAR-TM adapters plus retained GoF Rosenblatt orchestration."""

from __future__ import annotations

import numpy as np

from pyscarcopula._utils import clip_rosenblatt_output
from pyscarcopula.numerical._scar_ou_config import AutoTMConfig
from pyscarcopula.numerical._transition_methods import (
    normalize_ou_grid_transition_method,
)
from pyscarcopula.numerical.gof_blocks import forward_block_size
from pyscarcopula.numerical.predictive_tm import tm_state_distribution
from pyscarcopula.numerical.tm_grid import TMGrid


def _config(
        transition_method, K, grid_range, grid_method, adaptive,
        pts_per_sigma, max_K, r_gh, gh_order):
    return AutoTMConfig(
        transition_method=transition_method,
        K=K,
        grid_range=grid_range,
        grid_method=grid_method,
        adaptive=adaptive,
        pts_per_sigma=pts_per_sigma,
        max_K=max_K,
        r_gh=r_gh,
        gh_order=gh_order,
    )


def tm_loglik(
        kappa, mu, nu, u, copula, K=300, grid_range=5.0,
        grid_method="auto", adaptive=True, pts_per_sigma=4,
        transition_method="matrix", max_K=None, r_gh=3.0, gh_order=5):
    """Return native negative SCAR-TM-OU log-likelihood."""
    from pyscarcopula.numerical import _cpp_scar_ou
    return _cpp_scar_ou.neg_loglik(
        kappa, mu, nu, u, copula,
        _config(
            transition_method, K, grid_range, grid_method, adaptive,
            pts_per_sigma, max_K, r_gh, gh_order))


def tm_forward_predictive_mean(
        kappa, mu, nu, u, copula, K=300, grid_range=5.0,
        grid_method="auto", adaptive=True, pts_per_sigma=4,
        transition_method="matrix", max_K=None, r_gh=3.0, gh_order=5):
    """Return native one-step predictive copula-parameter means."""
    from pyscarcopula.numerical import _cpp_scar_ou
    return _cpp_scar_ou.predictive_mean(
        kappa, mu, nu, u, copula,
        _config(
            transition_method, K, grid_range, grid_method, adaptive,
            pts_per_sigma, max_K, r_gh, gh_order))


def tm_forward_mixture_h(
        kappa, mu, nu, u, copula, K=300, grid_range=5.0,
        grid_method="auto", adaptive=True, pts_per_sigma=4,
        transition_method="matrix", max_K=None, r_gh=3.0, gh_order=5,
        state_cache=None, current_cache_key=None, next_cache_key=None):
    """Return the native mixture h-function."""
    from pyscarcopula.numerical import _cpp_scar_ou
    config = _config(
        transition_method, K, grid_range, grid_method, adaptive,
        pts_per_sigma, max_K, r_gh, gh_order)
    values = _cpp_scar_ou.mixture_h(
        kappa, mu, nu, u, copula, config)
    if state_cache is not None:
        if current_cache_key is not None:
            state_cache[current_cache_key] = _cpp_scar_ou.state_distribution(
                kappa, mu, nu, u, copula, config, horizon="current")
        if next_cache_key is not None:
            state_cache[next_cache_key] = _cpp_scar_ou.state_distribution(
                kappa, mu, nu, u, copula, config, horizon="next")
    return values


def tm_xT_distribution(
        kappa, mu, nu, u, copula, K=300, grid_range=5.0,
        grid_method="auto", adaptive=True, pts_per_sigma=4,
        transition_method="matrix", max_K=None, r_gh=3.0, gh_order=5):
    """Return the native posterior state distribution at the final time."""
    return tm_state_distribution(
        kappa, mu, nu, u, copula, K, grid_range,
        grid_method, adaptive, pts_per_sigma,
        transition_method=transition_method, max_K=max_K,
        r_gh=r_gh, gh_order=gh_order, horizon="current")


def _h_block_on_grid(copula, u_block, r_grid):
    n_block = len(u_block)
    K = len(r_grid)
    u2_grid = np.repeat(u_block[:, 1], K)
    u1_grid = np.repeat(u_block[:, 0], K)
    r_eval = np.tile(r_grid, n_block)
    return copula.h(u2_grid, u1_grid, r_eval).reshape(n_block, K)


def tm_forward_rosenblatt(
        kappa, mu, nu, u, copula, K=300, grid_range=5.0,
        grid_method="auto", adaptive=True, pts_per_sigma=4,
        transition_method="matrix", max_K=None, r_gh=3.0, gh_order=5):
    """Retained GoF orchestration for the mixture Rosenblatt transform."""
    transition_method = normalize_ou_grid_transition_method(transition_method)
    grid = TMGrid(
        kappa, mu, nu, len(u), K, grid_range,
        grid_method, adaptive, pts_per_sigma,
        transition_method=transition_method, max_K=max_K,
        r_gh=r_gh, gh_order=gh_order)
    r_grid = copula.transform(grid.z + grid.mu)
    phi = grid.p0.copy()
    out = np.empty((len(u), 2), dtype=np.float64)
    out[:, 0] = u[:, 0]
    block_size = forward_block_size(grid.K)
    prepare_cache = getattr(copula, "prepare_emission_cache", None)
    cache = None if prepare_cache is None else prepare_cache(u)

    for start in range(0, len(u), block_size):
        stop = min(len(u), start + block_size)
        u_block = u[start:stop]
        if cache is None:
            fi_block = copula.copula_grid_batch(
                u_block, grid.z + grid.mu)
        else:
            fi_block = copula.copula_grid_batch(
                u_block, grid.z + grid.mu,
                t_index=start, cache=cache)
        h_block = _h_block_on_grid(copula, u_block, r_grid)
        for local, index in enumerate(range(start, stop)):
            weights = grid.predictive_weights_from_phi(phi)
            out[index, 1] = np.sum(h_block[local] * weights)
            if index < len(u) - 1:
                phi = grid.advance_forward_phi(phi, fi_block[local])
    return clip_rosenblatt_output(out)


__all__ = [
    "tm_loglik",
    "tm_forward_predictive_mean",
    "tm_forward_rosenblatt",
    "tm_forward_mixture_h",
    "tm_xT_distribution",
]
