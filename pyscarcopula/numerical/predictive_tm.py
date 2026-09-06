"""Predictive sampling helpers and native SCAR-TM state adapter."""

from __future__ import annotations

import numpy as np

def tm_state_distribution(kappa, mu, nu, u, copula, K=300, grid_range=5.0,
                          grid_method='auto', adaptive=True, pts_per_sigma=4,
                          transition_method='matrix', max_K=None,
                          r_gh=3.0, gh_order=5,
                          horizon='current'):
    """Return the native posterior distribution of ``x_T`` or ``x_{T+1}``."""
    from pyscarcopula._native import scar_ou as _cpp_scar_ou
    from pyscarcopula.numerical._scar_ou_config import AutoTMConfig

    config = AutoTMConfig(
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
    return _cpp_scar_ou.state_distribution(
        kappa, mu, nu, u, copula, config, horizon=horizon)


def sample_grid_distribution(z_grid, prob, n, rng, mode='histogram'):
    """Sample states from a discrete grid distribution.

    ``grid`` returns atoms exactly on ``z_grid``. ``histogram`` treats each
    grid point as the center of a local cell and samples uniformly inside
    that cell, preserving cell masses while removing grid atoms.
    """
    mode = 'grid' if mode is None else str(mode).lower()
    if mode not in {'grid', 'histogram'}:
        raise ValueError("predictive_r_mode must be 'grid' or 'histogram'")
    from pyscarcopula._native import scar_ou

    selection_draws = rng.uniform(0.0, 1.0, size=n)
    jitter_draws = (
        rng.uniform(0.0, 1.0, size=n)
        if mode == 'histogram' and len(z_grid) > 1
        else np.empty(0, dtype=np.float64)
    )
    values, _ = scar_ou.sample_state_distribution_fixed_draws(
        z_grid,
        prob,
        selection_draws,
        jitter_draws,
        mode=mode,
    )
    return values
