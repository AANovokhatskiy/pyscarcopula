"""Native SCAR-TM adapters."""

from __future__ import annotations

from pyscarcopula.numerical._scar_ou_config import AutoTMConfig
from pyscarcopula.numerical.predictive_tm import tm_state_distribution


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


def tm_forward_rosenblatt(
        kappa, mu, nu, u, copula, K=300, grid_range=5.0,
        grid_method="auto", adaptive=True, pts_per_sigma=4,
        transition_method="matrix", max_K=None, r_gh=3.0, gh_order=5):
    """Return the fully native bivariate mixture Rosenblatt transform."""
    from pyscarcopula.numerical import _cpp_scar_ou
    return _cpp_scar_ou.forward_rosenblatt(
        kappa, mu, nu, u, copula,
        _config(
            transition_method, K, grid_range, grid_method, adaptive,
            pts_per_sigma, max_K, r_gh, gh_order))


__all__ = [
    "tm_loglik",
    "tm_forward_predictive_mean",
    "tm_forward_rosenblatt",
    "tm_forward_mixture_h",
    "tm_xT_distribution",
]
