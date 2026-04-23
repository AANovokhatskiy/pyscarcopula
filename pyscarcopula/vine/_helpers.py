"""
vine._helpers — shared utility functions for vine copulas.

Includes r-generation for sample and predict, clipping, etc.
Used by both CVineCopula and RVineCopula.
"""

import numpy as np

from pyscarcopula.vine._edge import VineEdge, _get_alpha, _get_gas_params


def _clip_unit(x):
    """Clip to (eps, 1-eps) to avoid NaN in h-functions."""
    eps = 1e-10
    return np.clip(x, eps, 1.0 - eps)


def generate_r_for_sample(edge, n, rng):
    """Generate r trajectory for sample (model reproduction).

    MLE: constant r.
    SCAR: OU trajectory with dt = 1/(n-1).
    GAS: unsupported here; use vine stepwise GAS sampling.
    """
    from pyscarcopula.copula.independent import IndependentCopula
    from pyscarcopula._types import LatentResult, MLEResult, GASResult

    if isinstance(edge.copula, IndependentCopula):
        return np.zeros(n)

    if isinstance(edge.fit_result, MLEResult):
        return np.full(n, edge.fit_result.copula_param)

    if isinstance(edge.fit_result, LatentResult):
        alpha = _get_alpha(edge.fit_result)
        theta, mu, nu = alpha
        dt = 1.0 / (n - 1) if n > 1 else 1.0
        rho_ou = np.exp(-theta * dt)
        sigma_cond = np.sqrt(
            nu ** 2 / (2.0 * theta) * (1.0 - rho_ou ** 2))
        x_path = np.empty(n)
        x_path[0] = rng.normal(mu, nu / np.sqrt(2.0 * theta))
        for t in range(1, n):
            x_path[t] = (mu + rho_ou * (x_path[t - 1] - mu)
                         + sigma_cond * rng.standard_normal())
        return edge.copula.transform(x_path)

    if isinstance(edge.fit_result, GASResult):
        raise ValueError(
            "GAS sample paths require stepwise score updates and cannot be "
            "precomputed by generate_r_for_sample"
        )

    return edge.get_r_predict(n)


def generate_r_for_predict(edge, n, v_train_pair, K, grid_range, horizon='next',
                           state_cache=None, cache_key=None, rng=None,
                           **kwargs):
    """Generate r for predict (next-step conditional).

    MLE: constant r.
    SCAR-TM: mixture sampling from p(x_T | data) or p(x_{T+1} | data).
    GAS: last filtered value f_T.

    Parameters
    ----------
    edge : VineEdge
    n : int — number of samples
    v_train_pair : (T, 2) array or None — pseudo-observations for this edge
    K : int — grid size
    grid_range : float
    """
    from pyscarcopula.copula.independent import IndependentCopula
    from pyscarcopula._types import IndependentResult

    if isinstance(edge.copula, IndependentCopula):
        return np.zeros(n)

    if isinstance(edge.fit_result, IndependentResult):
        return np.zeros(n)

    if edge.fit_result is not None:
        from pyscarcopula.strategy._base import get_strategy_for_result
        strategy = get_strategy_for_result(
            edge.fit_result, K=K, grid_range=grid_range)
        return strategy.predictive_params(
            edge.copula,
            v_train_pair,
            edge.fit_result,
            n,
            horizon=horizon,
            state_cache=state_cache,
            cache_key=cache_key,
            rng=rng,
            predictive_r_mode=kwargs.get('predictive_r_mode'),
        )

    return edge.get_r_predict(n)
