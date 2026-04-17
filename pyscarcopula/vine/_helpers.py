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
    GAS: unconditional mean f_bar.
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
        p = edge.fit_result.params
        omega, _, beta = p.omega, p.alpha, p.beta
        if abs(beta) < 1.0 - 1e-8:
            f_bar = omega / (1.0 - beta)
        else:
            f_bar = omega
        r_bar = edge.copula.transform(np.array([f_bar]))[0]
        return np.full(n, r_bar)

    return edge.get_r_predict(n)


def generate_r_for_predict(edge, n, v_train_pair, K, grid_range, horizon='next',
                           state_cache=None, cache_key=None):
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
    from pyscarcopula._types import LatentResult, MLEResult, GASResult

    if isinstance(edge.copula, IndependentCopula):
        return np.zeros(n)

    if isinstance(edge.fit_result, MLEResult):
        return np.full(n, edge.fit_result.copula_param)

    if isinstance(edge.fit_result, LatentResult):
        if v_train_pair is not None:
            cached = None
            if state_cache is not None and cache_key is not None:
                cached = state_cache.get(cache_key)
            if cached is None:
                alpha = _get_alpha(edge.fit_result)
                theta, mu, nu = alpha
                from pyscarcopula.numerical.predictive_tm import tm_state_distribution
                cached = tm_state_distribution(
                    theta, mu, nu, v_train_pair, edge.copula, K, grid_range,
                    horizon=horizon)
                if state_cache is not None and cache_key is not None:
                    state_cache[cache_key] = cached
            z_grid, prob = cached
            idx = np.random.choice(len(z_grid), size=n, p=prob)
            return edge.copula.transform(z_grid[idx])
        else:
            return edge.get_r_predict(n)

    if isinstance(edge.fit_result, GASResult):
        r_last = getattr(edge.fit_result, 'r_last', None)
        if r_last is not None and r_last != 0.0:
            return np.full(n, r_last)
        if v_train_pair is not None:
            from pyscarcopula.numerical.gas_filter import gas_filter
            p = edge.fit_result.params
            scaling = getattr(edge.fit_result, 'scaling', 'unit')
            _, r_path, _ = gas_filter(
                p.omega, p.alpha, p.beta,
                v_train_pair, edge.copula, scaling)
            return np.full(n, r_path[-1])
        else:
            p = edge.fit_result.params
            f_bar = p.omega / (1.0 - p.beta) if abs(p.beta) < 0.999 else p.omega
            return np.full(n, edge.copula.transform(np.array([f_bar]))[0])

    return edge.get_r_predict(n)
