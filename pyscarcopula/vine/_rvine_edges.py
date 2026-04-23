"""Edge-level operations for the refactored RVineCopula."""

import numpy as np

from pyscarcopula.copula.independent import IndependentCopula
from pyscarcopula._types import (
    DEFAULT_CONFIG,
    GASResult,
    IndependentResult,
    LatentResult,
    MLEResult,
)


def _edge_h(edge, u_conditioned, u_given, config=None):
    """Compute h(u_conditioned | u_given) for an RVine pair edge."""
    if isinstance(edge.copula, IndependentCopula):
        return np.asarray(u_conditioned, dtype=np.float64).copy()

    cfg = {} if config is None else dict(config)
    r = cfg.get('r')
    if r is not None:
        return edge.copula.h(u_conditioned, u_given, r)

    result = getattr(edge, 'fit_result', None)
    if result is None:
        r = np.full(len(np.atleast_1d(u_conditioned)), edge.param)
        return edge.copula.h(u_conditioned, u_given, r)
    if isinstance(result, IndependentResult):
        return np.asarray(u_conditioned, dtype=np.float64).copy()

    u_pair = np.column_stack((u_given, u_conditioned))
    strategy = _strategy_for_result(result, config=config)
    return strategy.mixture_h(edge.copula, u_pair, result)


def _edge_h_inverse(edge, v, u_given, config=None):
    """Compute inverse h for an RVine pair edge.

    ``config`` may contain a precomputed ``r`` array. If omitted, ``r`` is
    generated with the same model-reproduction rules used by CVine sampling.
    """
    if isinstance(edge.copula, IndependentCopula):
        return np.asarray(v, dtype=np.float64).copy()

    cfg = {} if config is None else dict(config)
    result = getattr(edge, 'fit_result', None)
    if result is None:
        r = cfg.get('r')
        if r is None:
            r = np.full(len(np.atleast_1d(v)), edge.param)
        return edge.copula.h_inverse(v, u_given, r)
    if isinstance(result, IndependentResult):
        return np.asarray(v, dtype=np.float64).copy()

    r = cfg.get('r')
    if r is None:
        rng = cfg.get('rng')
        if rng is None:
            rng = np.random.default_rng()
        r = _edge_r_for_sample(edge, len(np.atleast_1d(v)), rng)

    return edge.copula.h_inverse(v, u_given, r)


def _edge_r_for_sample(edge, n, rng=None):
    """Generate an r trajectory for unconditional model sampling."""
    if rng is None:
        rng = np.random.default_rng()

    if isinstance(edge.copula, IndependentCopula):
        return np.zeros(n, dtype=np.float64)

    result = getattr(edge, 'fit_result', None)
    if result is None:
        return np.full(n, edge.param, dtype=np.float64)

    if isinstance(result, IndependentResult):
        return np.zeros(n, dtype=np.float64)

    if isinstance(result, MLEResult):
        return np.full(n, result.copula_param, dtype=np.float64)

    if isinstance(result, LatentResult):
        theta, mu, nu = result.params.values
        dt = 1.0 / (n - 1) if n > 1 else 1.0
        rho_ou = np.exp(-theta * dt)
        sigma_cond = np.sqrt(
            nu ** 2 / (2.0 * theta) * (1.0 - rho_ou ** 2))

        x_path = np.empty(n, dtype=np.float64)
        x_path[0] = rng.normal(mu, nu / np.sqrt(2.0 * theta))
        for t in range(1, n):
            x_path[t] = (
                mu
                + rho_ou * (x_path[t - 1] - mu)
                + sigma_cond * rng.standard_normal()
            )
        return edge.copula.transform(x_path)

    if isinstance(result, GASResult):
        raise ValueError(
            "GAS sample paths require stepwise score updates and cannot be "
            "precomputed by _edge_r_for_sample"
        )

    raise TypeError(f"Unsupported fit_result type: {type(result).__name__}")


def _edge_r_for_predict(edge, n, u_train_pair=None, horizon='next',
                        rng=None, config=None, predictive_r_mode=None):
    """Generate an r vector for one-step predictive sampling."""
    if rng is None:
        rng = np.random.default_rng()
    if horizon in (1, '1'):
        horizon = 'next'
    elif horizon in (0, '0'):
        horizon = 'current'
    else:
        horizon = str(horizon).lower()

    if isinstance(edge.copula, IndependentCopula):
        return np.zeros(n, dtype=np.float64)

    result = getattr(edge, 'fit_result', None)
    if result is None:
        return np.full(n, edge.param, dtype=np.float64)

    if isinstance(result, IndependentResult):
        return np.zeros(n, dtype=np.float64)

    if isinstance(result, (MLEResult, GASResult, LatentResult)):
        strategy = _strategy_for_result(result, config=config)
        return strategy.predictive_params(
            edge.copula,
            u_train_pair,
            result,
            n,
            rng=rng,
            horizon=horizon,
            predictive_r_mode=predictive_r_mode,
        )

    raise TypeError(f"Unsupported fit_result type: {type(result).__name__}")


def _gas_update_from_last_observation(edge, f_t, u_pair, result, config):
    score_eps = config.gas_score_eps
    p = result.params

    f_plus = f_t + score_eps
    f_minus = f_t - score_eps
    r_t = float(edge.copula.transform(np.array([f_t]))[0])
    r_plus = float(edge.copula.transform(np.array([f_plus]))[0])
    r_minus = float(edge.copula.transform(np.array([f_minus]))[0])

    u1 = u_pair[:, 0]
    u2 = u_pair[:, 1]
    ll_t = float(edge.copula.log_pdf(u1, u2, np.array([r_t]))[0])
    ll_plus = float(edge.copula.log_pdf(u1, u2, np.array([r_plus]))[0])
    ll_minus = float(edge.copula.log_pdf(u1, u2, np.array([r_minus]))[0])

    nabla = (ll_plus - ll_minus) / (2.0 * score_eps)
    if result.scaling == 'fisher':
        d2 = (ll_plus - 2.0 * ll_t + ll_minus) / (score_eps ** 2)
        fisher = max(-d2, 1e-6)
        score = nabla / fisher
    else:
        score = nabla

    score = float(np.clip(score, -100.0, 100.0))
    f_new = p.omega + p.beta * f_t + p.alpha * score
    f_new = float(np.clip(f_new, -50.0, 50.0))
    r_new = float(edge.copula.transform(np.array([f_new]))[0])
    return f_new, r_new


def _edge_initial_gas_state(edge):
    """Return initial (f_t, r_t) for GAS model reproduction."""
    result = getattr(edge, 'fit_result', None)
    if not isinstance(result, GASResult):
        return None

    p = result.params
    if abs(p.beta) < 1.0 - 1e-8:
        f_t = p.omega / (1.0 - p.beta)
    else:
        f_t = p.omega
    r_t = float(edge.copula.transform(np.array([f_t]))[0])
    return float(f_t), r_t


def _edge_update_gas_state(edge, f_t, u_pair, config=None):
    """Advance GAS state after one generated edge observation."""
    result = getattr(edge, 'fit_result', None)
    if not isinstance(result, GASResult):
        raise TypeError("_edge_update_gas_state requires GASResult")

    cfg = config or DEFAULT_CONFIG
    f_new, r_new = _gas_update_from_last_observation(
        edge, f_t, u_pair, result, cfg)
    return f_new, r_new


def _is_gas_edge(edge):
    return isinstance(getattr(edge, 'fit_result', None), GASResult)


def _strategy_for_result(result, config=None):
    from pyscarcopula.strategy._base import get_strategy_for_result

    return get_strategy_for_result(result, config=config)
