"""Shared vine edge operations.

This module is the common dispatch layer for vine ``PairCopula`` edges. It
keeps method-specific behavior behind fitted ``FitResult`` objects instead of
duplicating strategy logic in each vine.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EdgeView:
    """Normalized read-only view over CVine and RVine edge objects."""

    copula: object
    fit_result: object = None
    param: float | None = None
    log_likelihood: float | None = None
    nfev: int | None = None
    tau: float | None = None
    tree: int | None = None
    idx: int | None = None

    @property
    def method(self):
        return edge_method(self)

    @property
    def n_params(self):
        return edge_n_params(self)


def edge_view(edge):
    """Return a normalized read-only view for any supported vine edge."""
    return EdgeView(
        copula=getattr(edge, 'copula', None),
        fit_result=getattr(edge, 'fit_result', None),
        param=getattr(edge, 'param', None),
        log_likelihood=getattr(edge, 'log_likelihood', None),
        nfev=getattr(edge, 'nfev', None),
        tau=getattr(edge, 'tau', None),
        tree=getattr(edge, 'tree', None),
        idx=getattr(edge, 'idx', None),
    )


def edge_copula(edge):
    """Return the copula object stored on an edge-like object."""
    return getattr(edge, 'copula', None)


def edge_result(edge):
    """Return the fitted result stored on an edge-like object."""
    return getattr(edge, 'fit_result', None)


def edge_method(edge):
    """Return the fitted method name for an edge-like object, or ``None``."""
    result = edge_result(edge)
    if result is None:
        return None
    return getattr(result, 'method', None)


def edge_param(edge, default=None):
    """Return a scalar edge parameter when the edge has a point parameter."""
    value = getattr(edge, 'param', None)
    if value is not None:
        return float(value)

    result = edge_result(edge)
    result_param = getattr(result, 'copula_param', None)
    if result_param is not None:
        return float(result_param)
    if _is_independent_copula(edge_copula(edge)):
        return 0.0
    return default


def edge_n_params(edge):
    """Return the number of fitted parameters for an edge-like object."""
    result = edge_result(edge)
    if result is not None:
        return int(getattr(result, 'n_params', 0))
    if _is_independent_copula(edge_copula(edge)):
        return 0
    return 1 if edge_param(edge) is not None else 0


def edge_is_independent(edge):
    """Return True when edge-like object represents an independent edge."""
    return (
        _is_independent_copula(edge_copula(edge))
        or type(edge_result(edge)).__name__ == 'IndependentResult'
    )


def edge_has_dynamic_params(edge):
    """Return True when an edge result carries strategy parameters."""
    return getattr(edge_result(edge), 'params', None) is not None


def result_param_items(result):
    """Return named strategy parameters from a fitted result."""
    params = getattr(result, 'params', None)
    if params is None:
        return ()
    names = getattr(params, 'names', ())
    values = getattr(params, 'values', ())
    return tuple(zip(names, values))


def strategy_for_result(result, config=None, **kwargs):
    """Instantiate the strategy associated with a fitted edge result."""
    from pyscarcopula.strategy._base import get_strategy_for_result

    return get_strategy_for_result(result, config=config, **kwargs)


def _strategy_kwargs(result, **kwargs):
    out = {}
    for name in (
            'K', 'grid_range', 'pts_per_sigma',
            'transition_method', 'max_K', 'r_gh', 'gh_order'):
        if name in kwargs:
            out[name] = kwargs[name]
    return out


def _is_independent_copula(copula):
    from pyscarcopula.copula.independent import IndependentCopula

    return isinstance(copula, IndependentCopula)


def _normalize_horizon(horizon):
    if horizon in (1, '1'):
        return 'next'
    if horizon in (0, '0'):
        return 'current'
    return str(horizon).lower()


def sample_r_path(copula, result, n, rng=None, param=None,
                  error_name='sample_r_path'):
    """Generate an edge parameter path for unconditional vine sampling."""
    if rng is None:
        rng = np.random.default_rng()

    if result is None:
        if param is None:
            raise TypeError("sample_r_path requires a fit result or param")
        return np.full(n, param, dtype=np.float64)
    if edge_is_independent(EdgeView(copula=copula, fit_result=result)):
        return np.zeros(n, dtype=np.float64)

    strategy = strategy_for_result(result)
    try:
        return strategy.model_sample_params(copula, result, n, rng=rng)
    except ValueError as exc:
        if error_name != 'sample_r_path':
            raise ValueError(f"{exc} by {error_name}") from exc
        raise


def predict_r_path(copula, result, n, u_train_pair=None, horizon='next',
                   rng=None, config=None, predictive_r_mode=None,
                   state_cache=None, cache_key=None, posterior_cache=None,
                   param=None, **kwargs):
    """Generate an edge parameter vector for predictive vine sampling."""
    if rng is None:
        rng = np.random.default_rng()
    horizon = _normalize_horizon(horizon)

    if result is None:
        if param is None:
            raise TypeError("predict_r_path requires a fit result or param")
        return np.full(n, param, dtype=np.float64)
    if edge_is_independent(EdgeView(copula=copula, fit_result=result)):
        return np.zeros(n, dtype=np.float64)

    strategy = strategy_for_result(
        result, config=config, **_strategy_kwargs(result, **kwargs))
    return strategy.predictive_params(
        copula,
        u_train_pair,
        result,
        n,
        rng=rng,
        horizon=horizon,
        predictive_r_mode=predictive_r_mode,
        state_cache=state_cache,
        cache_key=cache_key,
        posterior_cache=posterior_cache,
    )


def edge_mixture_h(copula, result, u_pair, config=None, **kwargs):
    """Compute h(u2 | u1) using the strategy matching ``result``."""
    strategy = strategy_for_result(
        result, config=config, **_strategy_kwargs(result, **kwargs))
    call_kwargs = {}
    for name in (
            'state_cache', 'current_cache_key', 'next_cache_key',
            'posterior_cache'):
        value = kwargs.get(name)
        if value is not None:
            call_kwargs[name] = value
    try:
        return strategy.mixture_h(copula, u_pair, result, **call_kwargs)
    except NotImplementedError:
        try:
            r = strategy.predictive_mean(copula, u_pair, result)
        except NotImplementedError:
            r = strategy.predictive_params(
                copula,
                u_pair,
                result,
                len(u_pair),
                rng=np.random.default_rng(0),
            )
        return copula.h(u_pair[:, 1], u_pair[:, 0], r)


def edge_log_likelihood(copula, result, u_pair, config=None, **kwargs):
    """Evaluate edge log-likelihood using the strategy matching ``result``."""
    strategy = strategy_for_result(
        result, config=config, **_strategy_kwargs(result, **kwargs))
    return strategy.log_likelihood(copula, u_pair, result)


def edge_model_sample_state(copula, result, config=None):
    """Return strategy-owned state for stepwise model sampling, if any."""
    if result is None:
        return None
    if edge_is_independent(EdgeView(copula=copula, fit_result=result)):
        return None
    strategy = strategy_for_result(result, config=config)
    return strategy.model_sample_state(copula, result)


def edge_condition_sample_state(copula, result, state, u_pair, config=None):
    """Advance strategy-owned model-sampling state after an edge observation."""
    strategy = strategy_for_result(result, config=config)
    return strategy.condition_state(copula, state, u_pair, result)


def edge_state_param(state):
    """Return scalar copula parameter represented by a strategy state."""
    return float(np.asarray(state.r, dtype=np.float64)[0])
