"""Edge-level operations for the refactored RVineCopula."""

import numpy as np

from pyscarcopula.copula.independent import IndependentCopula
from pyscarcopula.numerical import _cpp_scar_ou
from pyscarcopula.vine._edge_adapter import (
    edge_condition_sample_state,
    edge_log_likelihood,
    edge_mixture_h,
    edge_copula,
    edge_is_independent,
    edge_model_sample_state,
    edge_param,
    edge_result,
    edge_state_param,
    predict_r_path,
    sample_r_path,
    strategy_for_result,
)


def _backend_name(value):
    return str(value).lower() if value is not None else None


def _edge_cpp_requested(edge, cfg):
    backend = _backend_name(cfg.get('backend'))
    if backend == 'python':
        return False
    result_backend = _backend_name(getattr(edge_result(edge), 'backend', None))
    return backend in ('cpp', 'auto') or result_backend in ('cpp', 'auto')


def _edge_cpp_required(edge, cfg):
    backend = _backend_name(cfg.get('backend'))
    result_backend = _backend_name(getattr(edge_result(edge), 'backend', None))
    return backend == 'cpp' or result_backend == 'cpp'


def _try_cpp_h(edge, u_conditioned, u_given, r, cfg):
    if not _edge_cpp_requested(edge, cfg):
        return None
    try:
        return _cpp_scar_ou.copula_h(
            edge_copula(edge), u_conditioned, u_given, r)
    except _cpp_scar_ou.CppError:
        if _edge_cpp_required(edge, cfg):
            raise
    return None


def _try_cpp_h_inverse(edge, v, u_given, r, cfg):
    if not _edge_cpp_requested(edge, cfg):
        return None
    try:
        return _cpp_scar_ou.copula_h_inverse(
            edge_copula(edge), v, u_given, r)
    except _cpp_scar_ou.CppError:
        if _edge_cpp_required(edge, cfg):
            raise
    return None


def _edge_h(edge, u_conditioned, u_given, config=None, u_pair=None,
            state_cache=None, current_cache_key=None, next_cache_key=None,
            **strategy_kwargs):
    """Compute h(u_conditioned | u_given) for a pair edge."""
    copula = edge_copula(edge)
    if isinstance(copula, IndependentCopula):
        return np.asarray(u_conditioned, dtype=np.float64).copy()

    cfg = config if isinstance(config, dict) else {}
    r = cfg.get('r')
    if r is not None:
        cpp = _try_cpp_h(edge, u_conditioned, u_given, r, cfg)
        if cpp is not None:
            return cpp
        return copula.h(u_conditioned, u_given, r)

    result = edge_result(edge)
    if result is None:
        r = np.full(len(np.atleast_1d(u_conditioned)), edge_param(edge))
        cpp = _try_cpp_h(edge, u_conditioned, u_given, r, cfg)
        if cpp is not None:
            return cpp
        return copula.h(u_conditioned, u_given, r)
    if edge_is_independent(edge):
        return np.asarray(u_conditioned, dtype=np.float64).copy()

    if u_pair is None:
        u_pair = np.column_stack((u_given, u_conditioned))
    return edge_mixture_h(
        copula,
        result,
        u_pair,
        config=config,
        state_cache=state_cache,
        current_cache_key=current_cache_key,
        next_cache_key=next_cache_key,
        **strategy_kwargs)


def _edge_log_likelihood(edge, u_pair, config=None, **strategy_kwargs):
    """Compute log-likelihood for one pair edge using its fitted strategy."""
    copula = edge_copula(edge)
    if isinstance(copula, IndependentCopula):
        return 0.0

    result = edge_result(edge)
    r = edge_param(edge)
    if r is not None:
        u1 = u_pair[:, 0]
        u2 = u_pair[:, 1]
        return float(np.sum(
            copula.log_pdf(u1, u2, np.full(len(u1), float(r)))))

    return edge_log_likelihood(
        copula, result, u_pair, config=config, **strategy_kwargs)


def _edge_h_inverse(edge, v, u_given, config=None):
    """Compute inverse h for an RVine pair edge.

    ``config`` may contain a precomputed ``r`` array. If omitted, ``r`` is
    generated with the same model-reproduction rules used by CVine sampling.
    """
    copula = edge_copula(edge)
    if isinstance(copula, IndependentCopula):
        return np.asarray(v, dtype=np.float64).copy()

    cfg = config if isinstance(config, dict) else {}
    result = edge_result(edge)
    if result is None:
        r = cfg.get('r')
        if r is None:
            r = np.full(len(np.atleast_1d(v)), edge_param(edge))
        cpp = _try_cpp_h_inverse(edge, v, u_given, r, cfg)
        if cpp is not None:
            return cpp
        return copula.h_inverse(v, u_given, r)
    if edge_is_independent(edge):
        return np.asarray(v, dtype=np.float64).copy()

    r = cfg.get('r')
    if r is None:
        rng = cfg.get('rng')
        if rng is None:
            rng = np.random.default_rng()
        r = _edge_r_for_sample(edge, len(np.atleast_1d(v)), rng)

    cpp = _try_cpp_h_inverse(edge, v, u_given, r, cfg)
    if cpp is not None:
        return cpp
    return copula.h_inverse(v, u_given, r)


def _edge_r_for_sample(edge, n, rng=None):
    """Generate an r trajectory for unconditional model sampling."""
    if rng is None:
        rng = np.random.default_rng()

    result = edge_result(edge)
    return sample_r_path(
        edge_copula(edge),
        result,
        n,
        rng,
        param=edge_param(edge),
        error_name='_edge_r_for_sample',
    )


def _edge_r_for_predict(edge, n, u_train_pair=None, horizon='next',
                        rng=None, config=None, predictive_r_mode=None,
                        state_cache=None, cache_key=None):
    """Generate an r vector for one-step predictive sampling."""
    if rng is None:
        rng = np.random.default_rng()

    result = edge_result(edge)
    return predict_r_path(
        edge_copula(edge),
        result,
        n,
        u_train_pair=u_train_pair,
        horizon=horizon,
        rng=rng,
        config=config,
        predictive_r_mode=predictive_r_mode,
        state_cache=state_cache,
        cache_key=cache_key,
        param=edge_param(edge),
    )


def _edge_initial_model_state(edge, config=None):
    """Return strategy-owned state for stepwise model reproduction."""
    result = edge_result(edge)
    return edge_model_sample_state(edge_copula(edge), result, config=config)


def _edge_update_model_state(edge, state, u_pair, config=None):
    """Advance strategy-owned model state after one generated observation."""
    result = edge_result(edge)
    if state is None:
        raise TypeError("_edge_update_model_state requires a state")
    return edge_condition_sample_state(
        edge_copula(edge), result, state, u_pair, config=config)


def _edge_state_r(edge, state):
    """Return scalar r represented by a strategy-owned state."""
    return edge_state_param(state)


def _edge_requires_stepwise_sample(edge):
    """Return True when an edge strategy owns stepwise sampling state."""
    return _edge_initial_model_state(edge) is not None


def _strategy_for_result(result, config=None):
    return strategy_for_result(result, config=config)
