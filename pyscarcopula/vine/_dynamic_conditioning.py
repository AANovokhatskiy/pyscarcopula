"""Strategy-generic dynamic conditioning helpers for R-vine prediction."""

import numpy as np

from pyscarcopula._types import PredictiveState
from pyscarcopula.vine._edge_adapter import (
    edge_copula,
    edge_has_dynamic_params,
    edge_method,
    edge_result,
)
from pyscarcopula.vine._rvine_edges import (
    _edge_initial_model_state,
    _strategy_for_result,
)


def normalize_predict_horizon(horizon):
    """Normalize public horizon aliases to internal names."""
    if horizon in (1, '1'):
        return 'next'
    if horizon in (0, '0'):
        return 'current'
    return str(horizon).lower()


def predictive_state_cache_key(edge_key, horizon):
    """Return cache key for strategy predictive state reuse."""
    return 'predictive_state', edge_key, normalize_predict_horizon(horizon)


def predictive_state_equal(left, right):
    """Return True when two strategy-owned predictive states are equivalent."""
    if left.kind != right.kind:
        return False
    for name in ('r', 'prob', 'z_grid'):
        left_value = getattr(left, name, None)
        right_value = getattr(right, name, None)
        if left_value is None and right_value is None:
            continue
        if left_value is None or right_value is None:
            return False
        if not np.array_equal(
                np.asarray(left_value, dtype=np.float64),
                np.asarray(right_value, dtype=np.float64)):
            return False
    return True


def predictive_given_update_r(edge, u_train_pair, u_observed_pair,
                              n, horizon, rng, predictive_r_mode,
                              state_cache=None, cache_key=None,
                              strategy_for_result=None):
    """Condition one edge predictive state on a fully observed pair."""
    result = edge_result(edge)
    if not edge_has_dynamic_params(edge):
        return None
    if u_train_pair is None or len(u_train_pair) == 0:
        return None

    copula = edge_copula(edge)
    horizon = normalize_predict_horizon(horizon)
    if strategy_for_result is None:
        strategy_for_result = _strategy_for_result
    strategy = strategy_for_result(result)
    state = strategy.predictive_state(
        copula,
        u_train_pair,
        result,
        horizon=horizon,
        predictive_r_mode=predictive_r_mode,
        state_cache=state_cache,
        cache_key=cache_key,
    )
    conditioned = strategy.condition_state(
        copula,
        state,
        u_observed_pair,
        result,
    )
    if predictive_state_equal(conditioned, state):
        return None
    return strategy.sample_params(
        copula,
        conditioned,
        n,
        rng=rng,
        predictive_r_mode=predictive_r_mode,
    )


def dynamic_edge_update_from_observation(
        edge, r_current, u_observed_pair, u_train_pair,
        horizon, rng, predictive_r_mode, state_cache=None, cache_key=None,
        strategy_for_result=None):
    """Return updated r vector after conditioning one eligible dynamic edge."""
    result = edge_result(edge)
    copula = edge_copula(edge)
    horizon = normalize_predict_horizon(horizon)
    if _edge_initial_model_state(edge) is not None:
        if horizon == 'next':
            return None
        if strategy_for_result is None:
            strategy_for_result = _strategy_for_result
        strategy = strategy_for_result(result)
        state = PredictiveState(
            method=result.method,
            horizon=horizon,
            kind='point',
            r=np.array([float(np.asarray(r_current)[0])], dtype=np.float64),
        )
        conditioned_state = strategy.condition_state(
            copula,
            state,
            u_observed_pair,
            result,
        )
        return strategy.sample_params(
            copula,
            conditioned_state,
            len(r_current),
            rng=rng,
            predictive_r_mode=predictive_r_mode,
        )

    if edge_has_dynamic_params(edge):
        return predictive_given_update_r(
            edge,
            u_train_pair,
            u_observed_pair,
            len(r_current),
            horizon,
            rng,
            predictive_r_mode,
            state_cache=state_cache,
            cache_key=cache_key,
            strategy_for_result=strategy_for_result,
        )

    return None


def dynamic_edge_skip_reason(edge, train_pseudo, horizon):
    """Return diagnostics reason for an eligible but non-updated edge."""
    horizon = normalize_predict_horizon(horizon)
    if _edge_initial_model_state(edge) is not None and horizon == 'next':
        return 'next_horizon_would_advance_filter'
    if edge_has_dynamic_params(edge) and train_pseudo is None:
        return 'no_training_history'
    return 'unsupported_or_noop'


def dynamic_update_record(
        trees, key, edge, edge_map, r_before, r_after, status, reason=None):
    """Build one diagnostics record for a dynamic conditioning action."""
    orig_idx = edge_map[key]
    conditioned, conditioning = trees[key[0]][orig_idx]
    record = {
        'key': tuple(int(v) for v in key),
        'tree': int(key[0]),
        'col': int(key[1]),
        'conditioned': tuple(sorted(int(v) for v in conditioned)),
        'conditioning': tuple(sorted(int(v) for v in conditioning)),
        'method': str(edge_method(edge) or ''),
        'family': type(edge_copula(edge)).__name__,
        'status': status,
    }
    if reason is not None:
        record['reason'] = reason
    if r_before is not None:
        r_before = np.asarray(r_before, dtype=np.float64)
        record['r_before_mean'] = float(np.mean(r_before))
    if r_after is not None:
        r_after = np.asarray(r_after, dtype=np.float64)
        record['r_after_mean'] = float(np.mean(r_after))
    return record


def dynamic_skip_records(trees, pair_copulas, edge_map, r_all, reason):
    """Return skip diagnostics for all dynamic edges in a path."""
    records = []
    for key, edge in sorted(pair_copulas.items()):
        if not (
                _edge_initial_model_state(edge) is not None
                or edge_has_dynamic_params(edge)):
            continue
        records.append(dynamic_update_record(
            trees,
            key,
            edge,
            edge_map,
            r_all.get(key),
            None,
            'skipped',
            reason=reason,
        ))
    return records
