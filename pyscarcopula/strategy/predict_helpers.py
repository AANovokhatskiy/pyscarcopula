"""Helpers for predictive and conditional sampling."""

from __future__ import annotations

import numpy as np

from pyscarcopula.numerical._arrays import (
    validate_float64_allocation,
    validate_integer,
    validate_sampling_n_threads,
)
from pyscarcopula.strategy._base import (
    copula_dimension,
    has_dynamic_scalar_parameter,
    is_multivariate_copula,
    is_pair_copula,
    supports_conditional_sampling,
    reject_unknown_operation_kwargs,
)


def validate_given(given):
    """Normalize `given` to {int: float} with indices in {0, 1}."""
    if given is None:
        return {}

    if not isinstance(given, dict):
        raise TypeError("given must be a dict[int, float] or None")

    out = {}
    for key, value in given.items():
        if isinstance(key, (bool, np.bool_)) or not isinstance(
                key, (int, np.integer)):
            raise TypeError("given keys must be integers 0 or 1")
        idx = int(key)
        if idx not in (0, 1):
            raise ValueError(f"given key must be 0 or 1, got {key!r}")

        if (
                isinstance(value, (bool, np.bool_, str, bytes, complex,
                                   np.complexfloating))
                or not np.isscalar(value)):
            raise TypeError("given values must be numeric scalars")
        val = float(value)
        if not (0.0 < val < 1.0):
            raise ValueError(
                f"given[{idx}] must be in pseudo-observation space (0, 1), got {val}"
            )
        out[idx] = val
    return out


def conditional_sample_bivariate(
        copula, n, r, given=None, rng=None, *, config=None):
    """Sample from a bivariate copula with optional fixed coordinates."""
    from pyscarcopula._native.registry import registry_entry_for

    registry_entry_for(copula)
    if rng is None:
        rng = np.random.default_rng()

    given = validate_given(given)
    r_arr = np.atleast_1d(np.asarray(r, dtype=np.float64)).ravel()
    if r_arr.size == 1:
        r_arr = np.full(n, r_arr[0], dtype=np.float64)
    elif r_arr.size != n:
        raise ValueError(f"r must be scalar or length {n}, got shape {r_arr.shape}")

    if not given:
        return copula.sample_at_parameter(n, r_arr, rng=rng)

    samples = np.empty((n, 2), dtype=np.float64)

    if 0 in given:
        samples[:, 0] = given[0]
    if 1 in given:
        samples[:, 1] = given[1]

    if len(given) == 2:
        return samples

    z = rng.uniform(0.0, 1.0, size=n)

    from pyscarcopula._native import pair as copula_native
    given_coordinate, given_value = next(iter(given.items()))
    inverse_options = {}
    if config is not None:
        inverse_options = {
            'bisection_tol': config.bisection_tol,
            'bisection_maxiter': config.bisection_maxiter,
        }
    return copula_native.conditional_sample_from_uniforms(
        copula,
        z,
        r_arr,
        given_coordinate=given_coordinate,
        given_value=given_value,
        **inverse_options,
    )


def sample_predictive(
        copula, n, r, given=None, rng=None, d=None, *,
        n_threads=1, memory_budget_bytes=None, config=None):
    """Sample from a predictive parameter path.

    Conditional ``given`` sampling is delegated to the registered built-in
    model's native bivariate or multivariate implementation.
    """
    from pyscarcopula._native.registry import registry_entry_for

    entry = registry_entry_for(copula)
    n = validate_integer(n, "n", minimum=0)
    n_threads = validate_sampling_n_threads(n_threads)
    dimension = copula_dimension(copula) if d is None else d
    if dimension is not None:
        validate_float64_allocation(
            (n, dimension), name="predictive sample output",
            memory_budget_bytes=memory_budget_bytes)
    if is_multivariate_copula(copula) and has_dynamic_scalar_parameter(copula):
        return copula.sample_conditional(
            n, r=r, given=given, rng=rng, n_threads=n_threads,
            memory_budget_bytes=memory_budget_bytes)
    if given is None and d is not None and int(d) != 2 and hasattr(copula, "_R_path"):
        return copula.sample(n=n, df_path=r, rng=rng)

    static_options = {"n_threads": n_threads}
    if entry.native_id == "Gaussian" and memory_budget_bytes is not None:
        # Gaussian owns a workspace guard in addition to the API output guard.
        # Student's static sampler only supports the latter contract.
        static_options["memory_budget_bytes"] = memory_budget_bytes

    if given is None:
        if is_pair_copula(copula):
            return copula.sample_at_parameter(n, r, rng=rng)
        if not has_dynamic_scalar_parameter(copula):
            return copula.sample(n, rng=rng, **static_options)
        return copula.sample_at_parameter(n, r, rng=rng)

    if is_pair_copula(copula):
        return conditional_sample_bivariate(
            copula, n, r, given=given, rng=rng, config=config)

    if supports_conditional_sampling(copula):
        if not has_dynamic_scalar_parameter(copula):
            return copula.sample_conditional(
                n, given=given, rng=rng, **static_options)
        return copula.sample_conditional(n, r=r, given=given, rng=rng)

    if d is None:
        d = 2
    if int(d) != 2:
        raise NotImplementedError(
            "given= conditional sampling is only implemented for bivariate "
            "copulas and vine models"
        )
    return conditional_sample_bivariate(
        copula, n, r, given=given, rng=rng, config=config)


def predict_from_strategy(strategy, copula, u, result, n, rng=None, **kwargs):
    """Shared strategy predict implementation for predictive parameter paths."""
    reject_unknown_operation_kwargs(strategy, 'predict', kwargs)
    n = validate_integer(n, "n", minimum=0)
    n_threads = validate_sampling_n_threads(kwargs.get("n_threads", 1))
    d = copula_dimension(copula, u)
    # Reject oversized output before materializing a parameter path or using RNG.
    validate_float64_allocation(
        (n, d), name="predictive sample output",
        memory_budget_bytes=kwargs.get("memory_budget_bytes"))
    if rng is None:
        rng = np.random.default_rng()
    r = strategy.predictive_params(copula, u, result, n, rng=rng, **kwargs)
    return sample_predictive(
        copula, n, r, given=kwargs.get("given"), rng=rng, d=d,
        n_threads=n_threads,
        memory_budget_bytes=kwargs.get("memory_budget_bytes"),
        config=getattr(strategy, 'config', None))


def strategy_predict(strategy, copula, u, result, n, rng=None, **kwargs):
    """Default strategy method backed by the shared prediction workflow."""
    return predict_from_strategy(
        strategy, copula, u, result, n, rng=rng, **kwargs)


def sample_model_batches(
        copula, strategy, result, state, n, *, batch_rows, given, rng,
        n_threads, memory_budget_bytes):
    """Generate bounded model blocks, preserving the sequential GAS state."""
    if state is None:
        parameter_blocks = strategy.model_sample_params_batches(
            copula, result, n, rng=rng, batch_rows=batch_rows)
        for parameters in parameter_blocks:
            yield copula.sample_conditional(
                len(parameters), r=parameters, given=given, rng=rng,
                n_threads=n_threads, memory_budget_bytes=memory_budget_bytes)
        return

    current = state
    for start in range(0, n, batch_rows):
        count = min(batch_rows, n - start)
        block = np.empty((count, copula.dimension), dtype=np.float64)
        for row in range(count):
            parameter = strategy.sample_params(copula, current, 1, rng=rng)[0]
            observation = copula.sample_conditional(
                1, r=parameter, given=given, rng=rng,
                n_threads=n_threads, memory_budget_bytes=memory_budget_bytes)
            block[row] = observation[0]
            current = strategy.condition_state(
                copula, current, observation, result)
        yield block


def sample_predictive_batches(
        copula, strategy, state, n, *, batch_rows, given, rng,
        predictive_r_mode, n_threads, memory_budget_bytes):
    """Sample bounded row blocks from one frozen predictive state."""
    for start in range(0, n, batch_rows):
        count = min(batch_rows, n - start)
        parameters = strategy.sample_params(
            copula, state, count, rng=rng,
            predictive_r_mode=predictive_r_mode)
        yield copula.sample_conditional(
            count, r=parameters, given=given, rng=rng,
            n_threads=n_threads, memory_budget_bytes=memory_budget_bytes)


def predictive_params_from_state(
        strategy, copula, u, result, n, rng=None, **kwargs):
    """Sample parameters from a strategy-created predictive state."""
    state = strategy.predictive_state(copula, u, result, **kwargs)
    return strategy.sample_params(
        copula, state, n, rng=rng, **kwargs)


def predictive_params_from_state_with_rng(
        strategy, copula, u, result, n, rng=None, **kwargs):
    """State sampler variant that always supplies a generator."""
    if rng is None:
        rng = np.random.default_rng()
    return predictive_params_from_state(
        strategy, copula, u, result, n, rng=rng, **kwargs)
