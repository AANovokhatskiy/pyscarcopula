"""Helpers for predictive and conditional sampling."""

from __future__ import annotations

import numpy as np

from pyscarcopula.strategy._base import (
    copula_dimension,
    get_copula_capabilities,
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


def conditional_sample_bivariate(copula, n, r, given=None, rng=None):
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
    return copula_native.conditional_sample_from_uniforms(
        copula,
        z,
        r_arr,
        given_coordinate=given_coordinate,
        given_value=given_value,
    )


def sample_predictive(copula, n, r, given=None, rng=None, d=None):
    """Sample from a predictive parameter path.

    Conditional ``given`` sampling is delegated to the registered built-in
    model's native bivariate or multivariate implementation.
    """
    from pyscarcopula._native.registry import registry_entry_for

    registry_entry_for(copula)
    capabilities = get_copula_capabilities(copula)

    if given is None and d is not None and int(d) != 2 and hasattr(copula, "_R_path"):
        return copula.sample(n=n, df_path=r, rng=rng)

    if given is None:
        if capabilities is not None and capabilities.supports_pair_ops:
            return copula.sample_at_parameter(n, r, rng=rng)
        if (
                capabilities is not None
                and not capabilities.has_dynamic_scalar_parameter):
            return copula.sample(n, rng=rng)
        return copula.sample_at_parameter(n, r, rng=rng)

    if capabilities is not None and capabilities.supports_pair_ops:
        return conditional_sample_bivariate(
            copula, n, r, given=given, rng=rng)

    if capabilities is not None and capabilities.supports_conditional_sampling:
        if not capabilities.has_dynamic_scalar_parameter:
            return copula.sample_conditional(
                n, given=given, rng=rng)
        return copula.sample_conditional(n, r=r, given=given, rng=rng)

    if d is None:
        d = 2
    if int(d) != 2:
        raise NotImplementedError(
            "given= conditional sampling is only implemented for bivariate "
            "copulas and vine models"
        )
    return conditional_sample_bivariate(copula, n, r, given=given, rng=rng)


def predict_from_strategy(strategy, copula, u, result, n, rng=None, **kwargs):
    """Shared strategy predict implementation for predictive parameter paths."""
    if rng is None:
        rng = np.random.default_rng()
    r = strategy.predictive_params(copula, u, result, n, rng=rng, **kwargs)
    d = copula_dimension(copula, u)
    return sample_predictive(
        copula, n, r, given=kwargs.get("given"), rng=rng, d=d)
