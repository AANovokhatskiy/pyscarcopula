"""Helpers for predictive and conditional sampling."""

from __future__ import annotations

import numpy as np


def validate_given(given):
    """Normalize `given` to {int: float} with indices in {0, 1}."""
    if given is None:
        return {}

    if not isinstance(given, dict):
        raise TypeError("given must be a dict[int, float] or None")

    out = {}
    for key, value in given.items():
        try:
            idx = int(key)
        except Exception as exc:
            raise TypeError("given keys must be integers 0 or 1") from exc
        if idx not in (0, 1):
            raise ValueError(f"given key must be 0 or 1, got {key!r}")

        val = float(value)
        if not (0.0 < val < 1.0):
            raise ValueError(
                f"given[{idx}] must be in pseudo-observation space (0, 1), got {val}"
            )
        out[idx] = val
    return out


def conditional_sample_bivariate(copula, n, r, given=None, rng=None):
    """Sample from a bivariate copula with optional fixed coordinates."""
    if rng is None:
        rng = np.random.default_rng()

    given = validate_given(given)
    r_arr = np.atleast_1d(np.asarray(r, dtype=np.float64)).ravel()
    if r_arr.size == 1:
        r_arr = np.full(n, r_arr[0], dtype=np.float64)
    elif r_arr.size != n:
        raise ValueError(f"r must be scalar or length {n}, got shape {r_arr.shape}")

    if not given:
        return copula.sample(n, r_arr, rng=rng)

    samples = np.empty((n, 2), dtype=np.float64)

    if 0 in given:
        samples[:, 0] = given[0]
    if 1 in given:
        samples[:, 1] = given[1]

    if len(given) == 2:
        return samples

    z = rng.uniform(0.0, 1.0, size=n)

    if 0 in given:
        samples[:, 1] = copula.h_inverse(z, samples[:, 0], r_arr)
        return samples

    samples[:, 0] = copula.h_inverse(z, samples[:, 1], r_arr)
    return samples
