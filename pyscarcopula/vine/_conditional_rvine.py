"""Validation helpers for R-vine conditional prediction."""

from __future__ import annotations

import numpy as np


def validate_rvine_given(given, d):
    """Validate `given` for R-vine conditional predict."""
    if given is None:
        return {}

    if not isinstance(given, dict):
        raise TypeError("given must be a dict[int, float] or None")

    out = {}
    for key, value in given.items():
        if isinstance(key, (bool, np.bool_)) or not isinstance(
                key, (int, np.integer)):
            raise TypeError("given keys must be integers")
        idx = int(key)
        if idx < 0 or idx >= d:
            raise ValueError(f"given key must be in [0, {d - 1}], got {key!r}")
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
            raise TypeError("given values must be numeric scalars")
        val = float(value)
        if not (0.0 < val < 1.0):
            raise ValueError(
                f"given[{idx}] must be in pseudo-observation space (0, 1), got {val}"
            )
        out[idx] = val
    return out
