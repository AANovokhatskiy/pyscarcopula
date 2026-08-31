"""Shared utility functions for vine copulas."""

import numpy as np

from pyscarcopula._constants import PSEUDO_OBS_EPS
from pyscarcopula._utils import clip_pseudo_observations
from pyscarcopula.numerical._arrays import as_float64_array


def validate_vine_given(given, d):
    """Validate conditional pseudo-observations shared by vine types."""
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
            raise ValueError(
                f"given key must be in [0, {d - 1}], got {key!r}")
        if (
                isinstance(value, (bool, np.bool_, str, bytes, complex,
                                   np.complexfloating))
                or not np.isscalar(value)):
            raise TypeError("given values must be numeric scalars")
        val = float(value)
        if not (0.0 < val < 1.0):
            raise ValueError(
                f"given[{idx}] must be in pseudo-observation space (0, 1), "
                f"got {val}"
            )
        out[idx] = val

    return out


def _clip_unit(x):
    """Clip vine pseudo-observations before h-function evaluation."""
    return clip_pseudo_observations(x)


def _open_unit_uniform(rng, size):
    """Draw vine pseudo-observations inside the shared safe unit interval."""
    return rng.uniform(PSEUDO_OBS_EPS, 1.0 - PSEUDO_OBS_EPS, size=size)


def _prepared_open_unit_draws(value, shape, *, name):
    """Validate replay draws and own a C-contiguous float64 representation."""
    array = as_float64_array(value, name=name)
    if array.shape != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any((array <= 0.0) | (array >= 1.0)):
        raise ValueError(f"{name} must contain values in the open unit interval")
    return np.ascontiguousarray(array)
