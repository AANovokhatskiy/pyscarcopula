"""Shared utility functions for vine copulas."""

import numpy as np

from pyscarcopula._constants import PSEUDO_OBS_EPS
from pyscarcopula._utils import clip_pseudo_observations


def _clip_unit(x):
    """Clip vine pseudo-observations before h-function evaluation."""
    return clip_pseudo_observations(x)


def _open_unit_uniform(rng, size):
    """Draw vine pseudo-observations inside the shared safe unit interval."""
    return rng.uniform(PSEUDO_OBS_EPS, 1.0 - PSEUDO_OBS_EPS, size=size)


def _prepared_open_unit_draws(value, shape, *, name):
    """Validate replay draws and own a C-contiguous float64 representation."""
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} must contain real values, not complex values")
    array = np.asarray(array, dtype=np.float64)
    if array.shape != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any((array <= 0.0) | (array >= 1.0)):
        raise ValueError(f"{name} must contain values in the open unit interval")
    return np.ascontiguousarray(array)
