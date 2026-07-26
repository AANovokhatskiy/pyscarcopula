"""Array normalization helpers for numerical kernels."""

import numpy as np


def as_float64_array(value, *, name="array"):
    """Return a real float64 array without lossy complex coercion."""
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} must contain real values, not complex values")
    if type(array) is np.ndarray and array.dtype == np.float64:
        return array
    return np.asarray(array, dtype=np.float64)


def as_pseudo_observation_array(
        value, *, name="u", allow_boundary=True):
    """Return finite float64 pseudo-observations in the supported unit range."""
    array = as_float64_array(value, name=name)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if allow_boundary:
        outside = (array < 0.0) | (array > 1.0)
        interval = "[0, 1]"
    else:
        outside = (array <= 0.0) | (array >= 1.0)
        interval = "(0, 1)"
    if np.any(outside):
        raise ValueError(
            f"{name} must contain pseudo-observations in {interval}")
    return array


def validate_positive_int(value, name):
    """Validate and return a positive integer option."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result
