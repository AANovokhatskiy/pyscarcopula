"""Array normalization helpers for numerical kernels."""

import numpy as np


def as_float64_array(value):
    """Return a float64 ndarray without changing identity unnecessarily."""
    if type(value) is np.ndarray and value.dtype == np.float64:
        return value
    return np.asarray(value, dtype=np.float64)


def validate_positive_int(value, name):
    """Coerce and validate a positive integer-valued option."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
