"""Array normalization helpers for numerical kernels."""

import numpy as np


def as_float64_array(value):
    """Return a float64 ndarray without changing identity unnecessarily."""
    if type(value) is np.ndarray and value.dtype == np.float64:
        return value
    return np.asarray(value, dtype=np.float64)
