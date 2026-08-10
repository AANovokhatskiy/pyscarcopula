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


def validate_integer(value, name, *, minimum=0):
    """Validate an integer option with an inclusive lower bound."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        if minimum == 0:
            qualifier = "non-negative"
        elif minimum == 1:
            qualifier = "positive"
        else:
            qualifier = f"at least {minimum}"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def validate_sampling_n_threads(value):
    """Validate the public sampling thread-count contract."""
    result = validate_integer(value, "n_threads", minimum=1)
    if result > 256:
        raise ValueError("n_threads must be an integer in [1, 256]")
    return result


def validate_sampling_memory_budget(memory_budget_bytes, required, guidance):
    """Reject a sampling allocation that exceeds an optional byte budget."""
    if memory_budget_bytes is None:
        return
    budget = validate_integer(memory_budget_bytes, "memory_budget_bytes")
    if budget < int(required):
        raise MemoryError(
            f"sampling requires approximately {int(required)} bytes; "
            f"{guidance}")


def validate_float64_allocation(
        shape, *, name="array", memory_budget_bytes=None):
    """Validate a float64 allocation and return its required byte count."""
    elements = 1
    max_size = int(np.iinfo(np.intp).max)
    for extent in shape:
        if isinstance(extent, (bool, np.bool_)) or not isinstance(
                extent, (int, np.integer)):
            raise TypeError(f"{name} dimensions must be integers")
        extent = int(extent)
        if extent < 0:
            raise ValueError(f"{name} dimensions must be non-negative")
        if extent and elements > max_size // extent:
            raise MemoryError(f"{name} is too large to allocate")
        elements *= extent

    itemsize = np.dtype(np.float64).itemsize
    if elements > max_size // itemsize:
        raise MemoryError(f"{name} is too large to allocate")
    required = elements * itemsize

    if memory_budget_bytes is None:
        return required
    if (
            isinstance(memory_budget_bytes, (bool, np.bool_))
            or not isinstance(memory_budget_bytes, (int, np.integer))):
        raise TypeError("memory_budget_bytes must be an integer or None")
    memory_budget_bytes = int(memory_budget_bytes)
    if memory_budget_bytes < 0:
        raise ValueError("memory_budget_bytes must be non-negative")
    if required > memory_budget_bytes:
        raise MemoryError(
            f"{name} requires an estimated {required} bytes, exceeding "
            f"memory_budget_bytes={memory_budget_bytes}")
    return required
