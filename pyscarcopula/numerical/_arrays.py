"""Array normalization helpers for numerical kernels."""

import numpy as np

from pyscarcopula._native.threads import validate_n_threads


def _validate_real_object_array(array, *, name):
    """Inspect nested containers before casting, without recursive Python calls."""
    active = {id(array)}
    checked = set()
    stack = [(id(array), iter(array.flat))]
    while stack:
        identity, items = stack[-1]
        try:
            item = next(items)
        except StopIteration:
            active.remove(identity)
            checked.add(identity)
            stack.pop()
            continue
        if isinstance(item, (complex, np.complexfloating)) or (
                isinstance(item, np.ndarray) and
                np.issubdtype(item.dtype, np.complexfloating)):
            raise TypeError(f"{name} must contain real values, not complex values")
        if isinstance(item, np.ndarray) and item.dtype.kind == "O":
            children = iter(item.flat)
        elif isinstance(item, (list, tuple)):
            children = iter(item)
        else:
            continue
        identity = id(item)
        if identity in active:
            raise ValueError(f"{name} must not contain cyclic containers")
        if identity not in checked:
            active.add(identity)
            stack.append((identity, children))


def as_float64_array(value, *, name="array"):
    """Return a real float64 array without lossy complex coercion."""
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} must contain real values, not complex values")
    if array.dtype.kind == "O":
        _validate_real_object_array(array, name=name)
    if type(array) is np.ndarray and array.dtype == np.float64:
        return array
    return np.asarray(array, dtype=np.float64)


def as_float64_scalar(value, *, name="parameter"):
    """Return a real scalar; leave finiteness and domain rules to the caller.

    Zero-dimensional arrays are scalars; one-element vectors are not.
    Complex values are rejected even when their imaginary part is zero.
    """
    if type(value) is float:
        return value
    if isinstance(value, np.floating):
        return float(value)
    array = as_float64_array(value, name=name)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    return float(array)


def as_integer_array(value, *, name="array", dtype=np.int64):
    """Return exact signed integers without truncation, bool coercion or wrap.

    Model-specific bounds remain the caller's responsibility. Sequence inputs
    use object storage first so mixed Python bool/int values cannot lose their
    types during NumPy's inference. Existing integer arrays keep the fast path.
    """
    target = np.dtype(dtype)
    if target.kind != "i":
        raise TypeError("dtype must be a signed integer dtype")
    array = np.asarray(value, dtype=object) if isinstance(
        value, (list, tuple)) else np.asarray(value)
    bounds = np.iinfo(target)
    if array.dtype.kind == "O":
        for item in array.flat:
            if isinstance(item, (bool, np.bool_, np.timedelta64)) or not isinstance(
                    item, (int, np.integer)):
                raise TypeError(f"{name} must contain integer values")
            integer = int(item)
            if integer < bounds.min or integer > bounds.max:
                raise ValueError(f"{name} values must fit in {target.name}")
    elif array.dtype.kind in "iu":
        if not np.can_cast(array.dtype, target, casting="safe"):
            outside = np.any(array > bounds.max)
            if array.dtype.kind == "i":
                outside = outside or np.any(array < bounds.min)
            if outside:
                raise ValueError(f"{name} values must fit in {target.name}")
    else:
        raise TypeError(f"{name} must contain integer values")
    return np.asarray(array, dtype=target)


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
    return validate_n_threads(value)


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
