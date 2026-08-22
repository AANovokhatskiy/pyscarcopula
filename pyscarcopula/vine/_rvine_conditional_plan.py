"""Immutable executor-neutral conditional R-vine programs."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from types import MappingProxyType

import numpy as np


def _node_key(variable, conditioning=()):
    """Normalize one conditioned-variable key shared by R-vine plans."""
    return int(variable), frozenset(int(value) for value in conditioning)


def _freeze(value):
    """Return a deterministic immutable representation for cache identity."""
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


class FrozenConditionalPlan(tuple):
    """Tuple of read-only steps with a stable semantic cache signature."""

    def __new__(cls, steps, d):
        """Freeze the steps and attach their stable semantic digest."""
        immutable_steps = tuple(
            MappingProxyType(dict(step)) for step in steps
        )
        instance = super().__new__(cls, immutable_steps)
        instance.d = int(d)
        signature = (
            instance.d,
            tuple(_freeze(step) for step in immutable_steps),
        )
        instance.native_signature = signature
        instance.native_signature_digest = hashlib.blake2b(
            repr(signature).encode("utf-8"), digest_size=16
        ).digest()
        return instance
