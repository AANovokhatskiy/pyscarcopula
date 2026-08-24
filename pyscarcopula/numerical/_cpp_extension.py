"""Compatibility alias for :mod:`pyscarcopula._native._extension`.

Production ownership moved to ``_native`` in Stage 8.1.  The module alias is
kept only while existing internal adapters migrate through Stage 8.5.
"""

from __future__ import annotations

import sys

from pyscarcopula._native import _extension as _implementation
from pyscarcopula._native import errors as _errors


for _name in _errors.__all__:
    setattr(_implementation, _name, getattr(_errors, _name))

sys.modules[__name__] = _implementation
