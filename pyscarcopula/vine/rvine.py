"""Backward-compatible module alias for the generic vine runtime."""

import sys

from pyscarcopula.vine import vine as _canonical_module


# Preserve old imports and private monkeypatch-based integrations while all
# implementation and module state remain owned by ``pyscarcopula.vine.vine``.
sys.modules[__name__] = _canonical_module
