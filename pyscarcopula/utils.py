"""
Backward compatibility wrapper.

All functions have moved to pyscarcopula._utils.
This module re-exports them so that old code using
`from pyscarcopula.utils import pobs` still works.
"""

from pyscarcopula._utils import pobs, linear_least_squares  # noqa: F401
