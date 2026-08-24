"""Stable Python facade over the bundled native computational extension."""

from pyscarcopula._native._extension import available, load
from pyscarcopula._native.errors import (
    CppError,
    CppUnavailable,
    CppUnsupported,
    DEFAULT_STATUS_EXCEPTION_POLICY,
    FailureContext,
    LEGACY_CPP_STATUS_EXCEPTION_POLICY,
    NativeError,
    NativeUnavailable,
    NativeUnsupported,
    StatusExceptionPolicy,
    cpp_status_name,
    native_status_name,
    raise_for_status,
)
from pyscarcopula._native.registry import (
    STRATEGY_REQUIREMENTS,
    StrategyRequirements,
    descriptor_for,
    ensure_capability,
    is_registered_type,
    query_capability,
    registered_model_types,
    strategy_support,
)
from pyscarcopula._native.threads import (
    MAX_NATIVE_THREADS,
    MIN_NATIVE_THREADS,
    validate_n_threads,
)


__all__ = [
    "CppError",
    "CppUnavailable",
    "CppUnsupported",
    "DEFAULT_STATUS_EXCEPTION_POLICY",
    "FailureContext",
    "LEGACY_CPP_STATUS_EXCEPTION_POLICY",
    "MAX_NATIVE_THREADS",
    "MIN_NATIVE_THREADS",
    "NativeError",
    "NativeUnavailable",
    "NativeUnsupported",
    "StatusExceptionPolicy",
    "STRATEGY_REQUIREMENTS",
    "StrategyRequirements",
    "available",
    "cpp_status_name",
    "descriptor_for",
    "ensure_capability",
    "is_registered_type",
    "load",
    "native_status_name",
    "query_capability",
    "raise_for_status",
    "registered_model_types",
    "strategy_support",
    "validate_n_threads",
]
