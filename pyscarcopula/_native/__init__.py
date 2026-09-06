"""Stable Python facade over the bundled native computational extension."""

from pyscarcopula._native._extension import load
from pyscarcopula._native.errors import (
    DEFAULT_STATUS_EXCEPTION_POLICY,
    FailureContext,
    NATIVE_ADAPTER_STATUS_EXCEPTION_POLICY,
    NativeError,
    NativeUnavailable,
    NativeUnsupported,
    StatusExceptionPolicy,
    native_status_name,
    raise_for_status,
)
from pyscarcopula._native.registry import (
    STRATEGY_REQUIREMENTS,
    StrategyRequirements,
    descriptor_for,
    ensure_capability,
    is_registered_type,
    native_id_for,
    query_capability,
    registry_entry_for,
    registered_model_types,
    strategy_support,
)
from pyscarcopula._native.threads import (
    MAX_NATIVE_THREADS,
    MIN_NATIVE_THREADS,
    validate_n_threads,
)


__all__ = [
    "DEFAULT_STATUS_EXCEPTION_POLICY",
    "FailureContext",
    "NATIVE_ADAPTER_STATUS_EXCEPTION_POLICY",
    "MAX_NATIVE_THREADS",
    "MIN_NATIVE_THREADS",
    "NativeError",
    "NativeUnavailable",
    "NativeUnsupported",
    "StatusExceptionPolicy",
    "STRATEGY_REQUIREMENTS",
    "StrategyRequirements",
    "descriptor_for",
    "ensure_capability",
    "is_registered_type",
    "load",
    "native_id_for",
    "native_status_name",
    "query_capability",
    "raise_for_status",
    "registry_entry_for",
    "registered_model_types",
    "strategy_support",
    "validate_n_threads",
]
