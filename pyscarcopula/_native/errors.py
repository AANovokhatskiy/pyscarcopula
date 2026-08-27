"""Central exception and structured-status policy for native adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
import operator
from typing import Any, Mapping


class NativeError(RuntimeError):
    """Base error for bundled native kernel failures."""


class NativeUnavailable(NativeError):
    """Raised when the mandatory compiled extension cannot be imported."""


class NativeUnsupported(NativeError):
    """Raised when an exact model/capability is not implemented natively."""

    def __init__(
        self,
        message: str | None = None,
        *,
        descriptor: str | None = None,
        operation: str | None = None,
        dynamics: str | None = None,
        reason: str | None = None,
    ) -> None:
        self.descriptor = descriptor
        self.operation = operation
        self.dynamics = dynamics
        self.reason = reason
        if message is None:
            if None in (descriptor, operation, dynamics, reason):
                raise TypeError(
                    "structured NativeUnsupported requires descriptor, "
                    "operation, dynamics, and reason"
                )
            message = (
                f"{descriptor} does not support {operation} for "
                f"{dynamics}: {reason}"
            )
        super().__init__(message)


_STATUS_NAMES = {
    0: "ok",
    1: "null_pointer",
    2: "invalid_size",
    3: "invalid_family",
    4: "invalid_rotation",
    5: "invalid_transform",
    6: "invalid_parameter",
    7: "numerical_failure",
}


@dataclass(frozen=True)
class FailureContext:
    """Python representation of native failure location and diagnostics."""

    index: int = -1
    backend: str | None = None
    fallback: str | None = None
    locations: Mapping[str, int] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StatusExceptionPolicy:
    """Stable exception classes used for one adapter compatibility profile."""

    invalid: type[Exception]
    unsupported: type[Exception]
    numerical: type[Exception]
    other: type[Exception]


DEFAULT_STATUS_EXCEPTION_POLICY = StatusExceptionPolicy(
    invalid=ValueError,
    unsupported=NativeUnsupported,
    numerical=FloatingPointError,
    other=NativeError,
)
NATIVE_ADAPTER_STATUS_EXCEPTION_POLICY = StatusExceptionPolicy(
    invalid=NativeError,
    unsupported=NativeError,
    numerical=NativeError,
    other=NativeError,
)


def native_status_name(status: int) -> str:
    """Return the stable name for a native status code."""
    return _STATUS_NAMES.get(int(status), "unknown")


def _mapping_value(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _raise_structured(
    exception_type: type[Exception],
    message: str,
    *,
    status: int,
    operation: str,
    context: FailureContext,
) -> None:
    exception = exception_type(message)
    exception.status = status
    exception.operation = operation
    exception.failure_index = context.index
    exception.context = context
    exception.failure_context = context
    exception.diagnostics = context.diagnostics
    raise exception


def raise_for_status(
    result: Any,
    operation: str,
    *,
    prefix: str = "C++",
    failure_fields: Mapping[str, str] | None = None,
    context: FailureContext | None = None,
    exception_policy: StatusExceptionPolicy = DEFAULT_STATUS_EXCEPTION_POLICY,
    numerical_exception: type[Exception] | None = None,
) -> None:
    """Translate one mechanical native status using the shared policy.

    ``result`` may be a status integer, a pybind result mapping, or an object
    exposing ``status``.  Adapters only provide location labels; exception
    selection remains centralized here.
    """
    if isinstance(result, Mapping) or hasattr(result, "status"):
        status = int(_mapping_value(result, "status", -1))
    else:
        try:
            status = int(operator.index(result))
        except TypeError as exc:
            raise TypeError(
                "native result must be an integer status or expose status"
            ) from exc
    if status == 0:
        return

    message = (
        f"{prefix} {operation} failed: status={status} "
        f"({native_status_name(status)})"
    )
    fields = failure_fields or {
        "failure_index": "index",
        "failure_row": "row",
        "failure_edge": "edge",
        "failure_operation": "operation",
        "failure_coordinate": "coordinate",
    }
    for key, label in fields.items():
        value = int(_mapping_value(result, key, -1))
        if value >= 0:
            message += f", {label}={value}"

    result_locations = {}
    for key, label in fields.items():
        value = int(_mapping_value(result, key, -1))
        if value >= 0:
            result_locations[label] = value

    context_locations = {} if context is None else dict(context.locations)
    resolved_locations = {**result_locations, **context_locations}
    result_index = result_locations.get("index", -1)
    resolved_index = (
        context.index if context is not None and context.index >= 0
        else result_index
    )
    result_diagnostics = _as_mapping(
        _mapping_value(result, "diagnostics", {}))
    context_diagnostics = (
        {} if context is None else dict(context.diagnostics))
    resolved_context = FailureContext(
        index=resolved_index,
        backend=(
            context.backend if context is not None
            and context.backend is not None
            else _mapping_value(result, "backend")
        ),
        fallback=(
            context.fallback if context is not None
            and context.fallback is not None
            else _mapping_value(result, "fallback_reason")
        ),
        locations=resolved_locations,
        diagnostics={**result_diagnostics, **context_diagnostics},
    )
    if resolved_context.index >= 0 and "index" not in result_locations:
        message += f", index={resolved_context.index}"
    for label, value in resolved_context.locations.items():
        if label not in result_locations and int(value) >= 0:
            message += f", {label}={int(value)}"
    if resolved_context.backend not in (None, "", "unknown"):
        message += f", backend={resolved_context.backend}"
    if resolved_context.fallback not in (None, "", "unknown"):
        message += f", fallback={resolved_context.fallback}"

    if status in (2, 6):
        _raise_structured(
            exception_policy.invalid,
            message,
            status=status,
            operation=operation,
            context=resolved_context,
        )
    if status in (3, 4, 5):
        _raise_structured(
            exception_policy.unsupported,
            message,
            status=status,
            operation=operation,
            context=resolved_context,
        )
    if status == 7:
        exception = numerical_exception or exception_policy.numerical
        _raise_structured(
            exception,
            message,
            status=status,
            operation=operation,
            context=resolved_context,
        )
    _raise_structured(
        exception_policy.other,
        message,
        status=status,
        operation=operation,
        context=resolved_context,
    )


__all__ = [
    "DEFAULT_STATUS_EXCEPTION_POLICY",
    "FailureContext",
    "NATIVE_ADAPTER_STATUS_EXCEPTION_POLICY",
    "NativeError",
    "NativeUnavailable",
    "NativeUnsupported",
    "StatusExceptionPolicy",
    "native_status_name",
    "raise_for_status",
]
