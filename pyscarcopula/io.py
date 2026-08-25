"""JSON model persistence helpers."""

from __future__ import annotations

import copy
import importlib
import json
import math
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, overload

import numpy as np
from scipy.optimize import OptimizeResult


MODEL_FORMAT = "pyscarcopula-model"
_TYPE = "__pyscarcopula_type__"

ModelT = TypeVar("ModelT")


def _removed_method_identity(value: object) -> bool:
    if not isinstance(value, str):
        return False
    identity = "".join(character for character in value.casefold()
                       if character.isalnum())
    return identity in {"scarpou", "scarmou"}


def _reject_removed_persistence(payload: Any) -> None:
    """Reject removed Monte Carlo strategy artifacts without importing them."""
    if isinstance(payload, list):
        for item in payload:
            _reject_removed_persistence(item)
        return
    if not isinstance(payload, dict):
        return

    method = payload.get("method")
    if _removed_method_identity(method):
        raise ValueError(
            "Unsupported persisted model method: legacy SCAR Monte Carlo "
            "artifacts have no migration execution path")
    class_path = payload.get("class")
    if (
        isinstance(class_path, str)
        and class_path.startswith("pyscarcopula.strategy.scar_mc.")
    ):
        raise ValueError(
            "Unsupported persisted model format: legacy SCAR Monte Carlo "
            "strategy classes cannot be loaded")
    if payload.get(_TYPE) == "dict":
        for pair in payload.get("items", ()):
            if (
                isinstance(pair, list)
                and len(pair) == 2
                and pair[0] == "method"
                and _removed_method_identity(pair[1])
            ):
                raise ValueError(
                    "Unsupported persisted model method: legacy SCAR Monte "
                    "Carlo artifacts have no migration execution path")
    for value in payload.values():
        _reject_removed_persistence(value)


def _class_path(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _qualified_name(obj: object) -> str:
    return _class_path(type(obj))


def _resolve_class(path: str) -> type:
    module_name, _, qualname = path.rpartition(".")
    if not module_name or not qualname:
        raise ValueError(f"Invalid class path: {path!r}")
    if not (
        module_name.startswith("pyscarcopula.")
        or module_name == "pyscarcopula"
        or path == "scipy.optimize._optimize.OptimizeResult"
        or path == "scipy.optimize.OptimizeResult"
    ):
        raise ValueError(f"Unsupported persisted class: {path!r}")
    module = importlib.import_module(module_name)
    obj = module
    for part in qualname.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, type):
        raise TypeError(f"Persisted reference is not a class: {path!r}")
    return obj


def _without_training_data(model: object) -> object:
    model_copy = copy.deepcopy(model)
    if hasattr(model_copy, "_last_u"):
        setattr(model_copy, "_last_u", None)
    return model_copy


def _object_state(obj: object) -> dict[str, Any]:
    if hasattr(obj, "__getstate__"):
        state = obj.__getstate__()
        if state is None:
            state = obj.__dict__
    else:
        state = obj.__dict__
    if not isinstance(state, dict):
        raise TypeError(
            f"__getstate__ for {type(obj).__name__} must return a dict")
    return state


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        if math.isnan(obj):
            value = "nan"
        elif obj > 0:
            value = "inf"
        else:
            value = "-inf"
        return {_TYPE: "float", "value": value}
    if isinstance(obj, np.generic):
        return _to_jsonable(obj.item())
    if isinstance(obj, np.ndarray):
        return {
            _TYPE: "ndarray",
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "data": _to_jsonable(obj.tolist()),
        }
    if isinstance(obj, type):
        return {_TYPE: "class", "class": _class_path(obj)}
    if isinstance(obj, OptimizeResult):
        return {
            _TYPE: "optimize_result",
            "data": _to_jsonable(dict(obj)),
        }
    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            _TYPE: "dataclass",
            "class": _qualified_name(obj),
            "fields": {
                field.name: _to_jsonable(getattr(obj, field.name))
                for field in fields(obj)
            },
        }
    if isinstance(obj, tuple):
        return {_TYPE: "tuple", "items": [_to_jsonable(item) for item in obj]}
    if isinstance(obj, frozenset):
        return {
            _TYPE: "frozenset",
            "items": [_to_jsonable(item) for item in sorted(obj)],
        }
    if isinstance(obj, set):
        return {_TYPE: "set", "items": [_to_jsonable(item) for item in sorted(obj)]}
    if isinstance(obj, list):
        return [_to_jsonable(item) for item in obj]
    if isinstance(obj, dict):
        return {
            _TYPE: "dict",
            "items": [
                [_to_jsonable(key), _to_jsonable(value)]
                for key, value in obj.items()
            ],
        }
    if hasattr(obj, "__dict__"):
        state = _object_state(obj)
        return {
            _TYPE: "object",
            "class": _qualified_name(obj),
            "state": {
                key: _to_jsonable(value)
                for key, value in state.items()
            },
        }
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _from_jsonable(payload: Any, _removed_checked: bool = False) -> Any:
    if not _removed_checked:
        _reject_removed_persistence(payload)
    if isinstance(payload, list):
        return [_from_jsonable(item, True) for item in payload]
    if not isinstance(payload, dict) or _TYPE not in payload:
        return payload

    tag = payload[_TYPE]
    if tag == "ndarray":
        arr = np.asarray(
            _from_jsonable(payload["data"], True),
            dtype=np.dtype(payload["dtype"]),
        )
        return arr.reshape(tuple(payload["shape"]))
    if tag == "float":
        value = payload["value"]
        if value == "nan":
            return float("nan")
        if value == "inf":
            return float("inf")
        if value == "-inf":
            return float("-inf")
        raise ValueError(f"Unsupported persisted float value: {value!r}")
    if tag == "class":
        return _resolve_class(payload["class"])
    if tag == "optimize_result":
        return OptimizeResult(_from_jsonable(payload["data"], True))
    if tag == "dataclass":
        cls = _resolve_class(payload["class"])
        values = {
            key: _from_jsonable(value, True)
            for key, value in payload["fields"].items()
        }
        if (
            cls.__module__ == "pyscarcopula._types"
            and cls.__name__ in {"GASResult", "LatentResult"}
        ):
            values.pop("backend", None)
        return cls(**values)
    if tag == "tuple":
        return tuple(_from_jsonable(item, True) for item in payload["items"])
    if tag == "frozenset":
        return frozenset(
            _from_jsonable(item, True) for item in payload["items"])
    if tag == "set":
        return set(_from_jsonable(item, True) for item in payload["items"])
    if tag == "dict":
        return {
            _from_jsonable(key, True): _from_jsonable(value, True)
            for key, value in payload["items"]
        }
    if tag == "object":
        cls = _resolve_class(payload["class"])
        obj = cls.__new__(cls)
        state = {
            key: _from_jsonable(value, True)
            for key, value in payload["state"].items()
        }
        if hasattr(obj, "__setstate__"):
            obj.__setstate__(state)
        else:
            obj.__dict__.update(state)
        return obj
    raise ValueError(f"Unsupported JSON persistence tag: {tag!r}")


def save_model(model: object, path: str | Path, *, include_data: bool = False) -> None:
    """Persist a fitted model to ``path`` as JSON.

    Parameters
    ----------
    model : object
        Model instance to serialize.
    path : str or pathlib.Path
        Destination file path.
    include_data : bool, default False
        If False, drop cached training pseudo-observations stored as
        ``_last_u`` before writing. This reduces file size and avoids
        persisting the training sample. Fitted state, diagnostics, and cached
        likelihood values are still saved. Loaded dynamic models may require
        explicit data passed to prediction methods.
    """
    payload_model = model if include_data else _without_training_data(model)
    envelope = {
        "format": MODEL_FORMAT,
        "class": _qualified_name(payload_model),
        "include_data": bool(include_data),
        "state": _to_jsonable(payload_model),
    }
    _reject_removed_persistence(envelope)
    with Path(path).open("w", encoding="utf-8") as fh:
        json.dump(
            envelope,
            fh,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


@overload
def load_model(
    path: str | Path,
    *,
    expected_type: type[ModelT],
) -> ModelT:
    ...


@overload
def load_model(
    path: str | Path,
    *,
    expected_type: None = None,
) -> Any:
    ...


def load_model(
    path: str | Path,
    *,
    expected_type: type[ModelT] | None = None,
) -> ModelT | Any:
    """Load a model persisted by :func:`save_model`.

    Parameters
    ----------
    path : str or pathlib.Path
        Source JSON document.
    expected_type : type or None
        Optional runtime type constraint. Supplying it also gives static type
        checkers a precise return type.

    Returns
    -------
    object
        Reconstructed model instance.

    Raises
    ------
    ValueError
        If the document is not a supported pyscarcopula model format.
    TypeError
        If the reconstructed model is not an instance of ``expected_type``.
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        envelope = json.load(fh)

    if not isinstance(envelope, dict) or envelope.get("format") != MODEL_FORMAT:
        raise ValueError("Not a pyscarcopula model file")

    model = _from_jsonable(envelope.get("state"))
    declared_path = envelope.get("class")
    if not isinstance(declared_path, str):
        raise ValueError("Persisted model class must be a qualified name")
    declared_type = _resolve_class(declared_path)
    if not isinstance(model, declared_type):
        raise ValueError(
            "Persisted model class does not match serialized state: "
            f"declared {declared_type.__name__}, "
            f"restored {type(model).__name__}")
    if expected_type is not None and not isinstance(model, expected_type):
        raise TypeError(
            f"Expected {expected_type.__name__}, got {type(model).__name__}"
        )
    return model
