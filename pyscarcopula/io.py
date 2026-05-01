"""JSON model persistence helpers."""

from __future__ import annotations

import copy
import importlib
import json
import math
from dataclasses import fields, is_dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import OptimizeResult


MODEL_FORMAT = "pyscarcopula-model"
MODEL_FORMAT_VERSION = 2
_TYPE = "__pyscarcopula_type__"


def _package_version() -> str | None:
    try:
        return metadata.version("pyscarcopula")
    except metadata.PackageNotFoundError:
        return None


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
        return {
            _TYPE: "object",
            "class": _qualified_name(obj),
            "state": {
                key: _to_jsonable(value)
                for key, value in obj.__dict__.items()
            },
        }
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _from_jsonable(payload: Any) -> Any:
    if isinstance(payload, list):
        return [_from_jsonable(item) for item in payload]
    if not isinstance(payload, dict) or _TYPE not in payload:
        return payload

    tag = payload[_TYPE]
    if tag == "ndarray":
        arr = np.asarray(
            _from_jsonable(payload["data"]),
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
        return OptimizeResult(_from_jsonable(payload["data"]))
    if tag == "dataclass":
        cls = _resolve_class(payload["class"])
        values = {
            key: _from_jsonable(value)
            for key, value in payload["fields"].items()
        }
        return cls(**values)
    if tag == "tuple":
        return tuple(_from_jsonable(item) for item in payload["items"])
    if tag == "frozenset":
        return frozenset(_from_jsonable(item) for item in payload["items"])
    if tag == "set":
        return set(_from_jsonable(item) for item in payload["items"])
    if tag == "dict":
        return {
            _from_jsonable(key): _from_jsonable(value)
            for key, value in payload["items"]
        }
    if tag == "object":
        cls = _resolve_class(payload["class"])
        obj = cls.__new__(cls)
        obj.__dict__.update({
            key: _from_jsonable(value)
            for key, value in payload["state"].items()
        })
        return obj
    raise ValueError(f"Unsupported JSON persistence tag: {tag!r}")


def save_model(model: object, path: str | Path, *, include_data: bool = True) -> None:
    """Persist a fitted model to ``path`` as JSON.

    Parameters
    ----------
    model : object
        Model instance to serialize.
    path : str or pathlib.Path
        Destination file path.
    include_data : bool, default True
        If False, drop cached training pseudo-observations stored as
        ``_last_u`` before writing. This reduces file size and avoids
        persisting the training sample, but loaded dynamic models may require
        explicit data passed to prediction methods.
    """
    payload_model = model if include_data else _without_training_data(model)
    envelope = {
        "format": MODEL_FORMAT,
        "format_version": MODEL_FORMAT_VERSION,
        "package_version": _package_version(),
        "class": _qualified_name(payload_model),
        "include_data": bool(include_data),
        "state": _to_jsonable(payload_model),
    }
    with Path(path).open("w", encoding="utf-8") as fh:
        json.dump(
            envelope,
            fh,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


def load_model(path: str | Path, *, expected_type: type | None = None) -> Any:
    """Load a model persisted by :func:`save_model`."""
    with Path(path).open("r", encoding="utf-8") as fh:
        envelope = json.load(fh)

    if not isinstance(envelope, dict) or envelope.get("format") != MODEL_FORMAT:
        raise ValueError("Not a pyscarcopula model file")
    if envelope.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError(
            "Unsupported pyscarcopula model format version: "
            f"{envelope.get('format_version')!r}"
        )

    model = _from_jsonable(envelope.get("state"))
    if expected_type is not None and not isinstance(model, expected_type):
        raise TypeError(
            f"Expected {expected_type.__name__}, got {type(model).__name__}"
        )
    return model
