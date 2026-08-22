"""Typed loader for the conditional-sampling support matrix."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PublicSignature:
    """Expected public parameter names for one conditional entrypoint."""

    name: str
    parameters: tuple[str, ...]


@dataclass(frozen=True)
class ModelCase:
    """One canonical conditional-sampling runtime."""

    id: str
    class_module: str
    class_name: str
    category: str
    dimension: int | str
    conditional_entrypoints: tuple[str, ...]
    exactness: str
    methods: tuple[str, ...]
    rotations: tuple[int, ...]
    correlation_modes: tuple[str, ...]
    factor_estimation_modes: tuple[str, ...]
    structures: tuple[str, ...]
    oracles: tuple[str, ...]
    capability_flags: Mapping[str, bool]
    signatures: tuple[PublicSignature, ...]
    notes: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ModelCase":
        signatures = tuple(
            PublicSignature(name=name, parameters=tuple(parameters))
            for name, parameters in value["signatures"].items()
        )
        return cls(
            id=str(value["id"]),
            class_module=str(value["class_module"]),
            class_name=str(value["class_name"]),
            category=str(value["category"]),
            dimension=value["dimension"],
            conditional_entrypoints=tuple(value["conditional_entrypoints"]),
            exactness=str(value["exactness"]),
            methods=tuple(value["methods"]),
            rotations=tuple(value["rotations"]),
            correlation_modes=tuple(value["correlation_modes"]),
            factor_estimation_modes=tuple(value["factor_estimation_modes"]),
            structures=tuple(value["structures"]),
            oracles=tuple(value["oracles"]),
            capability_flags=dict(value["capability_flags"]),
            signatures=signatures,
            notes=str(value["notes"]),
        )


@dataclass(frozen=True)
class UnsupportedCase:
    """A documented model/configuration combination that must fail early."""

    id: str
    model_id: str
    probe: str
    configuration: Mapping[str, Any]
    expected_exception: str
    reason: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "UnsupportedCase":
        return cls(
            id=str(value["id"]),
            model_id=str(value["model_id"]),
            probe=str(value["probe"]),
            configuration=dict(value["configuration"]),
            expected_exception=str(value["expected_exception"]),
            reason=str(value["reason"]),
        )


@dataclass(frozen=True)
class ConditionalCase:
    """One model/method cell with its supported mode axes and API contract.

    The mode axes are deliberately not expanded into a Cartesian product:
    some dynamic methods support only a subset of a model's correlation or
    factor-estimation modes.  Those exclusions live in ``UnsupportedCase``.
    """

    id: str
    model_id: str
    method: str
    entrypoints: tuple[str, ...]
    rotations: tuple[int, ...]
    correlation_modes: tuple[str, ...]
    factor_estimation_modes: tuple[str, ...]
    structures: tuple[str, ...]
    exactness: str
    oracles: tuple[str, ...]


@dataclass(frozen=True)
class SupportRegistry:
    """The complete conditional-sampling support inventory."""

    schema_version: int
    scope: str
    models: tuple[ModelCase, ...]
    unsupported: tuple[UnsupportedCase, ...]

    @property
    def by_id(self) -> Mapping[str, ModelCase]:
        return {case.id: case for case in self.models}

    @property
    def conditional_cases(self) -> tuple[ConditionalCase, ...]:
        """Return the canonical positive model/method cells."""

        return tuple(
            ConditionalCase(
                id=f"{model.id}::{method}",
                model_id=model.id,
                method=method,
                entrypoints=model.conditional_entrypoints,
                rotations=model.rotations,
                correlation_modes=model.correlation_modes,
                factor_estimation_modes=model.factor_estimation_modes,
                structures=model.structures,
                exactness=model.exactness,
                oracles=model.oracles,
            )
            for model in self.models
            for method in model.methods
        )


def load_registry(path: Path | None = None) -> SupportRegistry:
    """Load and type-normalize the checked-in JSON registry."""

    if path is None:
        path = Path(__file__).with_name("support_matrix.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return SupportRegistry(
        schema_version=int(raw["schema_version"]),
        scope=str(raw["scope"]),
        models=tuple(
            ModelCase.from_mapping(value) for value in raw["model_cases"]
        ),
        unsupported=tuple(
            UnsupportedCase.from_mapping(value)
            for value in raw["unsupported_cases"]
        ),
    )


REGISTRY = load_registry()
