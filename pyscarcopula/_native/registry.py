"""Exact-type model registry and typed native capability facade."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

from pyscarcopula._native import _extension
from pyscarcopula._native.errors import NativeUnsupported


DescriptorBuilder = Callable[[Any, Any], Any]


@dataclass(frozen=True)
class RegistryEntry:
    """One exact Python type to native descriptor mapping."""

    native_id: str
    builder: DescriptorBuilder


@dataclass(frozen=True)
class StrategyRequirements:
    """Native operations required by one Python fit strategy."""

    dynamics: str
    operations: tuple[str, ...]


_BASE_FIT_OPERATIONS = (
    "parameter_transform_bounds_initialization",
    "likelihood_objective_gradient",
)


STRATEGY_REQUIREMENTS = {
    "MLE": StrategyRequirements("MLE", _BASE_FIT_OPERATIONS),
    "GAS": StrategyRequirements(
        "GAS", (*_BASE_FIT_OPERATIONS, "state_filter_smoother")),
    "SCAR-TM-OU": StrategyRequirements(
        "SCAR-TM-OU", (*_BASE_FIT_OPERATIONS, "state_filter_smoother")),
    "SCAR-TM-JACOBI": StrategyRequirements(
        "SCAR-TM-JACOBI",
        (*_BASE_FIT_OPERATIONS, "state_filter_smoother"),
    ),
}


_OPERATION_NAMES = {
    "parameter_transform_bounds_initialization":
        "ParameterTransformBoundsInitialization",
    "point_density_derivatives": "PointDensityDerivatives",
    "row_grid_density_gradient": "RowGridDensityGradient",
    "likelihood_objective_gradient": "LikelihoodObjectiveGradient",
    "state_filter_smoother": "StateFilterSmoother",
    "rosenblatt_residual": "RosenblattResidual",
    "radial_gof_summary": "RadialGofSummary",
    "unconditional_sampling_transform": "UnconditionalSamplingTransform",
    "conditional_sampling_transform": "ConditionalSamplingTransform",
    "arbitrary_conditional_mcmc": "ArbitraryConditionalMcmc",
    "edge_structure_selection_score": "EdgeStructureSelectionScore",
}

_DYNAMICS_NAMES = {
    "MLE": "Mle",
    "GAS": "Gas",
    "SCAR-TM-OU": "ScarTmOu",
    "SCAR-TM-JACOBI": "ScarTmJacobi",
}

_REGISTERED_NATIVE_IDS = (
    "Independent",
    "Clayton",
    "Frank",
    "Gumbel",
    "Joe",
    "BivariateGaussian",
    "Gaussian",
    "Student",
    "EquicorrGaussian",
    "StochasticStudent",
    "Vine",
)


def _dimension(model) -> int:
    value = getattr(model, "dimension", None)
    if value is None:
        value = getattr(model, "d", None)
    return 2 if value is None else int(value)


def _correlation_name(model) -> str:
    mode = getattr(model, "corr_mode", None)
    if mode is None:
        mode = getattr(model, "_corr_mode", "fixed")
    normalized = str(mode).lower()
    return {
        "dense": "Fixed",
        "fixed": "Fixed",
        "shrinkage": "Shrinkage",
        "cholesky": "DenseCholesky",
        "factor": "Factor",
    }.get(normalized, "Fixed")


def _factor_estimation_name(model) -> str:
    value = getattr(model, "factor_estimation", None)
    if value is None:
        value = getattr(model, "_factor_estimation", "two-stage")
    return "Joint" if str(value).lower() == "joint" else "TwoStage"


def _build_pair(model, module, native_id: str):
    return module.make_typed_model_descriptor(
        getattr(module.NativeModelId, native_id),
        2,
        module.CorrelationKind.NotApplicable,
        int(getattr(model, "rotate", 0)),
        module.FactorEstimationKind.TwoStage,
    )


def _build_gaussian(model, module):
    return module.make_typed_model_descriptor(
        module.NativeModelId.Gaussian,
        _dimension(model),
        getattr(module.CorrelationKind, _correlation_name(model)),
        0,
        getattr(
            module.FactorEstimationKind, _factor_estimation_name(model)),
    )


def _build_student(model, module):
    return module.make_typed_model_descriptor(
        module.NativeModelId.Student,
        _dimension(model),
        getattr(module.CorrelationKind, _correlation_name(model)),
        0,
        getattr(
            module.FactorEstimationKind, _factor_estimation_name(model)),
    )


def _build_equicorr(model, module):
    return module.make_typed_model_descriptor(
        module.NativeModelId.EquicorrGaussian,
        _dimension(model),
        module.CorrelationKind.Equicorrelation,
    )


def _build_stochastic_student(model, module):
    factor_estimation = _factor_estimation_name(model)
    correlation_name = _correlation_name(model)
    if correlation_name == "Factor" and factor_estimation == "Joint":
        correlation_name = "FactorJointDynamicEstimation"
    return module.make_typed_model_descriptor(
        module.NativeModelId.StochasticStudent,
        _dimension(model),
        getattr(module.CorrelationKind, correlation_name),
        0,
        getattr(module.FactorEstimationKind, factor_estimation),
    )


def _build_vine(model, module):
    return module.make_typed_model_descriptor(
        module.NativeModelId.Vine,
        _dimension(model),
        module.CorrelationKind.MixedPairEdges,
    )


@lru_cache(maxsize=1)
def _registry() -> dict[type, RegistryEntry]:
    from pyscarcopula.copula.clayton import ClaytonCopula
    from pyscarcopula.copula.elliptical import BivariateGaussianCopula
    from pyscarcopula.copula.frank import FrankCopula
    from pyscarcopula.copula.gumbel import GumbelCopula
    from pyscarcopula.copula.independent import IndependentCopula
    from pyscarcopula.copula.joe import JoeCopula
    from pyscarcopula.copula.multivariate.equicorr import (
        EquicorrGaussianCopula,
    )
    from pyscarcopula.copula.multivariate.gaussian import GaussianCopula
    from pyscarcopula.copula.multivariate.stochastic_student import (
        StochasticStudentCopula,
    )
    from pyscarcopula.copula.multivariate.student import StudentCopula
    from pyscarcopula.vine.vine import VineCopula

    pair_entries = {
        IndependentCopula: "Independent",
        ClaytonCopula: "Clayton",
        FrankCopula: "Frank",
        GumbelCopula: "Gumbel",
        JoeCopula: "Joe",
        BivariateGaussianCopula: "BivariateGaussian",
    }
    entries = {
        cls: RegistryEntry(
            native_id,
            lambda model, module, native_id=native_id:
                _build_pair(model, module, native_id),
        )
        for cls, native_id in pair_entries.items()
    }
    entries.update({
        GaussianCopula: RegistryEntry("Gaussian", _build_gaussian),
        StudentCopula: RegistryEntry("Student", _build_student),
        EquicorrGaussianCopula: RegistryEntry(
            "EquicorrGaussian", _build_equicorr),
        StochasticStudentCopula: RegistryEntry(
            "StochasticStudent", _build_stochastic_student),
        VineCopula: RegistryEntry("Vine", _build_vine),
    })
    registered_ids = tuple(entry.native_id for entry in entries.values())
    if registered_ids != _REGISTERED_NATIVE_IDS:
        raise RuntimeError(
            "Python native model registry is incomplete or out of order: "
            f"expected {_REGISTERED_NATIVE_IDS!r}, got {registered_ids!r}"
        )
    return entries


def is_registered_type(model_or_type) -> bool:
    """Return whether an exact Python type has a native descriptor builder."""
    model_type = model_or_type if isinstance(model_or_type, type) else type(model_or_type)
    return model_type in _registry()


def registry_entry_for(model) -> RegistryEntry:
    """Return the exact registry entry or reject an unregistered subclass."""
    entry = _registry().get(type(model))
    if entry is None:
        raise NativeUnsupported(
            f"{type(model).__name__} is not an exact registered native model"
        )
    return entry


def native_id_for(model) -> str:
    """Return the stable native model id for an exact registered type."""
    return registry_entry_for(model).native_id


def registered_model_types() -> tuple[type, ...]:
    """Return exact registered built-in types in deterministic ID order."""
    return tuple(_registry())


def descriptor_for(model):
    """Build the opaque C++ descriptor for an exact built-in model type."""
    entry = registry_entry_for(model)
    module = _extension.load()
    return entry.builder(model, module)


def _operation(module, value):
    if isinstance(value, module.NativeOperation):
        return value
    normalized = str(value).strip().lower().replace("-", "_")
    try:
        return getattr(module.NativeOperation, _OPERATION_NAMES[normalized])
    except (KeyError, AttributeError) as exc:
        raise ValueError(f"unknown native operation: {value!r}") from exc


def _dynamics(module, value):
    if isinstance(value, module.DynamicsKind):
        return value
    normalized = str(value).strip().upper().replace("_", "-")
    try:
        return getattr(module.DynamicsKind, _DYNAMICS_NAMES[normalized])
    except (KeyError, AttributeError) as exc:
        raise ValueError(f"unknown native dynamics: {value!r}") from exc


def query_capability(model, operation, dynamics="MLE"):
    """Return the C++ support decision for one concrete exact model."""
    module = _extension.load()
    return module.query_capability(
        descriptor_for(model),
        _operation(module, operation),
        _dynamics(module, dynamics),
    )


def ensure_capability(model, operation, dynamics="MLE"):
    """Return a supported decision or raise ``NativeUnsupported``."""
    info = query_capability(model, operation, dynamics)
    if not info.supported:
        raise NativeUnsupported(
            descriptor=type(model).__name__,
            operation=str(operation),
            dynamics=str(dynamics),
            reason=info.reason,
        )
    return info


def strategy_support(model, method):
    """Return the first failed strategy requirement, or a supported result.

    Unregistered model types are rejected before strategy dispatch.
    """
    if not is_registered_type(model):
        registry_entry_for(model)
    normalized = str(method).upper().replace("_", "-")
    requirements = STRATEGY_REQUIREMENTS.get(normalized)
    if requirements is None:
        # User-registered strategies remain a Python extension point during
        # the staged migration and have no built-in native requirement set.
        return None
    last = None
    for operation in requirements.operations:
        last = query_capability(model, operation, requirements.dynamics)
        if not last.supported:
            return last
    return last


__all__ = [
    "RegistryEntry",
    "STRATEGY_REQUIREMENTS",
    "StrategyRequirements",
    "descriptor_for",
    "ensure_capability",
    "is_registered_type",
    "native_id_for",
    "query_capability",
    "registry_entry_for",
    "registered_model_types",
    "strategy_support",
]
