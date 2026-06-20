"""Supported copula, rotation, transform, and strategy combinations."""

from dataclasses import dataclass

import numpy as np
import pytest

from pyscarcopula import (
    BivariateGaussianCopula,
    ClaytonCopula,
    EquicorrGaussianCopula,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    IndependentCopula,
    JoeCopula,
    StochasticStudentCopula,
    StudentCopula,
)
from pyscarcopula.numerical import _cpp_copula
from pyscarcopula.numerical._cpp_extension import CppUnsupported
from pyscarcopula.strategy._base import list_methods


ALL_METHODS = frozenset(
    {
        "MLE",
        "SCAR-TM-OU",
        "SCAR-TM-JACOBI",
        "SCAR-P-OU",
        "SCAR-M-OU",
        "GAS",
    }
)
BIVARIATE_DYNAMIC_METHODS = ALL_METHODS


@dataclass(frozen=True)
class SupportCase:
    name: str
    factory: object
    family: str
    transform: str
    rotation: int | None
    methods: frozenset[str]
    native_point_ops: bool
    native_gas: bool
    native_scar_ou: bool
    runtime_ready: bool = True


def _archimedean_cases():
    cases = []
    families = (
        ("clayton", ClaytonCopula),
        ("frank", FrankCopula),
        ("gumbel", GumbelCopula),
        ("joe", JoeCopula),
    )
    for family, cls in families:
        for transform in ("softplus", "xtanh"):
            rotations = (0,) if family == "frank" else (0, 90, 180, 270)
            for rotation in rotations:
                point = True
                cases.append(
                    SupportCase(
                        name=f"{family}-{transform}-r{rotation}",
                        factory=lambda cls=cls, transform=transform, rotation=rotation: cls(
                            rotate=rotation,
                            transform_type=transform,
                        ),
                        family=family,
                        transform=transform,
                        rotation=rotation,
                        methods=BIVARIATE_DYNAMIC_METHODS,
                        native_point_ops=point,
                        native_gas=point,
                        native_scar_ou=True,
                    )
                )
    return cases


SUPPORT_CASES = [
    SupportCase(
        name="independent",
        factory=IndependentCopula,
        family="independent",
        transform="constant",
        rotation=0,
        methods=frozenset({"MLE"}),
        native_point_ops=True,
        native_gas=True,
        native_scar_ou=True,
    ),
    *_archimedean_cases(),
    *[
        SupportCase(
            name=f"bivariate-gaussian-{transform}",
            factory=lambda transform=transform: BivariateGaussianCopula(
                transform_type=transform
            ),
            family="bivariate_gaussian",
            transform=transform,
            rotation=0,
            methods=BIVARIATE_DYNAMIC_METHODS,
            native_point_ops=True,
            native_gas=True,
            native_scar_ou=True,
        )
        for transform in ("softplus", "xtanh")
    ],
    SupportCase(
        name="gaussian",
        factory=GaussianCopula,
        family="gaussian",
        transform="matrix",
        rotation=None,
        methods=frozenset({"MLE"}),
        native_point_ops=False,
        native_gas=False,
        native_scar_ou=False,
    ),
    SupportCase(
        name="student",
        factory=StudentCopula,
        family="student",
        transform="matrix_df",
        rotation=None,
        methods=frozenset({"MLE"}),
        native_point_ops=False,
        native_gas=False,
        native_scar_ou=False,
    ),
    SupportCase(
        name="equicorr-gaussian",
        factory=lambda: EquicorrGaussianCopula(d=3),
        family="equicorr_gaussian",
        transform="equicorr",
        rotation=None,
        methods=frozenset({"MLE", "SCAR-TM-OU", "GAS"}),
        native_point_ops=False,
        native_gas=True,
        native_scar_ou=True,
    ),
    SupportCase(
        name="stochastic-student-uninitialized",
        factory=lambda: StochasticStudentCopula(d=3),
        family="stochastic_student",
        transform="student_df",
        rotation=None,
        methods=frozenset(
            {"MLE", "SCAR-TM-OU", "SCAR-P-OU", "SCAR-M-OU", "GAS"}
        ),
        native_point_ops=False,
        native_gas=True,
        native_scar_ou=True,
        runtime_ready=False,
    ),
    SupportCase(
        name="stochastic-student-ready",
        factory=lambda: StochasticStudentCopula(d=3, R=np.eye(3)),
        family="stochastic_student",
        transform="student_df",
        rotation=None,
        methods=frozenset(
            {"MLE", "SCAR-TM-OU", "SCAR-P-OU", "SCAR-M-OU", "GAS"}
        ),
        native_point_ops=False,
        native_gas=True,
        native_scar_ou=True,
    ),
]


def _is_supported(check, copula):
    try:
        check(copula)
    except CppUnsupported:
        return False
    return True


def test_strategy_registry_matches_declared_public_methods():
    assert frozenset(list_methods()) == ALL_METHODS


def test_support_matrix_enumerates_all_archimedean_variants():
    observed = {
        (case.family, case.transform, case.rotation)
        for case in SUPPORT_CASES
        if case.family in {"clayton", "frank", "gumbel", "joe"}
    }
    expected = {
        (family, transform, rotation)
        for family in ("clayton", "frank", "gumbel", "joe")
        for transform in ("softplus", "xtanh")
        for rotation in (
            (0,) if family == "frank" else (0, 90, 180, 270)
        )
    }
    assert observed == expected


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: FrankCopula(rotate=90), "Rotation not supported"),
        (lambda: BivariateGaussianCopula(rotate=90), "Rotation not supported"),
    ],
)
def test_invalid_public_rotation_combinations_raise_expected_errors(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize("case", SUPPORT_CASES, ids=lambda case: case.name)
def test_native_support_matches_declared_support_matrix(case):
    copula = case.factory()

    assert _is_supported(
        _cpp_copula.ensure_supported_for_copula_ops, copula
    ) is case.native_point_ops

    gas_ready = _is_supported(_cpp_copula.ensure_supported_for_gas, copula)
    scar_ready = _is_supported(
        _cpp_copula.ensure_supported_for_scar_ou, copula
    )
    assert gas_ready is (case.native_gas and case.runtime_ready)
    assert scar_ready is (case.native_scar_ou and case.runtime_ready)


def test_capability_and_runtime_readiness_are_separate():
    uninitialized = next(
        case
        for case in SUPPORT_CASES
        if case.name == "stochastic-student-uninitialized"
    )
    ready = next(
        case
        for case in SUPPORT_CASES
        if case.name == "stochastic-student-ready"
    )

    assert uninitialized.family == ready.family
    assert uninitialized.methods == ready.methods
    assert uninitialized.native_gas == ready.native_gas
    assert uninitialized.native_scar_ou == ready.native_scar_ou
    assert not uninitialized.runtime_ready
    assert ready.runtime_ready
