"""Stage-3 bivariate family, rotation, and dependence regimes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BivariateCase:
    family: str
    rotation: int
    regime: str
    parameter: float

    @property
    def id(self) -> str:
        return (
            f"{self.family}-r{self.rotation}-{self.regime}"
        )


ROTATIONS = {
    "independent": (0,),
    "gaussian": (0,),
    "clayton": (0, 90, 180, 270),
    "gumbel": (0, 90, 180, 270),
    "frank": (0,),
    "joe": (0, 90, 180, 270),
}

PARAMETER_REGIMES = {
    "independent": (("independent", 0.0),),
    "gaussian": (
        ("negative-strong", -0.75),
        ("weak", -0.15),
        ("medium", 0.50),
        ("positive-strong", 0.90),
    ),
    "clayton": (("weak", 0.08), ("medium", 0.80), ("strong", 3.0)),
    "gumbel": (("weak", 1.05), ("medium", 1.50), ("strong", 3.5)),
    "frank": (("weak", 0.20), ("medium", 2.50), ("strong", 8.0)),
    "joe": (("weak", 1.05), ("medium", 1.50), ("strong", 3.5)),
}

CONFIGURATIONS = tuple(
    (family, rotation)
    for family, rotations in ROTATIONS.items()
    for rotation in rotations
)

CASES = tuple(
    BivariateCase(family, rotation, regime, parameter)
    for family, rotation in CONFIGURATIONS
    for regime, parameter in PARAMETER_REGIMES[family]
)

MEDIUM_CASES = tuple(
    next(
        case
        for case in CASES
        if case.family == family
        and case.rotation == rotation
        and case.regime in ("independent", "medium")
    )
    for family, rotation in CONFIGURATIONS
)


def transposed_rotation(rotation: int) -> int:
    rotation = int(rotation)
    if rotation not in (0, 90, 180, 270):
        raise ValueError("rotation must be 0, 90, 180, or 270")
    return {0: 0, 90: 270, 180: 180, 270: 90}[rotation]

