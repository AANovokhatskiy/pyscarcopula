"""Independent bivariate CDF and directional conditional-CDF formulas.

This module intentionally has no pyscarcopula imports.  The formulas are
used as the primary Stage-3 oracle; pyvinecopulib is a separate external
cross-check rather than the only reference.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal, norm

from ._bivariate_cases import transposed_rotation


_FAMILIES = {
    "independent", "gaussian", "clayton", "gumbel", "frank", "joe"
}


def _validate_family_rotation(family: str, rotation: int) -> tuple[str, int]:
    family = str(family).lower()
    rotation = int(rotation)
    if family not in _FAMILIES:
        raise ValueError(f"unsupported reference family {family!r}")
    if rotation not in (0, 90, 180, 270):
        raise ValueError("rotation must be 0, 90, 180, or 270")
    if family in ("independent", "gaussian", "frank") and rotation != 0:
        raise ValueError(f"rotation {rotation} is unsupported for {family}")
    return family, rotation


def _open_unit(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if np.any(~np.isfinite(array)) or np.any(
            (array <= 0.0) | (array >= 1.0)):
        raise ValueError(f"{name} must be finite and in (0, 1)")
    return array


def base_cdf(u, v, parameter, family: str) -> np.ndarray:
    """Unrotated bivariate copula CDF from its defining formula."""

    family, _ = _validate_family_rotation(family, 0)
    u, v = np.broadcast_arrays(_open_unit(u, "u"), _open_unit(v, "v"))
    parameter = float(np.asarray(parameter, dtype=np.float64))
    if family == "independent":
        return u * v
    if family == "gaussian":
        z = np.column_stack((norm.ppf(u.ravel()), norm.ppf(v.ravel())))
        values = multivariate_normal.cdf(
            z, mean=np.zeros(2), cov=[[1.0, parameter], [parameter, 1.0]]
        )
        return np.asarray(values, dtype=np.float64).reshape(u.shape)
    if family == "clayton":
        return (
            np.power(u, -parameter)
            + np.power(v, -parameter)
            - 1.0
        ) ** (-1.0 / parameter)
    if family == "gumbel":
        radial = (
            np.power(-np.log(u), parameter)
            + np.power(-np.log(v), parameter)
        ) ** (1.0 / parameter)
        return np.exp(-radial)
    if family == "frank":
        numerator = np.expm1(-parameter * u) * np.expm1(-parameter * v)
        return -np.log1p(numerator / np.expm1(-parameter)) / parameter

    first = np.power(1.0 - u, parameter)
    second = np.power(1.0 - v, parameter)
    radial = first + second - first * second
    return 1.0 - np.power(radial, 1.0 / parameter)


def rotated_cdf(u, v, parameter, family: str, rotation: int) -> np.ndarray:
    """CDF after the public 0/90/180/270 coordinate rotations."""

    family, rotation = _validate_family_rotation(family, rotation)
    u, v = np.broadcast_arrays(_open_unit(u, "u"), _open_unit(v, "v"))
    if rotation == 0:
        return base_cdf(u, v, parameter, family)
    if rotation == 90:
        return v - base_cdf(1.0 - u, v, parameter, family)
    if rotation == 180:
        return (
            u + v - 1.0
            + base_cdf(1.0 - u, 1.0 - v, parameter, family)
        )
    return u - base_cdf(u, 1.0 - v, parameter, family)


def base_h_first_given_second(u, v, parameter, family: str) -> np.ndarray:
    """Unrotated H(U1 <= u | U2 = v) = partial C(u,v)/partial v."""

    family, _ = _validate_family_rotation(family, 0)
    u, v, parameter = np.broadcast_arrays(
        _open_unit(u, "u"),
        _open_unit(v, "v"),
        np.asarray(parameter, dtype=np.float64),
    )
    if np.any(~np.isfinite(parameter)):
        raise ValueError("parameter must be finite")
    if family == "independent":
        return u.copy()
    if family == "gaussian":
        z_u = norm.ppf(u)
        z_v = norm.ppf(v)
        return norm.cdf(
            (z_u - parameter * z_v) / np.sqrt(1.0 - parameter ** 2)
        )
    if family == "clayton":
        radial = (
            np.power(u, -parameter)
            + np.power(v, -parameter)
            - 1.0
        )
        return (
            np.power(v, -parameter - 1.0)
            * np.power(radial, -1.0 / parameter - 1.0)
        )
    if family == "gumbel":
        log_u = -np.log(u)
        log_v = -np.log(v)
        radial = np.power(log_u, parameter) + np.power(log_v, parameter)
        root = np.power(radial, 1.0 / parameter)
        return (
            np.exp(-root)
            * np.power(radial, 1.0 / parameter - 1.0)
            * np.power(log_v, parameter - 1.0)
            / v
        )
    if family == "frank":
        first = np.expm1(-parameter * u)
        second = np.expm1(-parameter * v)
        denominator = np.expm1(-parameter) + first * second
        return first * np.exp(-parameter * v) / denominator

    first = np.power(1.0 - u, parameter)
    second = np.power(1.0 - v, parameter)
    radial = first + second - first * second
    return (
        np.power(radial, 1.0 / parameter - 1.0)
        * np.power(1.0 - v, parameter - 1.0)
        * (1.0 - first)
    )


def rotated_h_first_given_second(
        u, v, parameter, family: str, rotation: int) -> np.ndarray:
    """Directional conditional CDF for the first coordinate."""

    family, rotation = _validate_family_rotation(family, rotation)
    u, v = np.broadcast_arrays(_open_unit(u, "u"), _open_unit(v, "v"))
    if rotation == 0:
        return base_h_first_given_second(u, v, parameter, family)
    if rotation == 90:
        return 1.0 - base_h_first_given_second(
            1.0 - u, v, parameter, family
        )
    if rotation == 180:
        return 1.0 - base_h_first_given_second(
            1.0 - u, 1.0 - v, parameter, family
        )
    return base_h_first_given_second(u, 1.0 - v, parameter, family)


def conditional_cdf(
        free_u,
        given_u,
        parameter,
        family: str,
        rotation: int,
        free_index: int) -> np.ndarray:
    """CDF of either free coordinate under the rotated copula."""

    if isinstance(free_index, (bool, np.bool_)) or int(free_index) not in (0, 1):
        raise ValueError("free_index must be 0 or 1")
    free_index = int(free_index)
    directional_rotation = (
        rotation if free_index == 0 else transposed_rotation(rotation)
    )
    return rotated_h_first_given_second(
        free_u, given_u, parameter, family, directional_rotation
    )

