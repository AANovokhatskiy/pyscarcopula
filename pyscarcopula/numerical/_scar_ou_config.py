"""Configuration shared by native SCAR-OU adapters and strategies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyscarcopula.numerical._transition_methods import (
    normalize_ou_transition_method,
)


@dataclass(frozen=True)
class AutoTMConfig:
    """Numerical configuration for native SCAR-TM-OU evaluation."""

    transition_method: str = "auto"
    small_kdt: float = 1e-2
    basis_order: int = 32
    quad_order: int | None = None
    K: int = 300
    grid_range: float = 5.0
    grid_method: str = "auto"
    adaptive: bool = True
    pts_per_sigma: int = 4
    max_K: int | None = 1000
    gh_order: int = 5
    r_gh: float = 3.0


CPP_MAX_GRID_SIZE = 100_000
CPP_MAX_DENSE_GRID_SIZE = 10_000
CPP_MAX_SPECTRAL_ORDER = 1_024


def _cpp_integer_option(name: str, value, minimum: int, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)):
        raise ValueError(
            f"{name} must be an integer in [{minimum}, {maximum}]")
    result = int(value)
    if result < minimum or result > maximum:
        raise ValueError(
            f"{name} must be in [{minimum}, {maximum}], got {result}")
    return result


def _cpp_finite_option(name: str, value, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{name} must be a {qualifier} number")
    return result


def validate_cpp_config(
        config: AutoTMConfig,
        *,
        transition_method: str | None = None) -> None:
    """Validate dimensions accepted by the C++ SCAR-OU implementation."""
    method = normalize_ou_transition_method(
        transition_method or config.transition_method)
    grid_limit = (
        CPP_MAX_DENSE_GRID_SIZE
        if method == "matrix"
        else CPP_MAX_GRID_SIZE
    )
    _cpp_integer_option("K", config.K, 2, grid_limit)
    if config.max_K is not None:
        _cpp_integer_option("max_K", config.max_K, 2, grid_limit)
    _cpp_integer_option(
        "pts_per_sigma", config.pts_per_sigma, 1, 2_147_483_647)
    _cpp_integer_option("gh_order", config.gh_order, 1, CPP_MAX_SPECTRAL_ORDER)

    basis_order = _cpp_integer_option(
        "basis_order", config.basis_order, 1, CPP_MAX_SPECTRAL_ORDER)
    if config.quad_order is None:
        quad_order = max(2 * basis_order + 16, 48)
        if quad_order > CPP_MAX_SPECTRAL_ORDER:
            raise ValueError(
                "quad_order derived from basis_order exceeds "
                f"{CPP_MAX_SPECTRAL_ORDER}; set a smaller basis_order or an "
                "explicit quad_order")
    else:
        quad_order = _cpp_integer_option(
            "quad_order", config.quad_order, 1, CPP_MAX_SPECTRAL_ORDER)
    if quad_order < basis_order:
        raise ValueError(
            f"quad_order must be at least basis_order ({basis_order})")

    _cpp_finite_option("grid_range", config.grid_range, positive=True)
    _cpp_finite_option("small_kdt", config.small_kdt)
    _cpp_finite_option("r_gh", config.r_gh)


def select_auto_backend(
        kappa: float,
        n_obs: int,
        config: AutoTMConfig | None = None) -> str:
    """Return the native backend selected before numerical fallbacks."""
    cfg = config or AutoTMConfig()
    method = normalize_ou_transition_method(cfg.transition_method)
    if method != "auto":
        return method
    kappa_dt = float(kappa) if n_obs <= 1 else float(kappa) / (n_obs - 1)
    return "local" if kappa_dt < cfg.small_kdt else "spectral"


__all__ = ["AutoTMConfig", "select_auto_backend", "validate_cpp_config"]
