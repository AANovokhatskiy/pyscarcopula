"""Cheap fitted-runtime factories for conditional API contract tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import norm

from pyscarcopula import (
    BivariateGaussianCopula,
    ClaytonCopula,
    CVineCopula,
    EquicorrGaussianCopula,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    IndependentCopula,
    JoeCopula,
    StochasticStudentCopula,
    StudentCopula,
    VineCopula,
)
from pyscarcopula._types import IndependentResult, MLEResult

from ._registry import REGISTRY


@dataclass(frozen=True)
class RuntimeHarness:
    """One fitted MLE runtime plus inputs needed by both public APIs."""

    model_id: str
    model: Any
    u_train: np.ndarray
    result: Any
    dimension: int
    parameter_path: float | np.ndarray | None = None


PAIR_CLASSES = {
    "bivariate-independent": IndependentCopula,
    "bivariate-gaussian": BivariateGaussianCopula,
    "bivariate-clayton": ClaytonCopula,
    "bivariate-gumbel": GumbelCopula,
    "bivariate-frank": FrankCopula,
    "bivariate-joe": JoeCopula,
}

PAIR_PARAMETERS = {
    "bivariate-gaussian": 0.45,
    "bivariate-clayton": 2.0,
    "bivariate-gumbel": 2.0,
    "bivariate-frank": 5.0,
    "bivariate-joe": 2.0,
}

RUNTIME_IDS = tuple(case.id for case in REGISTRY.models)


def correlation(d: int = 3) -> np.ndarray:
    """A deterministic, well-conditioned correlation matrix."""

    result = np.full((d, d), 0.2, dtype=np.float64)
    np.fill_diagonal(result, 1.0)
    return result


def observations(d: int = 3, n: int = 100) -> np.ndarray:
    """Deterministic pseudo-observations used only as prediction history."""

    rng = np.random.default_rng(20260813 + d)
    latent = rng.multivariate_normal(np.zeros(d), correlation(d), size=n)
    return np.clip(norm.cdf(latent), 1e-8, 1.0 - 1e-8)


def independent_vine_specs(d: int) -> list[list[tuple[type, int]]]:
    """Fixed all-independent edges keep contract tests fast and exact."""

    return [
        [(IndependentCopula, 0) for _ in range(d - tree - 1)]
        for tree in range(d - 1)
    ]


def _pair_harness(model_id: str) -> RuntimeHarness:
    model = PAIR_CLASSES[model_id]()
    u_train = observations(d=2)
    if model_id == "bivariate-independent":
        result = IndependentResult(
            log_likelihood=0.0,
            method="MLE",
            copula_name=model.name,
            success=True,
        )
    else:
        result = MLEResult(
            log_likelihood=0.0,
            method="MLE",
            copula_name=model.name,
            success=True,
            copula_param=PAIR_PARAMETERS[model_id],
        )
    model.fit_result = result
    model._last_u = u_train
    return RuntimeHarness(model_id, model, u_train, result, 2)


def build_runtime(model_id: str) -> RuntimeHarness:
    """Build one canonical, cheap conditional-sampling runtime."""

    if model_id in PAIR_CLASSES:
        return _pair_harness(model_id)

    u_train = observations()
    corr = correlation()

    if model_id == "multivariate-gaussian":
        model = GaussianCopula(d=3, R=corr)
        result = model.fit(u_train)
        return RuntimeHarness(model_id, model, u_train, result, 3)

    if model_id == "multivariate-student":
        model = StudentCopula(d=3, R=corr)
        result = model.fit(u_train)
        return RuntimeHarness(model_id, model, u_train, result, 3)

    if model_id == "multivariate-equicorr-gaussian":
        model = EquicorrGaussianCopula(d=3)
        result = MLEResult(
            log_likelihood=0.0,
            method="MLE",
            copula_name=model.name,
            success=True,
            copula_param=0.2,
        )
        model.fit_result = result
        model._last_u = u_train
        return RuntimeHarness(
            model_id, model, u_train, result, 3, parameter_path=0.2
        )

    if model_id == "multivariate-stochastic-student":
        model = StochasticStudentCopula(d=3, R=corr)
        result = MLEResult(
            log_likelihood=0.0,
            method="MLE",
            copula_name=model.name,
            success=True,
            copula_param=6.0,
        )
        model.fit_result = result
        model._last_u = u_train
        return RuntimeHarness(
            model_id, model, u_train, result, 3, parameter_path=6.0
        )

    specs = independent_vine_specs(3)
    if model_id == "vine-generic":
        model = VineCopula.cvine(d=3, order=[0, 1, 2]).fit(
            u_train, method="mle", copulas=specs
        )
        return RuntimeHarness(
            model_id, model, u_train, model.fit_result, 3
        )

    if model_id == "vine-legacy-cvine":
        model = CVineCopula().fit(
            u_train, method="mle", copulas=specs
        )
        return RuntimeHarness(
            model_id, model, u_train, model.fit_result, 3
        )

    raise KeyError(f"unknown conditional runtime {model_id!r}")


def build_unfitted_runtime(model_id: str):
    """Build the corresponding runtime without fitted/configured state."""

    if model_id in PAIR_CLASSES:
        return PAIR_CLASSES[model_id]()
    if model_id == "multivariate-gaussian":
        return GaussianCopula(d=3)
    if model_id == "multivariate-student":
        return StudentCopula(d=3)
    if model_id == "multivariate-equicorr-gaussian":
        return EquicorrGaussianCopula(d=3)
    if model_id == "multivariate-stochastic-student":
        return StochasticStudentCopula(d=3)
    if model_id == "vine-generic":
        return VineCopula.cvine(d=3, order=[0, 1, 2])
    if model_id == "vine-legacy-cvine":
        return CVineCopula()
    raise KeyError(f"unknown conditional runtime {model_id!r}")
