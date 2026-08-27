"""Architecture contracts for bivariate and multivariate copulas."""

import numpy as np
import pytest

from pyscarcopula import (
    BivariateCopula,
    BivariateGaussianCopula,
    ClaytonCopula,
    CopulaBase,
    EquicorrGaussianCopula,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    IndependentCopula,
    JoeCopula,
    MultivariateCopula,
    RVineCopula,
    StochasticStudentCopula,
    StudentCopula,
)
from pyscarcopula.api import fit
from pyscarcopula.strategy._base import (
    is_multivariate_copula,
)


BIVARIATE_CLASSES = (
    GumbelCopula,
    ClaytonCopula,
    FrankCopula,
    JoeCopula,
    IndependentCopula,
    BivariateGaussianCopula,
)


def _multivariate_instances():
    return (
        GaussianCopula(),
        StudentCopula(),
        EquicorrGaussianCopula(d=3),
        StochasticStudentCopula(d=3, R=np.eye(3)),
    )


@pytest.mark.parametrize("copula_class", BIVARIATE_CLASSES)
def test_built_in_pair_classes_use_bivariate_base(copula_class):
    assert issubclass(copula_class, BivariateCopula)
    assert issubclass(copula_class, CopulaBase)


@pytest.mark.parametrize("copula", _multivariate_instances())
def test_built_in_multivariate_classes_use_multivariate_base(copula):
    assert isinstance(copula, MultivariateCopula)
    assert isinstance(copula, CopulaBase)
    assert not isinstance(copula, BivariateCopula)
    assert is_multivariate_copula(copula)


@pytest.mark.parametrize("copula", _multivariate_instances())
def test_multivariate_classes_do_not_expose_pair_contract(copula):
    for name in ("rotate", "h", "h_inverse", "pdf"):
        assert not hasattr(copula, name)


@pytest.mark.parametrize(
    "candidate",
    (GaussianCopula, StudentCopula, EquicorrGaussianCopula,
     StochasticStudentCopula),
)
def test_multivariate_classes_are_rejected_as_vine_candidates(candidate):
    with pytest.raises(TypeError, match="cannot be used as a vine pair") as exc:
        RVineCopula(candidates=[candidate])
    if candidate is GaussianCopula:
        assert "BivariateGaussianCopula" in str(exc.value)


def test_dimension_two_multivariate_dispatch_does_not_use_shape_heuristic():
    rng = np.random.default_rng(1202)
    u = rng.uniform(0.05, 0.95, size=(50, 2))
    copula = EquicorrGaussianCopula(d=2)
    result = fit(copula, u, method="mle", maxiter=5)
    assert np.isfinite(result.log_likelihood)
    assert is_multivariate_copula(copula)


def test_declared_dimension_is_validated_before_strategy_dispatch():
    copula = BivariateGaussianCopula()
    u = np.full((10, 3), 0.5)
    with pytest.raises(ValueError, match="expects 2 columns"):
        fit(copula, u, method="mle")


def test_static_multivariate_model_rejects_dynamic_strategy():
    copula = GaussianCopula()
    u = np.full((10, 3), 0.5)
    with pytest.raises(TypeError, match="does not support GAS"):
        fit(copula, u, method="gas")
