"""Multivariate copula models and shared contracts."""

from pyscarcopula.copula.multivariate.base import MultivariateCopula
from pyscarcopula.copula.multivariate.equicorr import EquicorrGaussianCopula
from pyscarcopula.copula.multivariate.gaussian import GaussianCopula
from pyscarcopula.copula.multivariate.stochastic_student import (
    StochasticStudentCopula,
)
from pyscarcopula.copula.multivariate.student import StudentCopula

__all__ = (
    "MultivariateCopula",
    "GaussianCopula",
    "StudentCopula",
    "EquicorrGaussianCopula",
    "StochasticStudentCopula",
)
