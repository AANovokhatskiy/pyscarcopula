"""
pyscarcopula - stochastic copula models with Ornstein-Uhlenbeck latent process.

Usage:
    from pyscarcopula import GumbelCopula
    from pyscarcopula.api import fit, predictive_mean
    from pyscarcopula.stattests import gof_test

    copula = GumbelCopula(rotate=180)
    result = fit(copula, u, method='scar-tm-ou')
"""

# ruff: noqa: E402

from importlib import metadata as _metadata

__version__ = _metadata.version("pyscarcopula")
del _metadata

from pyscarcopula.copula.gumbel import GumbelCopula
from pyscarcopula.copula.frank import FrankCopula
from pyscarcopula.copula.joe import JoeCopula
from pyscarcopula.copula.clayton import ClaytonCopula
from pyscarcopula.copula.independent import IndependentCopula
from pyscarcopula.vine.cvine import CVineCopula
from pyscarcopula.vine.rvine import RVineCopula

from pyscarcopula.copula.elliptical import BivariateGaussianCopula
from pyscarcopula.copula.multivariate import (
    EquicorrGaussianCopula,
    GaussianCopula,
    StochasticStudentCopula,
    StudentCopula,
)
from pyscarcopula.copula.base import (
    BivariateCopula,
    CopulaBase,
    CopulaCapabilities,
)
from pyscarcopula.copula.multivariate import MultivariateCopula
from pyscarcopula._types import (
    LBFGSBConfig,
    MultivariateMLEResult,
    NumericalConfig,
    PredictConfig,
)
from pyscarcopula.io import load_model, save_model


__all__ = (
    # Archimedean
    'GumbelCopula', 'FrankCopula', 'JoeCopula', 'ClaytonCopula',
    # Special
    'IndependentCopula',
    # Elliptical
    'GaussianCopula', 'StudentCopula', 'BivariateGaussianCopula',
    # Dynamic multivariate
    'EquicorrGaussianCopula',
    'StochasticStudentCopula',
    # Base hierarchy and capability contract
    'CopulaBase', 'BivariateCopula', 'MultivariateCopula',
    'CopulaCapabilities',
    # Vine
    'CVineCopula',
    'RVineCopula',
    # Prediction options
    'PredictConfig',
    'MultivariateMLEResult',
    'LBFGSBConfig',
    'NumericalConfig',
    # Persistence
    'save_model', 'load_model',
    # Metadata
    '__version__',
)
