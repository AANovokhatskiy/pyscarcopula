"""
pyscarcopula — stochastic copula models with Ornstein-Uhlenbeck latent process.

Usage:
    from pyscarcopula import GumbelCopula
    from pyscarcopula.api import fit, smoothed_params
    from pyscarcopula.stattests import gof_test

    copula = GumbelCopula(rotate=180)
    result = fit(copula, u, method='scar-tm-ou')
"""

from pyscarcopula.copula.gumbel import GumbelCopula
from pyscarcopula.copula.frank import FrankCopula
from pyscarcopula.copula.joe import JoeCopula
from pyscarcopula.copula.clayton import ClaytonCopula
from pyscarcopula.copula.independent import IndependentCopula
from pyscarcopula.copula.vine import CVineCopula

from pyscarcopula.copula.elliptical import (
    BivariateGaussianCopula, GaussianCopula, StudentCopula
)


__all__ = (
    # Archimedean
    'GumbelCopula', 'FrankCopula', 'JoeCopula', 'ClaytonCopula',
    # Special
    'IndependentCopula',
    # Elliptical
    'GaussianCopula', 'StudentCopula', 'BivariateGaussianCopula',
    # Vine
    'CVineCopula',
)
