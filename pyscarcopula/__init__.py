"""
pyscarcopula — stochastic copula models with Ornstein-Uhlenbeck latent process.

Usage:
    from pyscarcopula import GumbelCopula
    from pyscarcopula.api import fit, smoothed_params
    from pyscarcopula.stattests import gof_test

    copula = GumbelCopula(rotate=180)
    result = fit(copula, u, method='scar-tm-ou')
"""

# Force single-threaded BLAS before any numpy import.
# SCAR-TM likelihood uses many small mat-vecs (K=300) inside a Python loop;
# multi-threaded BLAS adds overhead without benefit and steals cores from
# joblib parallelism in risk_metrics / rolling windows.
import os as _os
for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    _os.environ.setdefault(_var, '1')
del _os, _var

from pyscarcopula.copula.gumbel import GumbelCopula
from pyscarcopula.copula.frank import FrankCopula
from pyscarcopula.copula.joe import JoeCopula
from pyscarcopula.copula.clayton import ClaytonCopula
from pyscarcopula.copula.independent import IndependentCopula
from pyscarcopula.vine.cvine import CVineCopula
from pyscarcopula.vine.rvine import RVineCopula

from pyscarcopula.copula.elliptical import (
    BivariateGaussianCopula, GaussianCopula, StudentCopula
)
from pyscarcopula.copula.experimental.stochastic_student import StochasticStudentCopula
from pyscarcopula.copula.experimental.stochastic_student_dcc import StochasticStudentDCCCopula
from pyscarcopula._types import PredictConfig
from pyscarcopula.io import load_model, save_model


__all__ = (
    # Archimedean
    'GumbelCopula', 'FrankCopula', 'JoeCopula', 'ClaytonCopula',
    # Special
    'IndependentCopula',
    # Elliptical
    'GaussianCopula', 'StudentCopula', 'BivariateGaussianCopula',
    # Stochastic multivariate
    'StochasticStudentCopula',
    'StochasticStudentDCCCopula',
    # Vine
    'CVineCopula',
    'RVineCopula',
    # Prediction options
    'PredictConfig',
    # Persistence
    'save_model', 'load_model',
)
