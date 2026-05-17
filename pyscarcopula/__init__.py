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

# Force single-threaded BLAS before any numpy import.
# SCAR-TM likelihood uses many small mat-vecs inside a Python loop;
# multi-threaded BLAS adds overhead and competes with useful outer-level
# parallelism in risk_metrics / rolling windows. Users who intentionally
# want a different policy can set PYSCA_BLAS_THREADS before importing.
import os as _os
_blas_threads = _os.environ.get('PYSCA_BLAS_THREADS', '1')
for _var in (
    'OMP_NUM_THREADS',
    'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'BLIS_NUM_THREADS',
):
    _os.environ[_var] = _blas_threads
del _os, _var, _blas_threads

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
from pyscarcopula.copula.experimental.equicorr import EquicorrGaussianCopula
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
    # Experimental / stochastic multivariate
    'EquicorrGaussianCopula',
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
