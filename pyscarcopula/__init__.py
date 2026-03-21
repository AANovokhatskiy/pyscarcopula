import os as _os

# Limit BLAS to single thread to prevent oversubscription
# when using multiprocessing for parallel CVaR or vine fitting.
# Users can override by setting these env vars before importing.
for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    _os.environ.setdefault(_var, '1')
del _os, _var

from pyscarcopula.copula.gumbel import GumbelCopula
from pyscarcopula.copula.frank import FrankCopula
from pyscarcopula.copula.joe import JoeCopula
from pyscarcopula.copula.clayton import ClaytonCopula
from pyscarcopula.copula.independent import IndependentCopula
from pyscarcopula.copula.vine import CVineCopula

from pyscarcopula.copula.elliptical import (
    BivariateGaussianCopula, GaussianCopula, StudentCopula
)

from pyscarcopula.latent.ou_process import OULatentProcess
from pyscarcopula.latent.gas_process import GASProcess


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