"""
pyscarcopula.copula.experimental — experimental copula models.

These models are under active development and their API may change.
"""

from pyscarcopula.copula.experimental.equicorr import EquicorrGaussianCopula
from pyscarcopula.copula.experimental.stochastic_student import StochasticStudentCopula
from pyscarcopula.copula.experimental.stochastic_student_dcc import StochasticStudentDCCCopula

__all__ = [
    'EquicorrGaussianCopula',
    'StochasticStudentCopula',
    'StochasticStudentDCCCopula',
]
