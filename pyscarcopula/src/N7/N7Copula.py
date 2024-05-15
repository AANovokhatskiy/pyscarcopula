import sympy as sp
import numpy as np

from numba import njit
from pyscarcopula.src.ArchimedianCopula import ArchimedianCopula

class N7Copula(ArchimedianCopula):
    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.__sp_generator = - sp.log(self.r * self.t + (1 - self.r))
        self.__sp_inverse_generator = 1 / self.r * (sp.exp(-self.t) + self.r - 1)
        self.__name = 'N7 copula'

    @property
    def name(self):
        return self.__name
          
    @staticmethod
    @njit
    def transform(r):
        return np.exp(-r**2)
    
    @property
    def sp_generator(self):
        return self.__sp_generator
    
    @property
    def sp_inverse_generator(self):
        return self.__sp_inverse_generator
