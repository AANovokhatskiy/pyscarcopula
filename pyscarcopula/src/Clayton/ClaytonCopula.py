import sympy as sp
import numpy as np

from numba import njit
from pyscarcopula.src.ArchimedianCopula import ArchimedianCopula

class ClaytonCopula(ArchimedianCopula):
    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.__sp_generator = 1/self.r * (self.t**(-self.r) - 1)
        self.__sp_inverse_generator = (1 + self.t * self.r)**(-1/self.r)
        self.__name = 'Clayton copula'

    @property
    def name(self):
        return self.__name
        
    @staticmethod
    @njit
    def transform(r):
        return np.minimum(r**2 + 0.0001, 40)
    
    @property
    def sp_generator(self):
        return self.__sp_generator
    
    @property
    def sp_inverse_generator(self):
        return self.__sp_inverse_generator
