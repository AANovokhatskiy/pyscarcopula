import sympy as sp
import numpy as np

from numba import njit
from pyscarcopula.src.ArchimedianCopula import ArchimedianCopula

class GumbelCopula(ArchimedianCopula):
    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.__sp_generator = (-sp.log(self.t))**self.r
        self.__sp_inverse_generator = sp.exp(-self.t**(1/self.r))
        self.__name = 'Gumbel copula'

    @property
    def name(self):
        return self.__name
        
    @staticmethod
    @njit
    def transform(r):
        return 1 + r**2
    
    @property
    def sp_generator(self):
        return self.__sp_generator
    
    @property
    def sp_inverse_generator(self):
        return self.__sp_inverse_generator
