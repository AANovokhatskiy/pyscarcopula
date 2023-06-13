import sympy as sp
import numpy as np

from numba import njit
from pyscarcopula.src.ArchimedianCopula import ArchimedianCopula

class GumbelBarnettCopula(ArchimedianCopula):
    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.__sp_generator = sp.log(1 - self.r * sp.log(self.t))
        self.__sp_inverse_generator = sp.exp(1 / self.r * (1 - sp.exp(self.t)))
        self.__name = 'Gumbel-Barnett copula'

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
