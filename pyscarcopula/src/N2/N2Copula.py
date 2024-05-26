import sympy as sp
import numpy as np

from numba import njit
from pyscarcopula.src.ArchimedianCopula import ArchimedianCopula

class N2Copula(ArchimedianCopula):
    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.__sp_generator = (1 - self.t)**(self.r)
        self.__sp_inverse_generator = 1 - self.t**(1 / self.r)
        self.__name = 'N2 copula'

    @property
    def name(self):
        return self.__name
          
    @staticmethod
    @njit
    def transform(r):
        return r**2 - 1
    
    @property
    def sp_generator(self):
        return self.__sp_generator
    
    @property
    def sp_inverse_generator(self):
        return self.__sp_inverse_generator
