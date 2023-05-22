import sympy as sp
import numpy as np
from numba import njit

from pyscarcopulas.copula_src.ArchimedianCopula import ArchimedianCopula

class N22Copula(ArchimedianCopula):
    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.__sp_generator = sp.asin(1 - self.t**self.r)
        self.__sp_antigenerator = (1 - sp.sin(self.t))**(1 / self.r)
        
    @staticmethod
    @njit        
    def transform(r):
        return np.exp(- 0.1 * r**2)
    
    @property
    def sp_generator(self):
        return self.__sp_generator
    
    @property
    def sp_antigenerator(self):
        return self.__sp_antigenerator
    
    def pdf(self, data: np.array, r: np.array):
        func = self.np_pdf()
        res = func(data.T, r)
        return res