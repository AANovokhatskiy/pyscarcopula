import sympy as sp
import numpy as np

from numba import njit
from pyscarcopulas.copula_src.ArchimedianCopula import ArchimedianCopula

class JoeCopula(ArchimedianCopula):
    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.__sp_generator = - sp.log(1 - (1 - self.t)**self.r)
        self.__sp_antigenerator = 1 - (1 - sp.exp(-self.t))**(1 / self.r)
    
    @staticmethod
    @njit
    def transform(r):
        return np.cosh(r)
    
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