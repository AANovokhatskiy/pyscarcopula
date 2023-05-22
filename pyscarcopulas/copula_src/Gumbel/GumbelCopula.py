import sympy as sp
import numpy as np

from numba import njit
from pyscarcopulas.copula_src.ArchimedianCopula import ArchimedianCopula

class GumbelCopula(ArchimedianCopula):
    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.__sp_generator = (-sp.log(self.t))**self.r
        self.__sp_antigenerator = sp.exp(-self.t**(1/self.r))
    
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

        # '''threshold'''
        # tr = 10**(-10)
        # for i in range(0, self.dim):
        #     index = np.argwhere(p_data[i].ravel() < tr)
        #     if len(index) > 0:
        #         res[i] = 0
        return res