import sympy as sp
import numpy as np

from numba import njit
from functools import lru_cache

from pyscarcopula.src.ArchimedianCopula import ArchimedianCopula

class GenestGhoudiCopula(ArchimedianCopula):
    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.__sp_generator = (1 - self.t**(1 / self.r))**self.r
        self.__sp_inverse_generator = (1 - self.t**(1 / self.r))**self.r
        self.__name = 'Genest-Ghoudi copula'

    @property
    def name(self):
        return self.__name
        
    @staticmethod
    @njit
    def transform(r):
        return np.cosh(r)
    
    @property
    def sp_generator(self):
        return self.__sp_generator
    
    @property
    def sp_inverse_generator(self):
        return self.__sp_inverse_generator

    @lru_cache
    def np_pdf(self):
        '''Numpy pdf function from sympy expression'''
        expr = self.sp_pdf()
        u = sp.symbols('u0:%d'%(self.dim))
        r = sp.symbols('r')
        func = sp.lambdify((u, r), expr, modules = 'numpy', cse = True)
        return njit(func)


    @staticmethod
    @njit
    def restrict(arr, data, r, threshold):
        t = np.sum((1 - data**(1 / r))**r, axis=1)
        arr[t >= 1] = threshold
    
    #@njit
    def pdf(self, data: np.array, r: np.array):
        '''Numpy pdf function'''
        func = self.np_pdf()
        res = func(data.T, r)
        self.restrict(res, data, r, 0)
        return res