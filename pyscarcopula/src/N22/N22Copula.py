import sympy as sp
import numpy as np
from numba import njit
from functools import lru_cache

from pyscarcopula.src.ArchimedianCopula import ArchimedianCopula

class N22Copula(ArchimedianCopula):
    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.__sp_generator = sp.asin(1 - self.t**self.r)
        self.__sp_inverse_generator = (1 - sp.sin(self.t))**(1 / self.r)
        self.__name = 'N22 copula'

    @property
    def name(self):
        return self.__name
              
    @staticmethod
    @njit        
    def transform(r):
        return np.exp(- 0.1 * r**2)
    
    @property
    def sp_generator(self):
        return self.__sp_generator
    
    @property
    def sp_inverse_generator(self):
        return self.__sp_inverse_generator
    
    @staticmethod
    @njit
    def restrict(arr, data, r, threshold):
        t = np.sum(np.arcsin(1 - data**r), axis=1)
        arr[t >= np.pi / 2] = threshold
        
    def pdf(self, data: np.array, r: np.array):
        '''Numpy pdf function'''
        func = self.np_pdf()
        res = func(data.T, r)
        self.restrict(res, data, r, 1.0e-20)
        return res

    # @lru_cache
    # def sp_pdf(self):
    #     '''Sympy expression of copula's pdf'''
    #     u = sp.symbols('u0:%d'%(self.dim))
    #     params = [self.sp_generator.subs([(self.t, x)]) for x in u]
    #     func = self.sp_inverse_generator.subs([(self.t, sum(params))])
    #     func 
    #     for k in u:
    #         func = func.diff(k)
    #     #func = sp.trigsimp(sp.together(func))
    #     func = sp.together(func)
    #     func = sp.Piecewise(( 10**(-20)  , sum(params) >= np.pi / 2), (func, True)) #* (-1)**(self.dim)
    #     return func

    # @lru_cache
    # def sp_pdf(self):
    #     '''Sympy expression of copula's pdf'''
    #     u = sp.symbols('u0:%d'%(self.dim))
    #     params = [self.sp_generator.subs([(self.t, x)]) for x in u]
    #     #func = self.sp_inverse_generator.subs([(self.t, sum(params))])
    #     diff_inverse_generator = sp.together(self.sp_inverse_generator.diff((self.t, self.dim)))
    #     diff_generator = sp.together(self.sp_generator.diff(self.t, 1))
    #     func = diff_inverse_generator.subs([(self.t, sum(params))])
    #     for x in u:
    #         func = func * diff_generator.subs([(self.t, x)])
    #     func = sp.powsimp(func)
    #     func = sp.Piecewise((0, sum(params) > sp.pi / 2), (func, True))
    #     return func
