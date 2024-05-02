import sympy as sp
import numpy as np
from numba import njit
from functools import lru_cache

from pyscarcopula.src.ArchimedianCopula import ArchimedianCopula

class N20Copula(ArchimedianCopula):
    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.__sp_generator = sp.exp(self.t**(-self.r)) - sp.E
        self.__sp_inverse_generator = sp.log(self.t + sp.E)**(-1 / self.r)
        self.__name = 'N20 copula'

    @property
    def name(self):
        return self.__name
              
    @staticmethod
    @njit       
    def transform(r):
        return r**2
    
    @property
    def sp_generator(self):
        return self.__sp_generator
    
    @property
    def sp_inverse_generator(self):
        return self.__sp_inverse_generator
    
    # @lru_cache
    # def sp_pdf(self):
    #     '''Sympy expression of copula's pdf'''
    #     u = sp.symbols('u0:%d'%(self.dim))
    #     params = [self.sp_generator.subs([(self.t, x)]) for x in u]
    #     func = self.sp_inverse_generator.subs([(self.t, sum(params))])
    #     for k in u:
    #         func = func.diff(k)
    #     func = sp.together(func)

    #     '''threshold value'''
    #     tr = (0.01)**(self.r)
    #     params1 = [sp.Piecewise((tr, x > tr), (x, True)) for x in u] 
    #     func.subs([(u, params1)])
    #     return func
    
    #     c = sp.symbols('c')
    #     params1 = [sp.exp(x ** (-self.r)) for x in u]
    #     params2 = [sp.exp(x ** (-self.r) - c) for x in u]
    #     list1 = sp.log(sum(params1) - (self.dim - 1) * sp.E)
    #     list2 = sp.log(sum(params2) - (self.dim - 1) * sp.E) + c
    #     list3 = [x ** (-self.r) for x in u]
    #     func = func.subs([(list1, list2), (c, sp.Max(*list3))])
    #     func = sp.expand_log(sp.log(func), force = True)
    #     return func
