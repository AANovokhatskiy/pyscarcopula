import sympy as sp
import numpy as np

from typing import Literal

from numba import njit, jit
from pyscarcopula.ArchimedianCopula import ArchimedianCopula
from functools import lru_cache


# @jit(nopython = True, cache = True)
def bivariateClaytonPDF(u, r, rotate):
    u1 = u[0]
    u2 = u[1]

    if rotate == 0:
        v1 = u1
        v2 = u2
    elif rotate == 90:
        v1 = 1 - u1
        v2 = u2
    elif rotate == 180:
        v1 = 1 - u1
        v2 = 1 - u2
    elif rotate == 270:
        v1 = u1
        v2 = 1 - u2

    min_v = np.minimum(v1, v2)
    max_v = np.maximum(v1, v2)

    mask_r = r > 100

    mask1 = ~mask_r
    mask2 = mask_r

    res = np.zeros_like(r)

    temp1 = np.zeros_like(r)
    temp2 = np.zeros_like(r)
    temp3 = np.zeros_like(r)

    temp1[mask1] = v1[mask1]**(-r[mask1])
    temp2[mask1] = v2[mask1]**(-r[mask1])
    temp3[mask2] = (min_v[mask2] / max_v[mask2])**r[mask2]

    res[mask1] = (temp1[mask1] * temp2[mask1] / 
                  (v1[mask1] * v2[mask1]) *
                  (r[mask1] + 1) *
                  (-1 + temp1[mask1] + temp2[mask1])**(-2 - 1/r[mask1])
                 )
    
    res[mask2] = (min_v[mask2] /
                  (v1[mask2] * v2[mask2]) *
                  temp3[mask2] / 
                  (1 + temp3[mask2])**2 *
                  (r[mask2] - temp3[mask2] - (-1/r[mask2] - 1) * 1/2 * temp3[mask2]**2)
                 )
    
    return res


class ClaytonCopula(ArchimedianCopula):
    def __init__(self, dim: int = 2, rotate: Literal[0, 90, 180, 270] = 0) -> None:
        super().__init__(dim, rotate)
        self.__sp_generator = 1/self.r * (self.t**(-self.r) - 1)
        self.__sp_inverse_generator = (1 + self.t * self.r)**(-1/self.r)
        self.__name = 'Clayton copula'

    @property
    def name(self):
        return self.__name
        
    @staticmethod
    @njit
    def transform(r):
        return r * np.tanh(r) + 0.0001
    
    @property
    def sp_generator(self):
        return self.__sp_generator
    
    @property
    def sp_inverse_generator(self):
        return self.__sp_inverse_generator

    @lru_cache
    def sp_cdf(self):
        u = sp.symbols('u0:%d'%(self.dim), positive = True)
        if self.dim == 2 and self.rotate == 0:
            func = (-1 + u[0]**(-self.r) + u[1]**(-self.r))**(-1/self.r) 
        else:
            func = self.sp_cdf_from_generator()
        return func
    
    @lru_cache
    def sp_pdf(self):
        u = sp.symbols('u0:%d'%(self.dim), positive = True)

        if self.dim == 2 and self.rotate == 0:
            func = (1 + self.r) * u[0]**(-1 - self.r) * u[1]**(-1 - self.r)\
                 * (-1 + u[0]**(-self.r) + u[1]**(-self.r))**(-2 - 1/self.r)
        else:
            func = self.sp_pdf_from_generator()
        return func

    @staticmethod
    def psi(t, r):
        return (1 + t * r)**(-1/r)
     
    @staticmethod
    def V(N, r):
        res = np.random.gamma(1/r, scale = 1, size = N)
        return res
    
    @lru_cache
    def np_pdf(self):
        if self.dim == 2:
            rotate = self.rotate

            #@njit(cache = True)
            def bivariate_np_pdf(u, r):
                return bivariateClaytonPDF(u, r, rotate)
            
            func = bivariate_np_pdf
            return func
        else:
            '''Numpy pdf function from sympy expression'''
            expr = self.sp_pdf()
            u = sp.symbols('u0:%d'%(self.dim))
            r = sp.symbols('r')
            func = sp.lambdify((u, r), expr, modules = 'numpy', cse = True)
        return func #njit(func)