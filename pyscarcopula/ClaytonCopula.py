import sympy as sp
import numpy as np

from typing import Literal

from numba import njit, jit
from pyscarcopula.ArchimedianCopula import ArchimedianCopula
from functools import lru_cache


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

@njit
def ClaytonPDF2_jit(u, r, rotate):
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

    mask_r = r > 100

    mask1 = ~mask_r
    mask2 = mask_r

    res = 0.0

    if mask1:
        temp1 = v1**(-r)
        temp2 = v2**(-r)
        res = (temp1 * temp2 / (v1 * v2) * (r + 1) *
                    (-1 + temp1 + temp2)**(-2 - 1/r))
        return res
    
    if mask2:
        min_v = np.minimum(v1, v2)
        max_v = np.maximum(v1, v2)
        temp3 = (min_v / max_v)**r
        res = (min_v / (v1 * v2) *
                    temp3 / (1 + temp3)**2 * (r - temp3 - (-1/r - 1) * 1/2 * temp3**2))
        return res
    
    return res


class ClaytonCopula(ArchimedianCopula):
    def __init__(self, dim: int = 2, rotate: Literal[0, 90, 180, 270] = 0) -> None:
        super().__init__(dim, rotate)
        self.__name = 'Clayton copula'
        self.__bounds = [(0.0001, np.inf)]

    @property
    def name(self):
        return self.__name

    @property
    def bounds(self):
        return self.__bounds
        
    @staticmethod
    def transform(r):
        return r * np.tanh(r) + 0.0001

    @staticmethod
    @njit
    def transform_jit(r):
        return r * np.tanh(r) + 0.0001

    @staticmethod
    def inv_transform(r):
        return r

    @lru_cache
    def sp_generator(self):
        """
        Returns the symbolic generator function of the copula.

        Returns:
            sympy.Expr: The symbolic generator function.
        """
        t, r = sp.symbols('t r')
        
        result = 1/r * (t**(-r) - 1)
        return result
    
    @lru_cache
    def sp_inverse_generator(self):
        """
        Returns the symbolic inverse generator function of the copula.

        Returns:
            sympy.Expr: The symbolic inverse generator function.
        """
        t, r = sp.symbols('t r')

        result = (1 + t * r)**(-1/r)
        return result

    @lru_cache
    def sp_cdf(self):
        u = sp.symbols('u0:%d'%(self.dim), positive = True)
        t, r = sp.symbols('t r')

        if self.dim == 2 and self.rotate == 0:
            func = (-1 + u[0]**(-r) + u[1]**(-r))**(-1/r) 
        else:
            func = self.sp_cdf_from_generator()
        return func
    
    @lru_cache
    def sp_pdf(self):
        u = sp.symbols('u0:%d'%(self.dim), positive = True)
        t, r = sp.symbols('t r')

        if self.dim == 2 and self.rotate == 0:
            func = (1 + r) * u[0]**(-1 - r) * u[1]**(-1 - r)\
                 * (-1 + u[0]**(-r) + u[1]**(-r))**(-2 - 1/r)
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
    def np_pdf(self, numba_jit = False):
        if self.dim == 2:
            rotate = self.rotate

            if numba_jit == False:
                def bivariate_np_pdf(u, r):
                    return bivariateClaytonPDF(u, r, rotate)
                
                func = bivariate_np_pdf
            else:
                @njit
                def bivariate_np_pdf(u, r):
                    return ClaytonPDF2_jit(u, r, rotate)
                
                func = bivariate_np_pdf

            return func
        else:
            '''Numpy pdf function from sympy expression'''
            expr = self.sp_pdf()
            u = sp.symbols('u0:%d'%(self.dim))
            r = sp.symbols('r')
            func = sp.lambdify((u, r), expr, modules = 'numpy', cse = True)

            if numba_jit == True:
                func = njit(func)

        return func

    @lru_cache
    def np_log_pdf(self, numba_jit = False):
        if self.dim == 2:
            rotate = self.rotate

            if numba_jit == False:
                def bivariate_np_log_pdf(u, r):
                    return np.log(bivariateClaytonPDF(u, r, rotate))
                
                func = bivariate_np_log_pdf
            else:
                @njit
                def bivariate_np_log_pdf(u, r):
                    return np.log(ClaytonPDF2_jit(u, r, rotate))
                
                func = bivariate_np_log_pdf
            return func
        else:
            '''Numpy pdf function from sympy expression'''
            expr = sp.log(self.sp_pdf())
            u = sp.symbols('u0:%d'%(self.dim))
            r = sp.symbols('r')
            func = sp.lambdify((u, r), expr, modules = 'numpy', cse = True)

            if numba_jit == True:
                func = njit(func)

        return func

    @staticmethod
    def h_unrotated(u0, u1, r):
        eps = 1e-6
        _u0 = np.clip(u0, eps, 1 - eps)
        _u1 = np.clip(u1, eps, 1 - eps)

        # x0 = _u1**(-r)
        # x1 = x0 - 1 + _u0**(-r)
        # return x0 * x1**(-1/r) / (_u1 * x1)
        return _u1**(-r - 1) * (_u0**(-r) + _u1**(-r)- 1)**(-1-1/r)       
    
    @staticmethod
    def h_inverse_unrotated(u0, u1, r):
        eps = 1e-6
        _u0 = np.clip(u0, eps, 1 - eps)
        _u1 = np.clip(u1, eps, 1 - eps)

        # return ((_u1 ** (-r / (1 + r)) - 1) * _u0 ** r + 1) ** (-1 / r)
        return ((_u0 * _u1**(r + 1))**(-r / (1 + r)) + 1 - _u1**(-r))**(-1/r)