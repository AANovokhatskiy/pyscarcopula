import sympy as sp
import numpy as np

from typing import Literal

from numba import njit
from pyscarcopula.ArchimedianCopula import ArchimedianCopula
from functools import lru_cache


def bivariateFrankPDF(u, r):
    v1 = u[0]
    v2 = u[1]

    mask_r = r * (v1 + v2) > 300
    mask_diff = r * np.abs(v1 - v2) > 300

    mask1 = ~mask_r & ~mask_diff
    mask2 = mask_r & ~mask_diff
    mask3 = mask_diff

    res = np.zeros_like(r)

    temp1 = np.zeros_like(r)
    temp2 = np.zeros_like(r)
    temp3 = np.zeros_like(r)

    temp1[mask1] = np.exp(-r[mask1] * v1[mask1])
    temp2[mask1] = np.exp(-r[mask1] * v2[mask1])
    temp3[mask1] = np.exp(-r[mask1])

    res[mask1] = (
                r[mask1] * (1 - temp3[mask1]) * temp1[mask1] * temp2[mask1] / 
                (
                temp3[mask1] + temp1[mask1] * temp2[mask1] -
                temp1[mask1] - temp2[mask1]
                )**2
                )

    res[mask2] = r[mask2] / (2 + np.exp(r[mask2] * (v1[mask2] - v2[mask2])) +
                             np.exp(-r[mask2] * (v1[mask2] - v2[mask2]))
                             )
    
    res[mask3] = r[mask3] * np.exp(-r[mask3] * np.abs(v1[mask3] - v2[mask3]))

    return res

@njit
def FrankPDF2_jit(u, r):
    v1 = u[0]
    v2 = u[1]

    mask_r = r * (v1 + v2) > 300
    mask_diff = r * np.abs(v1 - v2) > 300

    mask1 = ~mask_r & ~mask_diff
    mask2 = mask_r & ~mask_diff
    mask3 = mask_diff

    res = 0.0

    if mask1:
        temp1 = np.exp(-r * v1)
        temp2 = np.exp(-r * v2)
        temp3 = np.exp(-r)
        res = (r * (1 - temp3) * temp1 * temp2 / 
                    (temp3 + temp1 * temp2 - temp1 - temp2)**2)
        return res
    
    if mask2:
        res = r / (2 + np.exp(r * (v1 - v2)) + np.exp(-r * (v1 - v2)))        
        return res
    
    if mask3:
        res = r * np.exp(-r * np.abs(v1 - v2))
        return res
    
    return res

# def bivariateFrankPDF(u, r):
#     """
#     Vectorized bivariate Frank PDF
#     """        
#     v1 = u[0]
#     v2 = u[1]
    
#     # Reshape for broadcasting with r
#     v1 = np.expand_dims(v1, tuple(range(1, r.ndim)))
#     v2 = np.expand_dims(v2, tuple(range(1, r.ndim)))
    
#     # Calculate masks
#     mask_r = r * (v1 + v2) > 300.0
#     mask_diff = r * np.abs(v1 - v2) > 300.0
    
#     mask1 = ~mask_r & ~mask_diff
#     mask2 = mask_r & ~mask_diff
#     mask3 = mask_diff
    
#     res = np.zeros_like(r)
    
#     # Case 1 calculations
#     temp1 = np.zeros_like(r)
#     temp2 = np.zeros_like(r)
#     temp3 = np.zeros_like(r)
    
#     np.putmask(temp1, mask1, np.exp(-r * v1))
#     np.putmask(temp2, mask1, np.exp(-r * v2))
#     np.putmask(temp3, mask1, np.exp(-r))
    
#     # Compute case 1 result
#     numerator = r * (1 - temp3) * temp1 * temp2
#     denominator = (temp3 + temp1 * temp2 - temp1 - temp2)**2
#     np.putmask(res, mask1, numerator / denominator)
    
#     # Case 2 calculations
#     diff = v1 - v2
#     case2 = r / (2 + np.exp(r * diff) + np.exp(-r * diff))
#     np.putmask(res, mask2, case2)
    
#     # Case 3 calculations
#     case3 = r * np.exp(-r * np.abs(v1 - v2))
#     np.putmask(res, mask3, case3)
    
#     return res
# from pyscarcopula.cython_ext import bivariate_frank_pdf


class FrankCopula(ArchimedianCopula):
    def __init__(self, dim: int = 2, rotate: Literal[0] = 0) -> None:
        super().__init__(dim, rotate)

        self.__rotatable = False
        self.__name = 'Frank copula'
        if rotate != 0:
            raise ValueError("Rotation is not supported for Frank copula.")
        self.__bounds = [(0.0001, np.inf)]

    @property
    def rotatable(self):
        return self.__rotatable

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
        
        result = -sp.log((sp.exp(-r* t) - 1) / (sp.exp(-r) - 1))
        return result
    
    @lru_cache
    def sp_inverse_generator(self):
        """
        Returns the symbolic inverse generator function of the copula.

        Returns:
            sympy.Expr: The symbolic inverse generator function.
        """
        t, r = sp.symbols('t r')

        result = - 1 / r* sp.log(1 + sp.exp(-t) * (sp.exp(-r) - 1))
        return result

    @lru_cache
    def sp_cdf(self):
        u = sp.symbols('u0:%d'%(self.dim), positive = True)
        t, r = sp.symbols('t r')

        if self.dim == 2:
            func = -sp.log(1 + (sp.exp(-r * u[0]) - 1) * (sp.exp(-r * u[1]) - 1) \
                           / (sp.exp(-r) - 1)) * 1/r
        else:
            func = self.sp_cdf_from_generator()
        return func
    
    @lru_cache
    def sp_pdf(self):
        u = sp.symbols('u0:%d'%(self.dim), positive = True)
        t, r = sp.symbols('t r')

        if self.dim == 2:
            func = sp.exp(r * (1 + u[0] + u[1])) * (sp.exp(r) - 1) * r\
            / (sp.exp(r * (u[0] + u[1])) - sp.exp(r) *\
                (-1 + sp.exp(r * u[0]) + sp.exp(r * u[1])))**2
        else:
            func = self.sp_pdf_from_generator()
        return func

    @staticmethod
    def psi(t, r):
        return -np.log(np.exp(-t) * (np.exp(-r) - 1) + 1) / r
     
    @staticmethod
    @njit
    def V_auxiliary(N, r):
        res = np.zeros(N)
        for k in range(0, N):
            p0 = (1 - np.exp(-r[k])) / r[k]
            u = np.random.uniform(0, 1)
            i = 1
            p = p0
            F = p
            while u > F:
                multiplier = (1 - np.exp(-r[k])) / (i + 1) * i
                p = multiplier * p
                F = F + p
                i = i + 1
            res[k] = i
        return res
    
    @staticmethod
    @njit
    def bivariate_sample(N, r):
        res = np.zeros((N, 2))
        for k in range(0, N):
            u = np.random.uniform(0, 1)
            v = np.random.uniform(0, 1)
            u0 = u

            t = np.exp(- r[k] * u0)
            p = np.exp(-r[k])

            f1 = v * (1 - p)
            f2 = (t + v * (1 - t))
            #safe calculations
            if np.abs(f1 - f2) < 1e-9:
                u1 = u0
            else:
                u1 = -np.log(1 - f1 / f2) / r[k]
            res[k][0] = u0
            res[k][1] = u1
        return res
    
    def V(self, N, r):  
        return self.V_auxiliary(N, r)

    def get_sample(self, size, r):
        _r = np.asarray(r, dtype=np.float64)
        
        if _r.ndim == 0:
            _r = np.full(size, _r)
        elif _r.shape == (1,):
            _r = np.full(size, _r[0])
        elif len(_r) != size:
            raise ValueError(f"Length of r ({len(_r)}) must match N ({size}) or be a scalar")
        
        if self.dim == 2:
            return self.bivariate_sample(size, _r)
        else:
            pseudo_obs = np.zeros((size, self.dim))

            x = np.random.uniform(0, 1, size = (size, self.dim))
            
            V_data = self.V(size, _r)

            for k in range(0, self.dim):
                pseudo_obs[:,k] = self.psi(-np.log(x[:,k]) / V_data, _r)

            return pseudo_obs

    @lru_cache
    def np_pdf(self, numba_jit = False):
        if self.dim == 2:
            if numba_jit == False:
                func = bivariateFrankPDF
            else:
                func = FrankPDF2_jit
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
            if numba_jit == False:
                def bivariate_np_log_pdf(u, r):
                    return np.log(bivariateFrankPDF(u, r))
                
                func = bivariate_np_log_pdf

            if numba_jit == True:
                @njit
                def bivariate_np_log_pdf(u, r):
                    return np.log(FrankPDF2_jit(u, r))
                
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
    def h_unrotated(u, v, r):
        eps = 1e-6
        _u = np.clip(u, eps, 1 - eps)
        _v = np.clip(v, eps, 1 - eps)

        # x1 = np.exp(r * _u)
        # x2 = np.exp(r * _v) 
        # x3 = np.exp(r)

        # res = x3 * (-1 + x1) / (-x1 * x2 + x3 * (-1 + x1 + x2))
        x1 = np.exp(-r * _u)
        x2 = np.exp(-r * _v) 
        x3 = np.exp(-r)

        res = x2 / ((1 - x3) / (1 - x1) + x2 - 1)

        return res
    
    @staticmethod
    def h_inverse_unrotated(u, v, r):
        eps = 1e-6
        _u = np.clip(u, eps, 1 - eps)
        _v = np.clip(v, eps, 1 - eps)

        # x1 = _u
        # x2 = np.exp(r * _v)
        # x3 = np.exp(r)

        # res = 1/r * (r - np.log(x3 * (1 - x1) + x2 * x1) + np.log(1 + x1 * (-1 + x2)))

        x1 = _u
        x2 = np.exp(-r * _v)
        x3 = np.exp(-r)

        res = -1/r * np.log(1 - (1 - x3) / ((1/x1 - 1) * x2 + 1))

        return res