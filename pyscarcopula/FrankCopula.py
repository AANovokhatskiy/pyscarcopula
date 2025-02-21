import sympy as sp
import numpy as np

from typing import Literal

from numba import njit
from pyscarcopula.ArchimedianCopula import ArchimedianCopula
from functools import lru_cache


#@jit(nopython = True, cache = True)
def bivariateFrankPDF(u, r, rotate):
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


class FrankCopula(ArchimedianCopula):
    def __init__(self, dim: int = 2, rotate: Literal[0, 90, 180, 270] = 0) -> None:
        super().__init__(dim, rotate)
        self.__sp_generator = -sp.log( (sp.exp(-self.r * self.t) - 1) / (sp.exp(-self.r) - 1) )
        self.__sp_inverse_generator = - 1 / self.r * sp.log(1 + sp.exp(-self.t) * (sp.exp(-self.r) - 1) )
        self.__name = 'Frank copula'

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
            func = -sp.log(1 + (sp.exp(-self.r * u[0]) - 1) * (sp.exp(-self.r * u[1]) - 1) \
                           / (sp.exp(-self.r) - 1)) * 1/self.r
        else:
            func = self.sp_cdf_from_generator()
        return func
    
    @lru_cache
    def sp_pdf(self):
        u = sp.symbols('u0:%d'%(self.dim), positive = True)

        if self.dim == 2 and self.rotate == 0:
            func = sp.exp(self.r * (1 + u[0] + u[1])) * (sp.exp(self.r) - 1) * self.r\
            / (sp.exp(self.r * (u[0] + u[1])) - sp.exp(self.r) *\
                (-1 + sp.exp(self.r * u[0]) + sp.exp(self.r * u[1])))**2
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
        if isinstance(r, (int, float)):
            r_arr = np.ones(N) * r
        elif isinstance(r, np.ndarray):
            if len(r) == 1:
                r_arr = np.ones(N) * r[0]
            else:
                r_arr = r
        return self.V_auxiliary(N, r_arr)


    def get_sample(self, N, r):
        if self.dim == 2:
            if isinstance(r, (int, float)):
                r_arr = np.ones(N) * r
            elif isinstance(r, np.ndarray):
                if len(r) == 1:
                    r_arr = np.ones(N) * r[0]
                else:
                    r_arr = r
            return self.bivariate_sample(N, r_arr)
        else:
            pseudo_obs = np.zeros((N, self.dim))

            x = np.random.uniform(0, 1, size = (N, self.dim))
            
            V_data = self.V(N, r)

            for k in range(0, self.dim):
                pseudo_obs[:,k] = self.psi(-np.log(x[:,k]) / V_data, r)

            return pseudo_obs

    @lru_cache
    def np_pdf(self):
        if self.dim == 2:
            rotate = self.rotate

            #@njit(cache = True)
            def bivariate_np_pdf(u, r):
                return bivariateFrankPDF(u, r, rotate)
            
            func = bivariate_np_pdf
            return func
        else:
            '''Numpy pdf function from sympy expression'''
            expr = self.sp_pdf()
            u = sp.symbols('u0:%d'%(self.dim))
            r = sp.symbols('r')
            func = sp.lambdify((u, r), expr, modules = 'numpy', cse = True)
        return func #njit(func)