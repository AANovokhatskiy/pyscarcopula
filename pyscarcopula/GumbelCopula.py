import sympy as sp
import numpy as np

from typing import Literal
from functools import lru_cache

from numba import njit
from pyscarcopula.ArchimedianCopula import ArchimedianCopula

@njit
def generate_levy_stable(alpha, beta, loc = 0, scale = 1, size = 1):
    '''
    Weron, R. (1996). On the Chambers-Mallows-Stuck method for simulating skewed stable random variables
    Borak et. al. (2008), Stable Distributions
    '''

    V = np.random.uniform(-np.pi/2, np.pi/2, size = size)
    u = np.random.uniform(0, 1, size = size)
    W = -np.log(1 - u)

    indicator0 = (alpha != 1)
    indicator1 = np.invert(indicator0)

    B = np.arctan(beta * np.tan(np.pi/2 * alpha)) / alpha
    S = (1 + beta**2 * np.tan(np.pi/2 * alpha)**2)**(1 / (2 * alpha))

    X0 = S * np.sin(alpha * (V + B)) / np.cos(V)**(1/alpha) * (np.cos(V - alpha * (V + B)) / W)**((1 - alpha) / alpha)
    X1 = 2 / np.pi * ((np.pi/2 + beta * V) * np.tan(V) - beta * np.log(np.pi/2 * W * np.cos(V) / (np.pi / 2 + beta * V)))

    X = X0 * indicator0 + X1 * indicator1

    Y0 = scale * X + loc
    Y1 = scale * X + 2 / np.pi * beta * scale * np.log(scale) + loc

    Y = Y0 * indicator0 + Y1 * indicator1
        
    return Y

def bivariateGumbelPDF(u, r, rotate):
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

    p1 = -np.log(v1)
    p2 = -np.log(v2)

    '''consider multiple cases'''

    '''first when r is large (boundary value r == 100)'''
    mask_r_ub = r > 100

    '''second when u1 == u2'''
    mask_eq = v1 == v2

    max_p = np.maximum(p1, p2)
    min_p = np.minimum(p1, p2)

    val1 = np.zeros_like(r, dtype=np.float64)
    val1_temp = np.zeros_like(r, dtype=np.float64)

    val2 = np.zeros_like(r, dtype=np.float64)
    res = np.zeros_like(r, dtype=np.float64)

    temp_eq = np.zeros_like(r, dtype=np.float64)
    
    '''calculation for mask_eq'''
    temp_eq[mask_eq] = 2**(1/r[mask_eq]) * p1[mask_eq]
    
    res[mask_eq] = (1 / (v1[mask_eq]**2) * 
                    np.exp(-temp_eq[mask_eq]) *
                    (-1 + r[mask_eq] + temp_eq[mask_eq]) * 
                    2**(1/r[mask_eq] - 2) / p1[mask_eq])
    
    mask1 = ~mask_eq & mask_r_ub
    mask2 = ~mask_eq & ~mask_r_ub
    

    '''series expansion for part of multipliers in pdf for large r'''
    val1_temp[mask1] = (min_p[mask1] / max_p[mask1])**r[mask1]
    val1[mask1] = max_p[mask1] * (1 + 1/r[mask1] * val1_temp[mask1] + 
                                1/2 * 1/r[mask1] * (1/r[mask1] - 1) * val1_temp[mask1]**2)

    val2[mask2] = (p1[mask2]**r[mask2] + p2[mask2]**r[mask2])**(1/r[mask2])

    '''calculation for large r'''
    res[mask1] = (1 / (v1[mask1] * v2[mask1]) * 
                1 / (p1[mask1] * p2[mask1]) * 
                np.exp(-val1[mask1]) *
                (-1 + r[mask1] + val1[mask1]) * 
                val1[mask1] *
                val1_temp[mask1]  /
                (1 + 2 * val1_temp[mask1] + val1_temp[mask1]**2)
                )

    '''usual calculation in other cases'''
    temp2 = np.zeros_like(r, dtype=np.float64)
    temp2[mask2] = (min_p[mask2] / max_p[mask2])**r[mask2]

    res[mask2] = (1 / (v1[mask2] * v2[mask2]) * 
                1 / (p1[mask2] * p2[mask2]) * 
                np.exp(-val2[mask2]) *
                (-1 + r[mask2] + val2[mask2]) * 
                val2[mask2] *
                temp2[mask2] * 
                (1 + temp2[mask2])**(-2)
                )
    return res

@njit
def GumbelPDF2_jit(u, r, rotate):
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

    p1 = -np.log(v1)
    p2 = -np.log(v2)

    res = 0.0

    mask_r_ub = r > 100

    mask_eq = v1 == v2
    mask1 = ~mask_eq & mask_r_ub
    mask2 = ~mask_eq & ~mask_r_ub

    if mask_eq:
        temp_eq = 2**(1/r) * p1
        
        res = (1 / (v1**2) * np.exp(-temp_eq) * (-1 + r + temp_eq) * 
                        2**(1/r - 2) / p1)
        return res
    
    if mask1:
        max_p = np.maximum(p1, p2)
        min_p = np.minimum(p1, p2)
        val1_temp = (min_p / max_p)**r
        val1 = max_p * (1 + 1/r * val1_temp + 1/2 * 1/r * (1/r - 1) * val1_temp**2)

        val2 = (p1**r + p2**r)**(1/r)

        res = (1 / (v1 * v2) * 1 / (p1 * p2) * np.exp(-val1) *
                    (-1 + r + val1) * val1 * val1_temp  /
                    (1 + 2 * val1_temp + val1_temp**2)
                    )
        return res
    
    if mask2:
        max_p = np.maximum(p1, p2)
        min_p = np.minimum(p1, p2)
        val2 = (p1**r + p2**r)**(1/r)
        temp2 = (min_p / max_p)**r

        res = (1 / (v1 * v2) * 1 / (p1 * p2) * 
                    np.exp(-val2) * (-1 + r + val2) * 
                    val2 * temp2 * (1 + temp2)**(-2)
                    )
        return res
    return res


class GumbelCopula(ArchimedianCopula):
    def __init__(self, dim: int = 2, rotate: Literal[0, 90, 180, 270] = 0) -> None:
        super().__init__(dim, rotate)
        self.__name = 'Gumbel copula'
        self.__bounds = [(1.0001, np.inf)]

    @property
    def name(self):
        return self.__name

    @property
    def bounds(self):
        return self.__bounds
    
    @staticmethod
    def transform(r):
        return r * np.tanh(r) + 1.0001

    @staticmethod
    @njit
    def transform_jit(r):
        return r * np.tanh(r) + 1.0001

    @staticmethod
    def inv_transform(r):
        return r - 1

    @lru_cache
    def sp_generator(self):
        """
        Returns the symbolic generator function of the copula.

        Returns:
            sympy.Expr: The symbolic generator function.
        """
        t, r = sp.symbols('t r')
        
        result = (-sp.log(t))**r
        return result
    
    @lru_cache
    def sp_inverse_generator(self):
        """
        Returns the symbolic inverse generator function of the copula.

        Returns:
            sympy.Expr: The symbolic inverse generator function.
        """
        t, r = sp.symbols('t r')

        result = sp.exp(-t**(1/r))
        return result

    @staticmethod
    def psi(t, r):
        return np.exp(-t**(1 / r))
     
    @staticmethod
    def V(N, r):
        res = generate_levy_stable(alpha = 1/r, beta = 1, loc = 0, scale = np.cos(np.pi / (2 * r))**r, size = N)
        return res

    @lru_cache
    def np_pdf(self, numba_jit = False):
        if self.dim == 2:
            rotate = self.rotate

            if numba_jit == False:
                def bivariate_np_pdf(u, r):
                    return bivariateGumbelPDF(u, r, rotate)
                
                func = bivariate_np_pdf
            else:
                @njit
                def bivariate_np_pdf(u, r):
                    return GumbelPDF2_jit(u, r, rotate)
                
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
                    return np.log(bivariateGumbelPDF(u, r, rotate))
                
                func = bivariate_np_log_pdf
            else:
                @njit
                def bivariate_np_log_pdf(u, r):
                    return np.log(GumbelPDF2_jit(u, r, rotate))
                
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
        
        x0 = np.log(_u1)
        x1 = (-x0)**r
        x2 = x1 + (-np.log(_u0))**r
        x3 = x2**(r**(-1.0))
        return -x1*x3*np.exp(-x3)/(_u1*x0*x2)
    
    @staticmethod
    def h_inverse_unrotated(u1, u2, r):
        raise NotImplementedError("Not implemented for Gumbel copula")