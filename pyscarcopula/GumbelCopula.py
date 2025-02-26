import sympy as sp
import numpy as np

from typing import Literal
from functools import lru_cache

from numba import njit, jit
from pyscarcopula.ArchimedianCopula import ArchimedianCopula

@njit
def generate_levy_stable(alpha, beta, loc = 0, scale = 1, size = 1):
    # Weron, R. (1996). On the Chambers-Mallows-Stuck method for simulating skewed stable random variables
    # Borak et. al. (2008), Stable Distributions

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

# @jit(nopython = True, cache = True)
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


class GumbelCopula(ArchimedianCopula):
    def __init__(self, dim: int = 2, rotate: Literal[0, 90, 180, 270] = 0) -> None:
        super().__init__(dim, rotate)
        self.__sp_generator = (-sp.log(self.t))**self.r
        self.__sp_inverse_generator = sp.exp(-self.t**(1/self.r))
        self.__name = 'Gumbel copula'

    @property
    def name(self):
        return self.__name
        
    @staticmethod
    @njit
    def transform(r):
        return r * np.tanh(r) + 1.0001
    
    @property
    def sp_generator(self):
        return self.__sp_generator
    
    @property
    def sp_inverse_generator(self):
        return self.__sp_inverse_generator

    @staticmethod
    def psi(t, r):
        return np.exp(-t**(1 / r))
     
    @staticmethod
    def V(N, r):
        res = generate_levy_stable(alpha = 1/r, beta = 1, loc = 0, scale = np.cos(np.pi / (2 * r))**r, size = N)
        return res

    @lru_cache
    def np_pdf(self):
        if self.dim == 2:
            rotate = self.rotate

            # @njit(cache = True)
            def bivariate_np_pdf(u, r):
                return bivariateGumbelPDF(u, r, rotate)
            
            func = bivariate_np_pdf
            return func
        else:
            '''Numpy pdf function from sympy expression'''
            expr = self.sp_pdf()
            u = sp.symbols('u0:%d'%(self.dim))
            r = sp.symbols('r')
            func = sp.lambdify((u, r), expr, modules = 'numpy', cse = True)
        # return njit(func)
        return func