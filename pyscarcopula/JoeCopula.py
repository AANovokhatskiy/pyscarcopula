import sympy as sp
import numpy as np

from typing import Literal
from functools import lru_cache

from numba import njit
from pyscarcopula.ArchimedianCopula import ArchimedianCopula


class JoeCopula(ArchimedianCopula):
    def __init__(self, dim: int = 2, rotate: Literal[0, 90, 180, 270] = 0) -> None:
        super().__init__(dim, rotate)
        self.__name = 'Joe copula'

    @property
    def name(self):
        return self.__name
          
    @staticmethod
    def transform(r):
        return r * np.tanh(r) + 1.0001
    
    @lru_cache
    def sp_generator(self):
        """
        Returns the symbolic generator function of the copula.

        Returns:
            sympy.Expr: The symbolic generator function.
        """
        t, r = sp.symbols('t r')
        
        result = - sp.log(1 - (1 - t)**r)
        return result
    
    @lru_cache
    def sp_inverse_generator(self):
        """
        Returns the symbolic inverse generator function of the copula.

        Returns:
            sympy.Expr: The symbolic inverse generator function.
        """
        t, r = sp.symbols('t r')

        result = 1 - (1 - sp.exp(-t))**(1 / r)
        return result

    @staticmethod
    def psi(t, r):
        return 1 - (1 - np.exp(-t))**(1/r)
     
    @staticmethod
    @njit
    def V_auxiliary(N, r):
        res = np.zeros(N)
        for k in range(0, N):
            p0 = 1/r[k]
            u = np.random.uniform(0, 1)
            i = 1
            p = p0
            F = p
            while u > F:
                multiplier = (-1) * (p0 - (i + 1) + 1) / (i + 1)
                p = multiplier * p
                F = F + p
                i = i + 1
                # if i > 10000:
                #     u = np.random.uniform(0, 1)
                #     i = 1
                #     p = p0
                #     F = p
            res[k] = i
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
    
    @staticmethod
    def h(u0, u1, r):
        eps = 1e-6
        _u0 = np.clip(u0, eps, 1 - eps)
        _u1 = np.clip(u1, eps, 1 - eps)

        x0 = 1 - _u1
        x1 = x0**r
        x2 = 1 - (1 - _u0)**r
        x3 = -x2 * (1 - x1) + 1
        return x1 * x2 * x3**(r**(-1.0)) / (x0 * x3)

    @staticmethod
    def h_inverse(u1, u2, r):
        pass