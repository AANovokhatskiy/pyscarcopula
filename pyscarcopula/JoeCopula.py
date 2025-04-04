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
        return self.V_auxiliary(N, r)
    
    @staticmethod
    def h_unrotated(u, v, r):
        eps = 1e-6
        _u = np.clip(u, eps, 1 - eps)
        _v = np.clip(v, eps, 1 - eps)

        x1 = (1 - _u)**r
        x2 = (1 - _v)**r
        x3 = -1 + x1
        return -(x3 * (x1 - x3 * x2)**(-1 + 1/r) * x2 / (1 - _v))

    @staticmethod
    def h_inverse_unrotated(u1, u2, r):
        raise NotImplementedError("Not implemented for Joe copula")