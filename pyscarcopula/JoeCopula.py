import sympy as sp
import numpy as np

from typing import Literal

from numba import njit
from pyscarcopula.ArchimedianCopula import ArchimedianCopula


class JoeCopula(ArchimedianCopula):
    def __init__(self, dim: int = 2, rotate: Literal['0', '90', '180', '270'] = '0') -> None:
        super().__init__(dim, rotate)
        self.__sp_generator = - sp.log(1 - (1 - self.t)**self.r)
        self.__sp_inverse_generator = 1 - (1 - sp.exp(-self.t))**(1 / self.r)
        self.__name = 'Joe copula'

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