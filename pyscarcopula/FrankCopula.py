import sympy as sp
import numpy as np

from numba import njit
from pyscarcopula.ArchimedianCopula import ArchimedianCopula

class FrankCopula(ArchimedianCopula):
    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.__sp_generator = -sp.log( (sp.exp(-self.r * self.t) - 1) / (sp.exp(-self.r) - 1) )
        self.__sp_inverse_generator = - 1 / self.r * sp.log(1 + sp.exp(-self.t) * (sp.exp(-self.r) - 1) )
        self.__name = 'Frank copula'

    @property
    def name(self):
        return self.__name
        
    @staticmethod
    @njit
    def transform(r):
        eps = 0.001
        #return np.clip(r + eps * (1 - np.sign(r)**2), -40, 40)
        #return np.minimum(np.maximum(r + eps * (1 - np.sign(r)**2), -40), 40)
        return r**2 + eps
    
    @property
    def sp_generator(self):
        return self.__sp_generator
    
    @property
    def sp_inverse_generator(self):
        return self.__sp_inverse_generator

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
                if i > 1000:
                    u = np.random.uniform(0, 1)
                    i = 1
                    p = p0
                    F = p
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