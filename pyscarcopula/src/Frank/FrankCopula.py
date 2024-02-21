import sympy as sp
import numpy as np

from numba import njit
from pyscarcopula.src.ArchimedianCopula import ArchimedianCopula

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
        r0_index = np.argwhere(np.abs(r.ravel()) < 10**(-3))
        if len(r0_index) > 0 :
            for i in r0_index:
                r[i] = eps
            return r
        else:
            return r
    
    @property
    def sp_generator(self):
        return self.__sp_generator
    
    @property
    def sp_inverse_generator(self):
        return self.__sp_inverse_generator
