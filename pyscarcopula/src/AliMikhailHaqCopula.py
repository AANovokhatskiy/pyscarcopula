import sympy as sp
import numpy as np

from numba import njit
from pyscarcopula.src.ArchimedianCopula import ArchimedianCopula

class AliMikhailHaqCopula(ArchimedianCopula):
    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.__sp_generator = sp.log((1 - self.r * (1 - self.t)) / self.t)
        self.__sp_inverse_generator = (1 - self.r) / (sp.exp(self.t) - self.r)
        self.__name = 'Ali-Mikhail-Haq copula'

    @property
    def name(self):
        return self.__name

    @staticmethod
    @njit
    def transform(r):
        return np.tanh(r)
    
    @property
    def sp_generator(self):
        return self.__sp_generator
    
    @property
    def sp_inverse_generator(self):
        return self.__sp_inverse_generator
