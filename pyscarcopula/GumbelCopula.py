import sympy as sp
import numpy as np

from numba import njit
from pyscarcopula.ArchimedianCopula import ArchimedianCopula
#from scipy.stats import levy_stable
from pyscarcopula.marginal.stable import generate_levy_stable

class GumbelCopula(ArchimedianCopula):
    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.__sp_generator = (-sp.log(self.t))**self.r
        self.__sp_inverse_generator = sp.exp(-self.t**(1/self.r))
        self.__name = 'Gumbel copula'

    @property
    def name(self):
        return self.__name
        
    @staticmethod
    @njit
    def transform(r):
        return np.minimum(1 + r**2, 40)
        #return 1 + np.exp(r)
    
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
        #res = levy_stable.rvs(alpha = 1/r, beta = 1, loc = 0, scale = np.cos(np.pi / (2 * r))**r, size = N)
        res = generate_levy_stable(alpha = 1/r, beta = 1, loc = 0, scale = np.cos(np.pi / (2 * r))**r, size = N)
        return res