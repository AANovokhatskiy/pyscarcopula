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
        return r * np.tanh(r) + 0.0001
    
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
                # if i > 10000:
                #     u = np.random.uniform(0, 1)
                #     i = 1
                #     p = p0
                #     F = p
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
            if (u0 < 0 or u0 > 1) or (u1 < 0 or u1 > 1):
                print(u0, u1, r[k], u, v)
                raise ValueError('oppa')
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
        # M.Hofert, Sampling Archimedean copulas, 2008
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