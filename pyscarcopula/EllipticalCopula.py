from scipy.stats import multivariate_normal, norm, multivariate_t, chi2, t, kendalltau, invgamma
from pyscarcopula.ArchimedianCopula import ArchimedianCopula, pobs

import numpy as np
from scipy.optimize import minimize
from typing import Literal
from numba import njit

from scipy.special import ndtri
import numpy as np

from numba.extending import get_cython_function_address
from ctypes import CFUNCTYPE, c_double, c_int

addr = get_cython_function_address("scipy.special.cython_special", "ndtri")
functype = CFUNCTYPE(c_double, c_double)
ndtri_numba = functype(addr)

def bivariateGaussianPDF(u, r):
        u1 = u[0]
        u2 = u[1]

        x1, x2 = norm.ppf(u1), norm.ppf(u2)
        res = 1 / np.sqrt(1 - r**2) * np.exp(- 0.5 * (r**2 * (x1**2 + x2**2) - 2 * r * x1 * x2) / (1 - r**2))
        return res

def bivariateGaussianLogPDF(u, r):
        u1 = u[0]
        u2 = u[1]

        x1, x2 = norm.ppf(u1), norm.ppf(u2)
        res = -0.5 * np.log(1 - r**2) - 0.5 * (r**2 * (x1**2 + x2**2) - 2 * r * x1 * x2) / (1 - r**2)
        return res

@njit
def GaussianPDF2_jit(u, r):
        u1 = u[0]
        u2 = u[1]

        x1, x2 = ndtri_numba(u1), ndtri_numba(u2)
        res = 1 / np.sqrt(1 - r**2) * np.exp(- 0.5 * (r**2 * (x1**2 + x2**2) - 2 * r * x1 * x2) / (1 - r**2))
        return res

@njit
def GaussianLogPDF2_jit(u, r):
        u1 = u[0]
        u2 = u[1]

        x1, x2 = ndtri_numba(u1), ndtri_numba(u2)
        res = -0.5 * np.log(1 - r**2) - 0.5 * (r**2 * (x1**2 + x2**2) - 2 * r * x1 * x2) / (1 - r**2)
        return res

class BivariateGaussianCopula(ArchimedianCopula):
    def __init__(self, dim: int = 2, rotate: Literal[0] = 0) -> None:
        super().__init__(dim, rotate)
        self.__rotatable = False
        self.__name = 'bivariate Gaussian copula'
        if rotate != 0:
            raise ValueError("Rotation is not supported for bivariate Gaussian copula.")
        self.__bounds = [(-0.999, 0.999)]

    @property
    def rotatable(self):
        return self.__rotatable

    @property
    def name(self):
        return self.__name

    @property
    def bounds(self):
        return self.__bounds
        
    @staticmethod
    def transform(r):
        return 0.9999 * np.tanh(r/4)

    @staticmethod
    @njit
    def transform_jit(r):
        return 0.9999 * np.tanh(r/4)

    @staticmethod
    def inv_transform(r):
        return 4 * np.arctanh(r/0.9999)

    @staticmethod
    def h_unrotated(u1, u2, theta):
        eps = 1e-6
        _u1 = np.clip(u1, eps, 1 - eps)
        _u2 = np.clip(u2, eps, 1 - eps)

        return norm.cdf((norm.ppf(_u1) - theta * norm.ppf(_u2)) / np.sqrt(1 - theta**2))

    @staticmethod
    def h_inverse_unrotated(u1, u2, theta):
        eps = 1e-6
        _u1 = np.clip(u1, eps, 1 - eps)
        _u2 = np.clip(u2, eps, 1 - eps)

        return norm.cdf(norm.ppf(_u1) * np.sqrt(1 - theta**2) + theta * norm.ppf(_u2))

    def np_pdf(self, numba_jit = False):
        if self.dim == 2:
            if numba_jit == False:
                return bivariateGaussianPDF
            else:
                return GaussianPDF2_jit
        else:
            raise NotImplementedError("For multivariate case with dim > 2 use GaussianCopula class")

    def np_log_pdf(self, numba_jit = False):
        if self.dim == 2:
            if numba_jit == False:
                return bivariateGaussianLogPDF
            else:
                return GaussianLogPDF2_jit
        else:
            raise NotImplementedError("For multivariate case with dim > 2 use GaussianCopula class")

    def sp_generator(self):
        raise NotImplementedError("Not implemented for elliptical copula")
    
    def sp_inverse_generator(self):
        raise NotImplementedError("Not implemented for elliptical copula")
    
    def sp_cdf_from_generator(self):
        raise NotImplementedError("Not implemented for elliptical copula")
    
    def sp_cdf(self):
        raise NotImplementedError("Not implemented for elliptical copula")
     
    def sp_pdf_from_generator(self):
        raise NotImplementedError("Not implemented for elliptical copula") 
    
    def sp_pdf(self):
        raise NotImplementedError("Not implemented for elliptical copula")
    
    def np_cdf(self):
        raise NotImplementedError("Not implemented for elliptical copula")
    
    def cdf(self):
        raise NotImplementedError("Not implemented for elliptical copula")
    
    def V(self):
        raise NotImplementedError("Not implemented for elliptical copula")
    
    def get_sample(self, N, r):
        """
        Generates samples from the copula. Based on M.Hofert, Sampling Archimedean copulas, 2008

        Args:
            N (int): The number of samples to generate.
            r (np.ndarray or float): The copula parameter (can be an array or a scalar).

        Returns:
            np.ndarray: An array of shape (N, dim) containing the generated samples.
        """  
        z1 = np.random.normal(0, 1, size = N)
        z2 = np.random.normal(0, 1, size = N)
        x1 = z1
        x2 = r * z1 + np.sqrt(1 - r**2) * z2
        
        u = np.column_stack((norm.cdf(x1), norm.cdf(x2)))
        return u


def GaussianCopulaSample(size, cov):
    n = len(cov)
    mean = np.zeros(n)
    x = multivariate_normal.rvs(mean = mean, cov = cov, size = size)
    u = norm.cdf(x)
    return u

def GaussianCopulaFit(u):
    x = norm.ppf(u)
    cov = np.corrcoef(x.T)
    return cov

class GaussianCopula():
    def __init__(self):
        self.cov = None
    
    def fit(self, data, method = 'mle', to_pobs = False, **kwargs):
        u = data
        if to_pobs == True:
            u = pobs(data)

        if method.upper() == 'MLE':
            self.cov = GaussianCopulaFit(u)
        else:
            raise NotImplementedError(f"method {method} is not implemented")
        
        return self.cov
    
    def get_sample(self, size):
        return GaussianCopulaSample(size, self.cov)

    def get_predict(self, size):
        return self.get_sample(size)

    def mlog_likelihood(self, u, method = 'mle'):
        if method.upper() == 'MLE':
            if self.cov is None:
                raise ValueError(f"Unknown copula parameters. Fit copula first")
                
            dim = u.shape[1]

            x = norm.ppf(u)

            logpdf_dim_d = multivariate_normal.logpdf(x, mean = np.zeros(dim), cov = self.cov)
            logpdf_dim_1 = np.sum(norm.logpdf(x), axis = 1)
            logpdf = logpdf_dim_d - logpdf_dim_1
            result = -np.sum(logpdf)
        else:
            raise NotImplementedError(f"method {method} is not implemented")
        return result

    def sp_cdf(self):
        raise NotImplementedError("Not implemented for elliptical copula")

    def sp_pdf(self):
        raise NotImplementedError("Not implemented for elliptical copula")   
     
def StudentCopulaSample(size, shape, df):
    n = len(shape)
    mean = np.zeros(n)
    x = multivariate_t.rvs(loc = mean, shape = shape, df = df, size = size)
    u = t.cdf(x, df = df)
    return u

def FillSymmetric(dim, values):
    mat = np.zeros((dim, dim))
    z = np.ones(dim)

    i1, i2 = np.triu_indices(dim, 1)
    mat[i1, i2] = values

    mat += mat.T
    mat += np.diag(z)

    return mat

def IsPositiveDefinite(x):
    return np.all(np.linalg.eigvals(x) > 0)


def KendtallTauMatrix(x):
    dim = x.shape[1]
    P = np.zeros((dim, dim))
    for k in range(0, dim):
        for j in range(0, k):
            tau, pval = kendalltau(x[:,j], x[:,k])
            P[j][k] = np.sin(np.pi / 2 * tau)
    P += P.T
    P += np.diag(np.ones(dim))
    return P

def StudentCopulaMLogPDF(z, u):
    df = z[0]
    dim = u.shape[1]

    x = t.ppf(u, df = df)
    shape = KendtallTauMatrix(x)

    if IsPositiveDefinite(shape) == False:
        shape = GaussianCopulaFit(u)

    loc = np.zeros(dim)

    logpdf_dim_d = multivariate_t.logpdf(x, loc = loc, shape = shape, df = df)
    logpdf_dim_1 = np.sum(t.logpdf(x, df = df), axis = 1)
    logpdf = logpdf_dim_d - logpdf_dim_1
    result = -np.sum(logpdf)
    return result

def StudentCopulaFit(u):

    dim = u.shape[1]
    z0 = np.array([dim])

    fit_result = minimize(StudentCopulaMLogPDF, z0, 
                          args = (u, ),
                          bounds = [(0.001, np.inf)],
                          options = {'gtol': 1e-2, 'eps': 1e-4}
                          )
    
    df = fit_result.x
    x = t.ppf(u, df = df)
    shape = KendtallTauMatrix(x)

    return shape, df   

class StudentCopula():
    def __init__(self):
        self.shape = None
        self.df = None
    
    def fit(self, data, method = 'mle', to_pobs = False, **kwargs):
        u = data
        if to_pobs == True:
            u = pobs(data)

        if method.upper() == 'MLE':
            fit_result = StudentCopulaFit(u)
            self.shape = fit_result[0]
            self.df = fit_result[1][0]
        else:
            raise NotImplementedError(f"method {method} is not implemented")
        
        return self.shape, self.df
    
    def get_sample(self, size):
        return StudentCopulaSample(size, self.shape, self.df)

    def get_predict(self, size):
        return self.get_sample(size)

    def mlog_likelihood(self, u, method = 'mle'):
        if method.upper() == 'MLE':
            if self.shape is None:
                raise ValueError(f"Unknown copula parameters. Fit copula first")
            
            df = self.df
            shape = self.shape

            dim = u.shape[1]

            x = t.ppf(u, df = df)
            loc = np.zeros(dim)

            logpdf_dim_d = multivariate_t.logpdf(x, loc = loc, shape = shape, df = df)
            logpdf_dim_1 = np.sum(t.logpdf(x, df = df), axis = 1)
            logpdf = logpdf_dim_d - logpdf_dim_1
            result = -np.sum(logpdf)
        else:
            raise NotImplementedError(f"method {method} is not implemented")
        return result
    
    def sp_cdf(self):
        raise NotImplementedError("Not implemented for elliptical copula")

    def sp_pdf(self):
        raise NotImplementedError("Not implemented for elliptical copula")  