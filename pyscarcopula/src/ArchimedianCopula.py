import sympy as sp
import numpy as np
from numba import njit, jit
import math
from typing import Callable 
from functools import lru_cache
from scipy.optimize import minimize, Bounds

from pyscarcopula.sampler.sampler_ou import p_jit_mlog_likelihood_ou
from pyscarcopula.sampler.sampler_ou import m_jit_mlog_likelihood_ou
from pyscarcopula.sampler.sampler_ld import p_jit_mlog_likelihood_ld
from pyscarcopula.sampler.sampler_ds import p_jit_mlog_likelihood_ds
from pyscarcopula.sampler.sampler_ds import m_jit_mlog_likelihood_ds
from pyscarcopula.sampler.mle import jit_mlog_likelihood_mle

from pyscarcopula.auxiliary.funcs import pobs

class ArchimedianCopula:
    def __init__(self, dim: int) -> None:
        self.__dim = dim
        self.__t, self.__r = sp.symbols('t r')
        '''independent copula by default'''
        self.__sp_generator = - sp.log(self.t)
        self.__sp_inverse_generator = sp.exp(-self.t)
        self.__name = 'Independent copula'

    @property
    def name(self):
        return self.__name

    @property
    def t(self):
        return self.__t
    
    @property
    def r(self):
        return self.__r
    
    @property
    def sp_generator(self):
        '''generator function of copula'''
        return self.__sp_generator
    
    @property
    def sp_inverse_generator(self):
        '''inverse generator function of copula'''
        return self.__sp_inverse_generator
    
    @property
    def dim(self):
        return self.__dim

    @staticmethod
    def list_of_methods():
        l = ['MLE', 'SCAR-M-OU', 'SCAR-P-OU', 'SCAR-P-LD', 'SCAR-P-DS', 'SCAR-M-DS']
        return l

    @lru_cache
    def sp_cdf(self):
        '''Sympy expression of copula's cdf'''
        u = sp.symbols('u0:%d'%(self.dim))
        params = sum([self.sp_generator.subs([(self.t, x)]) for x in u])
        func = self.sp_inverse_generator.subs([(self.t, params)])
        return func
    
    @lru_cache
    def sp_pdf(self):
        '''Sympy expression of copula's pdf'''
        u = sp.symbols('u0:%d'%(self.dim))
        params = [self.sp_generator.subs([(self.t, x)]) for x in u]
        #func = self.sp_inverse_generator.subs([(self.t, sum(params))])
        diff_inverse_generator = sp.together(self.sp_inverse_generator.diff((self.t, self.dim)))
        diff_generator = sp.together(self.sp_generator.diff(self.t, 1))
        func = diff_inverse_generator.subs([(self.t, sum(params))])
        for x in u:
            func = func * diff_generator.subs([(self.t, x)])
        return sp.powsimp(func)
        
    @lru_cache
    def np_pdf(self):
        '''Numpy pdf function from sympy expression'''
        expr = self.sp_pdf()
        u = sp.symbols('u0:%d'%(self.dim))
        r = sp.symbols('r')
        func = sp.lambdify((u, r), expr, modules = 'numpy', cse = True)
        return njit(func)
   
    @lru_cache
    def np_cdf(self):
        '''Numpy cdf function from sympy expression'''
        expr = self.sp_cdf()
        u = sp.symbols('u0:%d'%(self.dim))
        r = sp.symbols('r')
        func = sp.lambdify((u, r), expr, modules = 'numpy', cse = True)
        return njit(func)

    def pdf(self, data: np.array, r: np.array):
        '''Numpy pdf function'''
        func = self.np_pdf()
        res = func(data.T, r)
        return res
    
    def cdf(self, data: np.array, r: np.array):
        '''Numpy cdf function'''
        func = self.np_cdf()
        res = func(data.T, r)
        return res
    
    @staticmethod
    def transform(r):
        '''Function that transfroms interval (-inf, inf) to avialable range for copula parameter.
        Function used for solving non-constrained minimization problem.
        '''
        return r

    def log_likelihood(self, data, r):
        '''Log of likelihood function
        
        Parameters.
        1. data - dataset for calculation. Type - Numpy Array
        2. r - copula parameter. Type - Numpy Array. 
        '''
        return np.sum(np.log(self.pdf(data, r)))

    @staticmethod
    def calculate_dwt(T: int, latent_process_tr: int, seed: int = None):
        '''Calculation of common random numbers (crn) with of T rows and latent_process_tr columns. Setting seed is also avilable'''
        shape = (T, latent_process_tr)
        if seed is None:
            dwt = np.random.normal(0 , 1 , size = (T, latent_process_tr) )
        else:
            rng = np.random.RandomState(seed)
            dwt = rng.normal(0 , 1 , size = (T, latent_process_tr) )
        return dwt * np.sqrt(1/T)

    def mlog_likelihood(self, alpha: np.array, data: np.array, latent_process_tr: int = 500, m_iters: int = 5,
        method: str = 'MLE', seed: int = None, dwt: np.array = None, 
        print_path: bool = False, init_state = None, max_log_lik_debug: int = -1000) -> float:

        if method.upper() not in self.list_of_methods():
            raise ValueError(f'given method {method} is not avialable. avialable methods: {self.list_of_methods()}')

        if method.upper() == 'MLE':
            res = jit_mlog_likelihood_mle(alpha, data.T, self.np_pdf(), self.transform)
            return res

        T = len(data)
        dt = 1 / T
        if dwt is None:
            dwt = self.calculate_dwt(T, latent_process_tr, seed)
        else:
            if dwt.shape != (T, latent_process_tr):
                raise ValueError(f"Common random numbers shape is not compatible to data: got {dwt.shape}, expected: {(T, latent_process_tr)}")

        if method.upper() == 'SCAR-P-OU':
            res = p_jit_mlog_likelihood_ou(alpha, data, dwt, latent_process_tr, print_path, self.np_pdf(), self.transform, init_state)
        elif method.upper() == 'SCAR-M-OU':
            res = m_jit_mlog_likelihood_ou(alpha, data, dwt, latent_process_tr, m_iters, print_path, self.np_pdf(), self.transform, init_state, max_log_lik_debug)
        elif method.upper() == 'SCAR-P-LD':
            res = p_jit_mlog_likelihood_ld(alpha, data, dwt, latent_process_tr, print_path, self.np_pdf(), self.transform, init_state)
        # if method == 'SCAR-P-DS':
        #     res = p_jit_mlog_likelihood_ds(alpha, data, dwt, latent_process_tr, print_path, self.np_pdf(), self.transform)
        # if method == 'SCAR-M-DS':
        #     res = m_jit_mlog_likelihood_ds(alpha, data, dwt, latent_process_tr, m_iters, print_path, self.np_pdf(), self.transform)
        return res

    def fit(self, data: np.array, latent_process_tr: int = 500, m_iters: int = 5, accuracy = 1e-5,
         method: str = 'MLE', alpha0: np.array = None, to_pobs = True, 
         seed: int = None, dwt: np.array = None, print_path: bool = False, init_state = None,
         max_log_lik_debug = -1000):

        if method.upper() not in self.list_of_methods():
            raise ValueError(f'given method {method} is not avialable. avialable methods: {self.list_of_methods()}')
    
        default_alpha0 = dict()
        default_alpha0['MLE'] = 1/2
        default_alpha0['SCAR-P-DS'] = np.array([0.05, 0.95, 0.05])
        default_alpha0['SCAR-M-DS'] = np.array([0.05, 0.95, 0.05])
        
        default_alpha0['SCAR-M-OU'] = np.array([1.0, 0.5, 0.05])
        default_alpha0['SCAR-P-OU'] = np.array([1.0, 0.5, 0.05])
        default_alpha0['SCAR-P-LD'] = np.array([1.0, 0.5, 0.05])

        if alpha0 is None:
            alpha = default_alpha0[method.upper()]
        else:
            alpha = alpha0

        if method.upper() == 'SCAR-M-OU':
            bounds = Bounds([-100.,-5,-5],[100., 5, 5])
            constr = {'type': 'ineq', 'fun': lambda x: np.abs(x[1]) - x[2]**2 - 0.001}
        else:
            bounds = None
            constr = None

        fit_data = data
        if to_pobs == True:
            fit_data = pobs(data)

        T = len(data)
        dt = 1 / T
        if dwt is None:
            dwt = self.calculate_dwt(T, latent_process_tr, seed)

        log_min = minimize(self.mlog_likelihood, alpha,
                                    args=(fit_data, latent_process_tr, m_iters, method.upper(), 
                                          seed, dwt, print_path, init_state, max_log_lik_debug),
                                    method='SLSQP',
                                    bounds = bounds,
                                    #constraints = constr,
                                    options={'ftol': accuracy} )
        log_min.name = self.name
        log_min.fun = -log_min.fun
        log_min.method = method
        if method.upper() == 'MLE':
            log_min.x_transformed = self.transform(log_min.x)
        return log_min
