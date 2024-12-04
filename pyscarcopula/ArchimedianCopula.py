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
        u = sp.symbols('u0:%d'%(self.dim), positive = True)
        params = sum([self.sp_generator.subs([(self.t, x)]) for x in u])
        func = self.sp_inverse_generator.subs([(self.t, params)])
        return func
    
    @lru_cache
    def sp_pdf(self):
        '''Sympy expression of copula's pdf'''
        u = sp.symbols('u0:%d'%(self.dim), positive = True)
        params = [self.sp_generator.subs([(self.t, x)]) for x in u]

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

    @staticmethod
    def psi(t, r):
        return np.exp(-t)
     

    @staticmethod
    def V(N, r):
        if isinstance(r, (int, float)):
            r_arr = np.ones(N) * 1
        else:
            r_arr = np.ones_like(r)
        return r_arr
    
    def get_sample(self, N, r):
        # M.Hofert, Sampling Archimedean copulas, 2008
        pseudo_obs = np.zeros((N, self.dim))

        x = np.random.uniform(0, 1, size = (N, self.dim))
        
        V_data = self.V(N, r)

        for k in range(0, self.dim):
            pseudo_obs[:,k] = self.psi(-np.log(x[:,k]) / V_data, r)

        return pseudo_obs
    
    def log_likelihood(self, data, r):
        '''Log of likelihood function
        
        Parameters.
        1. data - dataset for calculation. Type - Numpy Array
        2. r - copula parameter. Type - Numpy Array. 
        '''
        return np.sum(np.log(self.pdf(data, r)))

    @staticmethod
    def calculate_dwt(method, T: int, latent_process_tr: int, seed: int = None, dt: float = None):
        '''Calculation of common random numbers (crn) with of T rows and latent_process_tr columns. Setting seed is also avilable'''
        if seed is None:
            dwt = np.random.normal(0 , 1 , size = (T, latent_process_tr) )
        else:
            rng = np.random.RandomState(seed)
            dwt = rng.normal(0 , 1 , size = (T, latent_process_tr) )
        if method.upper() in ['SCAR-M-OU', 'SCAR-P-OU', 'SCAR-P-LD']:
            if dt is None:
                dt = 1.0 / (T - 1)
            result = dwt * np.sqrt(dt)
        elif method.upper() in ['SCAR-P-DS', 'SCAR-M-DS']:
            result = dwt
        else:
            result = None
        return result

    def mlog_likelihood(self, alpha: np.array, data: np.array, latent_process_tr: int = 500, M_iterations: int = 5,
        method: str = 'MLE', seed: int = None, dwt: np.array = None, 
        print_path: bool = False, init_state = None, max_log_lik_debug = -100000) -> float:

        if method.upper() not in self.list_of_methods():
            raise ValueError(f'given method {method} is not avialable. avialable methods: {self.list_of_methods()}')

        if method.upper() == 'MLE':
            res = jit_mlog_likelihood_mle(alpha, data.T, self.np_pdf(), self.transform)
            return res

        T = len(data)

        if dwt is None:
            dwt = self.calculate_dwt(method, T, latent_process_tr, seed)
        else:
            if dwt.shape != (T, latent_process_tr) and method.upper() != 'MLE':
                raise ValueError(f"Common random numbers shape is not compatible to data: got {dwt.shape}, expected: {(T, latent_process_tr)}")

        if method.upper() == 'SCAR-P-OU':
            res = p_jit_mlog_likelihood_ou(alpha, data, dwt, latent_process_tr, print_path, self.np_pdf(), self.transform, init_state)
        elif method.upper() == 'SCAR-M-OU':
            res = m_jit_mlog_likelihood_ou(alpha, data, dwt, latent_process_tr, M_iterations, print_path, self.np_pdf(), self.transform, init_state, max_log_lik_debug)[0]
        elif method.upper() == 'SCAR-P-LD':
            res = p_jit_mlog_likelihood_ld(alpha, data, dwt, latent_process_tr, print_path, self.np_pdf(), self.transform, init_state)
        elif method.upper() == 'SCAR-P-DS':
            res = p_jit_mlog_likelihood_ds(alpha, data, dwt, latent_process_tr, print_path, self.np_pdf(), self.transform)
        elif method.upper() == 'SCAR-M-DS':
            res = m_jit_mlog_likelihood_ds(alpha, data, dwt, latent_process_tr, M_iterations, print_path, self.np_pdf(), self.transform)
        else:
            raise ValueError(f"Method {method} is not implemented. Available methods = {self.list_of_methods}")
        return res

    def fit(self, data: np.array, latent_process_tr: int = 500, M_iterations: int = 5, accuracy = 1e-5,
         method: str = 'MLE', alpha0: np.array = None, to_pobs = True, 
         seed: int = None, dwt: np.array = None, print_path: bool = False, init_state = None,
         max_log_lik_debug = -100000):

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

        bounds = None
        constr = None

        num_method = 'L-BFGS-B'

        if method.upper() in ['SCAR-P-OU', 'SCAR-M-OU']:
            bounds = Bounds([-10.0, 0.01, 0.01], [10.0, 15.0, 5.0])
            #constr = {'type': 'ineq', 'fun': lambda x: np.abs(x[0]) - x[2]**2 - 0.001}
        elif method.upper() in ['SCAR-P-DS', 'SCAR-M-DS']:
            bounds = Bounds([-5.0,-0.9999,0.0], [5.0, 0.9999, 0.9999])

        if method.upper() == 'MLE':
            num_method = 'L-BFGS-B'

        fit_data = data
        if to_pobs == True:
            fit_data = pobs(data)

        T = len(data)
        if dwt is None:
            dwt = self.calculate_dwt(method, T, latent_process_tr, seed)

        log_min = minimize(self.mlog_likelihood, alpha,
                                    args=(fit_data, latent_process_tr, M_iterations, method.upper(), 
                                          seed, dwt, print_path, init_state, max_log_lik_debug),
                                    method = num_method,
                                    bounds = bounds,
                                    #constraints = constr,
                                    options={'ftol': accuracy, 'eps': accuracy} )
        log_min.name = self.name
        log_min.fun = -log_min.fun
        log_min.method = method
        
        if method.upper() == 'MLE':
            log_min.x_transformed = self.transform(log_min.x)

        if method.upper() != 'MLE':
            log_min.latent_process_tr = latent_process_tr

        if method.upper() == 'SCAR-M-OU' or method.upper() == 'SCAR-M-DS':
            log_min.m_iterations = M_iterations

        return log_min
