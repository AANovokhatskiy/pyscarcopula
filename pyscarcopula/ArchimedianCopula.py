import sympy as sp
import numpy as np
from numba import njit, jit
import math
from typing import Callable, Literal
import warnings

from functools import lru_cache
from scipy.optimize import minimize, Bounds

from pyscarcopula.sampler.sampler_ou import p_jit_mlog_likelihood_ou
from pyscarcopula.sampler.sampler_ou import m_jit_mlog_likelihood_ou
from pyscarcopula.sampler.sampler_ou import stationary_state_ou

from pyscarcopula.sampler.sampler_ld import p_jit_mlog_likelihood_ld
from pyscarcopula.sampler.sampler_ld import stationary_state_ld

from pyscarcopula.sampler.sampler_ds import p_jit_mlog_likelihood_ds
from pyscarcopula.sampler.sampler_ds import m_jit_mlog_likelihood_ds

from pyscarcopula.sampler.mle import jit_mlog_likelihood_mle

from pyscarcopula.auxiliary.funcs import pobs


class ArchimedianCopula:
    def __init__(self, dim: int = 2, rotate: Literal[0, 90, 180, 270] = 0) -> None:
        self.__dim = dim
        self.__rotate = rotate
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

    @property
    def rotate(self):
        return self.__rotate
    
    @staticmethod
    def list_of_methods():
        l = ['MLE', 'SCAR-M-OU', 'SCAR-P-OU', 'SCAR-P-LD', 'SCAR-P-DS', 'SCAR-M-DS', 'SCAR-S-OU']
        return l

    @lru_cache
    def sp_cdf_from_generator(self):
        '''Sympy expression of copula's cdf'''
        u = sp.symbols('u0:%d'%(self.dim), positive = True)
        u = list(u)

        if self.dim != 2 and self.rotate != 0:
            warnings.warn("Rotation is implemented only for dim = 2 and this parameter will be ignored")

        params = sum([self.sp_generator.subs([(self.t, x)]) for x in u])
        func = self.sp_inverse_generator.subs([(self.t, params)])

        if self.dim == 2 and self.rotate == 90:
            func = func.subs(u[0], 1 - u[0])
            func = u[1] - func
        elif self.dim == 2 and self.rotate == 180:
            func = func.subs(u[0], 1 - u[0])
            func = func.subs(u[1], 1 - u[1])
            func = u[0] + u[1] - 1 + func
        elif self.dim == 2 and self.rotate == 270:
            func = func.subs(u[1], 1 - u[1])
            func = u[0] - func
        
        return func

    @lru_cache
    def sp_cdf(self):
        return self.sp_cdf_from_generator()

    @lru_cache
    def sp_pdf_from_generator(self):
        '''Sympy expression of copula's pdf'''
        u = sp.symbols('u0:%d'%(self.dim), positive = True)
        params = [self.sp_generator.subs([(self.t, x)]) for x in u]

        if self.dim != 2 and self.rotate != 0:
            warnings.warn("Rotation is implemented only for dim = 2 and this parameter will be ignored")

        diff_inverse_generator = sp.together(self.sp_inverse_generator.diff((self.t, self.dim)))
        diff_generator = sp.together(self.sp_generator.diff(self.t, 1))

        func = diff_inverse_generator.subs([(self.t, sum(params))])

        for x in u:
            func = func * diff_generator.subs([(self.t, x)])
        
        func = sp.powsimp(func)

        if self.dim == 2 and self.rotate == 90:
            func = func.subs(u[0], 1 - u[0])
        elif self.dim == 2 and self.rotate == 180:
            func = func.subs(u[0], 1 - u[0])
            func = func.subs(u[1], 1 - u[1])
        elif self.dim == 2 and self.rotate == 270:
            func = func.subs(u[1], 1 - u[1])
    
        return func
    
    @lru_cache
    def sp_pdf(self):
        return self.sp_pdf_from_generator()
        
    @lru_cache
    def np_pdf(self):
        '''Numpy pdf function from sympy expression'''
        expr = self.sp_pdf()
        u = sp.symbols('u0:%d'%(self.dim))
        r = sp.symbols('r')
        func = sp.lambdify((u, r), expr, modules = 'numpy', cse = True)
        return func #njit(func)
   
    @lru_cache
    def np_cdf(self):
        '''Numpy cdf function from sympy expression'''
        expr = self.sp_cdf()
        u = sp.symbols('u0:%d'%(self.dim))
        r = sp.symbols('r')
        func = sp.lambdify((u, r), expr, modules = 'numpy', cse = True)
        return func #njit(func)

    def _broadcasting(self, u, r):
        """Broadcasts u and r to ensure compatible shapes."""
        u = np.asarray(u)

        if u.ndim == 1:
            u = u[np.newaxis, :]

        r = np.asarray(r)

        if r.ndim == 1 and len(r) == 1:
            r = r.item()
            r = np.asarray(r)

        if r.ndim == 0:
            r = np.full(u.shape[0], r)

        if u.shape[0] == 1 and r.ndim == 1:
            u = np.full((r.shape[0], u.shape[1]), u[0])

        if len(r) != u.shape[0] and not (u.shape[0] == 1 and r.ndim == 1):
            raise ValueError("The length of r must match the number of rows in u or be compatible for broadcasting.")

        return u, r

    def pdf(self, u: np.array, r: np.array):
        '''Numpy pdf function'''
        func = self.np_pdf()

        u, r = self._broadcasting(u, r)

        res = func(u.T, r)
        return res
    
    def cdf(self, u, r):
        '''Numpy cdf function'''
        func = self.np_cdf()

        u, r = self._broadcasting(u, r)
        
        ''' Fréchet-Hoeffding boundary'''
        ub = np.min(u, axis = 1)
        lb = np.maximum(1 - self.dim + np.sum(u, axis = 1), 0)
        
        res = func(u.T, r)
        res = np.clip(res, lb, ub)

        if res.size == 1:
            res = res.item()
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
        u = np.zeros((N, self.dim))

        x = np.random.uniform(0, 1, size = (N, self.dim))
        
        V_data = np.clip(self.V(N, r), 1e-20, 1e+20)
    
        for k in range(0, self.dim):
            u[:,k] = self.psi(-np.log(x[:,k]) / V_data, r)

        if self.dim == 2 and self.rotate == 90:
            u[:,0] = 1 - u[:,0]
        elif self.dim == 2 and self.rotate == 180:
            u[:,0] = 1 - u[:,0]
            u[:,1] = 1 - u[:,1]
        elif self.dim == 2 and self.rotate == 270:
            u[:,1] = 1 - u[:,1]
            
        return u
    
    def log_likelihood(self, u, r):
        '''Log of likelihood function
        
        Parameters.
        1. data - dataset for calculation. Type - Numpy Array
        2. r - copula parameter. Type - Numpy Array. 
        '''
        u, r = self._broadcasting(u, r)

        return np.sum(np.log(self.pdf(u, r)))

    @staticmethod
    def calculate_dwt(method: Literal['mle', 'scar-p-ou', 'scar-m-ou', 'scar-p-ld'],
                      T: int, latent_process_tr: int, seed: int = None, dt: float = None):
        '''Calculation of common random numbers (crn) with of T rows and latent_process_tr columns. Setting seed is also avilable'''

        if method.upper() in ['MLE']:
            return
    
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

    def mlog_likelihood(self, 
                        alpha: np.array, 
                        u: np.array, 
                        method: Literal['mle', 'scar-p-ou', 'scar-m-ou', 'scar-p-ld'] = 'mle',
                        latent_process_tr: int = 500, 
                        M_iterations: int = 5,
                        seed: int = None, 
                        dwt: np.array = None, 
                        stationary: bool = False,
                        print_path: bool = False, 
                        init_state = None) -> float:
        '''Calculation of -log_likelihood'''

        if method.upper() not in self.list_of_methods():
            raise ValueError(f'given method {method} is not avialable. avialable methods: {self.list_of_methods()}')

        alpha = np.asarray(alpha)
        u = np.asarray(u)
        
        if method.upper() == 'MLE':
            u, alpha = self._broadcasting(u, alpha)

            res = jit_mlog_likelihood_mle(alpha, u.T, self.np_pdf(), self.transform)
            return res
        
        T = len(u)
        if dwt is None:
            dwt = self.calculate_dwt(method, T, latent_process_tr, seed)

        if stationary == True and init_state is None:
            _latent_process_tr = dwt.shape[1]
            _seed = None
            if seed is not None:
                _seed = seed * 2

            if method.upper() in ['SCAR-P-OU', 'SCAR-M-OU']:
                init_state = stationary_state_ou(alpha, _latent_process_tr, _seed)
            elif method.upper() in ['SCAR-P-LD']:
                init_state = stationary_state_ld(alpha, _latent_process_tr, _seed)

        if method.upper() == 'SCAR-P-OU':
            res = p_jit_mlog_likelihood_ou(alpha, u, dwt, self.np_pdf(), self.transform, print_path, init_state)
        elif method.upper() == 'SCAR-M-OU':
            res, a1t, a2t = m_jit_mlog_likelihood_ou(alpha, u, dwt, M_iterations, self.np_pdf(), self.transform, print_path, init_state)
        elif method.upper() == 'SCAR-P-LD':
            res = p_jit_mlog_likelihood_ld(alpha, u, dwt, self.np_pdf(), self.transform, print_path, init_state)
        elif method.upper() == 'SCAR-P-DS':
            res = p_jit_mlog_likelihood_ds(alpha, u, dwt, self.np_pdf(), self.transform, print_path)
        elif method.upper() == 'SCAR-M-DS':
            res = m_jit_mlog_likelihood_ds(alpha, u, dwt, M_iterations, self.np_pdf(), self.transform, print_path)
        else:
            raise ValueError(f"Method {method} is not implemented. Available methods = {self.list_of_methods}")
        return res
    
    def fit(self, 
            data: np.array,
            method: Literal['mle', 'scar-p-ou', 'scar-m-ou', 'scar-p-ld'] = 'mle',
            alpha0: np.array = None,
            tol = 1e-2,
            to_pobs = True,
            latent_process_tr: int = 500,
            M_iterations: int = 5,
            seed: int = None,
            dwt: np.array = None,
            stationary: bool = False,
            print_path: bool = False,
            init_state: np.array = None,
            ):
        '''fit stochastic or classic copula

        methods: scar-p-ou, scar-m-ou, scar-p-ld, mle
        '''
        if method.upper() not in self.list_of_methods():
            raise ValueError(f'given method {method} is not avialable. avialable methods: {self.list_of_methods()}') 

        data = np.asarray(data)

        u = data
        if to_pobs == True:
            u = pobs(data)

        T = len(data)
        if dwt is None:
            dwt = self.calculate_dwt(method, T, latent_process_tr, seed)
 
        def minimize_mle():
            x0 = 1/2
            num_method = 'BFGS'

            log_min = minimize(self.mlog_likelihood, x0,
                               args = (u, 'mle'), 
                               method = num_method,
                               options={'gtol': 1e-4, 'eps': 1e-5}
                               )
            return log_min


        def minimize_scar():

            bounds = None

            if method.upper() in ['SCAR-P-OU', 'SCAR-M-OU', 'SCAR-P-LD']:
                bounds = Bounds([0.001, -np.inf, 0.001], [np.inf, np.inf, np.inf])

            elif method.upper() in ['SCAR-P-DS', 'SCAR-M-DS']:
                bounds = Bounds([0.01, -0.99, 0.0], [5.0, 0.99, 0.99])

            num_method = 'L-BFGS-B'
            eps = 1e-4

            _alpha0 = None

            if alpha0 is None:
                if method.upper() in ['SCAR-M-OU', 'SCAR-P-OU', 'SCAR-P-LD']:

                    log_min_mle = minimize_mle()
                    mu0 = log_min_mle.x[0]           
                    _alpha0 =  np.array([1.0, mu0, 1.0])
                elif method.upper() in ['SCAR-M-DS', 'SCAR-P-DS']:
                    _alpha0 = np.array([0.05, 0.95, 0.05])
            else:
                _alpha0 = alpha0

            log_min = minimize(self.mlog_likelihood, _alpha0,
                                    args = (u,
                                            method.upper(),
                                            latent_process_tr, 
                                            M_iterations,
                                            seed, 
                                            dwt, 
                                            stationary,
                                            print_path, 
                                            init_state),
                                    method = num_method,
                                    bounds = bounds,
                                    options={'gtol': tol, 'eps': eps})

            return log_min
        
        if method.upper() == 'MLE':
            log_min = minimize_mle()
        else:
            log_min = minimize_scar()

        log_min.name = self.name
        log_min.fun = -log_min.fun
        log_min.method = method
        
        if method.upper() == 'MLE':
            log_min.x_transformed = self.transform(log_min.x)

        if method.upper() not in ['MLE']:
            if dwt is not None:
                latent_process_tr = dwt.shape[1]
            log_min.latent_process_tr = latent_process_tr
            log_min.stationary = stationary

        if method.upper() in ['SCAR-M-OU', 'SCAR-M-DS']:
            log_min.M_iterations = M_iterations

        return log_min
