import sympy as sp
import numpy as np
#from sympy.utilities.autowrap import autowrap
from numba import njit, jit
import math
from typing import Callable 
from functools import lru_cache
from scipy.optimize import minimize
from pyscarcopulas.sampler.p_sampler.ps import get_avg_p_log_likelihood
from pyscarcopulas.sampler.m_sampler.ms import get_avg_m_log_likelihood
from pyscarcopulas.sampler.m_sampler.ms import linear_least_squares
from pyscarcopulas.sampler.m_sampler.ms import log_xi
from pyscarcopulas.sampler.mle.mle import get_mlog_likelihood

class ArchimedianCopula:
    def __init__(self, dim: int) -> None:
        self.__dim = dim
        self.__t, self.__r = sp.symbols('t r')
        self.__sp_generator = - sp.log(self.t)
        self.__sp_antigenerator = sp.exp(-self.t)
        self.__calculated_crns_flg = 0
        self.__crns = np.nan
        self.__current_seed = -1
        self.__current_shape = (-1, -1)

    @property
    def t(self):
        return self.__t
    
    @property
    def r(self):
        return self.__r
    
    @property
    def sp_generator(self):
        return self.__sp_generator
    
    @property
    def sp_antigenerator(self):
        return self.__sp_antigenerator
    
    @property
    def dim(self):
        return self.__dim

    @property
    def calculated_crns_flg(self):
        return self.__calculated_crns_flg 
    
    @calculated_crns_flg.setter
    def calculated_crns_flg(self, val: int):
        self.__calculated_crns_flg = val
    
    @property
    def crns(self):
        return self.__crns
    
    @crns.setter
    def crns(self, arr):
        self.__crns = arr

    @property
    def current_seed(self):
        return self.__current_seed
    
    @current_seed.setter
    def current_seed(self, val: int):
        self.__current_seed = val

    @property
    def current_shape(self):
        return self.__current_shape
    
    @current_shape.setter
    def current_shape(self, shape):
        self.__current_shape = shape

    @lru_cache
    def sp_cdf(self):
        u = sp.symbols('u0:%d'%(self.dim))
        params = sum([self.sp_generator.subs([(self.t, x)]) for x in u])
        func = self.sp_antigenerator.subs([(self.t, params)])
        return func
    
    @lru_cache
    def sp_pdf(self):
        u = sp.symbols('u0:%d'%(self.dim))
        params = [self.sp_generator.subs([(self.t, x)]) for x in u]
        func = self.sp_antigenerator.subs([(self.t, sum(params))])
        for k in u:
            func = func.diff(k)
        func = sp.together(func)
        return func
    
    @lru_cache
    def np_pdf(self):
        expr = self.sp_pdf()
        u = sp.symbols('u0:%d'%(self.dim))
        r = sp.symbols('r')
        func = sp.lambdify((u, r), expr, modules = 'numpy', cse = True)
        return njit(func)

    def pdf(self, data, r):
        pass
    
    @staticmethod
    def transform(r):
        pass

    def log_likelihood(self, data, r):
        return np.sum(np.log(self.pdf(data, r )))
    
    def calculate_crns(self, T: int, n_tr: int, seed: int = None):
        shape = (T, n_tr)
        if self.calculated_crns_flg == 0:
            if seed == None:
                self.crns = np.random.normal(0 , 1 , size = (T, n_tr) )
                self.current_shape = shape
                self.calculated_crns_flg = 1
            else:
                rng = np.random.RandomState(seed) 
                self.crns = rng.normal(0 , 1 , size = (T, n_tr) )
                self.current_shape = shape
                self.current_seed = seed
                self.calculated_crns_flg = 1
        else:
            if seed == None:
                self.crns = np.random.normal(0 , 1 , size = (T, n_tr) )
                self.current_shape = shape
                self.current_seed = -1
            else:
                if seed != self.current_seed or shape != self.current_shape:
                    rng = np.random.RandomState(seed) 
                    self.crns = rng.normal(0 , 1 , size = (T, n_tr) )
                    self.current_seed = seed
                    self.current_shape = shape
    
    def p_sampler(self, omega: np.array, T: int, n_tr: int, seed: int = None) -> np.array:
        '''generate lambda(t) from natural (p) sampler'''
        gamma, delta, nu = omega[0], omega[1], omega[2]
        self.calculate_crns(T, n_tr, seed)
        res = np.zeros(self.current_shape)
        res[0] = gamma / ( 1 - delta)
        for i in range(1, self.current_shape[0]):
            res[i] = gamma + delta * res[i - 1] + nu * self.crns[i]
        return res
    
    def m_sampler(self, omega: np.array, a: np.array, T: int, n_tr: int, seed: int = None) -> np.array:
        '''generate lambda(t) from importance (m) sampler using a1(t), a2(t) parameters from previos iterations'''
        gamma, delta, nu = omega[0], omega[1], omega[2]
        a1, a2 = a[:,1], a[:,2]
        self.calculate_crns(T, n_tr, seed)
        res = np.zeros(self.current_shape)
        res[0] = gamma / ( 1 - delta)
        var = nu**2 / ( 1 - 2 * nu**2 * a2)
        p1, p2, p3 = var * ( gamma/ nu**2 + a1), var * delta / nu**2, np.sqrt(var)
        for i in range(1, self.current_shape[0]):
            res[i] = p1[i] + p2[i] * res[i - 1] + p3[i] * self.crns[i]
        return res
    
    def p_mlog_likelihood(self, omega: np.array, data: np.array, n_tr: int, seed: int = None) -> float:
        '''Calculation of likelihood function by straightforward method (p sampler)'''

        '''initial data check'''
        if np.isnan(np.sum(omega)) == True:
            res = 10000
            print(omega, 'incorrect params', res)
            return res
        
        if np.abs(omega[2]) > 1 or np.abs(omega[1]) > 1:
            res = 10000 
            print(omega, 'params is out of bounds', res)
            return res

        lambda_data = self.p_sampler(omega, len(data), n_tr, seed)
        avg_log_likelihood = get_avg_p_log_likelihood(data.T, lambda_data, n_tr, self.np_pdf(), self.transform)
        res = - avg_log_likelihood
        # '''threshold for keeping numerical stability with low numbers'''
        # if avg_likelihood < 10**(-5):
        #     res = 10000 
        #     print(omega, 'threshold value', res)
        #     return res
        # res = -math.log(avg_likelihood) - corr

        if np.isnan(res) == True:
            print(omega, 'unknown error', res)
        else:
            print(omega, res)

        return res
    
    def m_mlog_likelihood(self, omega: np.array, data: np.array, n_tr: int, m_iters: int, seed: int = None) -> float:
        '''
        Calculation of likelihood function by efficient importance sampling method (m sampler)

        consider latent process for copula parameter: 
        lambda(t) = gamma + delta * lambda(t - 1) + nu * Z(t), where Z - N(0,1)
        copula parameter is psi( lambda(t) )

        Parameters:
        1. w - set [gamma, delta, nu]. Type - Numpy Array
        2. data - dataset for calculation. Type - two-dim Numpy Array
        3. crn - common random numbers. Type - Numpy Array
        4. n_tr - number of trajectories for sampling. Type - Int 
        5. m_iters - number of iterations for importance sampling algorithm. Type - Int
        6. copula - bivariate copula function. Type - Function
        7. psi - function of latent process. Type - Function
        '''
        
        '''initial data check'''
        if np.isnan(np.sum(omega)) == True:
            res = 10000
            print(omega, 'incorrect params', res)
            return res
        
        if np.abs(omega[2]) > 1 or np.abs(omega[1]) > 1:
            res = 10000
            print(omega, 'params is out of bounds', res)
            return res

        '''set initial parameters'''
        T = len(data)
        a = np.zeros( (T, 3) )
        log_xi_data = np.zeros((T, n_tr))

        for i in range(0, m_iters):
            if i == 0:
                '''generate lambda(t) from natural (p) sampler'''
                lambda_data = self.p_sampler(omega, len(data), n_tr, seed)
            else:
                '''generate lambda(t) from importance (m) sampler using a1(t), a2(t) parameters from previos iterations'''
                lambda_data = self.m_sampler(omega, a, len(data), n_tr, seed)
                if np.isnan(np.sum(lambda_data)) == True:
                    res = 10000
                    print(omega, 'm sampler nan', res)
                    return res
            '''solve linear least-squares problem for search optimal parameters [a] for each t in (T, 0)'''
            for t in range(T - 1, 0 , -1):
                copula_log_data = np.sum(np.log(self.np_pdf()(data[t], self.transform(lambda_data[t]) )) )
                
                '''set A and b in LS problem Ax = b'''
                A = np.dstack( (np.ones(n_tr) , lambda_data[t] , lambda_data[t]**2 ) )[0]           
                b = copula_log_data + log_xi_data[t]
                #print(lambda_data)
                '''solve problem Ax = b'''
                try:
                    a[t] = linear_least_squares(A, b)
                except:
                    res = 10000
                    print(omega, 'ls problem fail', res)
                    return res
                #a[t] = linear_least_squares(A, b)
                log_xi_data[t - 1] = log_xi(a[t][1:3], omega, lambda_data[t - 1] )
    
        avg_log_likelihood = get_avg_m_log_likelihood(omega, data, a, lambda_data, log_xi_data, n_tr, self.np_pdf(), self.transform)
        res = - avg_log_likelihood

        if np.isnan(res) == True:
            print(omega, 'unknown error', res)
        else:
            print(omega, res)

        return res

    def fit(self, data: np.array, n_tr: int = 500, m_iters: int = 5, method: str = 'SCAR-M', omega0: np.array = np.array([0.05, 0.95, 0.05]), seed: int = 10):
        
        '''MLE'''
        if method == 'MLE':
            r0 = self.transform(np.array([1])) 
            log_min_mle = minimize(get_mlog_likelihood, r0, 
                                args=(data.T, self.np_pdf(), self.transform), 
                                method='L-BFGS-B', 
                                options={'gtol': 1e-5, 'ftol': 1e-5} )
            log_min_mle.fun = -log_min_mle.fun
            log_min_mle.x = self.transform(log_min_mle.x)
            log_min_mle.method = method
            return log_min_mle
        
        '''Natural sampler'''
        if method == 'SCAR-P':
            self.calculate_crns(len(data), n_tr, seed)
            log_min_p = minimize(self.p_mlog_likelihood, omega0, 
                                        args=(data, n_tr, seed), 
                                        method='L-BFGS-B', 
                                        options={'gtol': 1e-5, 'ftol': 1e-5} )
            log_min_p.fun = -log_min_p.fun
            log_min_p.method = method
            return log_min_p
        
        '''Efficient importance sampler'''
        if method == 'SCAR-M':
            self.calculate_crns(len(data), n_tr, seed)
            log_min_m = minimize(self.m_mlog_likelihood, omega0, 
                                        args=(data, n_tr, m_iters, seed), 
                                        method='L-BFGS-B', 
                                        options={'gtol': 1e-5, 'ftol': 1e-5} )
            log_min_m.fun = -log_min_m.fun
            log_min_m.method = method
            return log_min_m
        else:
            raise ValueError(f'given method {method} is not avialable')