import sympy as sp
import numpy as np
#from sympy.utilities.autowrap import autowrap
from numba import njit, jit
import math
from typing import Callable 
from functools import lru_cache
from scipy.optimize import minimize
from pyscarcopula.sampler.p_sampler.ps import get_avg_p_log_likelihood
from pyscarcopula.sampler.m_sampler.ms import get_avg_m_log_likelihood
from pyscarcopula.sampler.m_sampler.ms import linear_least_squares
from pyscarcopula.sampler.m_sampler.ms import log_xi
from pyscarcopula.sampler.mle.mle import get_mlog_likelihood

class ArchimedianCopula:
    def __init__(self, dim: int) -> None:
        self.__dim = dim
        self.__t, self.__r = sp.symbols('t r')
        '''independent copula by default'''
        self.__sp_generator = - sp.log(self.t)
        self.__sp_inverse_generator = sp.exp(-self.t)
        self.__calculated_crns_flg = 0
        self.__crns = np.nan
        self.__current_seed = -1
        self.__current_shape = (-1, -1)
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
    
    def __calculate_crns(self, T: int, n_tr: int, seed: int = None):
        '''Calculation of common random numbers (crn) with of T rows and n_tr columns. Setting seed is also avilable'''
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
                pass
            else:
                if seed != self.current_seed or shape != self.current_shape:
                    rng = np.random.RandomState(seed) 
                    self.crns = rng.normal(0 , 1 , size = (T, n_tr) )
                    self.current_seed = seed
                    self.current_shape = shape
    
    def __delete_crns(self):
        self.__calculated_crns_flg = 0
        self.__crns = np.nan
        self.__current_seed = -1
        self.__current_shape = (-1, -1)

    def __p_sampler(self, omega: np.array, T: int, n_tr: int, seed: int = None) -> np.array:
        '''generate lambda(t) from natural (p) sampler'''
        gamma, delta, nu = omega[0], omega[1], omega[2]
        self.__calculate_crns(T, n_tr, seed)
        res = np.zeros(self.current_shape)
        res[0] = gamma / ( 1 - delta)
        for i in range(1, self.current_shape[0]):
            res[i] = gamma + delta * res[i - 1] + nu * self.crns[i]
        return res
    
    def __m_sampler(self, omega: np.array, a: np.array, T: int, n_tr: int, seed: int = None) -> np.array:
        '''generate lambda(t) from importance (m) sampler using a1(t), a2(t) parameters from previos iterations'''
        gamma, delta, nu = omega[0], omega[1], omega[2]
        a1, a2 = a[:,1], a[:,2]
        self.__calculate_crns(T, n_tr, seed)
        res = np.zeros(self.current_shape)
        res[0] = gamma / ( 1 - delta)
        var = nu**2 / ( 1 - 2 * nu**2 * a2)
        p1, p2, p3 = var * ( gamma/ nu**2 + a1), var * delta / nu**2, np.sqrt(var)
        for i in range(1, self.current_shape[0]):
            res[i] = p1[i] + p2[i] * res[i - 1] + p3[i] * self.crns[i]
        return res
    
    def p_mlog_likelihood(self, omega: np.array, data: np.array, n_tr: int, seed: int = None, print_path: bool = False) -> float:
        '''Calculation of likelihood function by straightforward method (p sampler)
        
        Parameters.
        1. Omega - set of [gamma, delta, nu] regression parameters. Type - Numpy Array
        2. data - dataset for calculation. Type - Numpy Array
        3. n_tr - number of trajectories for calculation of average likelihood. Type - int
        4. seed - set seed for random number generator. Type - int. None by default.
        5. print_path - print result of function calculation for manual control of possible mistakes. Type - bool. False by default.
        '''

        '''initial data check'''
        if np.isnan(np.sum(omega)) == True:
            res = 10000
            if print_path == True:
                print(omega, 'incorrect params', res)
            return res
        
        if np.abs(omega[2]) > 1 or np.abs(omega[1]) > 1:
            res = 10000 
            print(omega, 'params is out of bounds', res)
            return res

        lambda_data = self.__p_sampler(omega, len(data), n_tr, seed)
        avg_log_likelihood = get_avg_p_log_likelihood(data.T, lambda_data, n_tr, self.np_pdf(), self.transform)
        res = - avg_log_likelihood
        # '''threshold for keeping numerical stability with low numbers'''
        # if avg_likelihood < 10**(-5):
        #     res = 10000 
        #     print(omega, 'threshold value', res)
        #     return res
        # res = -math.log(avg_likelihood) - corr

        if np.isnan(res) == True:
            if print_path == True:
                print(omega, 'unknown error', res)
        else:
            if print_path == True:
                print(omega, res)
        return res
    
    def m_mlog_likelihood(self, omega: np.array, data: np.array, n_tr: int, m_iters: int, seed: int = None, print_path: bool = False) -> float:
        '''
        Calculation of likelihood function by efficient importance sampling method (m sampler)

        consider latent process for copula parameter: 
        lambda(t) = gamma + delta * lambda(t - 1) + nu * Z(t), where Z - N(0,1)
        copula parameter is self.transform( lambda(t) )

        Parameters.
        1. Omega - set of [gamma, delta, nu] regression parameters. Type - Numpy Array
        2. data - dataset for calculation. Type - Numpy Array
        3. n_tr - number of trajectories for calculation of average likelihood. Type - int
        4. seed - set seed for random number generator. Type - int. None by default.
        5. print_path - print result of function calculation for manual control of possible mistakes. Type - bool. False by default.
        '''
        
        '''initial data check'''
        if np.isnan(np.sum(omega)) == True:
            res = 10000
            if print_path == True:
                print(omega, 'incorrect params', res)
            return res
        
        if np.abs(omega[2]) > 1 or np.abs(omega[1]) > 1:
            res = 10000
            if print_path == True:
                print(omega, 'params is out of bounds', res)
            return res

        '''set initial parameters'''
        T = len(data)
        a = np.zeros( (T, 3) )
        log_xi_data = np.zeros((T, n_tr))

        for i in range(0, m_iters):
            if i == 0:
                '''generate lambda(t) from natural (p) sampler'''
                lambda_data = self.__p_sampler(omega, len(data), n_tr, seed)
            else:
                '''generate lambda(t) from importance (m) sampler using a1(t), a2(t) parameters from previos iterations'''
                lambda_data = self.__m_sampler(omega, a, len(data), n_tr, seed)
                if np.isnan(np.sum(lambda_data)) == True:
                    res = 10000
                    if print_path == True:
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
                    if print_path == True:
                        print(omega, 'ls problem fail', res)
                    return res
                #a[t] = linear_least_squares(A, b)
                log_xi_data[t - 1] = log_xi(a[t][1:3], omega, lambda_data[t - 1] )
    
        avg_log_likelihood = get_avg_m_log_likelihood(omega, data, a, lambda_data, log_xi_data, n_tr, self.np_pdf(), self.transform)
        res = - avg_log_likelihood

        if np.isnan(res) == True:
            if print_path == True:
                print(omega, 'unknown error', res)
        else:
            if print_path == True:
                print(omega, res)

        return res

    def fit(self, data: np.array, n_tr: int = 500, m_iters: int = 5,
             method: str = 'SCAR-M', omega0: np.array = np.array([0.05, 0.95, 0.05]), seed: int = None, print_path: bool = False):
        '''fit copula parameter
        
        # Parameters. 
        1. data - dataset for calculation. Type - Numpy Array.
        2. n_tr - number of trajectories for calculation of average likelihood. Type - int. Only for 'SCAR-P', 'SCAR-M'. By default - 500
        3. m_iters - number of m-sampling iterations. Type - int. Only for 'SCAR-M'. By default - 5
        4. method:
            a. MLE - maximum likelihood estimation. Returns float value of parameter `r`.

            b. SCAR-P - Natural sampler (p). Returns set of parameters omega. This method has a simplier implementation and works stable,
            but needs more trajectories (approx 10^5) to get more accurate reuslt

            c. SCAR-M (by default) - Efficient importance sampler (m). Optimized sampler. Method requires calculation of auxilary parameters 
            on each itaration. That allows to calculate significantly less trajectories (in most cases 200 is enough). 
            In rare cases could be unstable.

        5. omega0 - initial set of parameters omega. By default [0.05, 0.95, 0.05]. Set where second term close to 1 and others are close to 0
        gives as result almost stable regression similar to const parameter. That set is good starting point for minimization.
        Type - Numpy Array.
        6. seed - numpy random number generator parameter. Type - int. None by default
        7. print_path - print result of function calculation for manual control of possible mistakes. Type - bool. False by default. 
        Only for SCAR-M and SCAR-P.

        # Output
        Returns scipy.minimize dictionary with as result float value (for MLE) or set of regression parameters (for SCAR-P and SCAR-M) 
        and function value at this point. L-BFGS-B method is used.
        '''

        '''initial data check'''
        if self.__dim != len(data[0]):
            raise ValueError(f"Copula dim = {self.__dim} is not equal dim of data = {len(data[0])}. ")

        '''MLE'''
        if method.upper() == 'MLE':
            r0 = self.transform(np.array([1])) 
            log_min_mle = minimize(get_mlog_likelihood, r0, 
                                args=(data.T, self.np_pdf(), self.transform), 
                                method='L-BFGS-B', 
                                options={'gtol': 1e-5, 'ftol': 1e-5} )
            log_min_mle.name = self.name
            log_min_mle.fun = -log_min_mle.fun
            log_min_mle.x = self.transform(log_min_mle.x)
            log_min_mle.method = method
            return log_min_mle
        
        '''Natural sampler'''
        if method.upper() == 'SCAR-P':
            self.__calculate_crns(len(data), n_tr, seed)
            log_min_p = minimize(self.p_mlog_likelihood, omega0, 
                                        args=(data, n_tr, seed, print_path), 
                                        method='L-BFGS-B', 
                                        options={'gtol': 1e-5, 'ftol': 1e-5} )
            log_min_p.name = self.name
            log_min_p.fun = -log_min_p.fun
            log_min_p.method = method
            self.__delete_crns()
            return log_min_p
        
        '''Efficient importance sampler'''
        if method.upper() == 'SCAR-M':
            self.__calculate_crns(len(data), n_tr, seed)
            log_min_m = minimize(self.m_mlog_likelihood, omega0, 
                                        args=(data, n_tr, m_iters, seed, print_path), 
                                        method='L-BFGS-B', 
                                        options={'gtol': 1e-10, 'ftol': 1e-10} )
            log_min_m.name = self.name
            log_min_m.fun = -log_min_m.fun
            log_min_m.method = method
            self.__delete_crns()
            return log_min_m
        else:
            raise ValueError(f'given method {method} is not avialable')