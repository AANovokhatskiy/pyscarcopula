import sympy as sp
import numpy as np
from typing import Literal
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
    """
    A base class for representing and working with Archimedean copulas.

    This class provides a framework for defining Archimedean copulas,
    calculating their CDFs and PDFs symbolically and numerically,
    generating samples, and performing parameter estimation.
    """

    def __init__(self, dim: int = 2, rotate: Literal[0, 90, 180, 270] = 0) -> None:
        """
        Initializes an ArchimedianCopula object.

        Args:
            dim (int): The dimension of the copula (e.g., 2 for bivariate, 3 for trivariate). Defaults to 2.
            rotate (Literal[0, 90, 180, 270]): The rotation angle of the copula (only applicable for 2D).
                Possible values are 0, 90, 180, or 270 degrees. Defaults to 0.
        """

        self.__dim = dim
        self.__rotate = rotate

        if self.dim != 2 and self.rotate != 0:
            warnings.warn("Rotation is implemented only for dim = 2 and this parameter will be ignored")

        '''independent copula by default'''
        self.__name = 'Independent copula'

    @property
    def name(self):
        """
        Returns the name of the copula.

        Returns:
            str: The name of the copula.
        """
        return self.__name
   
    @lru_cache
    def sp_generator(self):
        """
        Returns the symbolic generator function of the copula.

        Returns:
            sympy.Expr: The symbolic generator function.
        """
        t, r = sp.symbols('t r')
        
        result = -sp.log(t)
        return result
    
    @lru_cache
    def sp_inverse_generator(self):
        """
        Returns the symbolic inverse generator function of the copula.

        Returns:
            sympy.Expr: The symbolic inverse generator function.
        """
        t, r = sp.symbols('t r')

        result = sp.exp(-t)
        return result
    
    @property
    def dim(self):
        """
        Returns the dimension of the copula.

        Returns:
            int: The dimension of the copula.
        """
        return self.__dim

    @property
    def rotate(self):
        """
        Returns the rotation angle of the copula.

        Returns:
            int: The rotation angle (0, 90, 180, or 270).
        """
        return self.__rotate
    
    @staticmethod
    def list_of_methods():
        """
        Returns a list of available methods for parameter estimation.

        Returns:
            list: A list of available methods ('MLE', 'SCAR-M-OU', 'SCAR-P-OU', 'SCAR-P-LD', 'SCAR-P-DS', 'SCAR-M-DS', 'SCAR-S-OU').
        """
        l = ['MLE', 'SCAR-M-OU', 'SCAR-P-OU', 'SCAR-P-LD', 'SCAR-P-DS', 'SCAR-M-DS', 'SCAR-S-OU']
        return l

    @lru_cache
    def sp_cdf_from_generator(self):
        """
        Generates the symbolic expression for the copula's CDF based on the generator.

        Returns:
            sympy.Expr: The symbolic expression for the copula's CDF.
        """
        u = sp.symbols('u0:%d'%(self.dim), positive = True)
        u = list(u)

        t, r = sp.symbols('t r')

        params = sum([self.sp_generator().subs([(t, x)]) for x in u])
        func = self.sp_inverse_generator().subs([(t, params)])

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
        """
        Returns the symbolic expression for the copula's CDF.

        Returns:
            sympy.Expr: The symbolic expression for the copula's CDF.
        """
        return self.sp_cdf_from_generator()

    @lru_cache
    def sp_pdf_from_generator(self):
        """
        Generates the symbolic expression for the copula's PDF based on the generator.

        Returns:
            sympy.Expr: The symbolic expression for the copula's PDF.
        """

        t, r = sp.symbols('t r')

        u = sp.symbols('u0:%d'%(self.dim), positive = True)
        params = [self.sp_generator().subs([(t, x)]) for x in u]

        diff_inverse_generator = sp.together(self.sp_inverse_generator().diff((t, self.dim)))
        diff_generator = sp.together(self.sp_generator().diff(t, 1))

        func = diff_inverse_generator.subs([(t, sum(params))])

        for x in u:
            func = func * diff_generator.subs([(t, x)])
        
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
        """
        Returns the symbolic expression for the copula's PDF.

        Returns:
            sympy.Expr: The symbolic expression for the copula's PDF.
        """
        return self.sp_pdf_from_generator()
        
    @lru_cache
    def np_pdf(self):
        """
        Converts the symbolic PDF to a numerical function that can be evaluated with NumPy. 
        For the convenience use `pdf` method

        Returns:
            Callable: A numerical function that takes (u, r) as input and returns the PDF value.

        Example:
            # pdf = self.np_pdf()(u.T, r)
        
        """
        expr = self.sp_pdf()
        u = sp.symbols('u0:%d'%(self.dim))
        r = sp.symbols('r')
        func = sp.lambdify((u, r), expr, modules = 'numpy', cse = True)
        return func

    @lru_cache
    def np_cdf(self):
        """
        Converts the symbolic CDF to a numerical function that can be evaluated with NumPy.
        For the convenience use `cdf` method

        Returns:
            Callable: A numerical function that takes (u, r) as input and returns the CDF value.

        Example:
            cdf = self.np_cdf()(u.T, r)

        """
        expr = self.sp_cdf()
        u = sp.symbols('u0:%d'%(self.dim))
        r = sp.symbols('r')
        func = sp.lambdify((u, r), expr, modules = 'numpy', cse = True)
        return func

    def _broadcasting(self, u, r):
        """
        Broadcasts u and r to ensure compatible shapes for numerical calculations.

        Args:
            u (np.ndarray): An array of uniform marginal values.
            r (np.ndarray or float): An array or a scalar representing the copula parameter.

        Returns:
            tuple: A tuple containing the broadcasted u and r arrays.

        Raises:
            ValueError: If the shapes of u and r are incompatible for broadcasting.
        """
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
        """
        Calculates the numerical PDF of the copula.

        Args:
            u (np.ndarray): An array of uniform marginal values.
            r (np.ndarray or float): An array or a scalar representing the copula parameter.

        Returns:
            np.ndarray: An array of PDF values or a single PDF value.
        """
        func = self.np_pdf()

        u, r = self._broadcasting(u, r)

        res = func(u.T, r)
        return res
    
    def cdf(self, u, r):
        """
        Calculates the numerical CDF of the copula.

        Args:
            u (np.ndarray): An array of uniform marginal values.
            r (np.ndarray or float): An array or a scalar representing the copula parameter.

        Returns:
            np.ndarray or float: An array of CDF values or a single CDF value.
        """
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
        """
        Function that transfroms interval (-inf, inf) to avialable range for copula parameter.

        Args:
            r (np.ndarray or float): The copula parameter to be transformed.

        Returns:
            np.ndarray or float: The transformed copula parameter.
        """
        return r

    @staticmethod
    def psi(t, r):
        """
        Placeholder for the 'psi' function used in sampling (default is exp(-t)).

        Args:
            t (float): Input value.
            r (float): Copula parameter.

        Returns:
            float: The result of the psi function.
        """
        return np.exp(-t)
     
    @staticmethod
    def V(N, r):
        """
        Placeholder for the 'V' function used in sampling (default is ones).

        Args:
            N (int): Number of samples
            r (float): Copula parameter or array of copula parameters

        Returns:
            np.ndarray: Array of ones with shape (N,) or shape of r
        """
        if isinstance(r, (int, float)):
            r_arr = np.ones(N) * 1
        else:
            r_arr = np.ones_like(r)
        return r_arr
    
    def get_sample(self, N, r):
        """
        Generates samples from the copula. Based on M.Hofert, Sampling Archimedean copulas, 2008

        Args:
            N (int): The number of samples to generate.
            r (np.ndarray or float): The copula parameter (can be an array or a scalar).

        Returns:
            np.ndarray: An array of shape (N, dim) containing the generated samples.
        """
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
        """
        Calculates the log-likelihood of the copula.

        Args:
            u (np.ndarray): An array of uniform marginal values.
            r (np.ndarray or float): An array or a scalar representing the copula parameter.

        Returns:
            float: The log-likelihood value.
        """
        u, r = self._broadcasting(u, r)

        return np.sum(np.log(self.pdf(u, r)))

    @staticmethod
    def calculate_dwt(method: Literal['mle', 'scar-p-ou', 'scar-m-ou', 'scar-p-ld'],
                      T: int, latent_process_tr: int, seed: int = None, dt: float = None):
        """
        Calculates increments of Wiener process for stochastic methods.

        Args:
            method (Literal['mle', 'scar-p-ou', 'scar-m-ou', 'scar-p-ld']): The estimation method.
            T (int): The number of time steps.
            latent_process_tr (int): The number of latent process trajectories.
            seed (int, optional): The random seed for generating common random numbers. Defaults to None.
            dt (float, optional): The time step (for 'SCAR-M-OU', 'SCAR-P-OU', 'SCAR-P-LD'). Defaults to None.

        Returns:
            np.ndarray: An array of Wiener process increments.
        """
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
        """
        Calculates the negative log-likelihood of the copula for a given dataset and parameters.

        This function supports both Maximum Likelihood Estimation (MLE) and stochastic methods
        for parameter estimation. For stochastic methods, it approximates the likelihood by
        integrating over the latent process trajectories.

        Args:
            alpha (np.ndarray): The parameter vector. The interpretation of the parameters depends
                on the chosen method:
                - 'mle': `alpha` should be a scalar representing the copula parameter (should also be non transformed: see `fit` method). 
                - 'scar-p-ou', 'scar-m-ou', 'scar-p-ld': `alpha` should be a vector `[theta, mu, nu]` where:
                    - `theta` is the rate of mean reversion.
                    - `mu` is the long-term mean.
                    - `nu` is the volatility of the latent process.
                - other scar methods: parameter set depends on method (see README for details).
            u (np.ndarray): An array of uniform marginal values (pseudo observations).
            method (Literal['mle', 'scar-p-ou', 'scar-m-ou', 'scar-p-ld']): The estimation method:
                - 'mle': Maximum Likelihood Estimation with a constant copula parameter.
                - 'scar-p-ou': Stochastic Copula with Autoregressive parameter using Ornstein-Uhlenbeck (OU) process.
                - 'scar-m-ou': Stochastic Copula with OU process and Monte Carlo importance sampling.
                - 'scar-p-ld': Stochastic Copula with Logistic Distribution transition density (experimental).
            latent_process_tr (int, optional): The number of latent process trajectories used for stochastic methods.
                Defaults to 500.
            M_iterations (int, optional): The number of Monte Carlo importance sampling steps (only for 'scar-m-ou').
                Defaults to 5.
            seed (int, optional): The random seed for generating Wiener process increments. Defaults to None.
            dwt (np.ndarray, optional): Pre-calculated Wiener process increments. If provided, the `calculate_dwt`
                function is not called. Defaults to None.
            stationary (bool, optional): If True, uses stationary distribution for generating initial latent process states.
                Defaults to False.
            print_path (bool, optional): If True, save and return all latent process values. Defaults to False.
            init_state (np.ndarray, optional): Initial state of the latent process. Defaults to None.

        Returns:
            float: The negative log-likelihood value.

        Raises:
            ValueError: If the given `method` is not available or dimension mismatch.
        """
        if self.dim != u.shape[1]:
            raise ValueError(f'Dimension of copula = {self.dim} does not correspond to dimension of data = {u.shape[1]}')

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
        """
        Fits the copula to the given data using the specified method.

        This function estimates the copula parameters, either as a constant value (for MLE)
        or as parameters of a latent stochastic process (for SCAR methods).

        Args:
            data (np.ndarray): The dataset to fit the copula to. It can be either log-returns or
                pseudo-observations (uniform marginals), depending on the `to_pobs` parameter.
            method (Literal['mle', 'scar-p-ou', 'scar-m-ou', 'scar-p-ld']): The parameter estimation method:
                - 'mle': Maximum Likelihood Estimation (constant parameter).
                - 'scar-p-ou': Stochastic Copula with OU process (stochastic parameter).
                - 'scar-m-ou': Stochastic Copula with OU process and Monte Carlo importance sampling (stochastic parameter).
                - 'scar-p-ld': Stochastic Copula with Logistic Distribution transition density(stochastic parameter, experimental).
                - 'scar-p-ds': Stochastic Copula with Discrete transition density(stochastic parameter, experimental).
                - 'scar-m-ds': Stochastic Copula with Discrete transition density and Monte Carlo importance sampling(stochastic parameter, experimental).
            alpha0 (np.ndarray, optional): The initial guess for the parameters. If None, a reasonable initial
                guess is calculated automatically. Defaults to None.
            tol (float, optional): The tolerance for the optimization stopping criterion (gradient norm). Only for stochastic models. Defaults to 1e-2.
            to_pobs (bool, optional): If True, the input data is transformed to pseudo-observations (uniform marginals).
                Defaults to True.
            latent_process_tr (int, optional): The number of latent process trajectories used for stochastic methods.
                Defaults to 500.
            M_iterations (int, optional): The number of Monte Carlo importance sampling steps (only for 'scar-m-ou' and 'scar-m-ds').
                Defaults to 5.
            seed (int, optional): The random seed for generating Wiener process increments. Defaults to None.
            dwt (np.ndarray, optional): Pre-calculated Wiener process increments. If provided, the `calculate_dwt`
                function is not called. Defaults to None.
            stationary (bool, optional): If True, uses stationary distribution for generating initial latent process states.
                Defaults to False.
            print_path (bool, optional): If True, save and return all latent process values. Defaults to False.
            init_state (np.ndarray, optional): Initial state of the latent process. Defaults to None.
        Returns:
            scipy.optimize.OptimizeResult: The optimization result object.
                - `x`: The estimated parameter(s). The real parameter is self.transform(x): see `x_transformed`. The transformation is used for solving unconstrained problem for stochastic models.
                - `fun`: The negative log-likelihood at the optimum.
                - `message`: Description of the cause of termination.
                - other attributes of `scipy.optimize.OptimizeResult` class.
                - `name`: The name of copula.
                - `method`: The method that was used to fit the copula.
                - `latent_process_tr`: The number of latent process trajectories (if any).
                - `stationary`: True if stationary state was used.
                - `M_iterations`: The number of M_iterations (if any).
                - `x_transformed`: Transformed (real) copula parameter (only for MLe method).
        Raises:
            ValueError: If the given `method` is not available.
        """        
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

            _seed = None

            if seed is not None:
                _seed = seed
            else:
                _seed = np.random.randint(1, 1000000)

            log_min = minimize(self.mlog_likelihood, _alpha0,
                                    args = (u,
                                            method.upper(),
                                            latent_process_tr, 
                                            M_iterations,
                                            _seed,
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
