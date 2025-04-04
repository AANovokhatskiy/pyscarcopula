import sympy as sp
import numpy as np
from typing import Literal
import warnings
from numba import njit

from functools import lru_cache
from scipy.optimize import minimize, Bounds

from pyscarcopula.sampler.sampler_ou import p_jit_mlog_likelihood_ou
from pyscarcopula.sampler.sampler_ou import m_jit_mlog_likelihood_ou
from pyscarcopula.sampler.sampler_ou import stationary_state_ou

from pyscarcopula.sampler.sampler_ld import p_jit_mlog_likelihood_ld
from pyscarcopula.sampler.sampler_ld import stationary_state_ld

from pyscarcopula.sampler.sampler_ds import p_jit_mlog_likelihood_ds
from pyscarcopula.sampler.sampler_ds import m_jit_mlog_likelihood_ds
from pyscarcopula.auxiliary.funcs import pobs
from pyscarcopula.metrics.latent_process import latent_process_final_state


from scipy.special import roots_hermite


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
        self.__rotatable = True
        self.rotate = rotate
        if rotate not in [0, 90, 180, 270]:
            raise ValueError(f"Only 0, 90, 180, 270 angles are supported, got {rotate}")
        
        if self.dim != 2 and self.rotate != 0:
            warnings.warn("Rotation is implemented only for dim = 2 and this parameter will be ignored")

        '''independent copula by default'''
        self.__name = 'Independent copula'
        self.__bounds = [(-np.inf, np.inf)]

        self.fit_result = None

    @property
    def rotatable(self):
        """
        Returns True if copula is rotatable, False - otherwise

        Returns:
            bool
        """
        return self.__rotatable

    @property
    def name(self):
        """
        Returns the name of the copula.

        Returns:
            str: The name of the copula.
        """
        return self.__name
   
    @property
    def bounds(self):
        """
        Returns the parameter bounds used for MLE optimization.

        Returns:
            Numpy array: parameter bounds.
        """
        return self.__bounds

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
    def np_pdf(self, numba_jit = False):
        """
        Converts the symbolic PDF to a numerical function that can be evaluated with NumPy. 
        For the convenience use `pdf` method

        Returns:
            Callable: A numerical function that takes (u, r) as input and returns the PDF value.

        Example:
            pdf = self.np_pdf()(u.T, r)
        
        """
        expr = self.sp_pdf()
        u = sp.symbols('u0:%d'%(self.dim))
        r = sp.symbols('r')
        func = sp.lambdify((u, r), expr, modules = 'numpy', cse = True)

        if numba_jit == True:
            func = njit(func)

        return func

    @lru_cache
    def np_log_pdf(self, numba_jit = False):
        """
        Converts the symbolic PDF to a numerical function that can be evaluated with NumPy. 
        For the convenience use `pdf` method

        Returns:
            Callable: A numerical function that takes (u, r) as input and returns the PDF value.

        Example:
            pdf = self.np_pdf()(u.T, r)
        
        """
        expr = sp.log(self.sp_pdf())
        u = sp.symbols('u0:%d'%(self.dim))
        r = sp.symbols('r')
        func = sp.lambdify((u, r), expr, modules = 'numpy', cse = True)

        if numba_jit == True:
            func = njit(func)

        return func

    @lru_cache
    def np_cdf(self, numba_jit = False):
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

        if numba_jit == True:
            func = njit(func)

        return func

    def _broadcasting(self, u, r):
        """
        Broadcasts u and r to ensure compatible shapes for numerical calculations,
        including support for r of shape (n,m).

        Args:
            u (np.ndarray): An array of uniform marginal values.
            r (np.ndarray or float): An array or a scalar representing the copula parameter.

        Returns:
            tuple: A tuple containing the broadcasted u and r arrays.

        Raises:
            ValueError: If the shapes of u and r are incompatible for broadcasting.
        """
        # u = np.asarray(u, dtype=np.float64)
        # u_copy = np.copy(u)

        # # Validate u shape
        # if u_copy.ndim == 1:
        #     if len(u_copy) != self.dim:
        #         raise ValueError(f"The dimension of u ({len(u_copy)}) must match the dimension of the copula ({self.dim})")
        #     u_copy = u_copy[np.newaxis, :]
        # elif u_copy.ndim == 2:
        #     if u_copy.shape[1] != self.dim:
        #         raise ValueError(f"The dimension of u ({u_copy.shape[1]}) must match the dimension of the copula ({self.dim})")
        # else:
        #     raise ValueError("u must be 1D or 2D array")

        # r = np.asarray(r, dtype=np.float64)
        # r_copy = np.copy(r)

        # # Handle scalar case
        # if r_copy.ndim == 0:
        #     if u_copy.shape[0] == 1:
        #         return u_copy, r_copy
        #     r_copy = np.full(u_copy.shape[0], r_copy)
        #     return u_copy, r_copy

        # # Handle 1D r case
        # if r_copy.ndim == 1:
        #     if len(r_copy) == 1:
        #         if u_copy.shape[0] == 1:
        #             return u_copy, r_copy[0] if r_copy.shape[0] == 1 else r_copy
        #         r_copy = np.full(u_copy.shape[0], r_copy[0])
        #         return u_copy, r_copy
            
        #     if u_copy.shape[0] == 1:
        #         u_copy = np.full((r_copy.shape[0], u_copy.shape[1]), u_copy[0])
        #         return u_copy, r_copy
            
        #     if len(r_copy) != u_copy.shape[0]:
        #         raise ValueError(f"Length of r ({len(r_copy)}) must match number of rows in u ({u_copy.shape[0]}) or be 1")
        #     return u_copy, r_copy

        # # Handle 2D r case (n,m)
        # if r_copy.ndim == 2:
        #     if u_copy.shape[0] == 1:
        #         # Broadcast u to match r's first dimension
        #         u_copy = np.full((r_copy.shape[0], u_copy.shape[1]), u_copy[0])
        #         return u_copy, r_copy
            
        #     if r_copy.shape[0] != u_copy.shape[0]:
        #         raise ValueError(f"First dimension of r ({r_copy.shape[0]}) must match number of rows in u ({u_copy.shape[0]}) when r is 2D")
        #     return u_copy, r_copy

        # raise ValueError("r must be scalar, 1D array, or 2D array")

        _u = np.asarray(u)
        _r = np.asarray(r)

        if _u.ndim == 1:
            _u = _u[np.newaxis, :]

        _r = np.asarray(_r)

        if _r.ndim == 1 and len(_r) == 1:
            _r = _r.item()
            _r = np.asarray(_r)

        if _r.ndim == 0:
            _r = np.full(_u.shape[0], _r)

        if _u.shape[0] == 1 and _r.ndim == 1:
            _u = np.full((_r.shape[0], _u.shape[1]), _u[0])

        if len(_r) != _u.shape[0] and not (_u.shape[0] == 1 and _r.ndim == 1):
            raise ValueError("The length of r must match the number of rows in u or be compatible for broadcasting.")

        return _u, _r
    
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

        _u, _r = self._broadcasting(u, r)
        if _r.ndim == 2:
            res = np.empty(shape = _r.shape, dtype=np.float64)
            for k in range(0, _r.shape[1]):
                res[:,k] = func(_u.T, _r[:,k])
        else:
            res = func(_u.T, _r)

        return res
    
    def log_pdf(self, u: np.array, r: np.array):
        """
        Calculates the numerical log-PDF of the copula.

        Args:
            u (np.ndarray): An array of uniform marginal values.
            r
        Returns:
            np.ndarray: An array of log-PDF values or a single log-PDF value.
        """
        func = self.np_log_pdf()

        _u, _r = self._broadcasting(u, r)
        if _r.ndim == 2:
            res = np.empty(shape = _r.shape, dtype=np.float64)
            for k in range(0, _r.shape[1]):
                res[:,k] = func(_u.T, _r[:,k])
        else:
            res = func(_u.T, _r)
            
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

        _u, _r = self._broadcasting(u, r)
        if _r.ndim == 2:
            res = np.empty(shape = _r.shape, dtype=np.float64)
            for k in range(0, _r.shape[1]):
                res[:,k] = func(_u.T, _r[:,k])
        else:
            res = func(_u.T, _r)
        
        ''' Fréchet-Hoeffding boundary'''
        ub = np.min(_u, axis = 1)
        lb = np.maximum(1 - self.dim + np.sum(_u, axis = 1), 0)
        res = np.clip(res, lb, ub)

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
    @njit
    def transform_jit(r):
        return r
    
    @staticmethod
    def inv_transform(r):
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
    
    def get_sample(self, size, r):
        """
        Generates samples from the copula. Based on M.Hofert, Sampling Archimedean copulas, 2008

        Args:
            N (int): The number of samples to generate.
            r (np.ndarray or float): The copula parameter (can be an array or a scalar).

        Returns:
            np.ndarray: An array of shape (N, dim) containing the generated samples.
        """
        u = np.zeros((size, self.dim))
        
        _r = np.asarray(r, dtype=np.float64)
        
        if _r.ndim == 0:
            _r = np.full(size, _r)
        elif _r.shape == (1,):
            _r = np.full(size, _r[0])
        elif len(_r) != size:
            raise ValueError(f"Length of r ({len(_r)}) must match N ({size}) or be a scalar")

        x = np.random.uniform(0, 1, size = (size, self.dim))
        
        V_data = np.clip(self.V(size, _r), 1e-20, 1e+20)
    
        for k in range(0, self.dim):
            u[:,k] = self.psi(-np.log(x[:,k]) / V_data, _r)

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

        return np.sum(self.log_pdf(u, r))

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
                        alpha: np.array = None, 
                        u: np.array = None, 
                        method: Literal['mle', 'scar-p-ou', 'scar-m-ou', 'scar-p-ld'] = 'mle',
                        latent_process_tr: int = 500, 
                        M_iterations: int = 3,
                        seed: int = None, 
                        dwt: np.array = None, 
                        stationary: bool = False,
                        print_path: bool = False,
                        kwargs = {}) -> float:
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

        Returns:
            float: The negative log-likelihood value.

        Raises:
            ValueError: If the given `method` is not available or dimension mismatch.
        """
        
        if u is None:
            raise ValueError("Please specify the dataset u")

        if alpha is None and self.fit_result is None:
            raise ValueError("Please specify the copula parameters alpha or fit the model first")
        
        if alpha is None:
            alpha = self.fit_result.x

        if self.dim != u.shape[1]:
            raise ValueError(f'Dimension of copula = {self.dim} does not correspond to dimension of data = {u.shape[1]}')

        if method.upper() not in self.list_of_methods():
            raise ValueError(f'given method {method} is not avialable. avialable methods: {self.list_of_methods()}')

        _alpha = np.copy(np.asarray(alpha))
        _u = np.copy(np.asarray(u))
        
        if method.upper() == 'MLE':
            _u, _alpha = self._broadcasting(u, alpha)
            res = -self.log_likelihood(_u, _alpha)

            return res
        
        T = len(u)
        if dwt is None:
            dwt = self.calculate_dwt(method, T, latent_process_tr, seed)

        # if stationary == True and init_state is None:
        #     _latent_process_tr = dwt.shape[1]
        #     _seed = None
        #     if seed is not None:
        #         _seed = seed * 2

        #     if method.upper() in ['SCAR-P-OU', 'SCAR-M-OU']:
        #         init_state = stationary_state_ou(_alpha, _latent_process_tr, _seed)
        #     elif method.upper() in ['SCAR-P-LD']:
        #         init_state = stationary_state_ld(_alpha, _latent_process_tr, _seed)

        if method.upper() == 'SCAR-P-OU': 
            res = p_jit_mlog_likelihood_ou(_alpha, _u, dwt, self.log_pdf, self.transform, print_path, stationary) #### self.np_log_pdf(), _u.T
        
        elif method.upper() == 'SCAR-M-OU':
            if 'z' in kwargs and 'w' in kwargs:
                z = kwargs.get('z')
                w = kwargs.get('w')
            else:
                z, w = roots_hermite(250)
                args = w > 1e-3
                w = w[args]
                z = z[args]

            res, a1t, a2t = m_jit_mlog_likelihood_ou(_alpha, _u, dwt, M_iterations, 
                                                     self.np_log_pdf(numba_jit = True),
                                                     self.transform_jit, print_path, z, w, stationary)
        elif method.upper() == 'SCAR-P-LD':
            res = p_jit_mlog_likelihood_ld(_alpha, _u, dwt, self.log_pdf, self.transform, print_path, stationary)
        elif method.upper() == 'SCAR-P-DS':
            res = p_jit_mlog_likelihood_ds(_alpha, _u, dwt, self.log_pdf, self.transform, print_path)
        elif method.upper() == 'SCAR-M-DS':
            res = m_jit_mlog_likelihood_ds(_alpha, _u, dwt, M_iterations, self.log_pdf, self.transform, print_path)
        else:
            raise ValueError(f"Method {method} is not implemented. Available methods = {self.list_of_methods}")
        return res
    
    def fit(self, 
            data: np.array,
            method: Literal['mle', 'scar-p-ou', 'scar-m-ou', 'scar-p-ld'] = 'mle',
            alpha0: np.array = None,
            tol = 1e-2,
            to_pobs = False,
            latent_process_tr: int = 500,
            M_iterations: int = 3,
            seed: int = None,
            dwt: np.array = None,
            stationary: bool = False,
            print_path: bool = False,
            kwargs = {}
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
                Defaults to False.
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

        data = np.copy(np.asarray(data, dtype = np.float64))

        u = data
        if to_pobs == True:
            u = pobs(data)

        T = len(data)
        if dwt is None:
            dwt = self.calculate_dwt(method, T, latent_process_tr, seed)
 
        def minimize_mle():
            x0 = self.transform(1/2)
            num_method = 'L-BFGS-B'

            log_min = minimize(self.mlog_likelihood, x0,
                               args = (u, 'mle'), 
                               method = num_method,
                               bounds = self.bounds,
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
                    mu0 = self.inv_transform(log_min_mle.x[0]) #log_min_mle.x[0] - self.transform(0)
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

            if method.upper() == 'SCAR-M-OU':
                if 'z' in kwargs and 'w' in kwargs:
                    pass
                else:
                    z, w = roots_hermite(250)
                    args = w > 1e-3
                    w = w[args]
                    z = z[args]
                    kwargs['z'] = z
                    kwargs['w'] = w

            log_min = minimize(self.mlog_likelihood, _alpha0,
                                    args = (u,
                                            method.upper(),
                                            latent_process_tr, 
                                            M_iterations,
                                            _seed,
                                            dwt,
                                            stationary,
                                            print_path, 
                                            kwargs),
                                    method = num_method,
                                    bounds = bounds,
                                    options={'gtol': tol, 'eps': eps, 'maxfun': 100})

            return log_min
        
        if method.upper() == 'MLE':
            log_min = minimize_mle()
        else:
            log_min = minimize_scar()

        log_min.name = self.name
        log_min.fun = -log_min.fun
        log_min.method = method
        
        if method.upper() not in ['MLE']:
            if dwt is not None:
                latent_process_tr = dwt.shape[1]
            log_min.latent_process_tr = latent_process_tr
            log_min.stationary = stationary

        if method.upper() in ['SCAR-M-OU', 'SCAR-M-DS']:
            log_min.M_iterations = M_iterations

        self.fit_result = log_min
        
        return log_min

    @staticmethod
    def h_unrotated(u, v, r):
        raise NotImplementedError("Not implemented for base class")


    @staticmethod
    def h_inverse_unrotated(u, v, r):
        raise NotImplementedError("Not implemented for base class")
    
    def h(self, u, v, r):
        if self.rotate == 0:
            return self.h_unrotated(u, v, r)
        elif self.rotate == 90:
            _u = 1 - u
            return 1 - self.h_unrotated(_u, v, r)
        elif self.rotate == 180:
            _u = 1 - u
            _v = 1 - v
            return 1 - self.h_unrotated(_u, _v, r)
        elif self.rotate == 270:
            _v = 1 - v
            return self.h_unrotated(u, _v, r)
        else:
            pass

    def h_inverse(self, u, v, r):
        if self.rotate == 0:
            return self.h_inverse_unrotated(u, v, r)
        elif self.rotate == 90:
            _u = 1 - u
            return 1 - self.h_inverse_unrotated(_u, v, r)
        elif self.rotate == 180:
            _u = 1 - u
            _v = 1 - v
            return 1 - self.h_inverse_unrotated(_u, _v, r)
        elif self.rotate == 270:
            _v = 1 - v
            return self.h_inverse_unrotated(u, _v, r)
        else:
            pass
    
    def get_predict(self, size):
        if self.fit_result is None:
            raise ValueError("Please fit the model first")
        
        if self.fit_result.method.upper() == 'MLE':
            state = self.fit_result.x
        else:
            state = latent_process_final_state(self.fit_result.x, 
                                               self.fit_result.method,
                                               size
                                              )
            state = self.transform(state)
        sample = self.get_sample(size, state)
        return sample
