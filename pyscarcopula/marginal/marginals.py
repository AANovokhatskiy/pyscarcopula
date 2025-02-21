from pyscarcopula.marginal.normal import jit_normal_marginals, jit_normal_rvs
from pyscarcopula.marginal.hyperbolic import hyperbolic_marginals, hyperbolic_rvs
from pyscarcopula.marginal.stable import stable_marginals, stable_rvs
from pyscarcopula.marginal.logistic import genlogistic_marginals, genlogistic_rvs
from pyscarcopula.marginal.johnsonsu import johnsonsu_marginals, johnsonsu_rvs
from pyscarcopula.marginal.laplace_asym import laplace_asymmetric_marginals, laplace_asymmetric_rvs
from scipy.stats import norm, laplace_asymmetric, johnsonsu, genlogistic
from pyscarcopula.marginal.inverse_CDF import inverse_CDF
from pyscarcopula.auxiliary.funcs import jit_pobs

from typing import Literal
import numpy as np

def fit_marginal(data,
        method: Literal['normal', 'hyperbolic', 'stable', 'logistic', 'johnsonsu', 'laplace'] = 'normal',
        window_len = None
    ):
    if window_len == None:
        window_len = len(data)

    available_methods = ['normal', 'hyperbolic', 'stable', 'logistic', 'johnsonsu', 'laplace']
    if method == 'normal':
        res = jit_normal_marginals(data, window_len)
    elif method == 'hyperbolic':
        res = hyperbolic_marginals(data, window_len)
    elif method == 'stable':
        res = stable_marginals(data, window_len)
    elif method == 'logistic':
        res = genlogistic_marginals(data, window_len)
    elif method == 'johnsonsu':
        res = johnsonsu_marginals(data, window_len)
    elif method == 'laplace':
        res = laplace_asymmetric_marginals(data, window_len)
    else:
        raise ValueError(f'given method {method} is not implemented. Available methods: {available_methods}')
    
    if window_len == len(data):
        res = res[-1]
    return res

def rvs(params, N, 
        method: Literal['normal', 'hyperbolic', 'stable', 'logistic', 'johnsonsu', 'laplace'] = 'normal'):
    available_methods = ['normal', 'hyperbolic', 'stable', 'logistic', 'johnsonsu', 'laplace']
    if method == 'normal':
        res = jit_normal_rvs(params, N)
    elif method == 'hyperbolic':
        res = hyperbolic_rvs(params, N)
    elif method == 'stable':
        res = stable_rvs(params, N)
    elif method == 'logistic':
        res = genlogistic_rvs(params, N)
    elif method == 'johnsonsu':
        res = johnsonsu_rvs(params, N)
    elif method == 'laplace':
        res = laplace_asymmetric_rvs(params, N)
    else:
        raise ValueError(f'given method {method} is not implemented. Available methods: {available_methods}')
    return res

def ppf(u, 
        method: Literal['normal', 'hyperbolic', 'stable', 'logistic', 'johnsonsu', 'laplace'] = 'normal',
        params = None, x = None):
    
    available_methods = ['normal', 'hyperbolic', 'stable', 'logistic', 'johnsonsu', 'laplace']
    
    dim = u.shape[1]
    res = np.zeros_like(u)

    if method == 'normal':
        for k in range(0, dim):
            res[:,k] = norm.ppf(u[:,k], *params[k])

    elif method == 'hyperbolic':
        for k in range(0, dim):
            res[:,k] = inverse_CDF(u[:,k], x[:,k])

    elif method == 'stable':
        for k in range(0, dim):
            res[:,k] = inverse_CDF(u[:,k], x[:,k])

    elif method == 'logistic':
        for k in range(0, dim):
            res[:,k] = genlogistic.ppf(u[:,k], *params[k])

    elif method == 'johnsonsu':
        for k in range(0, dim):
            res[:,k] = johnsonsu.ppf(u[:,k], *params[k])

    elif method == 'laplace':
        for k in range(0, dim):
            res[:,k] = laplace_asymmetric.ppf(u[:,k], *params[k])

    else:
        raise ValueError(f'given method {method} is not implemented. Available methods: {available_methods}')
    
    return res

def cdf(x, 
        method: Literal['normal', 'hyperbolic', 'stable', 'logistic', 'johnsonsu', 'laplace'] = 'normal',
        params = None):
    
    available_methods = ['normal', 'hyperbolic', 'stable', 'logistic', 'johnsonsu', 'laplace']
    
    dim = x.shape[1]
    res = np.zeros_like(x)

    if method == 'normal':
        for k in range(0, dim):
            res[:,k] = norm.cdf(x[:,k], *params[k])

    elif method == 'hyperbolic':
        res = jit_pobs(x)

    elif method == 'stable':
        res = jit_pobs(x)

    elif method == 'logistic':
        for k in range(0, dim):
            res[:,k] = genlogistic.cdf(x[:,k], *params[k])

    elif method == 'johnsonsu':
        for k in range(0, dim):
            res[:,k] = johnsonsu.cdf(x[:,k], *params[k])

    elif method == 'laplace':
        for k in range(0, dim):
            res[:,k] = laplace_asymmetric.cdf(x[:,k], *params[k])

    else:
        raise ValueError(f'given method {method} is not implemented. Available methods: {available_methods}')
    
    return res