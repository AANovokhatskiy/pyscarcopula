from pyscarcopula.marginal.normal import jit_normal_marginals, jit_normal_rvs
from pyscarcopula.marginal.hyperbolic import hyperbolic_marginals, hyperbolic_rvs
from pyscarcopula.marginal.stable import stable_marginals, stable_rvs
from typing import Literal

def get_marginals_params_params(data, window_len, method: Literal['normal', 'hyperbolic', 'stable'] = 'normal'):
    print('calc marginals params')
    available_methods = ['normal', 'hyperbolic', 'stable']
    if method == 'normal':
        res = jit_normal_marginals(data, window_len)
    elif method == 'hyperbolic':
        res = hyperbolic_marginals(data, window_len)
    elif method == 'stable':
        res = stable_marginals(data, window_len)
    else:
        raise ValueError(f'given method {method} is not implemented. Available methods: {available_methods}')
    return res

def get_rvs(params, N, method: Literal['normal', 'hyperbolic', 'stable'] = 'normal'):
    available_methods = ['normal', 'hyperbolic', 'stable']
    if method == 'normal':
        res = jit_normal_rvs(params, N)
    elif method == 'hyperbolic':
        res = hyperbolic_rvs(params, N)
    elif method == 'stable':
        res = stable_rvs(params, N)
    else:
        raise ValueError(f'given method {method} is not implemented. Available methods: {available_methods}')
    return res
