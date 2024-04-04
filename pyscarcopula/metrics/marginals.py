from pyscarcopula.marginal.normal import jit_normal_marginals, jit_normal_rvs
from pyscarcopula.marginal.hyperbolic import hyperbolic_marginals, hyperbolic_rvs

def get_marginals_params_params(data, window_len, method):
    print('calc marginals_params')
    available_methods = ['normal', 'hyperbolic']
    if method == 'normal':
        res = jit_normal_marginals(data, window_len)
    elif method == 'hyperbolic':
        res = hyperbolic_marginals(data, window_len)
    else:
        raise ValueError(f'given method {method} is not implemented. Available methods: {available_methods}')
    return res

def get_rvs(params, N, method):
    available_methods = ['normal', 'hyperbolic']
    if method == 'normal':
        res = jit_normal_rvs(params, N)
    elif method == 'hyperbolic':
        res = hyperbolic_rvs(params, N)
    else:
        raise ValueError(f'given method {method} is not implemented. Available methods: {available_methods}')
    return res
