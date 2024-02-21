import numpy as np
from scipy.stats import ks_2samp
from scipy.stats import cramervonmises_2samp
from scipy.stats import genhyperbolic
from multiprocessing import Pool, cpu_count
from functools import partial
from itertools import starmap

from pyscarcopula.aux_functions.funcs import transform_pseudo_obs


# def get_hyperbolic_params(data):
#     n = len(data)
#     out_dict = dict()
#     ks = np.zeros(n)
#     cvm = np.zeros(n)
#     fit_result = []
#     for i in range(0, n):
#         fit_result_i = genhyperbolic.fit(data[i], method = 'MLE')
#         r = genhyperbolic.rvs(*fit_result_i, size=1000, random_state=None)
#         #print(ks_2samp(data[i], r, alternative='two-sided', method='auto'))
#         #print(cramervonmises_2samp(data[i], r, method='auto'))
#         fit_result.append(fit_result_i)
#     out_dict['fit_result'] = fit_result
#     out_dict['KS_test'] = ks
#     out_dict['CVM_test'] = cvm
#     return out_dict


def get_hyperbolic_params(data):
    fit_result = genhyperbolic.fit(data, method = 'MLE')
    return fit_result

def get_hyperbolic_params_parallel(data):
    res = []
    p = cpu_count()
    with Pool(processes = p) as pool:
        fit_result = pool.map(get_hyperbolic_params, data)
        res.append(fit_result)
    return res[0]

def get_hyperbolic_rvs(params, N, random_state):
    rvs = genhyperbolic.rvs(*params, size=N, random_state=random_state)
    return rvs

def get_hyperbolic_rvs_parallel(params, N, random_state=None):
    if N > 10**5:
        '''parallel branch'''
        res = []
        p = cpu_count()
        with Pool(processes = p) as pool:
            rvs = pool.map(partial(get_hyperbolic_rvs, N = N, random_state = random_state), params)
            res.append(rvs)
        return np.array(res[0])
    else:
        '''no parallel computations at small N'''
        count_instruments = len(params)
        res = np.zeros((count_instruments, N))
        for k in range(0, count_instruments):
            res[k] = get_hyperbolic_rvs(params[k], N, random_state)
        return res

def get_hyperbolic_ppf(rvs, params, method):
    ppf = []
    if method == 'direct':
        ppf = genhyperbolic.ppf(rvs, *params)
    if method == 'pobs':
        ppf = transform_pseudo_obs(rvs)
    return ppf

def get_hyperbolic_ppf_parallel(rvs, params, method = 'pobs'):
    N = rvs.shape[1]
    if N > 10**5:
        '''parallel branch'''
        res = []
        p = cpu_count()
        with Pool(processes = p) as pool:
            ppf = pool.starmap(partial(get_hyperbolic_ppf, method = method), zip(rvs, params))
            res.append(ppf)
        return np.array(res[0])
    else:
        '''no parallel computations at small N'''
        count_instruments = len(params)
        res = np.zeros((count_instruments, N))
        for k in range(0, count_instruments):
            res[k] = get_hyperbolic_ppf(rvs[k], params[k], method)
        return res