import numpy as np
from numba import jit

@jit(nopython = True)
def jit_normal_marginals(data, window_len):
    count_days = len(data)
    count_instruments = len(data[0])
    res = np.zeros(shape=(count_days, count_instruments, 2))
    for i in range(0, count_days - window_len + 1):
        for j in range(0, count_instruments):
            res[i - 1 + window_len][j][0] = np.mean(data[i: window_len + i][:,j]) 
            res[i - 1 + window_len][j][1] = np.std(data[i: window_len + i][:,j]) 
    return res

@jit(nopython = True)
def jit_normal_rvs(params, N):
    count_instruments = len(params)
    res = np.zeros(shape=(N, count_instruments))
    for i in range(0, count_instruments):
        res[:,i] = np.random.normal(loc = params[i][0], scale=params[i][1], size = N)
    return res
