import numpy as np
from scipy.stats import johnsonsu
from joblib import Parallel, delayed

def fit_johnsonsu(data_slice):
    m = 4
    dim = len(data_slice[0])
    fit_result = np.zeros((dim, m))
    for k in range(0, dim):
        fit_result[k] = johnsonsu.fit(data_slice[:,k], method='MLE')
    return fit_result

def johnsonsu_marginals(data, window_len):
    T, dim = data.shape
    m = 4
    res = np.zeros((T, dim, m))
    iters = T - window_len + 1
    fit_results = Parallel(n_jobs=-1)(
        delayed(fit_johnsonsu)(data[i:i + window_len]) for i in range(0, iters)
    )
    for i in range(0, iters):
        idx = i + window_len - 1
        res[idx] = np.array(fit_results[i])

    return res

def johnsonsu_rvs(params, size):
    dim = len(params)
    res = np.zeros(shape=(size, dim))
    for i in range(dim):
        res[:, i] = johnsonsu.rvs(*params[i], size=size)
    return res