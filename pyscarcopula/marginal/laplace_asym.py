import numpy as np
from scipy.stats import laplace_asymmetric
from joblib import Parallel, delayed

def fit_laplace_asymmetric(data_slice):
    m = 3
    dim = len(data_slice[0])
    fit_result = np.zeros((dim, m))
    for k in range(0, dim):
        fit_result[k] = laplace_asymmetric.fit(data_slice[:,k], method='MLE')
    return fit_result

def laplace_asymmetric_marginals(data, window_len):
    T, dim = data.shape
    m = 3
    res = np.zeros((T, dim, m))
    iters = T - window_len + 1
    fit_results = Parallel(n_jobs=-1)(
        delayed(fit_laplace_asymmetric)(data[i:i + window_len]) for i in range(0, iters)
    )
    for i in range(0, iters):
        idx = i + window_len - 1
        res[idx] = np.array(fit_results[i])

    return res

def laplace_asymmetric_rvs(params, size):
    dim = len(params)
    res = np.zeros(shape=(size, dim))
    for i in range(dim):
        res[:, i] = laplace_asymmetric.rvs(*params[i], size=size)
    return res