import numpy as np

def inverse_CDF(u, x):
    res = np.quantile(x, u, method = 'median_unbiased')
    return res

def inverse_CDF_matrix(u, x):
    dim = u.shape[1]
    res = np.zeros_like(u)
    for k in range(0, dim):
        res[:,k] = inverse_CDF(u[:,k], x[:,k])
    return res