import numpy as np

def inverse_CDF(copula_sample, marginal_sample):
    res = np.quantile(marginal_sample, copula_sample, method = 'median_unbiased')
    return res

def inverse_CDF_matrix(copula_sample, marginal_sample):
    dim = copula_sample.shape[1]
    res = np.zeros_like(copula_sample)

    for k in range(0, dim):
        res[:,k] = inverse_CDF(copula_sample[:,k], marginal_sample[:,k])
    return res