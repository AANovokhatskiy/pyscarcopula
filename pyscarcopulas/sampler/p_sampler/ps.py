import numpy as np
from numba import jit, prange
import math

@jit(nopython = True, parallel = True)
def get_avg_p_log_likelihood(data, lambda_data, n_tr, pdf, transform):
    avg_likelihood = 0
    copula_log_data = np.zeros(n_tr)

    for k in prange(0, n_tr):
        copula_log_data[k] = np.sum(np.log(pdf(data, transform(lambda_data[:,k]))))

    '''trick for calculation large values. calculate e^(sum(log_cop) - corr) instead of e^(sum(log_cop)).
    Do inverse correction at the end of calculations'''
    corr = max(copula_log_data)
    avg_likelihood = np.sum(np.exp(copula_log_data - corr)) / n_tr
    return math.log(avg_likelihood) + corr