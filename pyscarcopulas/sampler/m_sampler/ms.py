import numpy as np
from numba import jit, prange

@jit(nopython=True)
def linear_least_squares(matA: np.array, matB: np.array) -> np.array:
    '''input  Ax = b
       output x = (A.T * A) ^ (-1) * A.T * b
    '''
    return np.linalg.inv( matA.T @ matA ) @ matA.T @ matB 

@jit(nopython=True)
def log_xi(x: np.array, params: np.array, xm1: np.array) -> np.array:
    gamma, delta, nu = params[0], params[1], params[2]
    a1, a2 = x[0], x[1]
    var = nu**2 / ( 1 - 2*nu**2 * a2)
    mu = var * ( (gamma + delta * xm1) / nu**2 + a1)
    return ( mu **2 / (2 * var) - 1 / (2 * nu**2 ) * (gamma + delta * xm1)**2 )

@jit(nopython=True, parallel = True)
def get_avg_m_log_likelihood(omega, data, a, lambda_data, log_xi_data, n_tr, pdf, transform):
    nu = omega[2]
    var = nu**2/ ( 1 - 2 * nu**2 * a[:,2])

    if (np.sum(np.abs(var)) - np.sum(var)) > 10**(-7):
        res = 10000
        print(omega, 'negative var', res)
        return res
    
    log_likelihood = np.zeros(n_tr)
    for k in prange(0, n_tr):
        copula_log_data = np.log(pdf(data.T , transform(lambda_data[:,k]) ))
        log_likelihood[k] = np.sum(copula_log_data + log_xi_data[:,k] - ( a[:,1] * lambda_data[:,k] + a[:,2] * lambda_data[:,k]**2 )\
                                         -0.5*np.log(nu**2)+0.5*np.log(var) )
    corr = max(log_likelihood)
    avg_likelihood = np.sum(np.exp(log_likelihood - corr)) / n_tr
    res = np.log(avg_likelihood) + corr
    return res