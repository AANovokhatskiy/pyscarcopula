import numpy as np
from numba import jit, prange
import math

@jit(nopython = True, cache = True)
def p_sampler_ds(alpha: np.array, crns: np.array) -> np.array:
    '''generate lambda(t) from natural (p) sampler'''
    gamma, delta, nu = alpha[0], alpha[1], alpha[2]
    T = len(crns)
    res = np.zeros(crns.shape)
    res[0] = gamma / ( 1 - delta)
    for i in range(1, T):
        res[i] = gamma + delta * res[i - 1] + nu * crns[i]
    return res

@jit(nopython = True, cache = True)
def p_sampler_no_hist_ds(alpha: np.array, T: int, N_mc: int) -> np.array:
    '''generate lambda(t) from natural (p) sampler'''
    gamma, delta, nu = alpha[0], alpha[1], alpha[2]
    res_im1 = np.ones(N_mc) * gamma / ( 1 - delta)
    for i in range(1, T):
        crns = np.random.normal(0, 1, size =  N_mc)
        res_i = gamma + delta * res_im1 + nu * crns
        res_im1 = res_i
    return res_i


@jit(nopython = True, cache = True)
def get_avg_p_log_likelihood(data, lambda_data, pdf, transform):
    avg_likelihood = 0

    latent_process_tr = lambda_data.shape[1]

    copula_log_data = np.zeros(latent_process_tr)

    for k in prange(0, latent_process_tr):
        copula_log_data[k] = np.sum(np.log(pdf(data, transform(lambda_data[:,k]))))

    '''trick for calculation large values. calculate e^(sum(log_cop) - corr) instead of e^(sum(log_cop)).
    Do inverse correction at the end of calculations'''
    corr = max(copula_log_data)
    avg_likelihood = np.sum(np.exp(copula_log_data - corr)) / latent_process_tr
    return math.log(avg_likelihood) + corr

@jit(nopython = True, cache = True)
def p_jit_mlog_likelihood_ds(alpha: np.array, data: np.array, crns: np.array,
                      pdf: callable, transform: callable, print_path: bool) -> float:
    
    '''initial data check'''
    if np.isnan(np.sum(alpha)) == True:
        res = 10000
        if print_path == True:
            print(alpha, 'incorrect params', res)
        return res
    
    if np.abs(alpha[2]) > 1 or np.abs(alpha[1]) > 1:
        res = 10000
        if print_path == True:
            print(alpha, 'params is out of bounds', res)
        return res

    lambda_data = p_sampler_ds(alpha, crns)
    avg_log_likelihood = get_avg_p_log_likelihood(data.T, lambda_data, pdf, transform)
    res = - avg_log_likelihood
    if np.isnan(res) == True:
        if print_path == True:
            print(alpha, 'unknown error', res)
    else:
        if print_path == True:
            print(alpha, res)
    return res

@jit(nopython=True, cache = True)
def linear_least_squares(matA: np.array, matB: np.array) -> np.array:
    '''input  Ax = b
       output x = (A.T * A) ^ (-1) * A.T * b
    '''
    return np.linalg.inv( matA.T @ matA ) @ matA.T @ matB 

@jit(nopython = True, cache = True)
def m_sampler_ds(alpha: np.array, a: np.array, crns: np.array) -> np.array:
    '''generate lambda(t) from importance (m) sampler using a1(t), a2(t) parameters from previos iterations'''
    gamma, delta, nu = alpha[0], alpha[1], alpha[2]
    a1, a2 = a[:,1], a[:,2]
    T = len(crns)
    res = np.zeros(crns.shape)
    res[0] = gamma / ( 1 - delta)
    var = nu**2 / ( 1 - 2 * nu**2 * a2)
    p1, p2, p3 = var * ( gamma/ nu**2 + a1), var * delta / nu**2, np.sqrt(var)
    for i in range(1, T):
        res[i] = p1[i] + p2[i] * res[i - 1] + p3[i] * crns[i]
    return res

@jit(nopython = True, cache = True)
def m_sampler_no_hist_ds(alpha: np.array, a: np.array, T: int, N_mc: int) -> np.array:
    '''generate lambda(t) from importance (m) sampler using a1(t), a2(t) parameters from previos iterations'''
    gamma, delta, nu = alpha[0], alpha[1], alpha[2]
    a1, a2 = a[:,1], a[:,2]
    res_im1 = np.ones(N_mc) * gamma / ( 1 - delta)
    var = nu**2 / ( 1 - 2 * nu**2 * a2)
    p1, p2, p3 = var * ( gamma/ nu**2 + a1), var * delta / nu**2, np.sqrt(var)
    for i in range(1, T):
        crns = np.random.normal(0, 1, size =  N_mc)
        res_i = p1[i] + p2[i] * res_im1 + p3[i] * crns
        res_im1 = res_i
    return res_i

@jit(nopython=True, cache = True)
def log_xi(x: np.array, params: np.array, xm1: np.array) -> np.array:
    gamma, delta, nu = params[0], params[1], params[2]
    a1, a2 = x[0], x[1]
    var = nu**2 / ( 1 - 2*nu**2 * a2)
    mu = var * ( (gamma + delta * xm1) / nu**2 + a1)
    return ( mu **2 / (2 * var) - 1 / (2 * nu**2 ) * (gamma + delta * xm1)**2 )


@jit(nopython=True, cache = True)
def get_avg_m_log_likelihood(omega, data, a, lambda_data, log_xi_data, pdf, transform):

    latent_process_tr = lambda_data.shape[1]

    nu = omega[2]
    var = nu**2/ ( 1 - 2 * nu**2 * a[:,2])

    if (np.sum(np.abs(var)) - np.sum(var)) > 10**(-7):
        res = 10000
        print(omega, 'negative var', res)
        return res
    
    log_likelihood = np.zeros(latent_process_tr)
    for k in prange(0, latent_process_tr):
        copula_log_data = np.log(pdf(data.T , transform(lambda_data[:,k]) ))
        log_likelihood[k] = np.sum(copula_log_data + log_xi_data[:,k] - ( a[:,1] * lambda_data[:,k] + a[:,2] * lambda_data[:,k]**2 )\
                                         -0.5 * np.log(nu**2) + 0.5 * np.log(var) )
    corr = max(log_likelihood)
    avg_likelihood = np.sum(np.exp(log_likelihood - corr)) / latent_process_tr
    res = np.log(avg_likelihood) + corr
    return res

@jit(nopython = True, cache = True)
def m_jit_mlog_likelihood_ds(alpha: np.array, data: np.array, crns: np.array, m_iters: int, 
                          pdf: callable, transform: callable, print_path: bool) -> float:
    
    latent_process_tr = crns.shape[1]
    
    '''initial data check'''
    if np.isnan(np.sum(alpha)) == True:
        res = 10000
        if print_path == True:
            print(alpha, 'incorrect params', res)
        return res
    
    if np.abs(alpha[2]) > 1 or np.abs(alpha[1]) > 1:
        res = 10000
        if print_path == True:
            print(alpha, 'params is out of bounds', res)
        return res

    '''set initial parameters'''
    T = len(data)
    a = np.zeros( (T, 3) )
    log_xi_data = np.zeros((T, latent_process_tr))

    for i in range(0, m_iters):
        if i == 0:
            '''generate lambda(t) from natural (p) sampler'''
            lambda_data = p_sampler_ds(alpha, crns)
        else:
            '''generate lambda(t) from importance (m) sampler using a1(t), a2(t) parameters from previos iterations'''
            lambda_data = m_sampler_ds(alpha, a, crns)
            if np.isnan(np.sum(lambda_data)) == True:
                res = 10000
                if print_path == True:
                    print(alpha, 'm sampler nan', res)
                return res
        '''solve linear least-squares problem for search optimal parameters [a] for each t in (T, 0)'''
        for t in range(T - 1, 0 , -1):
            copula_log_data = np.sum(np.log(pdf(data[t], transform(lambda_data[t]) )) )
            
            '''set A and b in LS problem Ax = b'''
            A = np.dstack( (np.ones(latent_process_tr) , lambda_data[t] , lambda_data[t]**2 ) )[0]           
            b = copula_log_data + log_xi_data[t]
            #print(lambda_data)
            '''solve problem Ax = b'''
            try:
                a[t] = linear_least_squares(A, b)
            except:
                res = 10000
                if print_path == True:
                    print(alpha, 'ls problem fail', res)
                return res
            #a[t] = linear_least_squares(A, b)
            log_xi_data[t - 1] = log_xi(a[t][1:3], alpha, lambda_data[t - 1] )

    avg_log_likelihood = get_avg_m_log_likelihood(alpha, data, a, lambda_data, log_xi_data, pdf, transform)
    res = - avg_log_likelihood

    if np.isnan(res) == True:
        if print_path == True:
            print(alpha, 'unknown error', res)
    else:
        if print_path == True:
            print(alpha, res)

    return res
