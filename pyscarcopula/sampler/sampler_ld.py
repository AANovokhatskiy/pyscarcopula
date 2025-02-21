import numpy as np
from numba import jit, prange
import math


@jit(nopython=True)
def init_state_ld(alpha, latent_process_tr):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]
    x0 = np.ones(latent_process_tr) * mu
    return x0


@jit(nopython=True,  cache = True)
def stationary_state_ld(alpha, latent_process_tr, seed = None):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]

    if seed is None:
        seed = np.random.randint(1, 1000000)
    rng = np.random.seed(seed)
    xs = mu
    s = nu**2 / (2 * theta) * (1 - np.exp(-2 * theta))
    u = np.random.uniform(0, 1, size = latent_process_tr)
    state = np.arctanh(2 * u - 1) * 2 * s + xs
    
    return state


@jit(nopython=True)
def p_sampler_ld(alpha, dwt, init_state = None):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]
    T = len(dwt)
    dt = 1 / (T - 1)
    latent_process_tr = len(dwt[0])
    xt = np.zeros((T, latent_process_tr))
    

    if init_state is None:
        xt[0] = init_state_ld(alpha, latent_process_tr)
    else:
        xt[0] = init_state
    D = nu**2 / 2
    for k in range(1, T):
        t = k / (T - 1)
        xs = mu + (xt[0] - mu) * np.exp(-theta * t)
        xs_dt = -theta * (xs - mu)
        sigma2 = D / theta * (1 - np.exp(- 2 * theta * t))
        sigma2_dt = 2 * D - 2 * theta * sigma2
        y = (xt[k - 1] - xs) / sigma2
        B = np.sqrt(12 * sigma2 * nu)
        A = y * sigma2_dt + xs_dt - B**2/(2 * sigma2) * np.tanh(y/2)
        xt[k] = xt[k - 1] + A * dt + B * dwt[k - 1]

    return xt


# @jit(nopython = True, parallel = True)
def get_avg_p_log_likelihood_ld(data, lambda_data, pdf, transform):
    avg_likelihood = 0

    latent_process_tr = lambda_data.shape[1]

    copula_log_data = np.zeros(latent_process_tr)

    for k in prange(0, latent_process_tr):
        copula_log_data[k] = np.sum(np.log(np.maximum(pdf(data, transform(lambda_data[:,k])), 1e-100)))

    '''trick for calculation large values. calculate e^(sum(log_cop) - corr) instead of e^(sum(log_cop)).
    Do inverse correction at the end of calculations'''
    corr = max(copula_log_data)
    avg_likelihood = np.sum(np.exp(copula_log_data - corr)) / latent_process_tr
    return math.log(avg_likelihood) + corr


# @jit(nopython=True)
def p_jit_mlog_likelihood_ld(alpha: np.array, data: np.array, dwt,
                      pdf: callable, transform: callable, print_path: bool, init_state = None) -> float:
    
    if np.isnan(np.sum(alpha)) == True:
        res = 10**10
        if print_path == True:
            print(alpha, 'incorrect params', res)
        return res

    lambda_data = p_sampler_ld(alpha, dwt, init_state)
    avg_log_likelihood = get_avg_p_log_likelihood_ld(data.T, lambda_data, pdf, transform)
    res = - avg_log_likelihood

    if np.isnan(res) == True:
        if print_path == True:
            res = 10**10
            print(alpha, 'unknown error', res)
    else:
        if print_path == True:
            print(alpha, res)
    return res


# @jit(nopython=True)
def jit_latent_process_conditional_expectation_p_ld(alpha, pobs_data, dwt, 
                                                    pdf, transform, init_state = None):

    latent_process_tr = dwt.shape[1]
    
    T = len(pobs_data)

    lambda_data = transform(p_sampler_ld(alpha, dwt, init_state))
    
    copula_log_data = np.zeros((T, latent_process_tr))

    for k in range(0, latent_process_tr):
        copula_log_data[:,k] = np.log(np.maximum(pdf(pobs_data.T, lambda_data[:,k]), 1e-100))
    
    copula_cs_log_data = np.zeros(latent_process_tr)
    
    latent_process_conditional_sample = np.zeros(T)
    for i in range(0, T):

        copula_cs_log_data += copula_log_data[i]
        xc = np.max(copula_cs_log_data)
        
        temp1 = np.exp(copula_cs_log_data - xc)

        avg_likelihood = np.sum(temp1) / latent_process_tr
        log_lik = np.log(avg_likelihood) + xc

        avg_expectation = np.sum(lambda_data[i] * temp1) / latent_process_tr
        avg_log_expectation = np.log(avg_expectation) + xc

        latent_process_conditional_sample[i] = avg_log_expectation - log_lik

    result = np.exp(latent_process_conditional_sample)
    
    return result