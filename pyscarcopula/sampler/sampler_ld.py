import numpy as np
from numba import jit, prange
import math


@jit(nopython=True, cache = True)
def p_sampler_init_state_ld(alpha, latent_process_tr):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]
    x0 = np.ones(latent_process_tr) * mu
    return x0


@jit(nopython=True, cache = True)
def p_sampler_ld(alpha, dwt, init_state = None):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]
    T = len(dwt)
    dt = 1 / (T - 1)
    latent_process_tr = len(dwt[0])
    xt = np.zeros((T, latent_process_tr))
    

    if init_state is None:
        xt[0] = p_sampler_init_state_ld(alpha, latent_process_tr)
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


@jit(nopython=True, cache = True)
def p_sampler_one_step_ld(alpha, dwt, dt, init_state):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]
    D = nu**2 / 2
    latent_process_tr = len(dwt)
    if init_state is None:
        x0 = p_sampler_init_state_ld(alpha, latent_process_tr)
    else:
        x0 = init_state

    t = 1 * dt
    xs = mu + (x0 - mu) * np.exp(-theta * t)
    xs_dt = -theta * (xs - mu)
    sigma2 = D / theta * (1 - np.exp(- 2 * theta * t))
    sigma2_dt = 2 * D - 2 * theta * sigma2
    y = (x0 - xs) / sigma2
    B = np.sqrt(12 * sigma2 * nu)
    A = y * sigma2_dt + xs_dt - B**2/(2 * sigma2) * np.tanh(y/2)
    x1 = x0 + A * dt + B * dwt

    return x1


@jit(nopython=True, cache = True)
def p_sampler_one_step_ld_rng(alpha, latent_process_tr, random_state, dt, init_state):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]
    D = nu**2 / 2

    sqrt_dt = np.sqrt(dt)
    rng = np.random.seed(random_state)

    if init_state is None:
        x0 = p_sampler_init_state_ld(alpha, latent_process_tr)
    else:
        x0 = init_state

    dwt = sqrt_dt * np.random.normal(0 , 1 , size = latent_process_tr)

    t = 1 * dt
    xs = mu + (x0 - mu) * np.exp(-theta * t)
    xs_dt = -theta * (xs - mu)
    sigma2 = D / theta * (1 - np.exp(- 2 * theta * t))
    sigma2_dt = 2 * D - 2 * theta * sigma2
    y = (x0 - xs) / sigma2
    B = np.sqrt(12 * sigma2 * nu)
    A = y * sigma2_dt + xs_dt - B**2/(2 * sigma2) * np.tanh(y/2)
    x1 = x0 + A * dt + B * dwt

    return x1


@jit(nopython=True, cache = True)
def p_sampler_no_hist_ld(alpha, dwt, dt, init_state):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]
    D = nu**2 / 2
    T = len(dwt)

    latent_process_tr = len(dwt[0])
    if init_state is None:
        x0 = p_sampler_init_state_ld(alpha, latent_process_tr)
    else:
        x0 = init_state

    xt_km1 = x0

    for k in range(1, T):
        t = k * dt
        xs = mu + (x0 - mu) * np.exp(-theta * t)
        xs_dt = -theta * (xs - mu)
        sigma2 = D / theta * (1 - np.exp(- 2 * theta * t))
        sigma2_dt = 2 * D - 2 * theta * sigma2
        y = (xt_km1 - xs) / sigma2
        B = np.sqrt(12 * sigma2 * nu)
        A = y * sigma2_dt + xs_dt - B**2/(2 * sigma2) * np.tanh(y/2)
        xt_k = xt_km1 + A * dt + B * dwt[k - 1]

        xt_km1 = xt_k

    return xt_k


@jit(nopython=True, cache = True)
def p_sampler_no_hist_ld_rng(alpha, latent_process_tr, random_states_sequence, dt, init_state):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]
    D = nu**2 / 2
    T = len(random_states_sequence)
    sqrt_dt = np.sqrt(dt)

    if init_state is None:
        x0 = p_sampler_init_state_ld(alpha, latent_process_tr)
    else:
        x0 = init_state

    xt_km1 = x0

    for k in range(1, T):
        rng = np.random.seed(random_states_sequence[k - 1])
        dwt = sqrt_dt * np.random.normal(0 , 1 , size = latent_process_tr)
        t = k * dt
        xs = mu + (xt_km1 - mu) * np.exp(-theta * t)
        xs_dt = -theta * (xs - mu)
        sigma2 = D / theta * (1 - np.exp(- 2 * theta * t))
        sigma2_dt = 2 * D - 2 * theta * sigma2
        y = (xt_km1 - xs) / sigma2
        B = np.sqrt(12 * sigma2 * nu)
        A = y * sigma2_dt + xs_dt - B**2/(2 * sigma2) * np.tanh(y/2)
        xt_k = xt_km1 + A * dt + B * dwt

        xt_km1 = xt_k
    return xt_k


@jit(nopython = True, parallel = True, cache = True)
def get_avg_p_log_likelihood_ld(data, lambda_data, latent_process_tr, pdf, transform):
    avg_likelihood = 0
    copula_log_data = np.zeros(latent_process_tr)

    for k in prange(0, latent_process_tr):
        copula_log_data[k] = np.sum(np.log(np.maximum(pdf(data, transform(lambda_data[:,k])), 1e-100)))

    '''trick for calculation large values. calculate e^(sum(log_cop) - corr) instead of e^(sum(log_cop)).
    Do inverse correction at the end of calculations'''
    corr = max(copula_log_data)
    avg_likelihood = np.sum(np.exp(copula_log_data - corr)) / latent_process_tr
    return math.log(avg_likelihood) + corr


@jit(nopython=True, cache = True)
def p_jit_mlog_likelihood_ld(alpha: np.array, data: np.array, dwt, latent_process_tr: int, 
                      print_path: bool, pdf: callable, transform: callable, init_state = None) -> float:
    if np.isnan(np.sum(alpha)) == True:
        res = 10**10
        if print_path == True:
            print(alpha, 'incorrect params', res)
        return res

    lambda_data = p_sampler_ld(alpha, dwt, init_state)
    avg_log_likelihood = get_avg_p_log_likelihood_ld(data.T, lambda_data, latent_process_tr, pdf, transform)
    res = - avg_log_likelihood

    if np.isnan(res) == True:
        if print_path == True:
            res = 10**10
            print(alpha, 'unknown error', res)
    else:
        if print_path == True:
            print(alpha, res)
    return res


#@jit(nopython=True, cache = True)
def jit_latent_process_conditional_expectation_p_ld(pdf, transform, pobs_data, alpha, latent_process_tr):
    T = len(pobs_data)
    dwt = np.random.normal(0, 1, size = (T, latent_process_tr)) * np.sqrt(1/T)

    lambda_data = transform(p_sampler_ld(alpha, dwt))
    
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