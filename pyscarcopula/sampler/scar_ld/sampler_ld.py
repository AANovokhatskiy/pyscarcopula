import numpy as np
from numba import jit
import math

@jit(nopython = True)
def get_avg_p_log_likelihood(data, lambda_data, n_tr, pdf, transform):
    avg_likelihood = 0
    copula_log_data = np.zeros(n_tr)

    for k in range(0, n_tr):
        copula_log_data[k] = np.sum(np.log(pdf(data, transform(lambda_data[:,k]))))

    '''trick for calculation large values. calculate e^(sum(log_cop) - corr) instead of e^(sum(log_cop)).
    Do inverse correction at the end of calculations'''
    corr = max(copula_log_data)
    avg_likelihood = np.sum(np.exp(copula_log_data - corr)) / n_tr
    return math.log(avg_likelihood) + corr

@jit(nopython=True)
def p_sampler_ld(alpha, dwt):
    mu, theta, nu = alpha[0], alpha[1], alpha[2]
    T = len(dwt)
    dt = 1 / T
    xt = np.zeros(dwt.shape)
    x0 = mu
    xt[0] = x0
    D = nu**2 / 2
    for k in range(1, T):
        t = k * dt
        xs = (x0 - mu) * np.exp(-theta * t)
        xs_dt = -theta * xs
        sigma2 = D / theta * (1 - np.exp(- 2 * theta * t))
        sigma2_dt = 2 * D - 2 * theta * sigma2
        y = (xt[k - 1] - mu - xs) / sigma2
        A = y * sigma2_dt + xs_dt - 1/2 * nu**2 * np.tanh(y/2)
        B = np.sqrt(nu**2 * sigma2)
        xt[k] = xt[k - 1] + A * dt + B * dwt[k]
    return xt

@jit(nopython=True)
def p_sampler_no_hist_ld(alpha, T, N_mc):
    mu, theta, nu = alpha[0], alpha[1], alpha[2]
    dt = 1 / T
    sqrt_dt = np.sqrt(dt)
    x0 = mu
    D = nu**2 / 2
    xt_km1 = np.ones(N_mc) * mu
    for k in range(1, T):
        dwt = sqrt_dt * np.random.normal(0, 1, size =  N_mc)
        t = k * dt
        xs = (x0 - mu) * np.exp(-theta * t)
        xs_dt = -theta * xs
        sigma2 = D / theta * (1 - np.exp(- 2 * theta * t))
        sigma2_dt = 2 * D - 2 * theta * sigma2
        y = (xt_km1 - mu - xs) / sigma2
        A = y * sigma2_dt + xs_dt - 1/2 * nu**2 * np.tanh(y/2)
        B = np.sqrt(nu**2 * sigma2)
        xt_k = xt_km1 + A * dt + B * dwt
    return xt_k

@jit(nopython=True)
def p_jit_mlog_likelihood_ld(alpha: np.array, data: np.array, dwt, n_tr: int, 
                      print_path: bool, pdf: callable, transform: callable) -> float:
    if np.isnan(np.sum(alpha)) == True:
        res = 10000
        if print_path == True:
            print(alpha, 'incorrect params', res)
        return res

    lambda_data = p_sampler_ld(alpha, dwt)
    avg_log_likelihood = get_avg_p_log_likelihood(data.T, lambda_data, n_tr, pdf, transform)
    res = - avg_log_likelihood

    if np.isnan(res) == True:
        if print_path == True:
            res = 10000
            print(alpha, 'unknown error', res)
    else:
        if print_path == True:
            print(alpha, res)
    return res



