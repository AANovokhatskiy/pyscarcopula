import numpy as np
from numba import jit, prange
import math
from typing import Literal


@jit(nopython=True, parallel = True, cache = True)
def p_sampler_ou(alpha, dwt, init_state = None):
    alpha1, alpha2, alpha3 = alpha[0], alpha[1], alpha[2]
    T = len(dwt)
    dt = 1 / T
    xt = np.zeros(dwt.shape)
    mu = -alpha1 / alpha2
    if init_state is None:
        xt[0] = mu
    else:
        xt[0] = init_state
    for k in range(1, T):
        xt[k] = xt[k - 1] + (alpha1 + alpha2 * xt[k - 1]) * dt + alpha3 * dwt[k]
    return xt


@jit(nopython=True, cache = True)
def p_sampler_init_state_ou(alpha, latent_process_tr):
    alpha1, alpha2, alpha3 = alpha[0], alpha[1], alpha[2]
    mu = -alpha1 / alpha2
    x0 = np.ones(latent_process_tr) * mu
    return x0


@jit(nopython=True, parallel = True, cache = True)
def p_sampler_one_step_ou(alpha, dwt, dt, init_state):
    alpha1, alpha2, alpha3 = alpha[0], alpha[1], alpha[2]
    x1 = init_state + (alpha1 + alpha2 * init_state) * dt + alpha3 * dwt
    return x1


@jit(nopython=True, parallel = True, cache = True)
def p_sampler_one_step_ou_rng(alpha, random_state, dt, init_state):
    latent_process_tr = len(init_state)
    alpha1, alpha2, alpha3 = alpha[0], alpha[1], alpha[2]
    sqrt_dt = np.sqrt(dt)
    rng = np.random.seed(random_state)
    dwt = sqrt_dt * np.random.normal(0 , 1 , size = latent_process_tr)
    x1 = init_state + (alpha1 + alpha2 * init_state) * dt + alpha3 * dwt
    return x1


@jit(nopython=True, parallel = True, cache = True)
def p_sampler_no_hist_ou(alpha, dwt, dt, init_state):
    alpha1, alpha2, alpha3 = alpha[0], alpha[1], alpha[2]
    T = len(dwt)
    xt_km1 = init_state

    for k in range(1, T):
        xt_k = xt_km1 + (alpha1 + alpha2 * xt_km1) * dt + alpha3 * dwt[k]
        xt_km1 = xt_k
    return xt_k


@jit(nopython=True, parallel = True, cache = True)
def p_sampler_no_hist_ou_rng(alpha, random_states_sequence, dt, init_state):
    alpha1, alpha2, alpha3 = alpha[0], alpha[1], alpha[2]
    latent_process_tr = len(init_state)
    T = len(random_states_sequence)
    sqrt_dt = np.sqrt(dt)
    mu = -alpha1 / alpha2
    xt_km1 = init_state

    for k in range(1, T):
        rng = np.random.seed(random_states_sequence[k])
        dwt = sqrt_dt * np.random.normal(0 , 1 , size = latent_process_tr)
        xt_k = xt_km1 + (alpha1 + alpha2 * xt_km1) * dt + alpha3 * dwt
        xt_km1 = xt_k
    return xt_k


@jit(nopython = True, cache = True, parallel = True)
def get_avg_p_log_likelihood_ou(data, lambda_data, latent_process_tr, pdf, transform):
    avg_likelihood = 0
    copula_log_data = np.zeros(latent_process_tr)

    for k in prange(0, latent_process_tr):
        copula_log_data[k] = np.sum(np.log(np.maximum(pdf(data, transform(lambda_data[:,k])), 1e-100)))

    '''trick for calculation large values. calculate e^(sum(log_cop) - corr) instead of e^(sum(log_cop)).
    Do inverse correction at the end of calculations'''
    corr = max(copula_log_data)
    avg_likelihood = np.sum(np.exp(copula_log_data - corr)) / latent_process_tr
    return math.log(avg_likelihood) + corr


@jit(nopython = True, cache = True)
def p_jit_mlog_likelihood_ou(alpha: np.array, data: np.array, dwt: np.array, latent_process_tr: int,
                      print_path: bool, pdf: callable, transform: callable, init_state: np.array = None) -> float:
    
    '''initial data check'''
    if np.isnan(np.sum(alpha)) == True:
        res = 10**10
        if print_path == True:
            print(alpha, 'incorrect params', res)
        return res
    
    lambda_data = p_sampler_ou(alpha, dwt, init_state)
    avg_log_likelihood = get_avg_p_log_likelihood_ou(data.T, lambda_data, latent_process_tr, pdf, transform)
    res = - avg_log_likelihood
    if np.isnan(res) == True:
        if print_path == True:
            print(alpha, 'unknown error', res)
    else:
        if print_path == True:
            print(alpha, res)
    return res


@jit(nopython=True, cache = True)
def linear_least_squares(matA: np.array, matB: np.array, alpha: float = 0.0) -> np.array:
    '''Ridge regression
       Input  Ax = b
       Output x = (A.T * A + alpha * I) ^ (-1) * A.T * b
    '''
    I = np.identity(len(matA[0]))
    I[0][0] = 0
    return np.linalg.inv( matA.T @ matA + alpha * I) @ matA.T @ matB 


@jit(nopython=True, cache = True)
def bounded_polynom_fit(x, y, dim, type: Literal['two-sided', 'left-sided', 'right-sided', 'no bounds'], ridge_alpha = 0.0):
    if type == 'two-sided':
        x0 = x[0]
        x1 = x[-1]
        y0 = y[0]
        y1 = y[-1]
        c0 = (y0 * x1 - y1 * x0) / (x1 - x0)
        c1 = (y1 - y0) / (x1 - x0)
        d1 = -x0 - x1
        d2 = x0 * x1
        A = np.zeros((len(x), dim))
        x_i = np.ones(len(x))
        for i in range(0, dim):
            A[:,i] = x_i * x * x + d1 * x_i * x + d2 * x
            x_i = x_i  * x
        A[:,0] += np.ones(len(x)) * c0
        A[:,1] += x * c1
        res = linear_least_squares(A, y, ridge_alpha)
        return res
    elif type == 'no bounds':
        fi = 1
        A = np.zeros((len(x), dim + fi))
        x_i = x
        for i in range(0, dim):
            A[:,i + fi] = x_i
            x_i = x_i  * x
        A[:,0] = np.ones(len(x))
        res = linear_least_squares(A, y, ridge_alpha)
        return res
    elif type == 'left-sided':
        x0 = x[0]
        y0 = y[0]
        A = np.zeros((len(x), dim))
        x_i = np.ones(len(x))
        for i in range(0, dim):
            A[:,i] = x_i * (x - x0)
            x_i = x_i  * x
        A[:,0] += np.ones(len(x)) * y0
        res = linear_least_squares(A, y, ridge_alpha)
        return res
    elif type == 'right-sided':
        x0 = x[-1]
        y0 = y[-1]
        A = np.zeros((len(x), dim))
        x_i = np.ones(len(x))
        for i in range(0, dim):
            A[:,i] = x_i * (x - x0)
            x_i = x_i  * x
        A[:,0] += np.ones(len(x)) * y0
        res = linear_least_squares(A, y, ridge_alpha)
        return res
    else:
        raise ValueError(f"type = {type} not implemented")


@jit(nopython=True, cache = True)
def bounded_polynom(x, y, coef, type: Literal['two-sided', 'left-sided', 'right-sided', 'no bounds']):
    if type == 'two-sided':
        dim = len(coef)
        x0 = x[0]
        x1 = x[-1]
        y0 = y[0]
        y1 = y[-1]
        c0 = (y0 * x1 - y1 * x0) / (x1 - x0)
        c1 = (y1 - y0) / (x1 - x0)
        d1 = -x0 - x1
        d2 = x0 * x1
        res = np.zeros(len(x))
        x_i = np.ones(len(x))
        for i in range(0, dim):
            res += coef[i] * x_i
            x_i = x_i * x
        return (x * x + d1 * x + d2) * res + c1 * x + c0
    elif type == 'no bounds':
        dim = len(coef)
        res = np.zeros(len(x))
        fi = 1
        for i in range(0, dim):
            res += coef[i] * x**(1 - fi + i)
        return res
    elif type == 'left-sided':
        dim = len(coef)
        x0 = x[0]
        y0 = y[0]
        res = np.zeros(len(x))
        x_i = np.ones(len(x))
        for i in range(0, dim):
            res += coef[i] * (x - x0) * x_i
            x_i = x_i * x
        return res + y0
    elif type == 'right-sided':
        dim = len(coef)
        x0 = x[-1]
        y0 = y[-1]
        res = np.zeros(len(x))
        x_i = np.ones(len(x))
        for i in range(0, dim):
            res += coef[i] * (x - x0) * x_i
            x_i = x_i * x
        return res + y0
    else:
        raise ValueError(f"type = {type} not implemented")


@jit(nopython=True, cache = True)
def mod_abs(x):
    b = 50
    res = x * np.tanh(b * x)
    return res


@jit(nopython=True, cache = True)
def correction(t_data, x_data, alpha):
    alpha1, alpha2, alpha3 = alpha[0], alpha[1], alpha[2]
    sigma2 = alpha3**2 / (- 2 * alpha2) * (1 - np.exp(2 * alpha2 * t_data)) + 0.0001
    #max_res = 1 / (2 * sigma2) - 0.001
    max_res = 0.0
    return -(mod_abs(max_res - x_data) - max_res - x_data) / 2


@jit(nopython=True, cache = True)
def log_norm_ou(alpha: np.array, a1: np.array, a2: np.array, t: np.array, x0: np.array):
    alpha1, alpha2, alpha3 = alpha[0], alpha[1], alpha[2]
    mu = -alpha1 / alpha2
    theta = -alpha2
    nu = alpha3
    D = nu**2/2
    sigma2 = D/theta * (1 - np.exp(-2 * theta * t))
    xs = (x0 - mu) * np.exp(-theta * t) + mu
    res = (a1**2 * sigma2 + 2 * a1 * xs + 2 * a2 * xs**2) / (2 - 4 * a2 * sigma2) - 0.5 * np.log(1 - 2*a2*sigma2)
    return res


@jit(nopython=True, cache = True)
def m_sampler_ou(alpha, a1t, a2t, dwt, init_state = None):
    alpha1, alpha2, alpha3 = alpha[0], alpha[1], alpha[2]
    T = len(dwt)
    dt = 1 / T
    xt = np.zeros(dwt.shape)
    mu = -alpha1 / alpha2
    theta = -alpha2
    nu = alpha3
    D = nu**2 / 2
    if init_state is None:
        xt[0] = mu
    else:
        xt[0] = init_state
    for i in range(1, T):
        a1, a2 = a1t[i], a2t[i]
        a1dt, a2dt =  (a1t[i] - a1t[i - 1]) / dt, (a2t[i] - a2t[i - 1]) / dt
        t = i/T
        sigma2 = D / theta * (1 - np.exp(- 2 * theta * t))
        p = (1 - 2 * a2 * sigma2)
        sigma2w = sigma2 / p
        xs = (xt[0] - mu) * np.exp(-theta * t) + mu
        xsw = (xs + a1 * sigma2) / p
        sigma2dt = nu**2 - 2 * theta * sigma2
        sigma2wdt = (sigma2dt + 2 * sigma2**2 * a2dt) / p**2
        xsdt = -theta * (xs - mu)
        xswdt = (xsdt + a1 * sigma2dt + a1dt * sigma2) / p + 2 * xsw * (a2dt * sigma2 + a2 * sigma2dt) / p
        B = nu
        A = xswdt - (xt[i - 1] - xsw) * (B**2 - sigma2wdt) / (2 * sigma2w)
        xt[i] = xt[i - 1] + A * dt + B * dwt[i]
    return xt


@jit(nopython=True, parallel = True, cache = True)
def m_jit_mlog_likelihood_ou(alpha, data, dwt, latent_process_tr, m_iters, print_path, pdf, transform, init_state = None):
    T = len(data)
    norm_log_data = np.zeros((T, latent_process_tr))
    dt = 1/T
    t_data = np.linspace(0, 1, T)
    a_data = np.zeros((T, 3))
    alpha1, alpha2, alpha3 = alpha[0], alpha[1], alpha[2]
    mu = - alpha1 / alpha2
    a1t = np.zeros(T)
    a2t = np.zeros(T)
    for j in range(0, m_iters):
        if j == 0:
            lambda_data = p_sampler_ou(alpha, dwt, init_state)
        else:
            lambda_data = m_sampler_ou(alpha, a1t, a2t, dwt, init_state)
            if np.isnan(np.sum(lambda_data)) == True:
                res = 10**10
                if print_path == True:
                    print(alpha, 'm sampler nan', res)
                return res
        for i in range(T - 1, 0 , -1):
            copula_log_data = np.log(np.maximum(pdf(data[i], transform(lambda_data[i])), 1e-100))
            A = np.dstack( ( np.ones(latent_process_tr) , lambda_data[i] , lambda_data[i]**2 ) )[0]
            b = copula_log_data + norm_log_data[i]
            sigma2 = alpha3**2 / (- 2 * alpha2) * (1 - np.exp(2 * alpha2 * (t_data[i])))
            try:
                a_data[i - 1] = linear_least_squares(A, b, 0.0) #alpha3
                a_data[i - 1][2] = np.minimum(a_data[i - 1][2], 0.0) #1/(2 * sigma2) - 0.001
            except:
                res = 10**10
                if print_path == True:
                    print(alpha, 'ls problem fail', res, i)
                return res
            norm_log_data[i - 1] = log_norm_ou(alpha, a_data[i - 1][1], a_data[i - 1][2], dt, lambda_data[i - 1])
        a_data_a1 = a_data[:,1].copy()
        a_data_a2 = a_data[:,2].copy()

        fit_type = 'right-sided'

        a1_params = bounded_polynom_fit(t_data, a_data_a1, dim = j, ridge_alpha = 0.0, type = fit_type)
        a2_params = bounded_polynom_fit(t_data, a_data_a2, dim = j, ridge_alpha = 0.0, type = fit_type)

        a1t = bounded_polynom(t_data, a_data_a1, a1_params, type = fit_type)
        a2t = bounded_polynom(t_data, a_data_a2, a2_params, type = fit_type)
        a2t = correction(t_data, a2t, alpha)

    log_likelihood = np.zeros(latent_process_tr)
    lambda_data = m_sampler_ou(alpha, a1t, a2t, dwt, init_state)
    for i in range(1, T):
        a1, a2 = a1t[i], a2t[i]
        norm_log_data[i] = log_norm_ou(alpha, a1, a2, dt, lambda_data[i - 1])
    for k in range(0, latent_process_tr):
        copula_log_data = np.log(np.maximum(pdf(data.T, transform(lambda_data[:,k]) ), 1e-100))
        g = (a1t * lambda_data[:,k]  + a2t * lambda_data[:,k]**2)
        log_likelihood[k] = np.sum(copula_log_data + norm_log_data[:,k] - g)
    xc = np.max(log_likelihood)
    avg_likelihood = np.sum(np.exp(log_likelihood - xc)) / latent_process_tr
    res = np.log(avg_likelihood) + xc
    res = -res
    if print_path == True:
        print(alpha, res)
    return res





