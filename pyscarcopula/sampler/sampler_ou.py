import numpy as np
from numba import jit, prange
from typing import Literal
import math
from pyscarcopula.auxiliary.funcs import linear_least_squares


@jit(nopython=True, cache = True)
def p_sampler_init_state_ou(alpha, latent_process_tr):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]

    x0 = np.ones(latent_process_tr) * mu
    return x0


@jit(nopython=True,  cache = True)
def p_sampler_ou(alpha, dwt, init_state = None):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]

    T = len(dwt)
    latent_process_tr = len(dwt[0])
    dt = 1 / (T - 1)
    xt = np.zeros((T, latent_process_tr))
    
    if init_state is None:
        xt[0] = p_sampler_init_state_ou(alpha, latent_process_tr)
    else:
        xt[0] = init_state

    for i in range(1, T):
        A = -theta * (xt[i - 1] - mu)
        B = nu

        xt[i] = xt[i - 1] + A * dt + B * dwt[i - 1]
    return xt


@jit(nopython=True,  cache = True)
def p_sampler_one_step_ou(alpha, dwt, dt, init_state):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]
    latent_process_tr = len(dwt)
    if init_state is None:
        x0 = p_sampler_init_state_ou(alpha, latent_process_tr)
    else:
        x0 = init_state    

    A = -theta * (x0 - mu)
    B = nu

    x1 = x0 + A * dt + B * dwt

    return x1


@jit(nopython=True,  cache = True)
def p_sampler_one_step_ou_rng(alpha, latent_process_tr, random_state, dt, init_state):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]
    if init_state is None:
        x0 = p_sampler_init_state_ou(alpha, latent_process_tr)
    else:
        x0 = init_state

    sqrt_dt = np.sqrt(dt)
    rng = np.random.seed(random_state)
    dwt = sqrt_dt * np.random.normal(0 , 1 , size = latent_process_tr)

    
    A = -theta * (x0 - mu)
    B = nu

    x1 = x0 + A * dt + B * dwt

    return x1


@jit(nopython=True,  cache = True)
def p_sampler_no_hist_ou(alpha, dwt, dt, init_state):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]
    T = len(dwt)

    latent_process_tr = len(dwt[0])
    if init_state is None:
        x0 = p_sampler_init_state_ou(alpha, latent_process_tr)
    else:
        x0 = init_state
    
    xt_km1 = x0

    for k in range(1, T):
        A = -theta * (xt_km1 - mu)
        B = nu
        
        xt_k = xt_km1 + A * dt + B * dwt[k - 1]

        xt_km1 = xt_k
    return xt_k


@jit(nopython=True,  cache = True)
def p_sampler_no_hist_ou_rng(alpha, latent_process_tr, random_states_sequence, dt, init_state):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]
    T = len(random_states_sequence)
    sqrt_dt = np.sqrt(dt)
    if init_state is None:
        x0 = p_sampler_init_state_ou(alpha, latent_process_tr)
    else:
        x0 = init_state
    
    xt_km1 = x0
    for k in range(1, T):
        rng = np.random.seed(random_states_sequence[k - 1])
        dwt = sqrt_dt * np.random.normal(0 , 1 , size = latent_process_tr)

        A = -theta * (xt_km1 - mu)
        B = nu
       
        xt_k = xt_km1 + A * dt + B * dwt

        xt_km1 = xt_k
    return xt_k


@jit(nopython = True, cache = True, parallel = True)
def get_avg_p_log_likelihood_ou(data, lambda_data, latent_process_tr, pdf, transform):
    avg_likelihood = 0
    copula_log_data = np.zeros(latent_process_tr)

    for k in prange(0, latent_process_tr):
        copula_log_data[k] = np.sum(np.log(np.maximum(pdf(data, transform(lambda_data[:,k])), 1e-100)))

    nan_idx = np.argwhere(np.isnan(copula_log_data)).flatten()
    if len(nan_idx) < 0.05 * latent_process_tr:
        copula_log_data = np.delete(copula_log_data, nan_idx)

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
        res = 10**10
        if print_path == True:
            print(alpha, 'unknown error', res)
    else:
        if print_path == True:
            print(alpha, res)
    return res


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
        A[:,0] += np.ones(len(x))
        A[:,1] += x * c1
        res = linear_least_squares(A, y - c0, ridge_alpha)
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
        res = linear_least_squares(A, y - y0, ridge_alpha)
        return res
    elif type == 'right-sided':
        x0 = x[-1]
        y0 = y[-1]
        A = np.zeros((len(x), dim))
        x_i = np.ones(len(x))
        for i in range(0, dim):
            A[:,i] = x_i * (x - x0)
            x_i = x_i  * x
        res = linear_least_squares(A, y - y0, ridge_alpha)
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
            res +=  coef[i] * x_i
            x_i = x_i * x
        return y0 + (x - x0) * res
    elif type == 'right-sided':
        dim = len(coef)
        x0 = x[-1]
        y0 = y[-1]
        res = np.zeros(len(x))
        x_i = np.ones(len(x))
        for i in range(0, dim):
            res += coef[i] * x_i
            x_i = x_i * x
        return y0 + (x - x0) * res
    else:
        raise ValueError(f"type = {type} not implemented")


@jit(nopython=True, cache = True)
def mod_abs(x):
    b = 1
    #res = x * (2 / (1 + np.exp(-b * x)) - 1)
    res = x * np.tanh(b * x)
    return res


@jit(nopython=True, cache = True)
def correction(t_data, x_data, alpha):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]
    sigma2 = nu**2 / (2 * theta) * (1 - np.exp(-2 * theta * t_data)) + 0.0001
    #ub = 0.0
    ub = np.maximum(1/(2 * sigma2) - 0.1, 0)
    exp_res = np.exp(-0.5*(ub - x_data))
    return 1 / (1 + exp_res) * x_data
    #return -(mod_abs(max_res - x_data) - max_res - x_data) / 2


@jit(nopython=True, cache = True)
def check_a2_bounds(alpha, x, t):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]
    sigma2 = nu**2 / (2 * theta) * (1 - np.exp(-2 * theta * t))
    ub = 1/(2 * sigma2 + 0.001)
    bound_check = True
    for k in range(0, len(x)):
        if x[k] >= ub[k]:
            bound_check = False
            return bound_check
    return bound_check


@jit(nopython=True, cache = True)
def log_norm_ou(alpha: np.array, a1: np.array, a2: np.array, t: np.array, x0: np.array):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]
    D = nu**2/2
    sigma2 = D/theta * (1 - np.exp(-2 * theta * t))
    xs = (x0 - mu) * np.exp(-theta * t) + mu
    res = (a1**2 * sigma2 + 2 * a1 * xs + 2 * a2 * xs**2) / (2 - 4 * a2 * sigma2) - 0.5 * np.log(1 - 2*a2*sigma2)
    return res


@jit(nopython=True, cache = True)
def m_sampler_ou(alpha, a1t, a2t, dwt, init_state = None):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]

    T = len(dwt)
    latent_process_tr = len(dwt[0])
    dt = 1 / (T - 1)
    xt = np.zeros((T, latent_process_tr))
    D = nu**2 / 2
    if init_state is None:
        xt[0] = p_sampler_init_state_ou(alpha, latent_process_tr)
    else:
        xt[0] = init_state

    Ito_integral_sum = np.zeros(latent_process_tr)
    for i in range(1, T):
        a1, a2 = a1t[i], a2t[i]

        t = i / (T - 1)
        sigma2 = D / theta * (1 - np.exp(- 2 * theta * t))
        p = (1 - 2 * a2 * sigma2)

        if i == 1:
            pm1 = 1
        else:
            tm1 = t - dt
            sigma2m1 = D / theta * (1 - np.exp(- 2 * theta * tm1))
            a2m1 = a2t[i - 1]
            pm1 = (1 - 2 * a2m1 * sigma2m1)

        xs = (xt[0] - mu) * np.exp(-theta * t) + mu
        xsw = (xs + a1 * sigma2) / p

        Determinated_part = xsw
        Ito_integral_sum = (Ito_integral_sum  * np.sqrt(pm1 / p) + nu / np.sqrt(p) * dwt[i - 1]) * np.exp(-theta * dt)
        xt[i] = Determinated_part + Ito_integral_sum
    return xt


@jit(nopython=True, cache = True)
def m_jit_mlog_likelihood_ou(alpha, data, dwt, latent_process_tr, m_iters, print_path, 
                             pdf, transform, init_state = None, max_log_lik_debug = -100000):
    T = len(data)
    norm_log_data = np.zeros((T, latent_process_tr))
    dt = 1/T
    t_data = np.linspace(0, 1, T)
    a_data = np.zeros((T, 3))
    theta, mu, nu = alpha[0], alpha[1], alpha[2]
    a1t = np.zeros(T)
    a2t = np.zeros(T)

    '''get latent process sample'''
    for j in range(0, m_iters):

        lambda_data = m_sampler_ou(alpha, a1t, a2t, dwt, init_state)

        '''check nan values'''
        if np.isnan(np.sum(lambda_data)) == True:
            res = float(10**10)
            if print_path == True:
                print(alpha, 'm sampler nan', res)
            return res, a1t, a2t
        
        # max_m = np.max(np.abs(lambda_data))
        # if max_m > 100:
        #     res = float(10**10)
        #     if print_path == True:
        #         print(alpha, 'm sampler bad trajectory', res)
        #     return res, a1t, a2t
        
        norm_log_data = np.zeros((T, latent_process_tr))

        '''set initial values: a(T)'''
        a_mean = np.zeros(3)

        a_mean[0] = np.mean(a_data[:,0])
        a_mean[1] = np.mean(a_data[:,1])

        '''consider upper bound for a2'''
        sigma2 = nu**2 / (2 * theta) * (1 - np.exp(-2 * theta))
        ub = np.maximum(1/(2 * sigma2) - 0.1, 0)
        a_mean[2] = np.minimum(np.mean(a_data[:,2]), ub)

        a_data = np.zeros((T, 3))
        a_data[-1] = a_mean

        '''solve ls problem'''
        for i in range(T - 1, 0 , -1):
            copula_log_data = np.log(np.maximum(pdf(data[i], transform(lambda_data[i])), 1e-100))
            A = np.dstack((np.ones(latent_process_tr) , lambda_data[i] , lambda_data[i]**2))[0]
            norm_log_data[i] = log_norm_ou(alpha, a_data[i][1], a_data[i][2], dt, lambda_data[i - 1])

            b = copula_log_data + norm_log_data[i]
            sigma2 = nu**2 / (2 * theta) * (1 - np.exp(-2 * theta * (t_data[i])))
            r = 0.0
            
            '''set upper bound for a2'''
            ub = np.maximum(1/(2 * sigma2) - 0.1, 0)
            try:
                a_data[i - 1] = linear_least_squares(A, b, r, pseudo_inverse = True)
                '''if nan use previous result'''
                if np.isnan(np.sum(a_data[i - 1])) == True:
                    a_data[i - 1] = a_data[i]
            except:
                res = 10**10
                if print_path == True:
                    print(alpha, 'ls problem fail', res, i)
                return res, a1t, a2t

            '''check a2 bounds'''
            a_data[i - 1][2] = np.minimum(a_data[i - 1][2], ub)

        a_data_a1 = a_data[:,1].copy()
        a_data_a2 = a_data[:,2].copy()

        '''set right bound for a2'''
        if j == m_iters - 1:
            val = np.minimum(np.mean(a_data_a2), 0)
            if a_data_a2[-1] > val:
                a_data_a2[-1] = val

        fit_type1 = 'right-sided'
        fit_type2 = 'right-sided'
        dim = j + 1

        '''check a2 lower then bounds and fit a2(t)'''
        rigde_alpha_list = np.array([0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0])
        for r in rigde_alpha_list:
            a2_params = bounded_polynom_fit(t_data, a_data_a2, dim = dim, ridge_alpha = r, type = fit_type2)
            a2t = bounded_polynom(t_data, a_data_a2, a2_params, type = fit_type2)
            bound_check = check_a2_bounds(alpha, a2t, t_data)
            if bound_check == True:
                break
            else:
                continue
        if r == rigde_alpha_list[-1]:
            a2t = correction(t_data, a2t, alpha)

        '''fit a1(t)'''
        a1_params = bounded_polynom_fit(t_data, a_data_a1, dim = dim, ridge_alpha = r, type = fit_type1)
        a1t = bounded_polynom(t_data, a_data_a1, a1_params, type = fit_type1)

    '''get latent process sample'''
    lambda_data = m_sampler_ou(alpha, a1t, a2t, dwt, init_state)
    log_likelihood = np.zeros(latent_process_tr)
    norm_log_data = np.zeros((T, latent_process_tr))
    
    '''check nan values'''
    if np.isnan(np.sum(lambda_data)) == True:
        res = float(10**10)
        if print_path == True:
            print(alpha, 'm sampler nan', res)
        return res, a1t, a2t
    # max_m = np.max(np.abs(lambda_data))
    # if max_m > 100:
    #     res = float(10**10)
    #     if print_path == True:
    #         print(alpha, 'm sampler bad trajectory', res)
    #     return res, a1t, a2t     
    
    '''calculate normalizing factors'''
    for i in range(T - 1, 0, -1):
        a1, a2 = a1t[i], a2t[i]
        norm_log_data[i] = log_norm_ou(alpha, a1, a2, dt, lambda_data[i - 1])
    if init_state is None:
        x0 = p_sampler_init_state_ou(alpha, latent_process_tr)
    else:
        x0 = init_state
    norm_log_data[0] =  log_norm_ou(alpha, a1t[0], a2t[0], dt, x0)

    '''calculate log likelihood'''
    for k in range(0, latent_process_tr):
        copula_log_data = np.log(np.maximum(pdf(data.T, transform(lambda_data[:,k])), 1e-100))
        g = (a1t * lambda_data[:,k]  + a2t * lambda_data[:,k]**2)
        log_likelihood[k] = np.sum(copula_log_data + norm_log_data[:,k] - g)
    xc = np.max(log_likelihood)
    avg_likelihood = np.sum(np.exp(log_likelihood - xc)) / latent_process_tr
    res = np.log(avg_likelihood) + xc
    res = -res
    if res < max_log_lik_debug:
        res = float(10**10)
        if print_path == True:
            print(alpha, 'instability encountered', res)
        return res, a1t, a2t

    '''check nan values'''
    if np.isnan(res) == True:
        res = float(10**10)
        if print_path == True:
            print(alpha, 'unknown error', res)
        return res, a1t, a2t


    if print_path == True:
        print(alpha, res)
    return res, a1t, a2t


#@jit(nopython=True, cache = True)
def jit_latent_process_conditional_expectation_p_ou(pdf, transform, pobs_data, alpha, latent_process_tr):
    T = len(pobs_data)
    dwt = np.random.normal(0, 1, size = (T, latent_process_tr)) * np.sqrt(1/T)

    lambda_data = transform(p_sampler_ou(alpha, dwt))
    
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


#@jit(nopython=True, cache = True)
def jit_latent_process_conditional_expectation_m_ou(pdf, transform, pobs_data, alpha, latent_process_tr, m_iters):
    T = len(pobs_data)
    dwt = np.random.normal(0, 1, size = (T, latent_process_tr)) * np.sqrt(1/T)

    res, a1t, a2t = m_jit_mlog_likelihood_ou(alpha, pobs_data, 
                                             dwt, latent_process_tr, m_iters, False, 
                                             pdf, transform)


    lambda_data = m_sampler_ou(alpha, a1t, a2t, dwt)
    norm_log_data = np.zeros((T, latent_process_tr))
    g = np.zeros((T, latent_process_tr))

    dt = 1/T
    for i in range(T - 1, 0, -1):
        a1, a2 = a1t[i], a2t[i]
        norm_log_data[i] = log_norm_ou(alpha, a1, a2, dt, lambda_data[i - 1])
    
    x0 = p_sampler_init_state_ou(alpha, latent_process_tr)
    norm_log_data[0] =  log_norm_ou(alpha, a1t[0], a2t[0], dt, x0)

    for i in range(0, T):
        g[i] = (a1t[i] * lambda_data[i]  + a2t[i] * lambda_data[i]**2)

    copula_log_data = np.zeros((T, latent_process_tr))

    for k in range(0, latent_process_tr):
        copula_log_data_temp = np.log(np.maximum(pdf(pobs_data.T, transform(lambda_data[:,k])), 1e-100))
        copula_log_data[:,k] = copula_log_data_temp + norm_log_data[:,k] - g[:,k]
    
    copula_cs_log_data = np.zeros(latent_process_tr)
    
    latent_process_conditional_sample = np.zeros(T)
    for i in range(0, T):

        copula_cs_log_data += copula_log_data[i]

        xc = np.max(copula_cs_log_data)
        
        temp1 = np.exp(copula_cs_log_data - xc)

        avg_likelihood = np.sum(temp1) / latent_process_tr
        log_lik = np.log(avg_likelihood) + xc

        avg_expectation = np.sum(transform(lambda_data[i]) * temp1) / latent_process_tr
        avg_log_expectation = np.log(avg_expectation) + xc

        latent_process_conditional_sample[i] = avg_log_expectation - log_lik

    result = np.exp(latent_process_conditional_sample)
    
    return result