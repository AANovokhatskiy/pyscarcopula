import numpy as np
from numba import jit, prange
from typing import Literal
import math
from pyscarcopula.auxiliary.funcs import linear_least_squares


@jit(nopython=True)
def init_state_ou(alpha, latent_process_tr):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]

    x0 = np.ones(latent_process_tr) * mu
    return x0

@jit(nopython=True,  cache = True)
def stationary_state_ou(alpha, latent_process_tr, seed = None):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]

    if seed is None:
        seed = np.random.randint(1, 1000000)
    rng = np.random.seed(seed)
    xs = mu
    sigma2 = nu**2 / (2 * theta)
    state = np.random.normal(loc = mu, scale = np.sqrt(sigma2), size = latent_process_tr)    
    
    return state

# @jit(nopython = True, cache = True, parallel = True)
def get_avg_p_log_likelihood_ou(u, lambda_data, pdf, transform):
    avg_likelihood = 0
    latent_process_tr = lambda_data.shape[1]

    copula_log_data = np.zeros(latent_process_tr)

    for k in prange(0, latent_process_tr):
        copula_log_data[k] = np.sum(np.log(np.maximum(pdf(u, transform(lambda_data[:,k])), 1e-100)))

    '''trick for calculation large values. calculate e^(sum(log_cop) - corr) instead of e^(sum(log_cop)).
    Do inverse correction at the end of calculations'''
    corr = max(copula_log_data)
    avg_likelihood = np.sum(np.exp(copula_log_data - corr)) / latent_process_tr
    return math.log(avg_likelihood) + corr


# @jit(nopython = True)
def p_jit_mlog_likelihood_ou(alpha: np.array, u: np.array, dwt: np.array,
                      pdf: callable, transform: callable, print_path: bool, init_state: np.array = None) -> float:
    
    '''initial data check'''
    if np.isnan(np.sum(alpha)) == True:
        res = 10**10
        if print_path == True:
            print(alpha, 'incorrect params', res)
        return res
    
    T = dwt.shape[0]
    a1t = np.zeros(T)
    a2t = np.zeros(T)
    lambda_data = m_sampler_ou(alpha, a1t, a2t, dwt, init_state)

    avg_log_likelihood = get_avg_p_log_likelihood_ou(u.T, lambda_data, pdf, transform)
    res = - avg_log_likelihood
    if np.isnan(res) == True:
        res = 10**10
        if print_path == True:
            print(alpha, 'unknown error', res)
    else:
        if print_path == True:
            print(alpha, res)
    return res


@jit(nopython=True)
def bounded_polynom_fit(x, y, dim, type: Literal['two-sided', 'left-sided', 'right-sided', 'no bounds'], ridge_alpha = 0.0):
    if ridge_alpha == 0.0:
        pseudo_inverse = True
    else:
        pseudo_inverse = False
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
        res = linear_least_squares(A, y - c0, ridge_alpha, pseudo_inverse)
        return res
    elif type == 'no bounds':
        fi = 1
        A = np.zeros((len(x), dim + fi))
        x_i = x
        for i in range(0, dim):
            A[:,i + fi] = x_i
            x_i = x_i  * x
        A[:,0] = np.ones(len(x))
        res = linear_least_squares(A, y, ridge_alpha, pseudo_inverse)
        return res
    elif type == 'left-sided':
        x0 = x[0]
        y0 = y[0]
        A = np.zeros((len(x), dim))
        x_i = np.ones(len(x))
        for i in range(0, dim):
            A[:,i] = x_i * (x - x0)
            x_i = x_i  * x
        res = linear_least_squares(A, y - y0, ridge_alpha, pseudo_inverse)
        return res
    elif type == 'right-sided':
        x0 = x[-1]
        y0 = y[-1]
        A = np.zeros((len(x), dim))
        x_i = np.ones(len(x))
        for i in range(0, dim):
            A[:,i] = x_i * (x - x0)
            x_i = x_i  * x
        res = linear_least_squares(A, y - y0, ridge_alpha, pseudo_inverse)
        return res
    else:
        raise ValueError(f"type = {type} is not implemented")


@jit(nopython=True)
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
        raise ValueError(f"type = {type} is not implemented")


@jit(nopython=True)
def check_a2_bounds(alpha, a2t, t):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]

    '''sigma2(t = 0) = 0, then upper bound = +inf. Dont check it.'''
    sigma2 = nu**2 / (2 * theta) * (1 - np.exp(-2 * theta * t[1:]))
    ub = 1/(2 * sigma2)
    
    bound_check = True

    for k in range(1, len(a2t)):
        '''sigma2 is less on 1 element than a2t'''
        if a2t[k] >= ub[k - 1]:
            bound_check = False
            return bound_check
        
    return bound_check


@jit(nopython=True)
def log_norm_ou(alpha: np.array, a1: np.array, a2: np.array, t: np.array, x0: np.array):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]
    D = nu**2/2
    sigma2 = D/theta * (1 - np.exp(-2 * theta * t))
    xs = (x0 - mu) * np.exp(-theta * t) + mu
    res = (a1**2 * sigma2 + 2 * a1 * xs + 2 * a2 * xs**2) / (2 - 4 * a2 * sigma2) - 0.5 * np.log(1 - 2 * a2 * sigma2)
    return res


@jit(nopython = True)
def m_sampler_ou(alpha, a1t, a2t, dwt, init_state = None):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]
    D = nu**2 / 2

    T = len(dwt)
    latent_process_tr = len(dwt[0])
    dt = 1 / (T - 1)
    xt = np.zeros((T, latent_process_tr))

    if init_state is None:
        xt[0] = init_state_ou(alpha, latent_process_tr)
    else:
        xt[0] = init_state

    Ito_integral_sum = np.zeros(latent_process_tr)
    for i in range(1, T):
        j = i
        a1, a2 = a1t[j], a2t[j]

        t = i * dt
        sigma2 = D / theta * (1 - np.exp(- 2 * theta * t))
        p = (1 - 2 * a2 * sigma2)

        if i == 1:
            pm1 = 1
        else:
            tm1 = t - dt
            sigma2m1 = D / theta * (1 - np.exp(- 2 * theta * tm1))
            a2m1 = a2t[j - 1]
            pm1 = (1 - 2 * a2m1 * sigma2m1)

        xs = (xt[0] - mu) * np.exp(-theta * t) + mu
        xsw = (xs + a1 * sigma2) / p

        Determinated_part = xsw
        Ito_integral_sum = (Ito_integral_sum  * np.sqrt(pm1 / p) + nu / np.sqrt(p) * dwt[i - 1]) * np.exp(-theta * dt)
        xt[i] = Determinated_part + Ito_integral_sum
    return xt


def m_jit_mlog_likelihood_ou(alpha, u, dwt, M_iterations,
                             pdf, transform, 
                             print_path, init_state = None, max_log_lik_debug = None):
    T = len(u)
    dt = 1 / (T - 1)

    latent_process_tr = dwt.shape[1]
    norm_log_data = np.zeros((T, latent_process_tr))
    t_data = np.linspace(0, 1, T)

    a_data = np.zeros((T, 3))
    theta, mu, nu = alpha[0], alpha[1], alpha[2]
    
    # a1t = np.zeros(T)
    # a2t = np.zeros(T)
    a1t = 10 * np.ones(T)
    a2t = -5 * np.ones(T)

    a1_hat = np.zeros(T)
    a2_hat = np.zeros(T)


    '''get latent process sample'''
    for j in range(0, M_iterations):

        lambda_data = m_sampler_ou(alpha, a1t, a2t, dwt, init_state)

        '''check nan values'''
        if np.isnan(np.sum(lambda_data)) == True:
            res = float(10**10)
            if print_path == True:
                print(alpha, 'm sampler nan', res)
            return res, a1t, a2t
              
        norm_log_data = np.zeros((T, latent_process_tr))

        '''set initial values: a(T)'''
        a_mean = np.zeros(3)

        a_mean[0] = np.mean(a_data[:,0])
        a_mean[1] = np.mean(a_data[:,1])

        '''consider upper bound for a2'''
        sigma2 = nu**2 / (2 * theta) * (1 - np.exp(-2 * theta))
        #ub = np.maximum(1/(2 * sigma2) - 0.1, 0)
        ub = 0
        a_mean[2] = np.minimum(np.mean(a_data[:,2]), ub)
        
        a_data = np.zeros((T, 3))
        if j == 0:
            a_data[-1] = np.array([0, a1t[-1], a2t[-1]])
        else:
            a_data[-1] = a_mean

        '''solve ls problem'''
        for i in range(T - 1, 0 , -1):
            u1 = np.full((latent_process_tr, u.shape[1]), u[i])
            copula_log_data = np.log(np.maximum(pdf(u1.T, transform(lambda_data[i])), 1e-100)) #u[i]
            A = np.dstack((np.ones(latent_process_tr) , lambda_data[i] , lambda_data[i]**2))[0]
            norm_log_data[i] = log_norm_ou(alpha, a_data[i][1], a_data[i][2], dt, lambda_data[i - 1]) #i - 1

            b = copula_log_data + norm_log_data[i]
            sigma2 = nu**2 / (2 * theta) * (1 - np.exp(-2 * theta * (t_data[i])))
            r = 0.0
            
            '''set upper bound for a2'''
            ub = np.maximum(1/(2 * sigma2) - 0.001, 0)
            try:
                lr = linear_least_squares(A, b, 0, pseudo_inverse = True)
                '''if nan use previous result'''
                if np.isnan(np.sum(lr)) == True:
                    a_data[i - 1] = a_data[i]
                else:
                    a_data[i - 1] = lr
            except:
                res = 10**10
                if print_path == True:
                    print(alpha, 'ls problem fail', res, i)
                return res, a1t, a2t

            '''check a2 bounds'''
            a_data[i - 1][2] = np.minimum(a_data[i - 1][2], ub)

        a1_hat = a_data[:,1].copy()
        a2_hat = a_data[:,2].copy()

        fit_type1 = 'no bounds'
        fit_type2 = 'no bounds'

        dim = j + 1

        '''check a2 lower then bounds and fit a2(t)'''
        bound_check = False
        rigde_alpha_list = np.array([0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0])
        for r in rigde_alpha_list:
            a2_params = bounded_polynom_fit(t_data, a2_hat, dim = dim, ridge_alpha = r, type = fit_type2)
            a2t = bounded_polynom(t_data, a2_hat, a2_params, type = fit_type2)
            bound_check = check_a2_bounds(alpha, a2t, t_data)
            if bound_check == True:
                break
        if bound_check == False:
            a1t = np.zeros(T)
            a2t = np.zeros(T)
            break
        else:
            '''fit a1(t)'''
            a1_params = bounded_polynom_fit(t_data, a1_hat, dim = dim, ridge_alpha = r, type = fit_type1)
            a1t = bounded_polynom(t_data, a1_hat, a1_params, type = fit_type1)


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
    
    '''calculate normalizing factors'''
    for i in range(T - 1, 0, -1):
        a1, a2 = a1t[i], a2t[i]
        norm_log_data[i] = log_norm_ou(alpha, a1, a2, dt, lambda_data[i - 1])
        
    if init_state is None:
        x0 = init_state_ou(alpha, latent_process_tr)
    else:
        x0 = init_state
    norm_log_data[0] =  log_norm_ou(alpha, a1t[0], a2t[0], dt, x0)

    '''calculate log likelihood'''
    for k in range(0, latent_process_tr):
        copula_log_data = np.log(np.maximum(pdf(u.T, transform(lambda_data[:,k])), 1e-100))
        g = (a1t * lambda_data[:,k]  + a2t * lambda_data[:,k]**2)
        log_likelihood[k] = np.sum(copula_log_data + norm_log_data[:,k] - g)
    xc = np.max(log_likelihood)
    avg_likelihood = np.sum(np.exp(log_likelihood - xc)) / latent_process_tr
    res = np.log(avg_likelihood) + xc
    res = -res
    
    if max_log_lik_debug is None:
        max_log_lik_debug = -100000
    
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


#@jit(nopython=True)
def jit_latent_process_conditional_expectation_p_ou(alpha, u, dwt, 
                                                    pdf, transform, init_state = None):
    latent_process_tr = dwt.shape[1]
    
    T = len(u)

    a1t = np.zeros(T)
    a2t = np.zeros(T)

    lambda_data = transform(m_sampler_ou(alpha, a1t, a2t, dwt, init_state))
    
    copula_log_data = np.zeros((T, latent_process_tr))

    for k in range(0, latent_process_tr):
        copula_log_data[:,k] = np.log(np.maximum(pdf(u.T, lambda_data[:,k]), 1e-100))
    
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


#@jit(nopython=True)
def jit_latent_process_conditional_expectation_m_ou(alpha, u, M_iterations, dwt, 
                                                    pdf, transform, init_state = None):

    latent_process_tr = dwt.shape[1]

    T = len(u)

    res, a1t, a2t = m_jit_mlog_likelihood_ou(alpha, u, 
                                             dwt, M_iterations, 
                                             pdf, transform, False, init_state)


    lambda_data = m_sampler_ou(alpha, a1t, a2t, dwt, init_state)
    norm_log_data = np.zeros((T, latent_process_tr))
    g = np.zeros((T, latent_process_tr))

    dt = 1 / (T - 1)
    #dt = 1 / T

    '''calculate normalizing factors'''
    for i in range(T - 1, 0, -1):
        a1, a2 = a1t[i], a2t[i]
        norm_log_data[i] = log_norm_ou(alpha, a1, a2, dt, lambda_data[i - 1])
        
    if init_state is None:
        x0 = init_state_ou(alpha, latent_process_tr)
    else:
        x0 = init_state
    norm_log_data[0] =  log_norm_ou(alpha, a1t[0], a2t[0], dt, x0)

    for i in range(0, T):
        g[i] = (a1t[i] * lambda_data[i] + a2t[i] * lambda_data[i]**2)

    copula_log_data = np.zeros((T, latent_process_tr))

    for k in range(0, latent_process_tr):
        copula_log_data_temp = np.log(np.maximum(pdf(u.T, transform(lambda_data[:,k])), 1e-100))
        copula_log_data[:,k] = copula_log_data_temp + norm_log_data[:,k] - g[:,k]
    
    copula_cs_log_data = np.zeros(latent_process_tr)
    
    latent_process_conditional_sample = np.zeros(T)
    for i in range(0, T):

        # if i > 0:
        #     copula_cs_log_data += copula_log_data[i - 1]
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