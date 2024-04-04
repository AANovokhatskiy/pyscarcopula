import numpy as np
from numba import jit, prange
import math


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
        copula_log_data[k] = np.sum(np.log(pdf(data, transform(lambda_data[:,k]))))

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
        res = 10000
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
def linear_least_squares(matA: np.array, matB: np.array) -> np.array:
    '''input  Ax = b
       output x = (A.T * A) ^ (-1) * A.T * b
    '''
    return np.linalg.inv( matA.T @ matA ) @ matA.T @ matB 


@jit(nopython=True, cache = True)
def poly_fit(x, y, dim, fit_intercept = True):
    '''least squares fit y = f(x), where f(x) - polynom with dimension = dim'''
    fi = int(fit_intercept)
    A = np.zeros((len(x), dim + fi))
    for i in range(0, dim):
        A[:,i + fi] = x**(i + 1)
    if fit_intercept == True:
        A[:,0] = np.ones(len(x))
    res = linear_least_squares(A, y)
    return res


@jit(nopython=True, cache = True)
def poly(data, coef, intercept = True):
    '''returns polynom of data (c0 + c1 t + c2 t^2 + ...) with coeficients = coef. 
    If intercept == True: first coef considered as free parameter c0; Otherwise - as c1.'''
    dim = len(coef)
    res = np.zeros(len(data))
    fi = int(intercept)
    for i in range(0, dim):
        res += coef[i] * data**(1 - fi + i)
    return res


@jit(nopython=True, cache = True)
def poly_corr(t_data, coef, alpha, intercept):
    '''correct polynom fit result to be below a threshold that raises from norm constant'''
    res = poly(t_data, coef, intercept)
    alpha1, alpha2, alpha3 = alpha[0], alpha[1], alpha[2]
    sigma2 = alpha3**2 / (- 2 * alpha2) * (1 - np.exp(2 * alpha2 * t_data)) + 0.0001
    max_res = 1 / (2 * sigma2) - 1
    exp_res = np.exp(-0.05*(max_res - res))
    return 1 / (1 + exp_res) * res


@jit(nopython=True, cache = True)
def moving_average(a, n = 3):
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    ret = ret[n - 1:] / n
    lin_arr1 = np.linspace(a[0], ret[0], n//2)
    index = len(a) - (len(lin_arr1) + len(ret))
    lin_arr2 = np.linspace(ret[-1], a[-1], index)
    ret = np.concatenate((lin_arr1, ret, lin_arr2 ))
    return ret


@jit(nopython=True, cache = True)
def correction(t_data, x_data, alpha):
    '''correct arbitrary function to be below a normalizing threshold'''
    alpha1, alpha2, alpha3 = alpha[0], alpha[1], alpha[2]
    sigma2 = alpha3**2 / (- 2 * alpha2) * (1 - np.exp(2 * alpha2 * t_data)) + 0.0001
    max_res = 1 / (2 * sigma2) - 1
    exp_res = np.exp(-0.05*(max_res - x_data))
    return 1 / (1 + exp_res) * x_data


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
        xs = (xt[0] - mu) * np.exp(-theta * t)
        xsdt = -theta * xs
        sigma2 = D / theta * (1 - np.exp(- 2 * theta * t))
        sigma2dt = nu**2 - 2 * theta * sigma2
        p = (1 - 2 * a2 * sigma2)
        sigma2w = sigma2 / p
        sigma2wdt = (sigma2dt + 2 * sigma2**2 * a2dt) / p**2
        xsw = (xs + a1 * sigma2) / p
        xswdt = (xsdt + a1 * sigma2dt + a1dt * sigma2) / p + 2 * xsw * (a2dt * sigma2 + a2 * sigma2dt) / p
        B = nu
        A = xswdt - (xt[i - 1] - mu - xsw) * (B**2 - sigma2wdt) / (2 * sigma2w)
        xt[i] = xt[i - 1] + A * dt + B * dwt[i]
    return xt


@jit(nopython=True, cache = True)
def log_norm_ou(alpha: np.array, a1: np.array, a2: np.array, t: np.array, x0: np.array):
    alpha1, alpha2, alpha3 = alpha[0], alpha[1], alpha[2]
    mu = -alpha1 / alpha2
    xs = (x0 - mu) * np.exp(alpha2 * t)
    sigma2 = alpha3**2 / (- 2 * alpha2) * (1 - np.exp(2 * alpha2 * t))
    res = (2*xs*(a1 + a2*xs) + a1**2*sigma2) / (2 - 4*a2*sigma2) - 0.5 * np.log(1 - 2*a2*sigma2)
    return res


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
                res = 10000
                if print_path == True:
                    print(alpha, 'm sampler nan', res)
                return res
        for i in range(1, T):
            copula_log_data = np.log(pdf(data[i], transform(lambda_data[i])))
            A = np.dstack( ( np.ones(latent_process_tr) , (lambda_data[i] - mu) , (lambda_data[i] - mu)**2 ) )[0]
            b = copula_log_data + norm_log_data[i - 1]
            sigma2 = alpha3**2 / (- 2 * alpha2) * (1 - np.exp(2 * alpha2 * (t_data[i])))
            try:
                a_data[i] = linear_least_squares(A, b)
                a_data[i][2] = np.minimum(a_data[i][2], 1/(2 * sigma2) - 10)
                a_data[i] = np.maximum(np.minimum(a_data[i], 30),-30)
            except:
                res = 10000
                if print_path == True:
                    print(alpha, 'ls problem fail', res, i)
                return res
            norm_log_data[i] = log_norm_ou(alpha, a_data[i][1], a_data[i][2], dt, lambda_data[i - 1])
        a_data_a1 = a_data[:,1].copy()
        a_data_a2 = a_data[:,2].copy()
        a1_params = poly_fit(t_data, a_data_a1, dim = 2, fit_intercept = False)
        a2_params = poly_fit(t_data, a_data_a2, dim = 2, fit_intercept = False)
        a1t = poly(t_data, a1_params, intercept = False)
        a2t = poly_corr(t_data, a2_params, alpha, intercept = False)
        ##corr_a1 = np.exp(- polyd(t_data, *a1_params)**2 / T**2) -- corr on da1/dt
        #n = T//10
        #a1t = moving_average(a_data_a1, n = n) #50
        #a2t = correction(t_data, moving_average(a_data_a2, n = n), alpha)

    log_likelihood = np.zeros(latent_process_tr)
    lambda_data = m_sampler_ou(alpha, a1t, a2t, dwt, init_state)
    for i in range(1, T):
        a1, a2 = a1t[i], a2t[i]
        norm_log_data[i] = log_norm_ou(alpha, a1, a2, dt, lambda_data[i - 1])
    for k in prange(0, latent_process_tr):
        copula_log_data = np.log(pdf(data.T, transform(lambda_data[:,k]) ))
        g = (a1t * (lambda_data[:,k] - mu)  + a2t * (lambda_data[:,k] - mu)**2)
        log_likelihood[k] = np.sum(copula_log_data + norm_log_data[:,k] - g)
    xc = np.max(log_likelihood)
    avg_likelihood = np.sum(np.exp(log_likelihood - xc)) / latent_process_tr
    res = np.log(avg_likelihood) + xc
    res = -res
    if print_path == True:
        print(alpha, res)
    return res






