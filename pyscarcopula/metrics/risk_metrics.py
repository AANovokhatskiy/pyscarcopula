import numpy as np
from scipy.optimize import Bounds, minimize

from numba import jit, prange
from tqdm import tqdm

from pyscarcopula.metrics.latent_process import get_latent_process_params, latent_process_init_state, latent_process_sampler_rng, latent_process_sampler_one_step_rng
from pyscarcopula.metrics.marginals import get_marginals_params_params, get_rvs

from pyscarcopula.aux_functions.funcs import jit_pobs
import gc


@jit(nopython=True, parallel = True, cache = True)
def loss_func(log_returns, weight):
    n = len(log_returns)
    m = len(log_returns[0])
    portfolio_return = np.zeros(n)
    for k in prange(0, n):
        temp = 0.0
        for j in range(0, m):
            temp += np.exp(log_returns[k][j]) * weight[j]
        portfolio_return[k] = temp
    loss = np.ones(n) - portfolio_return
    # loss = 1 - np.exp(log_returns) @ weight
    return loss


@jit(nopython=True, parallel = True, cache = True)
def F_cvar_wq(x, gamma, log_returns, copula_pdf_data):
    q = x[0]
    weight = x[1:]
    F = 0.0
    n = len(copula_pdf_data)
    m = len(log_returns[0])
    for k in prange(0, n):
        loss = 0.0
        for j in range(0, m):
            loss += np.exp(log_returns[k][j]) * weight[j]
        loss = np.maximum(1.0 - loss - q, 0.0)
        F += copula_pdf_data[k] * loss
    F = q + 1 / (1 - gamma) * F / n
    return F


@jit(nopython=True, parallel = True, cache = True)
def F_cvar_q(q, gamma, loss, copula_pdf_data):
    n = len(copula_pdf_data)
    mean = 0.0
    for k in prange(0, n):
        mean += copula_pdf_data[k] * np.maximum(loss[k] - q[0], 0.0)
    mean = mean / n
    F = q[0] + 1 / (1 - gamma) * mean
    return F


def calculate_cvar(copula,
                   latent_process_params,
                   latent_process_type,
                   marginals_params,
                   marginals_params_method,
                   gamma,
                   window_len,
                   MC_iterations,
                   portfolio_weight):

    print('calc portfolio cvar')
    T = len(marginals_params)
    dt = 1/window_len
    dim = len(marginals_params[0])

    random_state_sequence = np.random.choice(range(1, 10000 * T), size = T, replace = False)

    var_data = np.zeros(T)
    cvar_data = np.zeros(T)
    dt = 1.0/window_len
    iters = T - window_len + 1

    for k in tqdm(range(0, iters)):
        idx = k + window_len - 1
        rvs = get_rvs(marginals_params[idx], MC_iterations, method = marginals_params_method)
        loss = loss_func(rvs, portfolio_weight)
        pseudo_obs = jit_pobs(rvs)
        del rvs
        if k == 0:
            init_state = latent_process_init_state(latent_process_params[idx][1:], latent_process_type, MC_iterations)
            current_state = latent_process_sampler_rng(latent_process_params[idx][1:],
                                                       latent_process_type,
                                                       random_state_sequence[0:idx],
                                                       dt,
                                                       init_state)
        else:
            current_state = latent_process_sampler_one_step_rng(latent_process_params[idx][1:],
                                                                latent_process_type,
                                                                random_state_sequence[idx],
                                                                dt,
                                                                current_state)

        copula_pdf_data = copula.np_pdf()(pseudo_obs.T, copula.transform(current_state))
        del pseudo_obs
        x0 = 0
        min_result = minimize(F_cvar_q, x0 = x0,
                                        args=(gamma, loss, copula_pdf_data),
                                        method='SLSQP',
                                        tol = 1e-7)
        del copula_pdf_data
        del loss
        var_data[idx] = min_result.x[0]
        cvar_data[idx] = min_result.fun
        collected = gc.collect()

    return var_data, cvar_data, portfolio_weight


def calculate_cvar_optimal_portfolio(copula,
                                     latent_process_params,
                                     latent_process_type,
                                     marginals_params,
                                     marginals_params_method,
                                     gamma,
                                     window_len,
                                     MC_iterations):

    print('calc portfolio optimization')
    T = len(marginals_params)
    dt = 1/window_len
    dim = len(marginals_params[0])

    eq_weight = np.ones(dim) / dim

    random_state_sequence = np.random.choice(range(1, 10000 * T), size = T, replace = False)

    var_data = np.zeros(T)
    cvar_data = np.zeros(T)
    weight_data = np.zeros((T, dim))
    dt = 1.0/window_len
    iters = T - window_len + 1

    constr = {'type': 'eq', 'fun': lambda x: np.sum(x[1:]) - 1}
    lb = np.zeros(dim + 1)
    lb[0] = -1
    rb =  np.ones(dim + 1)
    bounds = Bounds(lb, rb)
    x0 = np.array([0, *eq_weight])

    for k in tqdm(range(0, iters)):
        idx = k + window_len - 1
        rvs = get_rvs(marginals_params[idx], MC_iterations, method = marginals_params_method)
        pseudo_obs = jit_pobs(rvs)
        

        if k == 0:
            init_state = latent_process_init_state(latent_process_params[idx][1:], latent_process_type, MC_iterations)
            current_state = latent_process_sampler_rng(latent_process_params[idx][1:],
                                                       latent_process_type,
                                                       random_state_sequence[0:idx],
                                                       dt,
                                                       init_state)
        else:
            current_state = latent_process_sampler_one_step_rng(latent_process_params[idx][1:],
                                                                latent_process_type,
                                                                random_state_sequence[idx],
                                                                dt,
                                                                current_state)

        copula_pdf_data = copula.np_pdf()(pseudo_obs.T, copula.transform(current_state))
        del pseudo_obs

        min_result = minimize(F_cvar_wq, x0 = x0,
                                        args=(gamma, rvs, copula_pdf_data),
                                        method='SLSQP',
                                        bounds = bounds,
                                        constraints = constr,
                                        tol = 1e-7)
        del copula_pdf_data
        var_data[idx] = min_result.x[0]
        cvar_data[idx] = min_result.fun
        weight_data[idx] = min_result.x[1:dim+1]
        x0 = np.array(min_result.x)
        collected = gc.collect()

    return var_data, cvar_data, weight_data


def risk_metrics(copula, 
                 data, 
                 window_len, 
                 gamma, 
                 MC_iterations, 
                 marginals_params_method,
                 latent_process_type, 
                 latent_process_tr = 500,
                 optimize_portfolio = True, 
                 portfolio_weight = None):
    '''calculate risk metrics VaR and CVaR and optimize portfolio weights'''

    T = len(data)
    if window_len > T:
        raise ValueError(f'Length of window = {window_len} is more than length of data = {T}')

    dwt = np.random.normal(0, 1, size = (T, latent_process_tr)) * np.sqrt(1.0/window_len)
    latent_process_params = get_latent_process_params(copula, data, latent_process_type.upper(), window_len, dwt)
    del dwt

    # latent_process_params = np.zeros((T, 4))
    #
    # for k in range(0, T):
    #     latent_process_params[k] = np.array([-519.014, 0.399247, 0.775835, 0.05])

    marginals_params = get_marginals_params_params(data, window_len, marginals_params_method)

    gamma_list = []
    MC_iterations_list = []
    if hasattr(gamma, '__iter__'):
        gamma_list = gamma
    else:
        gamma_list = [gamma]

    if hasattr(MC_iterations, '__iter__'):
        MC_iterations_list = MC_iterations
    else:
        MC_iterations_list = [MC_iterations]   

    res = dict()   
    for gamma_i in gamma_list:
        for MC_j in MC_iterations_list:
            print(f"gamma = {gamma_i}, MC_iterations = {MC_j}")
            if optimize_portfolio == True:
                var_data, cvar_data, weight_data = calculate_cvar_optimal_portfolio(copula, 
                                                                                    latent_process_params, 
                                                                                    latent_process_type.upper(),
                                                                                    marginals_params, 
                                                                                    marginals_params_method,
                                                                                    gamma_i, 
                                                                                    window_len,
                                                                                    MC_j)
            else:
                var_data, cvar_data, weight_data = calculate_cvar(copula, 
                                                                  latent_process_params, 
                                                                  latent_process_type.upper(),
                                                                  marginals_params, 
                                                                  marginals_params_method,
                                                                  gamma_i, 
                                                                  window_len,
                                                                  MC_j, 
                                                                  portfolio_weight)
            res[gamma_i] = dict()
            res[gamma_i][MC_j] = dict()
            res[gamma_i][MC_j]['var'] = var_data
            res[gamma_i][MC_j]['cvar'] = cvar_data
            res[gamma_i][MC_j]['weight'] = weight_data
    return res
