import numpy as np
from scipy.optimize import Bounds, minimize

from numba import jit, prange
from tqdm import tqdm

from pyscarcopula.metrics.latent_process import get_latent_process_params, latent_process_init_state, latent_process_sampler_rng, latent_process_sampler_one_step_rng
from pyscarcopula.metrics.marginals import get_marginals_params_params, get_rvs

from pyscarcopula.auxiliary.funcs import jit_pobs
import gc
import pandas as pd

from pyscarcopula.marginal.inverse_CDF import inverse_CDF_matrix

from time import time
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


@jit(nopython=True, parallel = True, cache = True)
def F_cvar_q_no_copula(q, gamma, loss):
    n = len(loss)
    mean = 0.0
    for k in prange(0, n):
        mean += np.maximum(loss[k] - q[0], 0.0)
    mean = mean / n
    F = q[0] + 1 / (1 - gamma) * mean
    return F


@jit(nopython=True, parallel = True, cache = True)
def F_cvar_wq_no_copula(x, gamma, log_returns):
    q = x[0]
    weight = x[1:]
    mean = 0.0
    n = len(log_returns)
    m = len(log_returns[0])
    for k in prange(0, n):
        loss = 0.0
        for j in range(0, m):
            loss += np.exp(log_returns[k][j]) * weight[j]
        loss = np.maximum(1.0 - loss - q, 0.0)
        mean += loss
    mean = mean / n
    F = q + 1 / (1 - gamma) * mean
    return F


@jit(nopython=True, parallel = True, cache = True)
def calculate_copula_pdf_data(data, state, pdf, transform):
    n = len(data)
    copula_pdf_data = np.zeros(n)
    if len(state) == 1:
        for k in prange(0, n):
            copula_pdf_data[k] = pdf(data[k], transform(state[0]))
    else:
         for k in prange(0, n):
            copula_pdf_data[k] = pdf(data[k], transform(state[k]))       
    return copula_pdf_data


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
    dt = 1.0 / (window_len - 1)
    iters = T - window_len + 1

    for k in tqdm(range(0, iters)):
        idx = k + window_len - 1
        if k == 0:
            init_state = latent_process_init_state(latent_process_params[idx][1:], latent_process_type, MC_iterations)
            current_state = latent_process_sampler_rng(latent_process_params[idx][1:],
                                                       latent_process_type,
                                                       MC_iterations,
                                                       random_state_sequence[0:idx],
                                                       dt,
                                                       init_state)
        else:
            current_state = latent_process_sampler_one_step_rng(latent_process_params[idx][1:],
                                                                latent_process_type,
                                                                MC_iterations,
                                                                random_state_sequence[idx],
                                                                dt,
                                                                current_state)

        copula_sample = copula.get_sample(MC_iterations, copula.transform(current_state))
        rvs = get_rvs(marginals_params[idx], MC_iterations, method = marginals_params_method)

        inv_cdf_copula_sample = inverse_CDF_matrix(copula_sample, rvs)
        del rvs
        del copula_sample
        loss = loss_func(inv_cdf_copula_sample, portfolio_weight)
        del inv_cdf_copula_sample

        x0 = 0
        min_result = minimize(F_cvar_q_no_copula, x0 = x0,
                                        args=(gamma, loss),
                                        method='SLSQP',
                                        tol = 1e-7)
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
    dt = 1 / (window_len - 1)
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

        if k == 0:
            init_state = latent_process_init_state(latent_process_params[idx][1:], latent_process_type, MC_iterations)
            current_state = latent_process_sampler_rng(latent_process_params[idx][1:],
                                                       latent_process_type,
                                                       MC_iterations,
                                                       random_state_sequence[0:idx],
                                                       dt,
                                                       init_state)
        else:
            current_state = latent_process_sampler_one_step_rng(latent_process_params[idx][1:],
                                                                latent_process_type,
                                                                MC_iterations,
                                                                random_state_sequence[idx],
                                                                dt,
                                                                current_state)

        copula_sample = copula.get_sample(MC_iterations, copula.transform(current_state))
        rvs = get_rvs(marginals_params[idx], MC_iterations, method = marginals_params_method)

        rvs = inverse_CDF_matrix(copula_sample, rvs)

        min_result = minimize(F_cvar_wq_no_copula, x0 = x0,
                                        args=(gamma, rvs),
                                        method='SLSQP',
                                        bounds = bounds,
                                        constraints = constr,
                                        tol = 1e-7)
        del rvs
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
                 M_iterations = 5,
                 seed = None,
                 optimize_portfolio = True, 
                 portfolio_weight = None, 
                 pre_calc_latent_process_params = None,
                 save_logs = False):
    '''calculate risk metrics VaR and CVaR and optimize portfolio weights'''

    T = len(data)
    if window_len > T:
        raise ValueError(f'Length of window = {window_len} is more than length of data = {T}')

    if pre_calc_latent_process_params is None:
        if seed is None:
            s = np.random.randint(0, 100000)
        else:
            s = seed

        dt = 1.0 / (window_len - 1)

        print(f"seed = {s}")
        dwt = copula.calculate_dwt(latent_process_type.upper(), T, latent_process_tr, seed, dt)
        latent_process_params = get_latent_process_params(copula, data, latent_process_type.upper(), 
                                                          window_len, dwt, M_iterations, save_logs)
        del dwt
    else:
        latent_process_params = pre_calc_latent_process_params
    
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
