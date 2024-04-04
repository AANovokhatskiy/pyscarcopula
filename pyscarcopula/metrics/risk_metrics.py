import numpy as np
from scipy.optimize import Bounds, minimize

from numba import njit, jit, prange
from tqdm import tqdm

from pyscarcopula.metrics.latent_process import get_latent_process_params, latent_process_init_state, latent_process_sampler_rng, latent_process_sampler_one_step_rng
from pyscarcopula.metrics.marginals import get_marginals_params_params, get_rvs

from pyscarcopula.aux_functions.funcs import jit_pobs

from multiprocessing import Pool, cpu_count, RawArray
from functools import partial
import gc

@jit(nopython=True, parallel = True, cache = True)
def loss_func(rvs, weight):
    loss = 1 - np.exp(rvs) @ weight
    return loss

@jit(nopython=True, parallel = True, cache = True)
def F_cvar_wq(x, gamma, exp_log_returns, copula_pdf_data):
    q = x[0]
    weight = x[1:]
    F = 0
    n = len(copula_pdf_data)
    m = len(exp_log_returns[0])
    for k in prange(0, n):
        loss = 0
        for j in range(0, m):
            loss += exp_log_returns[k][j] * weight[j]
        loss = np.maximum(1 - loss - q, 0)
        F += copula_pdf_data[k] * loss
    F = q + 1 / (1 - gamma) * F / n
    return F

@jit(nopython=True, parallel = True, cache = True)
def F_cvar_q(q, gamma, loss, copula_pdf_data):
    #F = q + 1 / (1 - gamma) * np.mean(copula_pdf_data * np.maximum(loss - q, 0))
    n = len(copula_pdf_data)
    mean = 0.0
    for k in prange(0, n):
        mean += copula_pdf_data[k] * np.maximum(loss[k] - q[0], 0.0)
    mean = mean / n
    F = q[0] + 1 / (1 - gamma) * mean
    return F

'''start experimental'''
@jit(nopython=True, cache = True, nogil = True)
def neutralize(arr):
    n = len(arr)
    avg_arr = np.sum(arr) / n
    return arr - np.ones(n) * avg_arr

@jit(nopython=True, cache = True, nogil = True)
def normalize(arr):
    #norm_arr = np.linalg.norm(arr)
    norm_arr = np.sum(np.abs(arr))
    if norm_arr == 0:
        return arr
    else:
        return arr / norm_arr
    
@jit(nopython=True, cache = True, nogil = True)
def mod_abs(x):
    b = 50
    res = x * (2 / (1 + np.exp(-b * x)) - 1)
    return res

@jit(nopython=True, cache = True, nogil = True)
def mod_abs_d(x):
    b = 50
    res = (b * x + np.sinh(b * x)) / (1 + np.cosh(b * x))
    return res

@jit(nopython=True, cache = True, nogil = True)
def F_cvar_wq2(x, gamma, exp_log_returns, copula_pdf_data):
    q = x[0]
    weight = x[1:]
    #loss = -(returns_data @ weight)
    loss = np.sum(weight) - exp_log_returns @ weight
    F = q + 1 / (1 - gamma) * np.mean(copula_pdf_data * 1/2 * (np.abs(loss - q) + loss) ) - q / (2 * (1 - gamma))
    return F

@jit(nopython=True, cache = True, nogil = True)
def F_cvar_wq2_jac(x, gamma, exp_log_returns, copula_pdf_data):
    q = x[0]
    weight = x[1:]
    loss = np.sum(weight) - exp_log_returns @ weight
    res = np.zeros(len(x))
    p1 = mod_abs_d(loss - q) 
    df_dq = 1 - 1 / (2 * (1 - gamma)) - 1 / (1 - gamma) * np.mean(copula_pdf_data * 1/2 * p1 )
    res[0] = df_dq
    for i in range(0, len(x) - 1):
        dloss_dwi = weight[i] - exp_log_returns[:,i]
        df_dwi =  1 / (1 - gamma) * np.mean(copula_pdf_data * 1/2 * (dloss_dwi * p1 + dloss_dwi) )
        res[i] = df_dwi
    return res
'''end experimental'''


def calculate_cvar(copula,
                   latent_process_params,
                   latent_process_type,
                   marginals_params,
                   marginals_params_method,
                   gamma,
                   window_len,
                   MC_iterations,
                   portfolio_weight = None):

    print('calc portfolio')
    T = len(marginals_params)
    dt = 1/window_len
    dim = len(marginals_params[0])

    if portfolio_weight is None:
        weight = np.ones(dim) / dim
    else:
        weight = portfolio_weight

    random_state_sequence = np.random.choice(range(1, 1000 * T), size = T, replace = False)

    var_data = np.zeros(T)
    cvar_data = np.zeros(T)
    dt = 1.0/window_len
    iters = T - window_len + 1

    for k in tqdm(range(0, iters)):
        idx = k + window_len - 1
        rvs = get_rvs(marginals_params[idx], MC_iterations, method = marginals_params_method)
        loss = loss_func(rvs, weight)
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
                                        tol = 1e-5)
        del copula_pdf_data
        del loss
        var_data[idx] = min_result.x[0]
        cvar_data[idx] = min_result.fun
        collected = gc.collect()

    return var_data, cvar_data, weight


def optimal_portfolo(gamma, window_len, MC_iterations, marginals_params,
                                        latent_process_params, copula, latent_process_type, marginals_params_method, portfolio_type, cpu_processes = None):
    '''calculate optimal portfolio weights and its risk metrics VaR and CVaR
    portfolio_type = 'I' -- investment portfolio; w_i > 0, Sum(w_i) = 1
    portfolio_type = 'R' -- risk-neutral portfolio; Sum(w_i) = 0

    '''

    print('calc portfolio')

    T = len(marginals_params)
    dim = len(marginals_params[0])
    count_marginal_params = len(marginals_params[0][0])
    count_latent_process_params = len(latent_process_params[0])

    '''initialize optimization params'''
    if portfolio_type == 'I':
        eq_weight = np.ones(dim) / dim
        x0 = np.array([0, *eq_weight])
        constr = {'type': 'eq', 'fun': lambda x: np.sum(x[1:]) - 1}
        lb = np.zeros(dim + 1)
        lb[0] = -1
        lb[1:] = lb[1:]
        rb =  np.ones(dim + 1)
        bounds = Bounds(lb, rb)

    if portfolio_type == 'R':
        weight = normalize(neutralize( np.random.uniform(-1, 1, dim) ) )
        constr1 = {'type': 'eq', 'fun': lambda x: np.sum(np.abs(x[1:])) - 1}
        constr2 = {'type': 'eq', 'fun': lambda x: np.sum(x[1:]) - 0}
        constr = [constr1, constr2]
        x0 = np.array([0, *weight])

    shared_marginals_params = RawArray('d', marginals_params.ravel())
    shared_latent_process_params = RawArray('d', latent_process_params.ravel())
    shared_var = RawArray('d', T)
    shared_cvar = RawArray('d', T)
    shared_weight = RawArray('d', T * dim)

    global optimize_single
    def optimize_single(idx):
        i1 = (idx - 1 + window_len) * dim * count_marginal_params
        i2 = i1 + dim * count_marginal_params

        local_marginals_params = np.array(shared_marginals_params[i1:i2]).reshape((dim, count_marginal_params))
        rvs = get_rvs(local_marginals_params, MC_iterations, method = marginals_params_method)
        pseudo_obs = jit_pobs(rvs)
        i3 = (idx - 1 + window_len) * count_latent_process_params
        i4 = i3 + count_latent_process_params
        local_latent_process_params = np.array(shared_latent_process_params[i3:i4])[1:]
        copula_pdf_data = latent_process_sampler(copula.np_pdf(), copula.transform,
                         local_latent_process_params, pseudo_obs, latent_process_type, window_len, MC_iterations)
        del pseudo_obs
        rvs = np.exp(rvs)
        if portfolio_type == 'I': 
            min_weight = minimize(F_cvar_wq, x0 = x0, 
                                            args = (gamma, rvs, copula_pdf_data), 
                                            method = 'SLSQP', 
                                            bounds = bounds,
                                            tol = 1e-7,
                                            constraints = constr)
        if portfolio_type == 'R':
            min_weight = minimize(F_cvar_wq2, x0 = x0, 
                                            args = (gamma, rvs, copula_pdf_data), 
                                            method = 'SLSQP', 
                                            jac = F_cvar_wq2_jac,
                                            tol = 1e-3,
                                            constraints = constr)            
        del rvs
        del copula_pdf_data
        
        i1 = idx - 1 + window_len
        i2 = i1 + dim

        shared_var[i1] = min_weight.x[0]
        shared_cvar[i1] = min_weight.fun

        i3 = (idx - 1 + window_len) *  dim
        i4 = i3 + dim
        shared_weight[i3:i4] = min_weight.x[1:]
        del min_weight
        collected = gc.collect()    
    
    if cpu_processes is None:
        processes = cpu_count()
    else:
        processes = cpu_processes
    pool = Pool(processes = processes)   

    iters = T - window_len + 1
    with pool:
        generator = tqdm(pool.imap(optimize_single, range(0, iters)), total = iters)
        for i in generator:
            continue

    del optimize_single
    shared_var = np.frombuffer(shared_var)
    shared_cvar = np.frombuffer(shared_cvar)
    shared_weight = np.frombuffer(shared_weight).reshape((T,dim))
    return shared_var, shared_cvar, shared_weight


def risk_metrics(copula, 
                 data, 
                 window_len, 
                 gamma, 
                 MC_iterations, 
                 marginals_params_method,
                 latent_process_type, 
                 latent_process_tr = 500,
                 optimize_portfolio = True, 
                 portfolio_type = 'I',
                 portfolio_weight = None, 
                 cpu_processes = None):
    '''calculate risk metrics VaR and CVaR and optimize portfolio weights'''

    T = len(data)
    if window_len > T:
        raise ValueError(f'Length of window = {window_len} is more than length of data = {T}')

    # dwt = copula.calculate_dwt(T, latent_process_tr) * np.sqrt(T/window_len)
    dwt = np.random.normal(0, 1, size = (T, latent_process_tr)) * np.sqrt(1.0/window_len)
    latent_process_params = get_latent_process_params(copula, data, latent_process_type, window_len, dwt)
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
                var_data, cvar_data, weight_data = optimal_portfolo(gamma_i, window_len, MC_j,
                                                                    marginals_params, latent_process_params, 
                                                                    copula, latent_process_type.upper(),
                                                                    marginals_params_method, portfolio_type, 
                                                                    cpu_processes)
            else:
                var_data, cvar_data, weight_data = calculate_cvar(copula, latent_process_params, latent_process_type.upper(),
                                                                  marginals_params, marginals_params_method,
                                                                  gamma_i, window_len,
                                                                  MC_j, portfolio_weight)
            res[gamma_i] = dict()
            res[gamma_i][MC_j] = dict()
            res[gamma_i][MC_j]['var'] = var_data
            res[gamma_i][MC_j]['cvar'] = cvar_data
            res[gamma_i][MC_j]['weight'] = weight_data
    return res
