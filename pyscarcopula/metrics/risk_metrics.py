import numpy as np
from scipy.optimize import Bounds, minimize

from numba import njit, jit, prange
from tqdm import tqdm

from pyscarcopula.sampler.scar_ou.sampler_ou import p_sampler_no_hist_ou, p_sampler_1_step_ou, p_sampler_init_state
from pyscarcopula.sampler.scar_ld.sampler_ld import p_sampler_no_hist_ld
from pyscarcopula.sampler.scar_ds.sampler_ds import p_sampler_no_hist_ds

from pyscarcopula.marginal.norm import jit_norm_marginals, jit_norm_rvs
from pyscarcopula.aux_functions.funcs import jit_pobs

from multiprocessing import Pool, cpu_count, RawArray
from functools import partial

#from memory_profiler import profile
import gc

'''optimization for F(w, q), F(q)'''
@jit(nopython=True, cache = True, nogil = True)
def F_cvar_wq(x, gamma, exp_log_returns, copula_pdf_data):
    q = x[0]
    weight = x[1:]
    F = 0
    n = len(copula_pdf_data)
    m = len(exp_log_returns[0])
    for k in range(0, n):
        loss = 0
        for j in range(0, m):
            loss += exp_log_returns[k][j] * weight[j]
        loss = np.maximum(1 - loss - q, 0)
        F += copula_pdf_data[k] * loss
    F = q + 1 / (1 - gamma) * F / n

    #loss = 1 - exp_log_returns @ weight    
    #F = q + 1 / (1 - gamma) * np.mean(copula_pdf_data * np.maximum(loss - q, 0))
    return F

@jit(nopython=True, cache = True)#, nogil = True)
def F_cvar_q(q, gamma, loss, copula_pdf_data):
    #F = q + 1 / (1 - gamma) * np.mean(copula_pdf_data * np.maximum(loss - q, 0))
    n = len(copula_pdf_data)
    mean = 0.0
    for k in range(0, n):
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


def get_marginals_params_params(data, window_len, method):
    print('calc marginals_params')
    available_methods = ['norm']
    if method == 'norm':
        res = jit_norm_marginals(data, window_len)
    else:
        raise ValueError(f'given method {method} is not implemented. Available methods: {available_methods}')
    return res

def get_rvs(params, N, method):
    available_methods = ['norm']
    if method == 'norm':
        res = jit_norm_rvs(params, N)
    else:
        raise ValueError(f'given method {method} is not implemented. Available methods: {available_methods}')
    return res

'''
def get_latent_process_params(data, window_len, latent_process_tr, copula, method, cpu_processes = None):
    #calculate copula params

    print('calc copula params')
    T = len(data)
    #initialize resulting array. First element is copula log_lik.
    #   Others -- copula params (alpha1, alpha2, alpha3)
    #res = np.zeros((T, 4))

    #set special initial for LD sampler
    if method == 'SCAR-P-LD' or method == 'SCAR-M-LD':
        alpha0 = np.array([0.5, 0.5, 0.5])
    else:
        alpha0 = np.array([0.05, 0.95, 0.05])

    #if M sampler fails -> try again with P sampler
    additional_methods = {}
    additional_methods['SCAR-M-OU'] = 'SCAR-P-OU'
    additional_methods['SCAR-P-OU'] = 'SCAR-P-OU'
    additional_methods['SCAR-P-LD'] = 'SCAR-P-LD'
    additional_methods['SCAR-M-DS'] = 'SCAR-P-DS'
    additional_methods['SCAR-P-DS'] = 'SCAR-P-DS'

    global fit_copula
    def fit_copula(idx):
        if method == 'MLE':
            pobs = jit_pobs(data[idx : idx + window_len])
            cop_fit_result = copula.fit(pobs, method = method, latent_process_tr = latent_process_tr, accuracy = 1e-5, to_pobs = False)
            res = np.array([cop_fit_result.fun, *cop_fit_result.x, 0, 0])
        else:
            pobs = jit_pobs(data[idx : idx + window_len])
            cop_fit_result = copula.fit(pobs, method = method, latent_process_tr = latent_process_tr, accuracy = 1e-3,
                m_iters=5, alpha0 = alpha0, to_pobs = False)
            if cop_fit_result.fun == -10000.0:
                cop_fit_result = copula.fit(pobs, method = additional_methods[method], latent_process_tr = latent_process_tr, accuracy= 1e-3, m_iters=5)
            res = np.array([cop_fit_result.fun, *cop_fit_result.x])
            #alpha0 = cop_fit_result.x
        collected = gc.collect()    
        return res
    

    if cpu_processes is None:
        processes = cpu_count()
    else:
        processes = cpu_processes
        
    pool = Pool(processes = processes)
    fit_data = []
    iters = T - window_len + 1
    with pool:
        fit_results = tqdm(pool.imap(fit_copula, range(0, iters)), total = iters)
        for result in fit_results:
            fit_data.append(result)
    del fit_copula

    fit_data = np.array(fit_data)
    fit_data = np.concatenate((np.zeros(shape=(window_len - 1, 4)), fit_data))
    return fit_data
'''
def get_latent_process_params(copula, data, window_len, crns, latent_process_tr, method, cpu_processes = None):
    print('calc copula params')

    '''set special initial for LD sampler'''
    if method == 'SCAR-P-LD' or method == 'SCAR-M-LD':
        alpha0 = np.array([0.5, 0.5, 0.5])
    else:
        alpha0 = np.array([0.05, 0.95, 0.05])

    '''if M sampler fails -> try again with P sampler'''
    additional_methods = {}
    additional_methods['SCAR-M-OU'] = 'SCAR-P-OU'
    additional_methods['SCAR-P-OU'] = 'SCAR-P-OU'
    additional_methods['SCAR-P-LD'] = 'SCAR-P-LD'
    additional_methods['SCAR-M-DS'] = 'SCAR-P-DS'
    additional_methods['SCAR-P-DS'] = 'SCAR-P-DS'

    T = len(data)
    dim = len(data[0])
    #crns = copula.calculate_crns(T + window_len, latent_process_tr)
    prev_result = None
    lp_mat_dim = 4
    shared_latent_process_params = RawArray('d', T * lp_mat_dim)
    shared_crns = RawArray('d', crns.flatten())
    shared_data = RawArray('d', data.flatten())

    global get_latent_process_params_single
    def get_latent_process_params_single(idx):
        i1 = idx * dim
        i2 = i1 +  dim * window_len
        current_data = np.array(shared_data[i1:i2]).reshape((window_len, dim))
        pobs = jit_pobs(current_data)
        if method == 'MLE':
            cop_fit_result = copula.fit(pobs, method = method, accuracy = 1e-5, to_pobs = False)
            result = np.array([*cop_fit_result.x, 0, 0])
        else:
            i3 = idx * latent_process_tr
            i4 = i3 + window_len * latent_process_tr
            current_crns = np.array(shared_crns[i3:i4]).reshape((window_len, latent_process_tr))
            cop_fit_result = copula.fit(pobs, alpha0 = alpha0, method = method,
                                        latent_process_tr = latent_process_tr, accuracy = 1e-3,
                                        m_iters=5,
                                        to_pobs = False, crns = current_crns)
            if cop_fit_result.fun == -10000.0:
                cop_fit_result = copula.fit(pobs, alpha0 = alpha0, method = additional_methods[method],
                                            latent_process_tr = latent_process_tr, accuracy= 1e-3,
                                            m_iters=5, crns = current_crns)
            result = np.array([*cop_fit_result.x])
        i5 = (idx + window_len - 1) * lp_mat_dim
        i6 = i5 + lp_mat_dim
        shared_latent_process_params[i5:i6] = np.array([cop_fit_result.fun, *result])
        collected = gc.collect()    

    iters = T - window_len + 1
    if cpu_processes is None:
        processes = cpu_count()
    else:
        processes = cpu_processes
        
    pool = Pool(processes = processes)

    #compile
    get_latent_process_params_single(0)

    with pool:
        generator = tqdm(pool.imap(get_latent_process_params_single, range(0, iters)), total = iters)
        for i in generator:
            continue  
    del get_latent_process_params_single

    res = np.frombuffer(shared_latent_process_params).reshape((T, lp_mat_dim))
    return res


@jit(nopython = True, cache = True)
def latent_process_sampler(latent_process_params, latent_process_type, random_states_sequence, MC_iterations, init_state = None):
    if latent_process_type == 'MLE':
        current_state = np.array([latent_process_params[0]])
    if latent_process_type == 'SCAR-P-OU':
        current_state = p_sampler_no_hist_ou(latent_process_params, random_states_sequence, MC_iterations, init_state)
    if latent_process_type == 'SCAR-M-OU':
        current_state = p_sampler_no_hist_ou(latent_process_params, random_states_sequence, MC_iterations, init_state)
#     if method == 'SCAR-P-LD':
#         current_state = p_sampler_no_hist_ld(init_state, latent_process_params, crns)
#     if method == 'SCAR-P-DS':
#         current_state = p_sampler_no_hist_ds(init_state, latent_process_params, crns)
# -    if method == 'SCAR-M-DS':
#         current_state = p_sampler_no_hist_ds(init_state, latent_process_params, crns)
    #copula_pdf_data = pdf(pseudo_obs.T, transform(p_data))
    return current_state


@jit(nopython = True, cache = True)
def latent_process_sampler_1_step(latent_process_params, latent_process_type, random_state, MC_iterations, prev_state, dt):
    '''calculate copula values from random pseudo_obs'''
    if latent_process_type == 'MLE':
        current_state = np.array([latent_process_params[0]])
    if latent_process_type == 'SCAR-P-OU':
        current_state = p_sampler_1_step_ou(latent_process_params, random_state, MC_iterations, prev_state, dt)
    if latent_process_type == 'SCAR-M-OU':
        current_state = p_sampler_1_step_ou(latent_process_params, random_state, MC_iterations, prev_state, dt)
#     if method == 'SCAR-P-LD':
#         current_state = p_sampler_no_hist_ld(latent_process_params, crns)
#     if method == 'SCAR-P-DS':
#         current_state = p_sampler_no_hist_ds(latent_process_params, crns)
# -    if method == 'SCAR-M-DS':
#         current_state = p_sampler_no_hist_ds(latent_process_params, crns)
    return current_state


@jit(nopython = True, cache = True)
def latent_process_init_state(latent_process_params, latent_process_type, MC_iterations):
    if latent_process_type == 'MLE':
        init_state = np.array([latent_process_params[0]])
    if latent_process_type == 'SCAR-P-OU':
        init_state = p_sampler_init_state(latent_process_params, MC_iterations)
    if latent_process_type == 'SCAR-M-OU':
        init_state = p_sampler_init_state(latent_process_params, MC_iterations)
    return init_state


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


'''
def optimal_cvar(copula, latent_process_params, latent_process_type,
                 marginals_params, marginals_params_method,
                 gamma, window_len, MC_iterations,
                 portfolio_weight = None, cpu_processes = None):
    #calculate risk metrics VaR and CVaR for portfolio with fixed weights
    print('calc portfolio')

    T = len(marginals_params)
    dim = len(marginals_params[0])
    count_marginal_params = len(marginals_params[0][0])
    count_latent_process_params = len(latent_process_params[0])

    weight = []
    if portfolio_weight is None:
        weight = np.ones(dim) / dim
    else:
        weight = portfolio_weight

    x0 = 0

    shared_marginals_params = RawArray('d', marginals_params.ravel())
    shared_latent_process_params = RawArray('d', latent_process_params.ravel())
    shared_var = RawArray('d', T)
    shared_cvar = RawArray('d', T)

    global optimize_single
    def optimize_single(idx):
        i1 = (idx - 1 + window_len) * dim * count_marginal_params
        i2 = i1 + dim * count_marginal_params

        local_marginals_params = np.array(shared_marginals_params[i1:i2]).reshape((dim, count_marginal_params))

        rvs = get_rvs(local_marginals_params, MC_iterations, method = marginals_params_method)
        loss = 1 - np.exp(rvs) @ weight
        pseudo_obs = jit_pobs(rvs)
        del rvs 
        i3 = (idx - 1 + window_len) * count_latent_process_params
        i4 = i3 + count_latent_process_params
        local_latent_process_params = np.array(shared_latent_process_params[i3:i4])[1:]


        if idx == 0:
            current_state = latent_process_sampler(latent_process_params, method, crns)
        else:
            prev_latent_process_params = np.array(shared_latent_process_params[i3 - count_latent_process_params:i3])[1:]
            init_state = latent_process_sampler_1_step(latent_process_params, method, crns, dt, prev_state)
        copula_pdf_data = latent_process_sampler(copula.np_pdf(), copula.transform,
                         local_latent_process_params, pseudo_obs, latent_process_type, window_len, MC_iterations)
        
        min_alpha = minimize(F_cvar_q, x0 = x0, 
                                       args=(gamma, loss, copula_pdf_data),
                                       method='SLSQP',
                                       tol = 1e-5)
        
        del copula_pdf_data

        i1 = idx - 1 + window_len
        shared_var[i1] = min_alpha.x
        shared_cvar[i1] = min_alpha.fun

        del min_alpha
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

    return shared_var, shared_cvar, weight
'''

def calculate_cvar(copula, latent_process_params, latent_process_type,
                   marginals_params, marginals_params_method,
                   gamma, window_len,
                   MC_iterations, portfolio_weight = None,
                   cpu_processes = None):
    print('calc portfolio')
    T = len(marginals_params)
    dt = 1/window_len
    dim = len(marginals_params[0])
    count_marginal_params = len(marginals_params[0][0])
    count_latent_process_params = len(latent_process_params[0])

    if portfolio_weight is None:
        weight = np.ones(dim) / dim
    else:
        weight = portfolio_weight

    shared_marginals_params = RawArray('d', marginals_params.flatten())
    shared_latent_process_params = RawArray('d', latent_process_params.flatten())
    shared_var = RawArray('d', T)
    shared_cvar = RawArray('d', T)

    random_state_sequence = np.random.choice(range(1, 100 * T), size = T, replace = False)
    shared_random_state_sequence = RawArray('i', random_state_sequence)

    global calculate_cvar_single
    def calculate_cvar_single(idx):
        i0 = idx - 1 + window_len
        i1 = i0 * dim * count_marginal_params
        i2 = i1 + dim * count_marginal_params
        current_marginals_params = np.array(shared_marginals_params[i1:i2]).reshape((dim, count_marginal_params))

        rvs = get_rvs(current_marginals_params, MC_iterations, method = marginals_params_method)
        loss = 1 - np.exp(rvs) @ weight
        pseudo_obs = jit_pobs(rvs)
        del rvs
        i3 = i0 * count_latent_process_params
        i4 = i3 + count_latent_process_params
        current_latent_process_params = np.array(shared_latent_process_params[i3:i4])[1:]

        current_state = latent_process_sampler(current_latent_process_params, latent_process_type, 
                                               shared_random_state_sequence[idx : idx + window_len], MC_iterations)
        copula_pdf_data = copula.np_pdf()(pseudo_obs.T, copula.transform(current_state))
        del pseudo_obs
        del current_state

        x0 = 0
        min_result = minimize(F_cvar_q, x0 = x0,
                                       args=(gamma, loss, copula_pdf_data),
                                       method='SLSQP',
                                       tol = 1e-5)

        del copula_pdf_data
        del loss

        shared_var[i0] = min_result.x
        shared_cvar[i0] = min_result.fun
        collected = gc.collect()

    iters = T - window_len + 1
    if cpu_processes is None:
        processes = cpu_count()
    else:
        processes = cpu_processes
    pool = Pool(processes = processes) 

    #compile
    calculate_cvar_single(0)

    with pool:
        generator = tqdm(pool.imap(calculate_cvar_single, range(0, iters)), total = iters)
        for i in generator:
            continue  
    del calculate_cvar_single

    return np.frombuffer(shared_var), np.frombuffer(shared_cvar), weight

def risk_metrics(copula, data, window_len, 
                 gamma, MC_iterations, marginals_params_method,
                 latent_process_type, latent_process_tr = 500,
                 optimize_portfolio = True, portfolio_type = 'I',
                 portfolio_weight = None, cpu_processes = None):
    '''calculate risk metrics VaR and CVaR and optimize portfolio weights'''

    T = len(data)
    if window_len > T:
        raise ValueError(f'Length of window = {window_len} is more than length of data = {T}')

    crns = copula.calculate_crns(T, latent_process_tr)
    latent_process_params = get_latent_process_params(copula, data, window_len, crns, latent_process_tr, latent_process_type, cpu_processes)
    del crns
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
                                                                    copula, latent_process_type, 
                                                                    marginals_params_method, portfolio_type, 
                                                                    cpu_processes)
            else:
                var_data, cvar_data, weight_data = calculate_cvar(copula, latent_process_params, latent_process_type,
                                                                  marginals_params, marginals_params_method,
                                                                  gamma_i, window_len,
                                                                  MC_j, portfolio_weight, 
                                                                  cpu_processes)
            res[gamma_i] = dict()
            res[gamma_i][MC_j] = dict()
            res[gamma_i][MC_j]['var'] = var_data
            res[gamma_i][MC_j]['cvar'] = cvar_data
            res[gamma_i][MC_j]['weight'] = weight_data
    return res
