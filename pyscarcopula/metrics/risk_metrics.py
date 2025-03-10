import numpy as np
from scipy.optimize import Bounds, minimize
from typing import Literal
from numba import jit, prange
from tqdm import tqdm
import gc

from pyscarcopula.metrics.latent_process import get_latent_process_params, latent_process_final_state
from pyscarcopula.marginal.marginals import fit_marginal, rvs, ppf

@jit(nopython=True)
def loss_func(r, weight):
    n = len(r)
    m = len(r[0])
    portfolio_return = np.zeros(n)
    for k in range(0, n):
        temp = 0.0
        for j in range(0, m):
            temp += np.exp(r[k][j]) * weight[j]
        portfolio_return[k] = temp
    loss = np.ones(n) - portfolio_return
    return loss


@jit(nopython=True)
def F_cvar_q(q, gamma, loss):
    n = len(loss)
    mean = 0.0
    for k in prange(0, n):
        mean += np.maximum(loss[k] - q[0], 0.0)
    mean = mean / n
    F = q[0] + 1 / (1 - gamma) * mean
    return F


@jit(nopython=True)
def F_cvar_wq(x, gamma, r):
    q = x[0]
    weight = x[1:]
    mean = 0.0
    n = len(r)
    m = len(r[0])
    for k in range(0, n):
        loss = 0.0
        for j in range(0, m):
            loss += np.exp(r[k][j]) * weight[j]
        loss = np.maximum(1.0 - loss - q, 0.0)
        mean += loss
    mean = mean / n
    F = q + 1 / (1 - gamma) * mean
    return F


# @jit(nopython=True, parallel = True, cache = True)
# def F_cvar_wq_pdf(x, gamma, r, copula_pdf_data):
#     q = x[0]
#     weight = x[1:]
#     F = 0.0
#     n = len(copula_pdf_data)
#     m = len(r[0])
#     for k in range(0, n):
#         loss = 0.0
#         for j in range(0, m):
#             loss += np.exp(r[k][j]) * weight[j]
#         loss = np.maximum(1.0 - loss - q, 0.0)
#         F += copula_pdf_data[k] * loss
#     F = q + 1 / (1 - gamma) * F / n
#     return F


# @jit(nopython=True, parallel = True, cache = True)
# def F_cvar_q_pdf(q, gamma, loss, copula_pdf_data):
#     n = len(copula_pdf_data)
#     mean = 0.0
#     for k in prange(0, n):
#         mean += copula_pdf_data[k] * np.maximum(loss[k] - q[0], 0.0)
#     mean = mean / n
#     F = q[0] + 1 / (1 - gamma) * mean
#     return F


# @jit(nopython=True)
# def calculate_copula_pdf_data(data, state, pdf, transform):
#     n = len(data)
#     copula_pdf_data = np.zeros(n)
#     if len(state) == 1:
#         for k in range(0, n):
#             copula_pdf_data[k] = pdf(data[k], transform(state[0]))
#     else:
#          for k in range(0, n):
#             copula_pdf_data[k] = pdf(data[k], transform(state[k]))       
#     return copula_pdf_data


def calculate_cvar(copula,
                   latent_process_params,
                   latent_process_type,
                   marginals_params,
                   marginals_method,
                   gamma,
                   window_len,
                   MC_iterations,
                   portfolio_weight):

    T = len(marginals_params)
    dim = len(marginals_params[0])

    var = np.zeros(T)
    cvar = np.zeros(T)
    iters = T - window_len + 1

    for k in tqdm(range(0, iters)):
        idx = k + window_len - 1       
        current_state = latent_process_final_state(latent_process_params[idx][1:], latent_process_type, MC_iterations)
        u = copula.get_sample(MC_iterations, copula.transform(current_state))

        if marginals_method in ['hyperbolic', 'stable']:
            r_sample = rvs(marginals_params[idx], MC_iterations, method = marginals_method)
            r = ppf(u, marginals_method, None, r_sample)
            del r_sample
        else:
            r = ppf(u, marginals_method, marginals_params[idx])

        del u
        loss = loss_func(r, portfolio_weight)
        del r

        x0 = 0
        min_result = minimize(F_cvar_q, x0 = x0,
                                        args=(gamma, loss),
                                        method='SLSQP',
                                        tol = 1e-7)
        del loss
        var[idx] = min_result.x[0]
        cvar[idx] = min_result.fun
        collected = gc.collect()

    return var, cvar, portfolio_weight


def calculate_cvar_optimal_portfolio(copula,
                                     latent_process_params,
                                     latent_process_type,
                                     marginals_params,
                                     marginals_method,
                                     gamma,
                                     window_len,
                                     MC_iterations):

    T = len(marginals_params)
    dim = len(marginals_params[0])

    eq_weight = np.ones(dim) / dim

    var = np.zeros(T)
    cvar = np.zeros(T)
    weight_data = np.zeros((T, dim))
    iters = T - window_len + 1

    constr = {'type': 'eq', 'fun': lambda x: np.sum(x[1:]) - 1}
    lb = np.zeros(dim + 1)
    lb[0] = -1
    rb =  np.ones(dim + 1)
    bounds = Bounds(lb, rb)
    x0 = np.array([0, *eq_weight])

    for k in tqdm(range(0, iters)):
        idx = k + window_len - 1            
        current_state = latent_process_final_state(latent_process_params[idx][1:], latent_process_type, MC_iterations)
        u = copula.get_sample(MC_iterations, copula.transform(current_state))

        if marginals_method in ['hyperbolic', 'stable']:
            r_sample = rvs(marginals_params[idx], MC_iterations, method = marginals_method)
            r = ppf(u, marginals_method, None, r_sample)
            del r_sample
        else:
            r = ppf(u, marginals_method, marginals_params[idx])

        min_result = minimize(F_cvar_wq, x0 = x0,
                                        args=(gamma, r),
                                        method='SLSQP',
                                        bounds = bounds,
                                        constraints = constr,
                                        tol = 1e-7)
        del r
        var[idx] = min_result.x[0]
        cvar[idx] = min_result.fun
        weight_data[idx] = min_result.x[1:dim+1]
        x0 = np.array(min_result.x)
        collected = gc.collect()

    return var, cvar, weight_data

def risk_metrics(copula, 
                 data, 
                 window_len, 
                 gamma = 0.95, 
                 MC_iterations = 100000, 
                 marginals_method: Literal['normal', 'hyperbolic', 'stable', 'logistic', 'johnsonsu', 'laplace'] = 'normal',
                 latent_process_type: Literal['mle', 'scar-p-ou', 'scar-m-ou', 'scar-p-ld'] = 'mle', 
                 latent_process_tr = 500,
                 M_iterations = 5,
                 seed = None,
                 optimize_portfolio = True, 
                 portfolio_weight = None, 
                 pre_calc_latent_process_params = None,
                 pre_calc_marginals_params = None,
                 save_logs = False,
                 logs_path = None,
                 stationary = False):
    """
    Calculates risk metrics (Value at Risk (VaR) and Conditional Value at Risk (CVaR)) for a portfolio 
    and optionally optimizes portfolio weights based on CVaR minimization.

    This function utilizes copula models, marginal distributions, and latent processes to estimate
    risk metrics over a rolling window of historical data. It supports various marginal distribution
    models and latent process types. The function can either calculate risk metrics for a fixed
    portfolio weight or optimize the portfolio weight to minimize CVaR.

    Args:
        copula: An object representing a copula model (ArchimedianCopula or inherited classes), which must have methods such as `get_sample`
            and `transform` for sampling and state transformation, respectively, and `calculate_dwt`
            for generating a Wiener process increment matrix.
        data: A NumPy array of shape (T, dim) representing the historical time series data of log-returns, where T is
            the number of time steps and dim is the number of assets.
        window_len: An integer representing the length of the rolling window used to calculate risk metrics.
        gamma: A float or a list of floats in the range (0, 1) representing the confidence level(s)
            for VaR and CVaR calculations. Defaults to 0.95.
        MC_iterations: An integer or a list of integers representing the number of Monte Carlo iterations 
            for risk estimation. Defaults to 100000.
        marginals_method: A string (Literal) specifying the marginal distribution method to use.
            Options are 'normal', 'hyperbolic', 'stable', 'logistic', 'johnsonsu', and 'laplace'.
            Defaults to 'normal'.
        latent_process_type: A string (Literal) specifying the type of latent process to use.
            Options are 'mle', 'scar-p-ou', 'scar-m-ou', and 'scar-p-ld'. Defaults to 'mle'.
        latent_process_tr: An integer used as a parameter for the latent process dwt calculation. 
            Defaults to 500.
        M_iterations: An integer representing the number of iterations used in latent process
            parameter estimation. Defaults to 5.
        seed: An integer representing the random seed for reproducibility. If None, a random seed is used.
            Defaults to None.
        optimize_portfolio: A boolean indicating whether to optimize the portfolio weights to
            minimize CVaR. If False, risk metrics are calculated using the provided or default
            portfolio weights. Defaults to True.
        portfolio_weight: A NumPy array of shape (dim,) representing the fixed portfolio weights to use
            when `optimize_portfolio` is False. If None, an equal-weighted portfolio is used.
            Defaults to None.
        pre_calc_latent_process_params: An optional pre-calculated array of latent process parameters.
            If provided, latent process parameters are not calculated. Defaults to None.
        pre_calc_marginals_params: An optional pre-calculated array of marginal distribution parameters.
            If provided, marginal distribution parameters are not calculated. Defaults to None.
        save_logs: A boolean indicating whether to save logs during the calculation of latent process 
            params. Defaults to False.
        logs_path: An optional string specifying the path to save the log file. If None and save_logs is True, 
                   a default directory "logs" in the current working directory will be used. Defaults to None.
        stationary: A boolean indicating whether to enforce stationarity in the latent process. 
                    Defaults to False.
    Returns:
        A dictionary where keys are gamma values. Each gamma value contains another dictionary with
        MC_iterations values as key. Each MC_iteration contains dictionary of  following keys :
        - 'var': A NumPy array of length T containing the calculated VaR values.
        - 'cvar': A NumPy array of length T containing the calculated CVaR values.
        - 'weight': A NumPy array of shape (T, dim) or of length T, containing the portfolio weights
          used (either fixed or optimized).

    Raises:
        ValueError: If `window_len` is greater than the length of the data.
    """
    T = len(data)
    if window_len > T:
        raise ValueError(f'Length of window = {window_len} is more than length of data = {T}')

    if portfolio_weight is None:
        dim = data.shape[1]
        portfolio_weight = np.ones(dim) / dim

    print('calc marginals params')

    if pre_calc_marginals_params is None:
        marginals_params = fit_marginal(data, marginals_method, window_len)
    else:
        marginals_params = pre_calc_marginals_params

    if pre_calc_latent_process_params is None:
        if seed is None:
            s = np.random.randint(0, 100000)
        else:
            s = seed

        dt = 1.0 / (window_len - 1)

        print(f"seed = {s}")
        dwt = copula.calculate_dwt(latent_process_type.upper(), T, latent_process_tr, seed, dt)
        print('calc copula params')

        latent_process_params = get_latent_process_params(copula, 
                                                          data, 
                                                          latent_process_type.upper(), 
                                                          window_len, 
                                                          dwt, 
                                                          M_iterations,
                                                          stationary,
                                                          save_logs, 
                                                          logs_path,
                                                          # marginals_method,
                                                          # marginals_params
                                                          )
        del dwt
    else:
        latent_process_params = pre_calc_latent_process_params

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
    print('calc portfolio risks')
    for gamma_i in gamma_list:
        for MC_j in MC_iterations_list:
            print(f"gamma = {gamma_i}, MC_iterations = {MC_j}")
            if optimize_portfolio == True:
                var, cvar, weight_data = calculate_cvar_optimal_portfolio(copula, 
                                                                        latent_process_params, 
                                                                        latent_process_type.upper(),
                                                                        marginals_params, 
                                                                        marginals_method,
                                                                        gamma_i, 
                                                                        window_len,
                                                                        MC_j)
            else:
                var, cvar, weight_data = calculate_cvar(copula, 
                                                        latent_process_params, 
                                                        latent_process_type.upper(),
                                                        marginals_params, 
                                                        marginals_method,
                                                        gamma_i, 
                                                        window_len,
                                                        MC_j, 
                                                        portfolio_weight)
            res[gamma_i] = dict()
            res[gamma_i][MC_j] = dict()
            res[gamma_i][MC_j]['var'] = var
            res[gamma_i][MC_j]['cvar'] = cvar
            res[gamma_i][MC_j]['weight'] = weight_data
    return res
