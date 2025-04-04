from numba import jit
from tqdm import tqdm
import numpy as np
import pandas as pd
from pyscarcopula.sampler.sampler_ou import init_state_ou
from pyscarcopula.sampler.sampler_ld import init_state_ld
from pyscarcopula.auxiliary.funcs import pobs
from pyscarcopula.marginal.marginals import cdf

from time import localtime, strftime
import os

def latent_process_init_state(alpha, latent_process_type, MC_iterations):
    if latent_process_type.upper() == 'MLE':
        # init_state = np.array([alpha[0]])
        final_state = alpha
    elif latent_process_type.upper() in ['SCAR-P-OU', 'SCAR-M-OU', 'SCAR-S-OU']:
        init_state = init_state_ou(alpha, MC_iterations)
    elif latent_process_type.upper() == 'SCAR-P-LD':
        init_state = init_state_ld(alpha, MC_iterations)
    return init_state

def latent_process_final_state(alpha,
                               latent_process_type,
                               latent_process_tr):
    if latent_process_type.upper() == 'MLE':
        # final_state = np.array([alpha[0]])
        final_state = alpha
    elif latent_process_type.upper() in ['SCAR-P-OU', 'SCAR-M-OU']:
        theta, mu, nu = alpha[0], alpha[1], alpha[2]
        xs = mu
        sigma2 = nu**2 / (2 * theta) * (1 - np.exp(-2 * theta))
        final_state = np.random.normal(loc = mu, scale = np.sqrt(sigma2), size = latent_process_tr)
    elif latent_process_type.upper() == 'SCAR-S-OU':
        theta, mu, nu = alpha[0], alpha[1], alpha[2]
        xs = mu
        sigma2 = nu**2 / (2 * theta)
        final_state = np.random.normal(loc = mu, scale = np.sqrt(sigma2), size = latent_process_tr)
    elif latent_process_type.upper() == 'SCAR-P-LD':
        theta, mu, nu = alpha[0], alpha[1], alpha[2]
        xs = mu
        s = nu**2 / (2 * theta) * (1 - np.exp(-2 * theta))
        u = np.random.uniform(0, 1, size = latent_process_tr)
        final_state = np.arctanh(2 * u - 1) * 2 * s + xs
    return final_state    

def get_latent_process_params(copula, 
                              returns_data, method, 
                              window_len, dwt, M_iterations = 5,
                              stationary = False,
                              save_logs = False, logs_path = None,
                              marginals_method = None, marginals_params = None):
    """
    Estimates the parameters of a latent process model for a given copula over a rolling window of data.

    This function fits a latent process model to the dependence structure captured by a copula, using
    a rolling window approach. It supports various latent process types and estimation methods.
    The function also has the capability to save the estimated parameters to a log file.

    Args:
        copula: An object representing a copula model (ArchimedianCopula or inherited classes). It must have a `fit` method 
                that takes data, method of fitting, and other optional parameters.
        returns_data: A NumPy array of shape (T, dim) representing the historical time series data of log-returns, 
                    where T is the number of time steps and dim is the number of assets.
        method: A string specifying the type of latent process and estimation method to use. 
                Options include 'MLE', 'SCAR-P-OU', 'SCAR-M-OU', 'SCAR-S-OU', and 'SCAR-P-LD'.
        window_len: An integer representing the length of the rolling window used to estimate the parameters.
        dwt: A NumPy array representing the Wiener process increments,
             which are used for latent process types other than 'MLE'. 
        M_iterations: An integer representing the number of iterations used in the parameter estimation 
                    process for some latent process types. Defaults to 5.
        stationary: A boolean indicating whether to enforce stationarity in the latent process. 
                    Defaults to False.
        save_logs: A boolean indicating whether to save the estimated parameters to a log file.
                    Defaults to False.
        logs_path: An optional string specifying the path to save the log file. If None and save_logs is True, 
                   a default directory "logs" in the current working directory will be used. Defaults to None.
        marginals_method: A string specifying the method used for fitting marginal distributions. 
                          If None, the empirical CDF is used (via `pobs` function).
        marginals_params: If marginals_method is not None, it is array with precalculated parameters for marginal distributions.

    Returns:
        A NumPy array of shape (T, 4) where T is the length of `returns_data`. Each row represents the 
        estimated parameters for a given time step, with the first element being the optimization
        result (function value), and the next three elements representing the estimated parameters of latent process.

    Raises:
        ValueError: If the provided `method` is not recognized.

    """
    x = None
    T = len(returns_data)
    iters = T - window_len + 1

    latent_process_params = np.zeros((T, 4))

    if method.upper() == 'MLE':
        latent_process_tr = None
    else:
        latent_process_tr = len(dwt[0])

    for k in tqdm(range(0, iters)):
        idx = k + window_len - 1

        # u = pobs(returns_data[k:window_len + k])
        if marginals_method is None:
            u = pobs(returns_data[k:window_len + k])
        else:
            u = cdf(returns_data[k:window_len + k], marginals_method, marginals_params[idx])
        
        if method.upper() == 'MLE':
            cop_fit_result = copula.fit(u, method = method, to_pobs = False)
            latent_process_params[idx] = np.array([cop_fit_result.fun, *cop_fit_result.x, 0, 0])
            x = cop_fit_result.x
        else:
            cop_fit_result = copula.fit(u,
                                        alpha0 = x,
                                        method = method,
                                        M_iterations = M_iterations,
                                        to_pobs = False,
                                        dwt = dwt[k:window_len + k],
                                        stationary = stationary
                                        )
        
            if np.isnan(cop_fit_result.fun) == True or int(cop_fit_result.fun) == -10**10:
                latent_process_params[idx] = latent_process_params[idx - 1]
            else:
                x = np.array(cop_fit_result.x)
                latent_process_params[idx] = np.array([cop_fit_result.fun, *cop_fit_result.x])

    if save_logs == True:
        '''log copula parameters result'''
        df = pd.DataFrame(latent_process_params)
        copula_name = copula.name.split(' ')[0]
        current_time = strftime("%Y-%m-%d_%H%M%S", localtime())

        if logs_path is None:
            directory = "logs"
        else:
            directory = logs_path

        if not os.path.exists(directory):
            os.makedirs(directory)
        df.to_csv(f"{directory}/{copula_name}_{method}_{latent_process_tr}_{current_time}.csv", sep = ';')

    return latent_process_params

