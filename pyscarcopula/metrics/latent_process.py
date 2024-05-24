from numba import jit
from tqdm import tqdm
import numpy as np
import pandas as pd
from pyscarcopula.sampler.sampler_ou import p_sampler_no_hist_ou, p_sampler_no_hist_ou_rng, p_sampler_one_step_ou
from pyscarcopula.sampler.sampler_ou import p_sampler_one_step_ou_rng, p_sampler_init_state_ou
#from pyscarcopula.sampler.sampler_ou import m_sampler_ou_params

from pyscarcopula.sampler.sampler_ld import p_sampler_no_hist_ld, p_sampler_no_hist_ld_rng, p_sampler_one_step_ld, p_sampler_one_step_ld_rng, p_sampler_init_state_ld
from pyscarcopula.auxiliary.funcs import jit_pobs

from time import localtime, strftime
import os

@jit(nopython = True, cache = True)
def latent_process_init_state(latent_process_params, latent_process_type, MC_iterations):
    if latent_process_type == 'MLE':
        init_state = np.array([latent_process_params[0]])
    elif latent_process_type == 'SCAR-P-OU':
        init_state = p_sampler_init_state_ou(latent_process_params, MC_iterations)
    elif latent_process_type == 'SCAR-P-LD':
        init_state = p_sampler_init_state_ld(latent_process_params, MC_iterations)
    elif latent_process_type == 'SCAR-M-OU':
        init_state = p_sampler_init_state_ou(latent_process_params, MC_iterations)
    return init_state


@jit(nopython = True, cache = True)
def latent_process_sampler_one_step(latent_process_params,
                                    latent_process_type,
                                    dwt,
                                    dt,
                                    init_state):
    if latent_process_type == 'MLE':
        current_state = np.array([latent_process_params[0]])
    elif latent_process_type == 'SCAR-P-OU':
        current_state = p_sampler_one_step_ou(latent_process_params, dwt, dt, init_state)
    elif latent_process_type == 'SCAR-P-LD':
        current_state = p_sampler_one_step_ld(latent_process_params, dwt, dt, init_state)
    elif latent_process_type == 'SCAR-M-OU':
        current_state = p_sampler_one_step_ou(latent_process_params, dwt, dt, init_state)    
    return current_state


@jit(nopython = True, cache = True)
def latent_process_sampler_one_step_rng(latent_process_params,
                                        latent_process_type,
                                        latent_process_tr,
                                        random_state,
                                        dt,
                                        init_state):
    if latent_process_type == 'MLE':
        current_state = np.array([latent_process_params[0]])
    elif latent_process_type == 'SCAR-P-OU':
        current_state = p_sampler_one_step_ou_rng(latent_process_params, latent_process_tr, random_state, dt, init_state)
    elif latent_process_type == 'SCAR-P-LD':
        current_state = p_sampler_one_step_ld_rng(latent_process_params, latent_process_tr, random_state, dt, init_state)
    elif latent_process_type == 'SCAR-M-OU':
        current_state = p_sampler_one_step_ou_rng(latent_process_params, latent_process_tr, random_state, dt, init_state)
    return current_state


@jit(nopython = True, cache = True)
def latent_process_sampler(latent_process_params,
                               latent_process_type,
                               dwt,
                               dt,
                               init_state):
    if latent_process_type == 'MLE':
        current_state = np.array([latent_process_params[0]])
    elif latent_process_type == 'SCAR-P-OU':
        current_state = p_sampler_no_hist_ou(latent_process_params, dwt, dt, init_state)
    elif latent_process_type == 'SCAR-P-LD':
        current_state = p_sampler_no_hist_ld(latent_process_params, dwt, dt, init_state)
    elif latent_process_type == 'SCAR-M-OU':
        current_state = p_sampler_no_hist_ou(latent_process_params, dwt, dt, init_state)
    return current_state


@jit(nopython = True, cache = True)
def latent_process_sampler_rng(latent_process_params,
                               latent_process_type,
                               latent_process_tr,
                               random_states_sequence,
                               dt,
                               init_state):
    if latent_process_type == 'MLE':
        current_state = np.array([latent_process_params[0]])
    elif latent_process_type == 'SCAR-P-OU':
        current_state = p_sampler_no_hist_ou_rng(latent_process_params, latent_process_tr, random_states_sequence, dt, init_state)
    elif latent_process_type == 'SCAR-P-LD':
        current_state = p_sampler_no_hist_ld_rng(latent_process_params, latent_process_tr, random_states_sequence, dt, init_state)
    elif latent_process_type == 'SCAR-M-OU':
        current_state = p_sampler_no_hist_ou_rng(latent_process_params, latent_process_tr, random_states_sequence, dt, init_state)
    return current_state


def get_latent_process_params(copula, returns_data, method, window_len, dwt):
    print('calc copula params')

    #alpha0 = np.array([1.0, 0.5, 0.05])
    alpha0 = None
    T = len(returns_data)
    dt = 1/window_len
    iters = T - window_len + 1

    latent_process_params = np.zeros((T, 4))
    latent_process_tr = len(dwt[0])
    M = 5

    #for k in tqdm(range(0, iters)):
    for k in range(0, iters):
        print(f'========================={k}=========================')
        idx = k + window_len - 1
        pobs = jit_pobs(returns_data[k:window_len + k])
        if method == 'MLE':
            cop_fit_result = copula.fit(pobs, method = method, accuracy = 1e-5, to_pobs = False)
            latent_process_params[idx] = np.array([cop_fit_result.fun,*cop_fit_result.x, 0, 0])
        else:
            if k == 0:
                init_state = None
            else:
                init_state = latent_process_sampler_one_step(alpha0, method, dwt[k - 1], dt, init_state)
            cop_fit_result = copula.fit(pobs,
                                        alpha0 = alpha0,
                                        method = method,
                                        latent_process_tr = latent_process_tr,
                                        accuracy = 1e-3,
                                        m_iters = M,
                                        to_pobs = False,
                                        dwt = dwt[k:window_len + k],
                                        print_path = True,
                                        init_state = init_state)
            if np.isnan(cop_fit_result.fun) == True or int(cop_fit_result.fun) == -10**10:
                latent_process_params[idx] = latent_process_params[idx - 1]
                init_state = None
            else:
                alpha0 = np.array(cop_fit_result.x)
                latent_process_params[idx] = np.array([cop_fit_result.fun,*cop_fit_result.x])
        #print(init_state)
        print(latent_process_params[idx])

    '''log copula parameters result'''
    df = pd.DataFrame(latent_process_params)
    copula_name = copula.name.split(' ')[0]
    current_time = strftime("%Y-%m-%d_%H%M%S", localtime())

    directory = "logs"

    if not os.path.exists(directory):
        os.makedirs(directory)
    df.to_csv(f"{directory}/{copula_name}_{method}_{latent_process_tr}_{current_time}.csv", sep = ';')

    return latent_process_params

