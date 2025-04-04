import numpy as np
import sympy as sp

from pyscarcopula.sampler.sampler_ou import jit_latent_process_conditional_expectation_p_ou
from pyscarcopula.sampler.sampler_ou import jit_latent_process_conditional_expectation_m_ou
from pyscarcopula.sampler.sampler_ld import jit_latent_process_conditional_expectation_p_ld
from pyscarcopula.auxiliary.funcs import pobs

from scipy.stats import chi2, norm, cramervonmises_2samp
from scipy.special import roots_hermite

from functools import lru_cache
from numba import njit

from pyscarcopula.sampler.sampler_ou import stationary_state_ou
from pyscarcopula.sampler.sampler_ld import stationary_state_ld



@lru_cache
def Rosenblatt_transform_funcs(copula):
    funcs = []
    expr = copula.sp_cdf()

    d = copula.dim
    u = sp.symbols('u0:%d'%(d), positive = True)
    r = sp.symbols('r')

    for i in range(1, d):
        expr = expr.diff(u[i - 1], 1)
        expr_func = sp.powsimp(expr)
        expr_func = njit(sp.lambdify((u, r), expr, modules = 'numpy', cse = True))
        funcs.append(expr_func)
    return funcs 

def Rosenblatt_transform(copula, pobs_data, param):
    T = len(pobs_data)
    d = copula.dim
    res = np.zeros_like(pobs_data)

    funcs = Rosenblatt_transform_funcs(copula)
    eps = 1e-9

    for i in range(1, d):
        for k in range(0, T):
            arg1 = np.ones(d)
            arg1[0:i] = pobs_data[k][0:i]
            arg2 = np.ones(d)
            arg2[0:i + 1] = pobs_data[k][0:i + 1]

            arg1 = np.clip(arg1, eps, 1.0 - eps)
            arg2 = np.clip(arg2, eps, 1.0 - eps)

            res[k][i] = funcs[i - 1](arg2, param[k]) / funcs[i - 1](arg1, param[k])
    res[:,0] = pobs_data[:,0]
    return res


def cvm_test(pobs_data_rt, seed = None):
    T = len(pobs_data_rt)
    d = len(pobs_data_rt[0])

    y_data = np.zeros(T)
    for k in range(0, T):
        val = 0.0
        for j in range(0, d):
            val += norm.ppf(pobs_data_rt[k][j])**2
        y_data[k] = chi2.cdf(val, df = d)

    size = 1_000_000
    if seed is None:
        set1 = np.random.uniform(0, 1, size = size)
    else:
        rng = np.random.default_rng(seed)
        set1 = rng.random(size = size)

    set2 = y_data
    cvm_result = cramervonmises_2samp(set1, set2, method='auto')
    return cvm_result


def latent_process_conditional_expectation_p_ou(copula, pobs_data, fit_result):
    alpha = np.array(fit_result.x)
    # latent_process_tr = np.maximum(fit_result.latent_process_tr, 10000)
    latent_process_tr = fit_result.latent_process_tr

    stationary = fit_result.stationary

    dwt = copula.calculate_dwt(method = fit_result.method, 
                               T = len(pobs_data), latent_process_tr = latent_process_tr)

    result = jit_latent_process_conditional_expectation_p_ou(alpha, pobs_data, dwt, 
                                                             copula.log_pdf, copula.transform, 
                                                             stationary)
    return result


def latent_process_conditional_expectation_p_ld(copula, pobs_data, fit_result):
    alpha = np.array(fit_result.x)
    # latent_process_tr = np.maximum(fit_result.latent_process_tr, 10000)
    latent_process_tr = fit_result.latent_process_tr

    stationary = fit_result.stationary

    dwt = copula.calculate_dwt(method = fit_result.method, 
                               T = len(pobs_data), latent_process_tr = latent_process_tr)

    result = jit_latent_process_conditional_expectation_p_ld(alpha, pobs_data, dwt, 
                                                             copula.log_pdf, copula.transform,
                                                             stationary)
    return result


def latent_process_conditional_expectation_m_ou(copula, pobs_data, fit_result):
    alpha = np.array(fit_result.x)
    # latent_process_tr = np.maximum(fit_result.latent_process_tr, 500)
    latent_process_tr = fit_result.latent_process_tr
    
    M_iterations = fit_result.M_iterations
    
    stationary = fit_result.stationary

    dwt = copula.calculate_dwt(method = fit_result.method, 
                               T = len(pobs_data), latent_process_tr = latent_process_tr)

    z, w = roots_hermite(250)
    args = w > 1e-3
    w = w[args]
    z = z[args]

    result = jit_latent_process_conditional_expectation_m_ou(alpha, pobs_data, M_iterations, dwt,
                                                             copula.np_log_pdf(numba_jit = True), 
                                                             copula.transform_jit,
                                                             z, w,
                                                             stationary)
    return result


def get_smoothed_sample(copula, pobs_data, fit_result):
    T = len(pobs_data)

    if fit_result.method.lower() == 'mle':
        smoothed_sample = np.ones(T) * fit_result.x[0]
    elif fit_result.method.lower() == 'scar-p-ou':
        smoothed_sample = latent_process_conditional_expectation_p_ou(copula, pobs_data, fit_result)
    elif fit_result.method.lower() == 'scar-m-ou':
        smoothed_sample = latent_process_conditional_expectation_m_ou(copula, pobs_data, fit_result)
    elif fit_result.method.lower() == 'scar-p-ld':
        smoothed_sample = latent_process_conditional_expectation_p_ld(copula, pobs_data, fit_result)
    else:
        raise ValueError(f'method {fit_result.method} not implemented')
    return smoothed_sample


def gof_test(copula, data, fit_result, to_pobs = True):
    if to_pobs == True:
        pobs_data = pobs(data)
    else:
        pobs_data = data
    T = len(pobs_data)
    smoothed_sample = get_smoothed_sample(copula, pobs_data, fit_result)
    Rosenblatt_tranformed = Rosenblatt_transform(copula, pobs_data, smoothed_sample)
    cvm_test_result = cvm_test(Rosenblatt_tranformed)
    return cvm_test_result
