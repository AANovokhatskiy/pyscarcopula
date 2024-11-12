import numpy as np
import sympy as sp

from pyscarcopula.sampler.sampler_ou import jit_latent_process_conditional_expectation_p_ou
from pyscarcopula.sampler.sampler_ou import jit_latent_process_conditional_expectation_m_ou
from pyscarcopula.sampler.sampler_ld import jit_latent_process_conditional_expectation_p_ld
from pyscarcopula.auxiliary.funcs import pobs

from scipy.stats import chi2, norm, cramervonmises_2samp
from functools import lru_cache
from numba import njit


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

    for i in range(1, d):
        for k in range(0, T):
            arg1 = np.ones(d)
            arg1[0:i] = pobs_data[k][0:i]
            arg2 = np.ones(d)
            arg2[0:i + 1] = pobs_data[k][0:i + 1]
            res[k][i] = funcs[i - 1](arg2, param[k]) / funcs[i - 1](arg1, param[k])
    res[:,0] = pobs_data[:,0]
    return res


def cvm_test(pobs_data_rt):
    T = len(pobs_data_rt)
    d = len(pobs_data_rt[0])

    y_data = np.zeros(T)
    for k in range(0, T):
        val = 0.0
        for j in range(0, d):
            val += norm.ppf(pobs_data_rt[k][j])**2
        y_data[k] = chi2.cdf(val, df = d)

    set1 = np.random.uniform(0, 1, size = 10000)
    set2 = y_data
    cvm_result = cramervonmises_2samp(set1, set2, method='auto')
    return cvm_result


def latent_process_conditional_expectation_p_ou(copula, pobs_data, fit_result):
    alpha = np.array(fit_result.x)
    latent_process_tr = fit_result.latent_process_tr
    result = jit_latent_process_conditional_expectation_p_ou(copula.np_pdf(), copula.transform, 
                                                             pobs_data, alpha, latent_process_tr)
    return result


def latent_process_conditional_expectation_p_ld(copula, pobs_data, fit_result):
    alpha = np.array(fit_result.x)
    latent_process_tr = fit_result.latent_process_tr
    result = jit_latent_process_conditional_expectation_p_ld(copula.np_pdf(), copula.transform, 
                                                             pobs_data, alpha, latent_process_tr)
    return result


def latent_process_conditional_expectation_m_ou(copula, pobs_data, fit_result):
    alpha = np.array(fit_result.x)
    latent_process_tr = fit_result.latent_process_tr
    m_iters = fit_result.m_iterations
    result = jit_latent_process_conditional_expectation_m_ou(copula.np_pdf(), copula.transform, 
                                                             pobs_data, alpha, latent_process_tr, m_iters)
    return result


def get_smoothed_sample(copula, pobs_data, fit_result):
    T = len(pobs_data)

    if fit_result.method.lower() == 'mle':
        smoothed_sample = np.ones(T) * copula.transform(fit_result.x[0])
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
