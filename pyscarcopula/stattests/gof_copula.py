import numpy as np
import sympy as sp

from pyscarcopula.sampler.sampler_ou import p_sampler_ou, m_sampler_ou, m_jit_mlog_likelihood_ou
from pyscarcopula.sampler.sampler_ld import p_sampler_ld
from pyscarcopula.auxiliary.funcs import pobs

from scipy.stats import chi2, norm, cramervonmises_2samp

def Rosenblatt_transform(copula, pobs_data, param):
    T = len(pobs_data)
    d = copula.dim
    u = sp.symbols('u0:%d'%(d))
    r = sp.symbols('r')

    res = np.zeros_like(pobs_data)
    expr = copula.sp_cdf()
    for i in range(1, d):
        #expr = sp.together(expr.diff(u[i], 1))
        expr = expr.diff(u[i - 1], 1)
        expr_func = sp.lambdify((u, r), expr, modules = 'numpy', cse = True)
        for k in range(0, T):
            arg1 = np.ones(d)
            arg1[0:i] = pobs_data[k][0:i]
            arg2 = np.ones(d)
            arg2[0:i + 1] = pobs_data[k][0:i + 1]
            res[k][i] = expr_func(arg2, param[k]) / expr_func(arg1, param[k])
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

def get_smoothed_sample(copula, fit_result, T):
    lp = 10000

    if fit_result.method.lower() == 'mle':
        smoothed_sample = np.ones(T) * copula.transform(fit_result.x[0])
    elif fit_result.method.lower() == 'scar-p-ou':
        dwt = np.random.normal(0, 1, size = (T, lp)) * np.sqrt(1/T)
        # smoothed_sample = copula.transform(np.mean(sample, axis = 1))
        sample = copula.transform(p_sampler_ou(fit_result.x, dwt))
        smoothed_sample = np.mean(sample, axis = 1)
    elif fit_result.method.lower() == 'scar-m-ou':
        lp = 1000
        dwt = np.random.normal(0, 1, size = (T, lp)) * np.sqrt(1/T)
        res, a1t, a2t = m_jit_mlog_likelihood_ou(fit_result.x, pobs_data, dwt, lp, 10, False, copula.np_pdf(), copula.transform)
        sample = copula.transform(m_sampler_ou(fit_result.x, a1t, a2t, dwt))
        smoothed_sample = np.mean(sample, axis = 1)
    elif fit_result.method.lower() == 'scar-p-ld':
        dwt = np.random.normal(0, 1, size = (T, lp)) * np.sqrt(1/T)
        sample = copula.transform(p_sampler_ld(fit_result.x, dwt))
        smoothed_sample = np.mean(sample, axis = 1)
    else:
        raise ValueError(f'method {fit_result.method} not implemented')
    return smoothed_sample

def gof_test(copula, data, fit_result, to_pobs = True):
    if to_pobs == True:
        pobs_data = pobs(data)
    else:
        pobs_data = data
    T = len(pobs_data)
    smoothed_sample = get_smoothed_sample(copula, fit_result, T)
    Rosenblatt_tranformed = Rosenblatt_transform(copula, pobs_data, smoothed_sample)
    cvm_test_result = cvm_test(Rosenblatt_tranformed)
    return cvm_test_result
