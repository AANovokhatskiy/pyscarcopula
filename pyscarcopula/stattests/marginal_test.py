from scipy.stats import genhyperbolic, levy_stable, norm
from scipy.stats import ks_2samp
from scipy.stats import cramervonmises_2samp


def norm_stattest(fit_result, data):
    cvm_result = []
    for i in range(0, len(fit_result)):
        fittedRes = fit_result[i]
        r = norm.rvs(*fittedRes, size=10000, random_state=None)
        print("--------------")
        print(str(i) + "; Normal distribution; MLE")
        print(fittedRes)
        ks = ks_2samp(data[:,i], r, alternative='two-sided', method='auto')
        print(ks)
        cvm = cramervonmises_2samp(data[:,i], r, method='auto')
        print(cvm)
        cvm_result.append(cvm.pvalue)
    return cvm_result

def stable_stattest(fit_result, data):
    cvm_result = []
    for i in range(0, len(fit_result)):
        fittedRes = fit_result[i]
        r = levy_stable.rvs(*fittedRes, size=10000, random_state=None)
        print("--------------")
        print(str(i) + "; Stable distribution; MLE")
        print(fittedRes)
        ks = ks_2samp(data[:,i], r, alternative='two-sided', method='auto')
        print(ks)
        cvm = cramervonmises_2samp(data[:,i], r, method='auto')
        print(cvm)
        cvm_result.append(cvm.pvalue)
    return cvm_result

def gh_stattest(fit_result, data):
    cvm_result = []
    for i in range(0, len(fit_result)):
        fittedRes = fit_result[i]
        r = genhyperbolic.rvs(*fittedRes, size=10000, random_state=None)
        print("--------------")
        print(str(i) + "; Generalized hyperbolic distribution; MLE")
        print(fittedRes)
        ks = ks_2samp(data[:,i], r, alternative='two-sided', method='auto')
        print(ks)
        cvm = cramervonmises_2samp(data[:,i], r, method='auto')
        print(cvm)
        cvm_result.append(cvm.pvalue)
    return cvm_result
    
#import openturns as ot

# def meixner_stattest(fit_result, data):
#     cvm_result = []
#     for i in range(0, len(fit_result)):
#         fittedRes = fit_result[i]
#         random_sample = ot.MeixnerDistribution(*fittedRes).getSample(10000)
#         r = random_sample.asPoint()
#         print("--------------")
#         print(str(i) + "; Meixner distribution; MLE")
#         print(fittedRes)
#         ks = ks_2samp(data[:,i], r, alternative='two-sided', method='auto')
#         print(ks)
#         cvm = cramervonmises_2samp(data[:,i], r, method='auto')
#         print(cvm)
#         cvm_result.append(cvm.pvalue)
#     return cvm_result
