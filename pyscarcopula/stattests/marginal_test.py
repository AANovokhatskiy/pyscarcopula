from scipy.stats import genhyperbolic, levy_stable
from scipy.stats import ks_2samp
from scipy.stats import cramervonmises_2samp

def stable_stattest(fit_result, data):
    for i in range(0, len(fit_result)):
        fittedRes = fit_result[i]
        r = levy_stable.rvs(*fittedRes, size=10000, random_state=None)
        print("--------------")
        print(str(i) + "; Stable distribution")
        print(fittedRes)
        print(ks_2samp(data[:,i], r, alternative='two-sided', method='auto'))
        print(cramervonmises_2samp(data[:,i], r, method='auto'))

def gh_stattest(fit_result, data):
    for i in range(0, len(fit_result)):
        fittedRes = fit_result[i]
        r = genhyperbolic.rvs(*fittedRes, size=10000, random_state=None)
        print("--------------")
        print(str(i) + "; Generalized hyperbolic distribution")
        print(fittedRes)
        print(ks_2samp(data[:,i], r, alternative='two-sided', method='auto'))
        print(cramervonmises_2samp(data[:,i], r, method='auto'))

def meixner_stattest(fit_result, data):
    for i in range(0, len(fit_result)):
        fittedRes = fit_result[i]
        random_sample = ot.MeixnerDistribution(*fittedRes).getSample(10000)
        r = random_sample.asPoint()
        print("--------------")
        print(str(i) + "; Meixner distribution")
        print(fittedRes)
        print(ks_2samp(data[:,i], r, alternative='two-sided', method='auto'))
        print(cramervonmises_2samp(data[:,i], r, method='auto'))

