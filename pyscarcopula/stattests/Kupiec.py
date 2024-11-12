from scipy.stats import chi2
import numpy as np

def Kupiec_POF(returns_data, var_data, p):
    N = len(returns_data)
    x = np.sum(returns_data < var_data)
    print(f"N = {N}, Expected drawdowns = {np.round(p * N, 4)}, Observed drawdowns = {x}, x/N = {np.round(x/N, 4)}, p = {np.round(p, 4)}")
    critical_value = chi2.ppf(1 - p, 1)
    LR_POF = -2 * ((N - x) * np.log(1 - p) + x * np.log(p) - (N - x) * np.log(1 - x/N) - x * np.log(x/N))
    print(f"critical_value = {np.round(critical_value, 4)}, estimated_statistics = {np.round(LR_POF, 4)}, accept = {LR_POF < critical_value}")
    return LR_POF
