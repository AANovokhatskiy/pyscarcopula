from scipy.stats import chi2
import numpy as np

def Kupiec_POF(returns_data, var_data, p):
    N = len(returns_data)
    x = np.sum(returns_data < var_data)
    print(f"N = {N}, x = {x}, x/N = {x/N}, p = {p}")
    critical_value = chi2.ppf(1 - p, 1)
    LR_POF = -2 * ((N - x) * np.log(1 - p) + x * np.log(p) - (N - x) * np.log(1 - x/N) - x * np.log(x/N))
    print(f"critical_value = {critical_value:.3e}, estimated_statistics = {LR_POF:.3e}, accept = {LR_POF < critical_value}")
    return LR_POF
