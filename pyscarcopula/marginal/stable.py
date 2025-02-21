import numpy as np
from scipy.stats import levy_stable
from joblib import Parallel, delayed
from numba import njit


def fit_stable(data_slice):
    m = 4
    dim = len(data_slice[0])
    fit_result = np.zeros((dim, m))
    for k in range(0, dim):
        fit_result[k] = levy_stable.fit(data_slice[:,k], method='MLE')
    return fit_result

def stable_marginals(data, window_len):
    T, dim = data.shape
    m = 4
    res = np.zeros((T, dim, m))
    iters = T - window_len + 1
    fit_results = Parallel(n_jobs=-1)(
        delayed(fit_stable)(data[i:i + window_len]) for i in range(0, iters)
    )
    for i in range(0, iters):
        idx = i + window_len - 1
        res[idx] = np.array(fit_results[i])

    return res

def generate_batch(params, size):
    dim = len(params)
    batch_result = np.zeros(shape=(size, dim))
    for i in range(dim):
        batch_result[:, i] = levy_stable.rvs(*params[i], size=size)
    return batch_result

def stable_rvs(params, N, batch_size = 100000):
    dim = len(params)
    if N >= 100 * batch_size:
        num_batches = (N + batch_size - 1) // batch_size
        results = Parallel(n_jobs=-1)(delayed(generate_batch)(params, batch_size) for _ in range(num_batches))
        return np.vstack(results)[:N]
    else:
        return generate_batch(params, N)

@njit
def generate_levy_stable(alpha, beta, loc = 0, scale = 1, size = 1):
    # Weron, R. (1996). On the Chambers-Mallows-Stuck method for simulating skewed stable random variables
    # Borak et. al. (2008), Stable Distributions

    V = np.random.uniform(-np.pi/2, np.pi/2, size = size)
    u = np.random.uniform(0, 1, size = size)
    W = -np.log(1 - u)

    indicator0 = (alpha != 1)
    indicator1 = np.invert(indicator0)

    B = np.arctan(beta * np.tan(np.pi/2 * alpha)) / alpha
    S = (1 + beta**2 * np.tan(np.pi/2 * alpha)**2)**(1 / (2 * alpha))

    X0 = S * np.sin(alpha * (V + B)) / np.cos(V)**(1/alpha) * (np.cos(V - alpha * (V + B)) / W)**((1 - alpha) / alpha)
    X1 = 2 / np.pi * ((np.pi/2 + beta * V) * np.tan(V) - beta * np.log(np.pi/2 * W * np.cos(V) / (np.pi / 2 + beta * V)))

    X = X0 * indicator0 + X1 * indicator1

    Y0 = scale * X + loc
    Y1 = scale * X + 2 / np.pi * beta * scale * np.log(scale) + loc

    Y = Y0 * indicator0 + Y1 * indicator1
        
    return Y