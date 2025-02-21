import numpy as np
from scipy.stats import genhyperbolic
from joblib import Parallel, delayed

def fit_hyperbolic(data_slice):
    m = 5
    dim = len(data_slice[0])
    fit_result = np.zeros((dim, m))
    for k in range(0, dim):
        fit_result[k] = genhyperbolic.fit(data_slice[:,k], method='MLE')
    return fit_result

def hyperbolic_marginals(data, window_len):
    T, dim = data.shape
    m = 5
    res = np.zeros((T, dim, m))
    iters = T - window_len + 1
    fit_results = Parallel(n_jobs=-1)(
        delayed(fit_hyperbolic)(data[i:i + window_len]) for i in range(0, iters)
    )
    for i in range(0, iters):
        idx = i + window_len - 1
        res[idx] = np.array(fit_results[i])

    return res

def generate_batch(params, size):
    dim = len(params)
    batch_result = np.zeros(shape=(size, dim))
    for i in range(dim):
        batch_result[:, i] = genhyperbolic.rvs(*params[i], size=size)
    return batch_result

def hyperbolic_rvs(params, N, batch_size = 20000):
    dim = len(params)

    if dim > 4:
        if N >= 100 * batch_size:
            num_batches = (N + batch_size - 1) // batch_size
            results = Parallel(n_jobs=-1)(delayed(generate_batch)(params, batch_size) for _ in range(num_batches))
            return np.vstack(results)[:N]
        else:
            return generate_batch(params, N)
    else:
        return generate_batch(params, N)