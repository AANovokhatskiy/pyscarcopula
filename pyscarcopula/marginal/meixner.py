import numpy as np
import openturns as ot
from joblib import Parallel, delayed

def fit_meixner(data_slice, m = 4):
    dim = len(data_slice[0])
    fit_result = np.zeros((dim, m))
    ln = len(data_slice)
    ot.ResourceMap.SetAsUnsignedInteger('DistributionFactory-DefaultBootstrapSize', 1000)
    ot.Log.Show(ot.Log.NONE)
    for k in range(0, dim):
        data = data_slice[:,k].reshape((ln, 1))
        sample = ot.Sample(data)
        try:
            fitted_data = ot.MeixnerDistributionFactory().build(sample)
            args = fitted_data.getParameter()
        except:
            factory = ot.MaximumLikelihoodFactory(ot.MeixnerDistribution())
            fitted_data = factory.build(sample)
            args = fitted_data.getParameter()
        fit_result[k] = np.array([args[0], args[1], args[2], args[3]])
    return fit_result

def meixner_marginals(data, window_len):
    T, dim = data.shape
    m = 4
    res = np.zeros((T, dim, m))
    iters = T - window_len + 1
    fit_results = Parallel(n_jobs=-1)(
        delayed(fit_meixner)(data[i:i + window_len]) for i in range(0, iters)
    )
    for i in range(0, iters):
        idx = i + window_len - 1
        res[idx] = np.array(fit_results[i])

    return res

def generate_batch(params, size):
    dim = len(params)
    batch_result = np.zeros(shape=(size, dim))
    for i in range(dim):
        batch_result[:, i] = ot.MeixnerDistribution(*params[i]).getSample(size).asDataFrame().values.flatten()
    return batch_result

def meixner_rvs(params, N, batch_size = 10000):
    dim = len(params)
    if N >= 100 * batch_size:
        num_batches = (N + batch_size - 1) // batch_size
        results = Parallel(n_jobs=-1)(delayed(generate_batch)(params, batch_size) for _ in range(num_batches))
        return np.vstack(results)[:N]
    else:
        return generate_batch(params, N)