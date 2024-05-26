import numpy as np
from scipy.stats import genhyperbolic
from tqdm import tqdm
from multiprocessing import RawArray, Pool, cpu_count


def hyperbolic_marginals(data, window_len):
    T = len(data)
    dim = len(data[0])
    m = 5
    res = RawArray('d', T * dim * m)
    shared_data = RawArray('d', (data.T).flatten())
    global fit_single
    def fit_single(i):
        idx = i - 1 + window_len
        for j in range(0, dim):
            i1 = i + j * T
            i2 = i1 + window_len
            fit_result = genhyperbolic.fit(shared_data[i1:i2], method = 'MLE')
            i3 = idx * dim * m + j * m
            i4 = i3 + m
            res[i3:i4] = fit_result
    iters = T - window_len + 1
    pool = Pool(processes = cpu_count())   
    with pool:
        generator = tqdm(pool.imap(fit_single, range(0, iters)), total = iters)
        for i in generator:
            continue
    res = np.frombuffer(res).reshape((T, dim, m))
    return res


def hyperbolic_rvs(params, N):
    dim = len(params)
    res = np.zeros(shape=(N, dim))
    for i in range(0, dim):
        res[:,i] = genhyperbolic.rvs(*params[i], size = N)
    return res