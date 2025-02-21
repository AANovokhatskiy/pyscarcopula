import numpy as np
from numba import jit

#@jit(nopython = True, cache = True)
def jit_mlog_likelihood_mle(r, data, pdf, transform):
    log_likelihood = np.sum(np.log(np.maximum(pdf(data, transform(r)), 1e-100) ) )
    return -log_likelihood
