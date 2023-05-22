import numpy as np
from numba import jit

@jit(nopython = True)
def get_mlog_likelihood(r, data, pdf, transform):
    log_likelihood = np.sum(np.log(pdf(data, transform(r) )) )
    return -log_likelihood