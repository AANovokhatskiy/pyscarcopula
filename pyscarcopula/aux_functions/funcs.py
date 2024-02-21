import numpy as np
from numba import jit

def rank(arr_data):
    n = len(arr_data)
    order = arr_data.argsort()
    ranks = order.argsort()
    ranks = (ranks + 1)/(n + 1)
    return ranks

def pobs(arr_data):
    '''transform data to pseudo observations'''
    if arr_data.ndim == 1:
        return rank(arr_data)
    
    res = np.zeros_like(arr_data)
    dim = arr_data.shape[1]
    for k in range(0, dim):
        res[:,k] = rank(arr_data[:,k])
    return res

#@jit(nopython = True)
def jit_rank(arr_data):
    n = len(arr_data)
    order = arr_data.argsort()
    ranks = order.argsort()
    ranks = (ranks + 1)/(n + 1)
    return ranks

#@jit(nopython = True)
def jit_pobs(arr_data):
    '''transform data to pseudo observations'''
    if arr_data.ndim == 1:
        return jit_rank(arr_data)
    
    res = np.zeros_like(arr_data)
    dim = arr_data.shape[1]
    for k in range(0, dim):
        res[:,k] = jit_rank(arr_data[:,k])
    return res