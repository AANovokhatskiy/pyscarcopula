import numpy as np
from numba import jit, prange

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

@jit(nopython = True)
def jit_rank(arr_data):
    n = len(arr_data)
    order = arr_data.argsort()
    ranks = order.argsort()
    ranks = (ranks + 1)/(n + 1)
    return ranks

@jit(nopython = True)
def jit_pobs(arr_data):
    '''transform data to pseudo observations'''
    if arr_data.ndim == 1:
        return jit_rank(arr_data)
    
    res = np.zeros_like(arr_data)
    dim = arr_data.shape[1]
    for k in prange(0, dim):
        res[:,k] = jit_rank(arr_data[:,k])
    return res

@jit(nopython=True)
def linear_least_squares(matA: np.array, matB: np.array, ridge_alpha = 0.0, pseudo_inverse = False) -> np.array:
    '''Ridge regression
       Input  Ax = b
       Output x = (A.T * A + alpha * I)^(-1) * A.T * b

       Solution with pseudoinverse matrix:
       Input Ax = b
       Output x = A^(+) * b
       where A^(+) - pseudoinverse matrix
    '''
    if pseudo_inverse == False:
        I = np.identity(len(matA[0]))
        I[0][0] = 0
        result = np.linalg.inv(matA.T @ matA + ridge_alpha * I) @ matA.T @ matB
    else:
        result = np.linalg.pinv(matA) @ matB
    return result