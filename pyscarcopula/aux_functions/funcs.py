import numpy as np

def rank(arr_data, left_bound = 0, right_bound = 1):
    '''Ranking operation. Used for calculation of pseudo observations.
    
    Parameters. 
    1. arr_data - one-dim numerical massive. Type - Numpy array.
    
    For numerical stability:

    2. left_bound - rank changing of minimal element (rank equals to 0) on left_bound value. Type - float.
    3. right_bound - rank changing of maximal element (rank equals to 1) on right_bound value. Type - float.

    '''
    n = len(arr_data)
    order = arr_data.argsort()
    ranks = order.argsort()
    ranks = ranks/(n - 1)
    if left_bound > 0:
        ranks[np.argwhere(ranks == 0)] = left_bound
    if right_bound < 1:
        ranks[np.argwhere(ranks == 1)] = right_bound
    return ranks


def transform_pseudo_obs(arr_data):
    #type(returns_data)
    if len(arr_data.shape) == 1:
        dim = 1
        return rank(arr_data, 0.00001, 0.99999)
    else:
        dim = arr_data.shape[1]
        res = np.zeros_like(arr_data)
        for k in range(0, dim):
            res[:,k] = rank(arr_data[:,k], 0.00001, 0.99999)
        return res