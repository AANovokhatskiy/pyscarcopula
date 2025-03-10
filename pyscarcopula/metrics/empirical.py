import numpy as np 

def var_emp_window(arr, gamma, window_len):
    res = np.zeros_like(arr)
    T = len(arr)
    iters = T - window_len + 1
    for k in range(0, iters):
        idx = k + window_len - 1
        res[idx] = np.quantile(arr[k:window_len + k], gamma)
    return res

def cvar_emp_window(arr, gamma, window_len):
    res = np.zeros_like(arr)
    T = len(arr)
    iters = T - window_len + 1
    for k in range(0, iters - 1):
        idx = k + window_len - 1
        data = arr[k:window_len + k]
        q = np.quantile(data, gamma)
        q_idx = np.argwhere(data <= q)
        mean_value = np.mean(data[q_idx])
        res[idx] = mean_value
    return res
