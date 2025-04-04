import numpy as np
from numba import njit, jit, prange
from typing import Literal
import math
from pyscarcopula.auxiliary.funcs import linear_least_squares
from scipy.signal import savgol_filter

@njit
def Ex_OU(alpha, t, a1t = 0, a2t = 0, a10 = 0, a20 = 0, stationary = False):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]

    Dx0 = 0.0
    Mx0 = mu
    if stationary == True:
        Dx0 = nu**2 / (2 * theta)

    # Dx0 = np.var(np.asarray(x0))
    # Mx0 = np.mean(np.asarray(x0))
    
    sigma2t = sigma2_OU(alpha, t, stationary)

    pt = 1 - 2 * a2t * sigma2t
    xs = (Mx0 - mu) * np.exp(-theta * t) + mu
    xsw = (xs + a1t * sigma2t) / pt
    
    st = np.exp(-theta * t) / np.sqrt(pt)
    Mx = xsw + Mx0 * st - st * (Mx0 + a10 * Dx0) / np.sqrt(1 - 2 * a20 * Dx0)

    return Mx

@njit
def sigma2_OU(alpha, t, stationary = False):
    Dx0 = 0.0
    theta, mu, nu = alpha[0], alpha[1], alpha[2]
    if stationary == True:
        Dx0 = nu**2 / (2 * theta)
    
    p3 = nu**2 / (2 * theta) * (np.exp(2 * theta * t) - 1)
    D = np.exp(-2 * theta * t) * (Dx0 + p3)

    return D

@njit
def Dx_OU(alpha, t, a2t = 0, stationary = False):
    Dx0 = 0.0
    theta, mu, nu = alpha[0], alpha[1], alpha[2]
    if stationary == True:
        Dx0 = nu**2 / (2 * theta)
    
    sigma2t = sigma2_OU(alpha, t, stationary)

    pt = 1 - 2 * a2t * sigma2t
    p3 = nu**2 / (2 * theta) * (np.exp(2 * theta * t) - 1)
    D = np.exp(-2 * theta * t) / pt * (Dx0 + p3)

    return D

@njit
def cov_OU(alpha, t, s, a2t = 0, a2s = 0, stationary = False):
    Dx0 = 0.0
    theta, mu, nu = alpha[0], alpha[1], alpha[2]
    if stationary == True:
        Dx0 = nu**2 / (2 * theta)

    sigma2t = sigma2_OU(alpha, t, stationary)
    sigma2s = sigma2_OU(alpha, s, stationary)

    pt = np.sqrt(1 - 2 * a2t * sigma2t)
    ps = np.sqrt(1 - 2 * a2s * sigma2s)
    p3 = nu**2 / (2 * theta) * (np.exp(2 * theta * np.minimum(t, s)) - 1)
    cov = np.exp(-theta * (t + s)) / (pt * ps) * (Dx0 + p3)

    return cov

@njit
def init_state_ou(alpha, latent_process_tr):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]

    x0 = np.ones(latent_process_tr) * mu
    return x0

@njit
def stationary_state_ou(alpha, latent_process_tr, seed = None):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]

    if seed is None:
        seed = np.random.randint(1, 1000000)
        
    rng = np.random.seed(seed)
    xs = mu
    sigma2 = nu**2 / (2 * theta)
    state = np.random.normal(loc = mu, scale = np.sqrt(sigma2), size = latent_process_tr)    
    
    return state

# @jit(nopython = True, cache = True, parallel = True)
def get_avg_p_log_likelihood_ou(u, lambda_data, log_pdf, transform):
    avg_likelihood = 0
    latent_process_tr = lambda_data.shape[1]

    copula_log_data = np.zeros(latent_process_tr)

    for k in prange(0, latent_process_tr):
        copula_log_data[k] = np.sum(log_pdf(u, transform(lambda_data[:,k])))

    '''trick for calculation large values. calculate e^(sum(log_cop) - corr) instead of e^(sum(log_cop)).
    Do inverse correction at the end of calculations'''
    corr = max(copula_log_data)
    avg_likelihood = np.sum(np.exp(copula_log_data - corr)) / latent_process_tr
    return math.log(avg_likelihood) + corr


# # @jit(nopython=True)
def p_jit_mlog_likelihood_ou(alpha: np.array, u: np.array, dwt: np.array,
                      log_pdf: callable, transform: callable, print_path: bool, stationary: bool = False) -> float:
    
    #initial data check
    if np.isnan(np.sum(alpha)) == True:
        res = 10**10
        if print_path == True:
            print(alpha, 'incorrect params', res)
        return res
    
    T = dwt.shape[0]
    a1t = np.zeros(T)
    a2t = np.zeros(T)

    if stationary == True:
        init_state = stationary_state_ou(alpha, dwt.shape[1])
    else:
        init_state = init_state_ou(alpha, dwt.shape[1])
    
    lambda_data = m_sampler_ou(alpha, a1t, a2t, dwt, init_state)

    avg_log_likelihood = get_avg_p_log_likelihood_ou(u, lambda_data, log_pdf, transform)
    res = - avg_log_likelihood
    if np.isnan(res) == True:
        res = 10**10
        if print_path == True:
            print(alpha, 'unknown error', res)
    else:
        if print_path == True:
            print(alpha, res)
    return res


# @jit(nopython=True)
def bounded_polynom_fit(x, y, dim, type: Literal['two-sided', 'left-sided', 'right-sided', 'no bounds'], ridge_alpha = 0.0):
    if ridge_alpha == 0.0:
        pseudo_inverse = True
    else:
        pseudo_inverse = False
    if type == 'two-sided':
        x0 = x[0]
        x1 = x[-1]
        y0 = y[0]
        y1 = y[-1]
        c0 = (y0 * x1 - y1 * x0) / (x1 - x0)
        c1 = (y1 - y0) / (x1 - x0)
        d1 = -x0 - x1
        d2 = x0 * x1
        A = np.zeros((len(x), dim))
        x_i = np.ones(len(x))
        for i in range(0, dim):
            A[:,i] = x_i * x * x + d1 * x_i * x + d2 * x
            x_i = x_i  * x
        A[:,0] += np.ones(len(x))
        A[:,1] += x * c1
        res = linear_least_squares(A, y - c0, ridge_alpha, pseudo_inverse)
        return res
    elif type == 'no bounds':
        fi = 1
        A = np.zeros((len(x), dim + fi))
        x_i = x
        for i in range(0, dim):
            A[:,i + fi] = x_i
            x_i = x_i  * x
        A[:,0] = np.ones(len(x))
        res = linear_least_squares(A, y, ridge_alpha, pseudo_inverse)
        return res
    elif type == 'left-sided':
        x0 = x[0]
        y0 = y[0]
        A = np.zeros((len(x), dim))
        x_i = np.ones(len(x))
        for i in range(0, dim):
            A[:,i] = x_i * (x - x0)
            x_i = x_i  * x
        res = linear_least_squares(A, y - y0, ridge_alpha, pseudo_inverse)
        return res
    elif type == 'right-sided':
        x0 = x[-1]
        y0 = y[-1]
        A = np.zeros((len(x), dim))
        x_i = np.ones(len(x))
        for i in range(0, dim):
            A[:,i] = x_i * (x - x0)
            x_i = x_i  * x
        res = linear_least_squares(A, y - y0, ridge_alpha, pseudo_inverse)
        return res
    else:
        raise ValueError(f"type = {type} is not implemented")


# @jit(nopython=True)
def bounded_polynom(x, y, coef, type: Literal['two-sided', 'left-sided', 'right-sided', 'no bounds']):
    if type == 'two-sided':
        dim = len(coef)
        x0 = x[0]
        x1 = x[-1]
        y0 = y[0]
        y1 = y[-1]
        c0 = (y0 * x1 - y1 * x0) / (x1 - x0)
        c1 = (y1 - y0) / (x1 - x0)
        d1 = -x0 - x1
        d2 = x0 * x1
        res = np.zeros(len(x))
        x_i = np.ones(len(x))
        for i in range(0, dim):
            res += coef[i] * x_i
            x_i = x_i * x
        return (x * x + d1 * x + d2) * res + c1 * x + c0
    elif type == 'no bounds':
        dim = len(coef)
        res = np.zeros(len(x))
        fi = 1
        for i in range(0, dim):
            res += coef[i] * x**(1 - fi + i)
        return res
    elif type == 'left-sided':
        dim = len(coef)
        x0 = x[0]
        y0 = y[0]
        res = np.zeros(len(x))
        x_i = np.ones(len(x))
        for i in range(0, dim):
            res +=  coef[i] * x_i
            x_i = x_i * x
        return y0 + (x - x0) * res
    elif type == 'right-sided':
        dim = len(coef)
        x0 = x[-1]
        y0 = y[-1]
        res = np.zeros(len(x))
        x_i = np.ones(len(x))
        for i in range(0, dim):
            res += coef[i] * x_i
            x_i = x_i * x
        return y0 + (x - x0) * res
    else:
        raise ValueError(f"type = {type} is not implemented")


# @jit(nopython=True)
def check_a2_bounds(alpha, a2t):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]

    t = np.linspace(0, 1, len(a2t))

    '''sigma2(t = 0) = 0, then upper bound = +inf. Dont check it.'''
    sigma2 = nu**2 / (2 * theta) * (1 - np.exp(-2 * theta * t[1:]))
    ub = 1/(2 * sigma2)
    
    bound_check = True

    for k in range(1, len(a2t)):
        '''sigma2 is less on 1 element than a2t'''
        if a2t[k] >= ub[k - 1]:
            bound_check = False
            return bound_check
        
    return bound_check

@jit(nopython=True)
def check_sigma2w_derivative(alpha, a2t):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]

    t = np.linspace(0, 1, len(a2t))
    dt = 1 / (len(t) - 1)

    sigma2 = nu**2 / (2 * theta) * (1 - np.exp(-2 * theta * t))
    p = (1 - 2 * a2t * sigma2)
    sigma2w = sigma2 / p
    Bt2 = nu**2 / p

    check = True

    for k in range(1, len(a2t)):
        if Bt2[k] <= (sigma2w[k] - sigma2w[k - 1]) / dt:
            check = False
            return check
        
    return check


@jit(nopython=True)
def log_norm_ou(alpha: np.array, a1: np.array, a2: np.array, t: np.array, x0: np.array):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]
    D = nu**2/2
    sigma2 = D/theta * (1 - np.exp(-2 * theta * t))
    xs = (x0 - mu) * np.exp(-theta * t) + mu
    res = (a1**2 * sigma2 + 2 * a1 * xs + 2 * a2 * xs**2) / (2 - 4 * a2 * sigma2) - 0.5 * np.log(1 - 2 * a2 * sigma2)
    return res


@njit
def m_sampler_ou(alpha, a1t, a2t, dwt, init_state = None):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]
    D = nu**2 / 2

    T = len(dwt)
    latent_process_tr = len(dwt[0])
    dt = 1 / (T - 1)
    xt = np.zeros((T, latent_process_tr))

    if init_state is None:
        xt[0] = init_state_ou(alpha, latent_process_tr)
    else:
        xt[0] = init_state

    Dx0 = np.var(np.asarray(xt[0]))
    Mx0 = np.mean(np.asarray(xt[0]))

    Ito_integral_sum = np.zeros(latent_process_tr)
    for i in range(1, T):
        j = i
        a1, a2 = a1t[j], a2t[j]

        t = i * dt
        sigma2 = D / theta * (1 - np.exp(- 2 * theta * t)) + Dx0 * np.exp(-2 * theta * t)
        p = (1 - 2 * a2 * sigma2)

        if i == 1:
            pm1 = 1
        else:
            tm1 = t - dt
            sigma2m1 = D / theta * (1 - np.exp(- 2 * theta * tm1)) + Dx0 * np.exp(-2 * theta * tm1)
            a2m1 = a2t[j - 1]
            pm1 = (1 - 2 * a2m1 * sigma2m1)

        xs = (Mx0 - mu) * np.exp(-theta * t) + mu
        # xs = (xt[0] - mu) * np.exp(-theta * t) + mu 
        xsw = (xs + a1 * sigma2) / p
        
        st = np.exp(-theta * t) / np.sqrt(p)
        Determinated_part = xsw + st * xt[0] - st * (Mx0 + a1t[0] * Dx0) / np.sqrt(1 - 2 * a2t[0] * Dx0)
        Ito_integral_sum = (Ito_integral_sum  * np.sqrt(pm1 / p) + nu / np.sqrt(p) * dwt[i - 1]) * np.exp(-theta * dt)
        xt[i] = Determinated_part + Ito_integral_sum
    return xt


@njit
def Q(s1, s2, rho):
    if rho == 0:
        return np.eye(2)
    
    p1 = np.sqrt(s1**4 + 2 * (-1 + 2 * rho**2) * s1**2 * s2**2 + s2**4)
    p2 = s1**2 - s2**2

    p3 = 2 * p2 / (p2 + p1)
    p4 = 2 * p2 / (-p2 + p1)

    p5 = np.sqrt(2 - p3)
    p6 = np.sqrt(2 + p4)
    
    p7 = 2 * rho * s1 * s2

    res = np.zeros((2, 2))
    res[0][0] = - (-p2 + p1) / (p7 * p5)
    res[0][1] = (p2 + p1) / (p7 * p6)
    res[1][0] = 1 / p5
    res[1][1] = 1 / p6

    return res

@njit
def L(s1, s2, rho):
    res = np.zeros((2, 2))
    if rho == 0:
        _s1 = np.maximum(1e-5, s1)
        _s2 = np.maximum(1e-5, s2)

        res[0][0] = 1/_s1**2
        res[1][1] = 1/_s2**2
        return res
    
    p1 = np.sqrt(s1**4 + 2 * (-1 + 2 * rho**2) * s1**2 * s2**2 + s2**4)
    p2 = s1**2 + s2**2

    res[0][0] = 2 / (p2 - p1)
    res[1][1] = 2 / (p2 + p1)

    return res

@njit
def Sigma(s1, s2, rho):
    res = np.zeros((2, 2))
    
    res[0][0] = s1**2
    res[0][1] = rho * s1 * s2
    res[1][0] = rho * s1 * s2
    res[1][1] = s2**2

    return res

@njit
def A(s1, s2, rho):
    matL = L(s1, s2, rho)
    matQ = Q(s1, s2, rho)
    matA = np.linalg.inv(np.sqrt(matL) @ matQ.T)

    return matA

@njit
def m_auxiliary_a_int2(alpha, u, M_iterations, z, w, log_pdf, transform, 
                       print_path = False, stationary = False):
    theta, mu, nu = alpha[0], alpha[1], alpha[2]

    T = len(u)
    n = len(z)
    dt = 1 / (T - 1)
    sqrt2 = np.sqrt(2)

    a_t = np.zeros((T, 3))

    for m in range(0, M_iterations):
        a_t[-1][0] = np.mean(a_t[:, 0])
        a_t[-1][1] = np.mean(a_t[:, 1])
        a_t[-1][2] = np.mean(a_t[:, 2])

        for k in range(T - 1, 0, -1):
            t = k * dt
            s = t - dt

            a1t, a2t = a_t[k][1], a_t[k][2]

            S, Sx, Sx2, Sx3, Sx4 = 0.0, 0.0, 0.0, 0.0, 0.0
            Sp, Spx, Spx2 = 0.0, 0.0, 0.0
            
            Dxt = Dx_OU(alpha, t, a2t, stationary = stationary)
            Dxs = Dx_OU(alpha, s, a2t, stationary = stationary)

            Ext = Ex_OU(alpha, t, a1t, a2t, stationary = stationary)
            Exs = Ex_OU(alpha, s, a1t, a2t, stationary = stationary)


            if Dxt == 0 or Dxs == 0:
                rho = 0
            else:
                rho = cov_OU(alpha, t, s, a2t, a2t, stationary = stationary) / np.sqrt(Dxt * Dxs)

            matA = A(np.sqrt(Dxt), np.sqrt(Dxs), rho)


            for i in range(0, n):
                for j in range(0, n):
                    xij = (matA[0][0] * z[i] + matA[0][1] * z[j]) * sqrt2 + Ext
                    x0ij = (matA[1][0] * z[i] + matA[1][1] * z[j]) * sqrt2 + Exs

                    log_g2 = log_norm_ou(alpha, a1t, a2t, dt, x0ij)

                    pij = log_pdf(u[k], transform(xij)) + log_g2

                    val = w[i] * w[j]

                    S += val
                    Sp += pij * val

                    val = xij * val
                    Sx += val
                    Spx += pij * val

                    val = xij * val
                    Sx2 += val
                    Spx2 += pij * val

                    val = xij * val
                    Sx3 += val

                    val = xij * val
                    Sx4 += val

            matS = np.array([[Sx, Sx2, Sx3], [Sx2, Sx3, Sx4], [S, Sx, Sx2]])
            vecS = np.array([Spx, Spx2, Sp])

            try:
                 a = np.linalg.solve(matS, vecS)
            except:
                if print_path == True:
                    print("ls problem fail", i, j, k)
                    a_t = np.zeros((T, 3))
                return a_t[:,1], a_t[:,2]
            
            if np.isnan(np.sum(a)) == True:
                a_t[k - 1] = a_t[k]
            else:
                sigma2 = nu**2 / (2 * theta) * (1 - np.exp(-2 * theta * t))
                ub = np.maximum(1/(2 * sigma2) - 0.01, 0)
                
                # ub = 0
                a_t[k - 1] = a
                a_t[k - 1][2] =  np.minimum(a_t[k - 1][2], ub)

        # deriv_check = check_sigma2w_derivative(alpha, a_t[:,2])
        # if deriv_check == False:
        #     if print_path == True:
        #         print("process parameters derivative warning")
        #         a_t = np.zeros((T, 3))
        #     return a_t[:,1], a_t[:,2]
               
    return a_t[:,1], a_t[:,2]


def m_auxiliary_a_MC(alpha, u, M_iterations, dwt, log_pdf, transform, 
                     print_path = False, stationary = False):
    T = len(u)
    dt = 1 / (T - 1)

    latent_process_tr = dwt.shape[1]
    norm_log_data = np.zeros((T, latent_process_tr))
    t_data = np.linspace(0, 1, T)

    a_data = np.zeros((T, 3))
    theta, mu, nu = alpha[0], alpha[1], alpha[2]
    
    a1t = np.zeros(T)
    a2t = np.zeros(T)
    a1t = 10 * np.ones(T)
    a2t = -5 * np.ones(T)

    if stationary == True:
        init_state = stationary_state_ou(alpha, dwt.shape[1])
    else:
        init_state = init_state_ou(alpha, dwt.shape[1])

    '''get latent process sample'''
    for j in range(0, M_iterations):
        lambda_data = m_sampler_ou(alpha, a1t, a2t, dwt, init_state)

        '''check nan values'''
        if np.isnan(np.sum(lambda_data)) == True:
            res = float(10**10)
            if print_path == True:
                print(alpha, 'm sampler nan', res)
            return res, a1t, a2t
              
        norm_log_data = np.zeros((T, latent_process_tr))

        '''set initial values: a(T)'''
        a_mean = np.zeros(3)

        a_mean[0] = np.mean(a_data[:,0])
        a_mean[1] = np.mean(a_data[:,1])

        '''consider upper bound for a2'''
        sigma2 = nu**2 / (2 * theta) * (1 - np.exp(-2 * theta))
        #ub = np.maximum(1/(2 * sigma2) - 0.1, 0)
        ub = 0
        a_mean[2] = np.minimum(np.mean(a_data[:,2]), ub)
        
        a_data = np.zeros((T, 3))
        if j == 0:
            a_data[-1] = np.array([0, a1t[-1], a2t[-1]])
        else:
            a_data[-1] = a_mean

        '''solve ls problem'''
        for i in range(T - 1, 0 , -1):
            u1 = np.full((latent_process_tr, u.shape[1]), u[i])
            copula_log_data = log_pdf(u1, transform(lambda_data[i])) #u[i]
            A = np.dstack((np.ones(latent_process_tr) , lambda_data[i] , lambda_data[i]**2))[0]
            norm_log_data[i] = log_norm_ou(alpha, a_data[i][1], a_data[i][2], dt, lambda_data[i - 1]) #i - 1

            b = copula_log_data + norm_log_data[i]
            sigma2 = nu**2 / (2 * theta) * (1 - np.exp(-2 * theta * (t_data[i])))
            r = 0.0
            
            '''set upper bound for a2'''
            ub = np.maximum(1/(2 * sigma2) - 0.001, 0)
            try:
                lr = linear_least_squares(A, b, 0, pseudo_inverse = True)
                '''if nan use previous result'''
                if np.isnan(np.sum(lr)) == True:
                    a_data[i - 1] = a_data[i]
                else:
                    a_data[i - 1] = lr
            except:
                res = 10**10
                if print_path == True:
                    print(alpha, 'ls problem fail', res, i)
                return res, a1t, a2t

            '''check a2 bounds'''
            a_data[i - 1][2] = np.minimum(a_data[i - 1][2], ub)

        a1_hat = a_data[:,1].copy()
        a2_hat = a_data[:,2].copy()

        fit_type1 = 'no bounds'
        fit_type2 = 'no bounds'

        dim = j + 1

        '''check a2 lower then bounds and fit a2(t)'''
        bound_check = False
        rigde_alpha_list = np.array([0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0])
        for r in rigde_alpha_list:
            a2_params = bounded_polynom_fit(t_data, a2_hat, dim = dim, ridge_alpha = r, type = fit_type2)
            a2t = bounded_polynom(t_data, a2_hat, a2_params, type = fit_type2)
            bound_check = check_a2_bounds(alpha, a2t)
            deriv_check = check_sigma2w_derivative(alpha, a2t)
            if (bound_check & deriv_check) == True:
                break
        if bound_check == False:
            a1t = np.zeros(T)
            a2t = np.zeros(T)
            break
        else:
            '''fit a1(t)'''
            a1_params = bounded_polynom_fit(t_data, a1_hat, dim = dim, ridge_alpha = r, type = fit_type1)
            a1t = bounded_polynom(t_data, a1_hat, a1_params, type = fit_type1)
    
    return a1t, a2t

@njit
def m_jit_mlog_likelihood_ou_aux(alpha, u, dwt, a1t, a2t,
                                jit_log_pdf, transform, 
                                print_path, stationary = False):
    
    T, latent_process_tr = dwt.shape
    dt = 1 / (T - 1)
    if stationary == True:
        init_state = stationary_state_ou(alpha, latent_process_tr)
    else:
        init_state = init_state_ou(alpha, latent_process_tr)

    '''get latent process sample'''
    lambda_data = m_sampler_ou(alpha, a1t, a2t, dwt, init_state)
    log_likelihood = np.zeros(latent_process_tr)
    norm_log_data = np.zeros((T, latent_process_tr))
    
    '''check nan values'''
    if np.isnan(np.sum(lambda_data)) == True:
        res = float(10**10)
        if print_path == True:
            print(alpha, 'm sampler nan', res)
        return res
    
    '''calculate normalizing factors'''
    for i in range(T - 1, 0, -1):
        a1, a2 = a1t[i], a2t[i]
        norm_log_data[i] = log_norm_ou(alpha, a1, a2, dt, lambda_data[i - 1])
        

    norm_log_data[0] =  log_norm_ou(alpha, a1t[0], a2t[0], dt, init_state)


    '''calculate log likelihood'''
    for k in range(0, latent_process_tr):
        copula_log_data = np.zeros(T)
        for i in range(0, T):
            copula_log_data[i] = jit_log_pdf(u[i], transform(lambda_data[i, k]))
        g = (a1t * lambda_data[:,k]  + a2t * lambda_data[:,k]**2)
        log_likelihood[k] = np.sum(copula_log_data + norm_log_data[:,k] - g)
    xc = np.max(log_likelihood)
    avg_likelihood = np.sum(np.exp(log_likelihood - xc)) / latent_process_tr
    res = np.log(avg_likelihood) + xc
    res = -res
    
    max_log_lik_debug = -100000
    
    if res < max_log_lik_debug:
        res = float(10**10)
        if print_path == True:
            print(alpha, 'instability encountered', res)
        return res

    '''check nan values'''
    if np.isnan(res) == True:
        res = float(10**10)
        if print_path == True:
            print(alpha, 'unknown error', res)
        return res

    if print_path == True:
        print(alpha, res)

    return res


def m_jit_mlog_likelihood_ou(alpha, u, dwt, M_iterations,
                             jit_log_pdf, transform, 
                             print_path, z, w, stationary = False):

    if np.isnan(np.sum(alpha)) == True:
        res = 10**10
        if print_path == True:
            print(alpha, 'incorrect params', res)
        return res, np.zeros(len(u)), np.zeros(len(u))
        

    method_importance_sampling = 'int2' #'MC'

    if method_importance_sampling == 'int2':
        a1t, a2t = m_auxiliary_a_int2(alpha, u, M_iterations, z, w,
                                    jit_log_pdf, transform,
                                    print_path, stationary)
        wl = len(u) // 10
        d = 2
        a1t = savgol_filter(a1t, wl, d)
        a2t = savgol_filter(a2t, wl, d)

    elif method_importance_sampling == 'MC':
        a1t, a2t = m_auxiliary_a_MC(alpha, u, M_iterations, dwt,
                                    jit_log_pdf, transform,
                                    print_path, stationary)
                                    
    res = m_jit_mlog_likelihood_ou_aux(alpha, u, dwt, a1t, a2t,
                                jit_log_pdf, transform, 
                                print_path, stationary)
    
    return res, a1t, a2t



## @jit(nopython=True)
def jit_latent_process_conditional_expectation_p_ou(alpha, u, dwt, 
                                                    log_pdf, transform, stationary = False):
    latent_process_tr = dwt.shape[1]
    
    T = len(u)

    a1t = np.zeros(T)
    a2t = np.zeros(T)

    if stationary == True:
        init_state = stationary_state_ou(alpha, dwt.shape[1])
    else:
        init_state = init_state_ou(alpha, dwt.shape[1])

    lambda_data = transform(m_sampler_ou(alpha, a1t, a2t, dwt, init_state))
    
    copula_log_data = np.zeros((T, latent_process_tr))

    for k in range(0, latent_process_tr):
        copula_log_data[:,k] = log_pdf(u, lambda_data[:,k])
    
    copula_cs_log_data = np.zeros(latent_process_tr)
    
    latent_process_conditional_sample = np.zeros(T)
    for i in range(0, T):
        
        copula_cs_log_data += copula_log_data[i]
        xc = np.max(copula_cs_log_data)
        
        temp1 = np.exp(copula_cs_log_data - xc)

        avg_likelihood = np.sum(temp1) / latent_process_tr
        log_lik = np.log(avg_likelihood) + xc

        avg_expectation = np.sum(lambda_data[i] * temp1) / latent_process_tr
        avg_log_expectation = np.log(avg_expectation) + xc

        latent_process_conditional_sample[i] = avg_log_expectation - log_lik

    result = np.exp(latent_process_conditional_sample)
    
    return result


## @jit(nopython=True)
def jit_latent_process_conditional_expectation_m_ou(alpha, u, M_iterations, dwt,
                                                    log_pdf, transform, z, w, stationary = False):

    latent_process_tr = dwt.shape[1]

    T = len(u)

    method_importance_sampling = 'int2' #'MC'

    if method_importance_sampling == 'int2':
        a1t, a2t = m_auxiliary_a_int2(alpha, u, M_iterations, z, w,
                                    log_pdf, transform,
                                    False, stationary)
        wl = len(u) // 10
        d = 2
        a1t = savgol_filter(a1t, wl, d)
        a2t = savgol_filter(a2t, wl, d)
    elif method_importance_sampling == 'MC':
        a1t, a2t = m_auxiliary_a_MC(alpha, u, M_iterations, dwt,
                                    log_pdf, transform,
                                    False, stationary)


    if stationary == True:
        init_state = stationary_state_ou(alpha, dwt.shape[1])
    else:
        init_state = init_state_ou(alpha, dwt.shape[1])

    lambda_data = m_sampler_ou(alpha, a1t, a2t, dwt, init_state)
    norm_log_data = np.zeros((T, latent_process_tr))
    g = np.zeros((T, latent_process_tr))

    dt = 1 / (T - 1)

    '''calculate normalizing factors'''
    for i in range(T - 1, 0, -1):
        a1, a2 = a1t[i], a2t[i]
        norm_log_data[i] = log_norm_ou(alpha, a1, a2, dt, lambda_data[i - 1])
        
    norm_log_data[0] =  log_norm_ou(alpha, a1t[0], a2t[0], dt, init_state)

    for i in range(0, T):
        g[i] = (a1t[i] * lambda_data[i] + a2t[i] * lambda_data[i]**2)

    copula_log_data = np.zeros((T, latent_process_tr))

    vlog_pdf = np.vectorize(log_pdf, signature="(n),()->()")

    for k in range(0, latent_process_tr):
        copula_log_data_temp = vlog_pdf(u, transform(lambda_data[:,k]))
        copula_log_data[:,k] = copula_log_data_temp + norm_log_data[:,k] - g[:,k]
    
    copula_cs_log_data = np.zeros(latent_process_tr)
    
    latent_process_conditional_sample = np.zeros(T)
    for i in range(0, T):

        # if i > 0:
        #     copula_cs_log_data += copula_log_data[i - 1]
        copula_cs_log_data += copula_log_data[i]
        
        xc = np.max(copula_cs_log_data)
        
        temp1 = np.exp(copula_cs_log_data - xc)

        avg_likelihood = np.sum(temp1) / latent_process_tr
        log_lik = np.log(avg_likelihood) + xc
    
        avg_expectation = np.sum(transform(lambda_data[i]) * temp1) / latent_process_tr
        avg_log_expectation = np.log(avg_expectation) + xc

        latent_process_conditional_sample[i] = avg_log_expectation - log_lik

    result = np.exp(latent_process_conditional_sample)
    
    return result