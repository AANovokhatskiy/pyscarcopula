import matplotlib.pyplot as plt
import numpy as np
from numba import jit
import matplotlib.animation as animation



@jit(nopython=True, cache = True)
def poly(data, coef, intercept = True):
    '''returns polynom of data (c0 + c1 t + c2 t^2 + ...) with coeficients = coef. 
    If intercept == True: first coef considered as free parameter c0; Otherwise - as c1.'''
    dim = len(coef)
    res = np.zeros(len(data))
    fi = int(intercept)
    for i in range(0, dim):
        res += coef[i] * data**(1 - fi + i)
    return res

# @jit(nopython=True, cache = True)
# def poly_corr(t_data, coef, alpha, intercept):
#     '''correct polynom fit result to be below a threshold that raises from norm constant'''
#     res = poly(t_data, coef, intercept)
#     alpha1, alpha2, alpha3 = alpha[0], alpha[1], alpha[2]
#     sigma2 = alpha3**2 / (- 2 * alpha2) * (1 - np.exp(2 * alpha2 * t_data)) + 0.0001
#     max_res = 1 / (2 * sigma2) - 1
#     exp_res = np.exp(-0.05*(max_res - res))
#     return 1 / (1 + exp_res) * res

@jit(nopython=True, cache = True)
def mod_abs(x):
    b = 50
    #res = x * (2 / (1 + np.exp(-b * x)) - 1)
    res = x * np.tanh(b * x)
    return res

@jit(nopython=True, cache = True)
def poly_corr(t_data, coef, alpha, intercept):
    '''correct polynom fit result to be below a threshold that raises from norm constant'''
    res = poly(t_data, coef, intercept)
    alpha1, alpha2, alpha3 = alpha[0], alpha[1], alpha[2]
    sigma2 = alpha3**2 / (- 2 * alpha2) * (1 - np.exp(2 * alpha2 * t_data)) + 0.0001
    max_res = 1 / (2 * sigma2) - 1
    # exp_res = np.exp(-1*(max_res - res))
    # return 1 / (1 + exp_res) * res
    return -(mod_abs(max_res - res) - max_res - res) / 2 - 1



@jit(nopython=True, cache = True)
def p_sampler_ou(alpha, dwt, init_state = None):
    alpha1, alpha2, alpha3 = alpha[0], alpha[1], alpha[2]
    T = len(dwt)
    dt = 1 / T
    xt = np.zeros(dwt.shape)
    mu = -alpha1 / alpha2
    if init_state is None:
        xt[0] = mu
    else:
        xt[0] = init_state
    for k in range(1, T):
        xt[k] = xt[k - 1] + (alpha1 + alpha2 * xt[k - 1]) * dt + alpha3 * dwt[k]
    return xt

@jit(nopython=True, cache = True)
def m_sampler_ou(alpha, a1t, a2t, dwt, init_state = None):
    alpha1, alpha2, alpha3 = alpha[0], alpha[1], alpha[2]
    T = len(dwt)
    dt = 1 / T
    xt = np.zeros(dwt.shape)
    mu = -alpha1 / alpha2
    theta = -alpha2
    nu = alpha3
    D = nu**2 / 2
    if init_state is None:
        xt[0] = mu
    else:
        xt[0] = init_state
    for i in range(1, T):
        a1, a2 = a1t[i], a2t[i]
        a1dt, a2dt =  (a1t[i] - a1t[i - 1]) / dt, (a2t[i] - a2t[i - 1]) / dt
        #a1dt, a2dt = 0, 0
        t = i/T
        sigma2 = D / theta * (1 - np.exp(- 2 * theta * t))
        p = (1 - 2 * a2 * sigma2)
        sigma2w = sigma2 / p
        #xs = mu + xt[0] * np.exp(-theta * t)
        xs = (xt[0] - mu) * np.exp(-theta * t) + mu
        xsw = (xs + a1 * sigma2) / p
        sigma2dt = nu**2 - 2 * theta * sigma2
        sigma2wdt = (sigma2dt + 2 * sigma2**2 * a2dt) / p**2
        xsdt = -theta * (xs - mu)
        xswdt = (xsdt + a1 * sigma2dt + a1dt * sigma2) / p + 2 * xsw * (a2dt * sigma2 + a2 * sigma2dt) / p
        B = nu / p
        A = xswdt - (xt[i - 1] - xsw) * (B**2 - sigma2wdt) / (2 * sigma2w)
        xt[i] = xt[i - 1] + A * dt + B * dwt[i]
    return xt

@jit(nopython=True, cache = True)
def pdf_p(x, t, alpha, x0):
    alpha1, alpha2, alpha3 = alpha[0], alpha[1], alpha[2]
    mu = -alpha1 / alpha2
    theta = -alpha2
    nu = alpha3
    D = alpha3**2/2
    if t == 0:
        t = t + 0.01
    sigma2 = D / theta * (1 - np.exp(- 2 * theta * t))
    xs = (x0 - mu) * np.exp(-theta * t) + mu
    return  1/np.sqrt(2 * np.pi * sigma2) * np.exp(-(x - xs)**2 / (2 * sigma2))

@jit(nopython=True, cache = True)
def pdf_m(x, t, alpha, x0, a1, a2):
    alpha1, alpha2, alpha3 = alpha[0], alpha[1], alpha[2]
    mu = -alpha1 / alpha2
    theta = -alpha2
    nu = alpha3
    D = alpha3**2/2
    if t == 0:
        t = t + 0.01
    sigma2 = D / theta * (1 - np.exp(- 2 * theta * t))
    p = (1 - 2 * a2 * sigma2)
    sigma2w = sigma2 / p
    xs = (x0 - mu) * np.exp(-theta * t) + mu
    xsw = (xs + a1 * sigma2) / p
    return  1/np.sqrt(2 * np.pi * sigma2w) * np.exp(-(x - xsw)**2 / (2 * sigma2w))

#alpha = np.array([0.3, -1.0, 0.25])
#alpha = np.array([0.11215126, 1.89076068, 0.15013735])
#alpha = np.array([-24.28 , -41.6  ,   1.503])
#alpha = np.array([-0.19788412, -0.67066287, -0.70314979])

alpha = np.array([5.421e-01,  9.751e-01,  1.024e-01])

mu = -alpha[0] / alpha[1]
x0 = 1.0 * mu
T = 600
latent_process_tr = 100000
dwt = np.random.normal(0, 1, size = (T, latent_process_tr)) * np.sqrt(1/T)
#p = p_sampler_ou(alpha, dwt, init_state = x0)

x_data = np.linspace(-2, 2, 200)

t_data = np.linspace(0, 1, T)
# a1t = 40 * np.sin(8 * np.pi/2 * t_data) + t_data + 1
# a2t = 1 - 30 * t_data + 4*t_data**2
# a1_params = np.array([ 1.22926716e+02, -9.18034709e+03,  1.49180472e+05, -1.09471747e+06,
#         4.37636070e+06, -1.03607933e+07,  1.49399148e+07, -1.28897980e+07,
#         6.11772940e+06, -1.22885234e+06])
# a2_params = np.array([ 4.38482739e+02, -2.31159093e+03,  1.88422857e+03,  3.46489738e+04,
#        -1.82846010e+05,  4.50592534e+05, -6.54475441e+05,  5.77984013e+05,
#        -2.87520608e+05,  6.15764018e+04])

# a1_params = np.array([ 1.76763087e+01,  2.50533310e+03, -4.28922699e+04,  3.13559813e+05,
#        -1.26658665e+06,  3.06559238e+06, -4.54539206e+06,  4.04058239e+06,
#        -1.97611072e+06,  4.08733689e+05])
# a2_params = np.array([ 2.75472229e+02, -1.26044675e+04,  1.43186561e+05, -8.91097795e+05,
#         3.39267752e+06, -8.08134979e+06,  1.19878006e+07, -1.07114900e+07,
#         5.26439491e+06, -1.09183911e+06])

# a1_params = np.array([ 3.00578682e+02, -2.52248329e+04,  3.37529806e+05, -2.08343265e+06,
#         7.31831076e+06, -1.57801116e+07,  2.12828558e+07, -1.74976652e+07,
#         8.01066331e+06, -1.56334820e+06])

# a2_params = np.array([ 1.64442288e+02, -4.92815874e+03,  7.89393336e+04, -6.52761609e+05,
#         2.97141973e+06, -7.90193195e+06,  1.25926609e+07, -1.18366580e+07,
#         6.04782490e+06, -1.29482758e+06])

# a1t = poly(t_data, a1_params, intercept = False)
# a2t = poly_corr(t_data, a2_params, alpha, intercept = False)

a1t = 14 * np.sin(8 * np.pi/2 * t_data)
a2t = -12 * np.cos(4 * np.pi/2 * t_data)**2

m = m_sampler_ou(alpha, a1t, a2t, dwt, init_state=x0)



fig, ax = plt.subplots()
# n, bins, patches = ax.hist(p[0], bins=x_data, density=True, 
#                            color='#0504aa', alpha=0.7, rwidth=0.85)  # Plot the initial normalized histogram with fixed x-range
# ax.plot(x_data, pdf_p(x_data, 0, alpha, x0))

n, bins, patches = ax.hist(m[0], bins=x_data, density=True, 
                           color='#0504aa', alpha=0.7, rwidth=0.85)  # Plot the initial normalized histogram with fixed x-range
ax.plot(x_data, pdf_m(x_data, 0, alpha, x0, a1t[0], a2t[0]))

def animate(i):
    ax.clear()  # Clear the previous histogram
    #n, bins, patches = ax.hist(p[i], bins=x_data, density=True, color='#0504aa', alpha=0.7, rwidth=0.85, label = 'SDE sample') # Plot the histogram at time step i
    n, bins, patches = ax.hist(m[i], bins=x_data, density=True, color='#0504aa', alpha=0.7, rwidth=0.85, label = 'SDE sample') # Plot the histogram at time step i
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    #ax.set_title(f'OU-process. Histogram at time step {i}', loc = 'left')
    ax.set_title(f'Modified OU-process. Histogram at time step {i}', loc = 'left')
    #ax.set_xlim(-2, 2)  # Set the x-range
    ax.set_ylim(0, 4)  # Set the x-range
    #ax.plot(x_data, pdf_p(x_data, i/T, alpha, x0), label = 'Fokker-Plank PDF', color = 'red')
    ax.plot(x_data, pdf_m(x_data, i/T, alpha, x0, a1t[i], a2t[i]), label = 'Fokker-Plank PDF', color = 'red')
    ax.legend(loc = 'upper right', fancybox = True, shadow = True) 
    ax.grid(True)

ani = animation.FuncAnimation(fig, animate, frames=T, interval=1, blit=False)

#ani.save('animated_histogram.gif', writer='imagemagick', fps=25)

plt.show()

