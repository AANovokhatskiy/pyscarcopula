- [About project](#about-project)
- [Install](#install)
- [Mathematical basis](#mathematical-basis)
  * [Marginal distributions](#marginal-distributions)
  * [Joint distribution](#joint-distribution)
  * [Risk metrics](#risk-metrics)
  * [Goodness of fit](#goodness-of-fit)
- [Examples](#examples)
  * [1. Read dataset and transform to log-returns](#read-dataset)
  * [2. Initialize copula object](#initialize-copula)
  * [3. Fit copula](#fit-copula)
  * [4. Sample from copula](#sample-copula)
  * [5. Calculate risk_metrics](#risk-metrics)
  
# About project
This project is made to calculate portfolio risk metrics Value-at-risk (VaR) and Conditional Value-at-risk (CVaR) using stochastic multivariate copula approach.
The main idea of calculations is the following. Consider a multivariate financial time-series (stock prices for example) and transform them to log-returns. Then we do
* Fit marginal distributions. Implemented: normal (best perfomance), Levy stable (heavy-tailed), generalized hyperbolic (heavy-tailed).
* Fit joint distribution. We use multivariate Archimedian copla approach. Implemented Frank, Gumbel, Clayton and Joe copulas. For this copulas we consider a classical model with constant parameter (MLE) and time-dependend model where Ornstein-Uhlenbeck process is used as a copula parameter. This extension of model could improve joint distribution fit about 15-20% more for log-likelihood.
* Calculate risk metrics. WE calculate metrics VaR and CVaR using famous paper Rockafellar, Uryasev Optimization of conditional value-at-risk (2000). Authors proved two theorems that provides one to calculate metrics VaR and CVaR by minimazing special function and also build a CVaR optimized portfolio.
* Assess goodness-of-fit (GoF) metrics. For copulate GoF we use Rosenblatt transform approach and calculate corresponding p-value; for risk metrics -- Kupiec likelihood ratio test and critical statistics.

More information see below in Mathematical basis section.

This project is made during the MSc program "Financial mathematics and financial technologies" in Sirius University, Sochi, Russia. 2023-2024.

# Install
Installation could be made using pip:
```
git clone https://github.com/AANovokhatskiy/pyscarcopula
cd pyscarcopula
pip install .
```

# Mathematical basis
## Marginal distributions
We consider the following distributions for logarithmic returns:

**Normal**
```math
    f(x, \mu, \sigma) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x- \mu)^2}{2 \sigma^2}}
```
The implementations is made manually and based on numba library. So this shows the best perfomance, but performs bad with heavy-tailed data.

**Levy stable**
```math
    f(x) = \frac{1}{2\pi} \int^{\infty}_{-\infty}\phi (t) e^{-ixt} dt
```

with 
```math
    \phi(t,\alpha,\beta, c, \mu) = e^{it\mu-|ct|^\alpha\left(1-i\beta sign(t) \Phi(\alpha, t)\right)}
```
Here we just use an existing scipy implementation. So see details in the corresponding page. Parallelization on all available cpu cores is made using joblib.

**Generalized hyperbolic**
```math
    f(x,\lambda,\alpha,\beta, \delta, \mu) = \frac{(\gamma /\delta)^{\lambda}}{\sqrt{2\pi} K_{\lambda}(\delta\gamma)}e^{\beta(x-\mu)} \cdot \frac{K_{\lambda-1/2}\left(\alpha\sqrt{\delta^2+(x-\mu)^2}\right)}{\left(\sqrt{\delta^2+(x-\mu)^2}/\alpha\right)^{\left(1/2-\lambda\right)}}

```
Here we also use an existing scipy implementation. Parallelization on all available cpu cores is made using joblib.

## Joint distribution
According to Sklar's theorem joint distribution could be constructed using special function of marginal distributions is named *copula*. 
```math
F\left(x_1, x_2, \ldots, x_d\right) = C\left( F_1 \left(x_1\right), F_2 \left(x_2\right), \ldots, F_d \left(x_d\right)\right)
```
Here we consider only a class of single-parameter copulas known as Archimedian. Denote $u_i = F_i(x_i)$, so this copulas could be constracted using generator function $\phi(u_i; \theta)$:
```math
C(u_1, ..., u_d) = \phi^{[-1]} \left( \phi(u_1; \theta) + ... + \phi(u_d; \theta) \right)
```

Here we use the following copulas
| Copula | $\phi(t; \theta)$ | $\phi^{-1}(t; \theta)$ | $\theta \in$ |
|---------------|---------------|---------------|---------------|
| Gumbel | $(-\log t)^\theta$ | $\exp{\left(-t^{1/\theta}\right)}$ |  $[1, \infty)$ |
| Frank | $-\log{\left(\frac{\exp{(-\theta t)} - 1}{\exp{(-\theta)} - 1}\right)}$ | $-\frac{1}{\theta} \log{\left(1 + \exp{(-t)} (\exp{(-\theta)} - 1)\right)}$ | $[0, \infty)$  |
| Joe | $-\log{\left( 1 - (1-t)^\theta \right)}$ | $1 - (1 - \exp{(-t)})^{(1 / \theta}$ | $[1, \infty)$ |
| Clayton | $\frac{1}{\theta}\left(t^{-\theta} - 1 \right)$ | $\left( 1 + t \theta\right)^{-1/\theta}$ |  $[0, \infty)$ |

Classical approach implies that $\theta$ is constant parameter. We can estimate it using Maximul likelihood method. Now consider that $\theta = \Lambda (x_t)$
where $x_t$ is Ornstein-Uhlenbeck process (this process cannot be observed and considered as latent process)
```math
dx_t = \left( \alpha_1 +\alpha_2 x_t \right) dt + \alpha_3 dW_t
```
and $\Lambda(x)$ - a suitable transformation that converts an $x_t$ values to appropriate copula parameter range. In such stochastic case we can write an expression for likelihood as an expectation and then use Monte-Carlo estimates for it. We don't give this precise expressions here for the simplicity. But we can solve such maximum likelihood problem and also implement some techniques for Monte-Carlo variance decreaseing. Again we don't give a precise expressions.

This approach for discrete case was proposed by Liesenfield and Richard, Univariate and Multivariate Stochastic Volatility Models: Estimation and Diagnostics (2003) and used in Hafner and Manner, Dynamic stochastic copula models: estimation, inference and applications (2012). Here we develop this approach in continious case.

## Risk metrics
Let $w$ - vector of portfolio weights, $\gamma$ - significance level, $f(r,w) = - (r,w)$ - portfolio loss function where $r$ - vector of portfolio return. Consider a function:
```math
F_{\gamma}(w, q) = q + \frac{1}{1 - \gamma} \int\limits_{\mathbb{R}^d} \left( f(w, r) - q \right)_{+} p(r) \, dr
```
Rockafellar and Uryasev in Optimization of conditional value-at-risk (2000) proved that:
```math
  \textnormal{CVaR}_{\gamma}(w) = \underset{q \in \mathbb{R}} {\min} \; F_{\gamma}(w, q),
  ```
 ```math
  \textnormal{VaR}_{\gamma}(w) =  \underset{q \in \mathbb{R}}{ \arg \min} \; F_{\gamma}(w, q)
  ```
```math
  \underset{w \in X}{\min} \; \textnormal{CVaR}_{\gamma}(w) = \underset{(w, q) \in X \times \mathbb{R}} {\min} \; F_{\gamma}(w, q)
  ````
Here we solve this minimizations problems numerically using Monte-Carlo estimates of function $F$. This calculations could be made in runnig window.

## Goodness of fit
### GoF Copula
For copula GoF we use Rosenblatt transform. For given multivariate pseudo observations dataset $U$ and copula $C$ could be constructed a new dataset $U'$ using Rosenblatt transform $R$. It can be shown that if copula parameters are found correctly then random variable $Y$
```math
Y = F_{\chi^2_d}\left( \sum\limits_{i = 1}^{d} \Phi^{-1}(u'_i)^2 \right)
```
is uniformly disturbed (see details in Hering and Hofert, Goodness-of-fit Tests for Archimedean
Copulas in High Dimensions (2015)). This fact could be checked using various methods. We use a scipy implementation of Cramer-Von-Mises test.

### GoF risk metrics
We could also check a statistical significance of found risk metrics estimates. A simple likelihood ratio test proposed by Kupiec (Techniques for verifying the accuracy of risk measurement models (1995)) is following. Consider statistics
```math
LR_{POF} = - 2 \log{\left( \frac{\left(1 - p\right)^{N-x} p^x}{\left(1 - \frac{x}{N} \right)^{N - x}\left(\frac{x}{N}\right)^x} \right)}
```
where $N$ - number of observations, $p = 1 - \gamma$ - significance level, $x$ - number of risk level breakdowns. This statistics has asymptotically $\chi^2$ distribution. So if statistics exceed the critical value then we reject our estimations of risk metrics.

# Examples
<a name="read-dataset"></a>
### 1. Read dataset and transform to log-returns
```python
import pandas as pd
import numpy as np

moex_data = pd.read_csv("data/moex_top.csv", index_col=0)
tickers = ['AFLT', 'LSRG', 'GAZP', 'NLMK']

returns_pd = np.log(moex_data[tickers] / moex_data[tickers].shift(1))[1:501]
returns = returns_pd.values
```

<a name="initialize-copula"></a>
### 2. Initialize copula object
```python
from pyscarcopula import GumbelCopula
copula = GumbelCopula(dim = 4)
```
Also available `GumbelCopula`, `JoeCopula`, `ClaytonCopula`, `FrankCopula`.

For initialized copula it is possible to show a cdf as a sympy expression:
```python
copula.sp_cdf()
```
with output
```math
e^{- \left(\left(- \log{\left(u_{0} \right)}\right)^{r} + \left(- \log{\left(u_{1} \right)}\right)^{r} + \left(- \log{\left(u_{2} \right)}\right)^{r} + \left(- \log{\left(u_{3} \right)}\right)^{r}\right)^{\frac{1}{r}}}
```

It is also possible to call *sp_pdf()* to show pdf expression. But the anwser would be much more complex.

<a name="fit-copula"></a>
### 3. Fit copula
```python
fit_result = copula.fit(data = returns, latent_process_tr = 500, M_iterations = 5, accuracy=1e-4,
                        method = 'scar-m-ou', seed = 333)
fit_result
```

Function *fit* description:
1. ***data*** -- initial dataset: log-returns or pseudo observations (see ***to_pobs*** description). Type *Numpy Array*. Required parameter.
2. ***method*** -- calculation method. Type *Literal*. Available methods: *mle*, *scar-p-ou*, *scar-m-ou*, *scar-p-ld*. Required parameter.
* mle - classic method with constant parameter
* scar-p-ou - stochastic model with Ornstein-Uhlenbeck process as a parameter. ***latent_process_tr*** = 10000 is good choice here
* scar-m-ou - stochastic model with Ornstein-Uhlenbeck process as a parameter and implemented importance sampling Monte-Carlo techniques. ***latent_process_tr*** = 500 is good choice here
* scar-p-ld - stochastic model with process with logistic distribution transition density (experimental). ***latent_process_tr*** = 10000 is good choice here
3. ***accuracy*** -- calculation accuracy. Type *float*. Optional parameter. Default value: $10^{-5}$ для *mle*, $10^{-4}$ for other methods.
4. ***latent_process_tr*** -- number of latent process trajectories (for *mle* is ignored). Type *int*. Optional parameter. Default value $500$.
5. ***M_iterations*** -- number of importance sampling steps (only for *scar-m-ou*). Type *int*. Optional parameter. Default value $5$.
6. ***to_pobs*** -- transform ***data*** to pseudo observations. Type *bool*. Optional parameter. Default value *True*.
7. ***dwt*** -- Wiener process values. Type *Numpy Array*. Optional parameter. Default value *None*. If *None* then function generate dataset automatically.
8. ***seed*** -- random state. Type *int*. Optional parameter. Default value *None*. If *None* then every run of program would unique. Parameter is ignored if ***dwt*** is set explicitly.
9. ***alpha0*** -- starting point for optimization problem. Type *Numpy Array*. Optional parameter.
10. ***init_state*** -- initial state of latent process. Type *Numpy Array*. Optional parameter.

fit result is mostly scipy.minimize output
```python
           message: CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH
           success: True
            status: 0
               fun: 184.55204171425615
                 x: [ 9.903e-01  6.082e-01  6.261e-01]
               nit: 6
               jac: [ 2.171e-01  3.244e-01 -9.893e-02]
              nfev: 32
              njev: 8
          hess_inv: <3x3 LbfgsInvHessProduct with dtype=float64>
              name: Gumbel copula
            method: scar-m-ou
 latent_process_tr: 500
      m_iterations: 5
```
Where ***fun*** is log likelihood and ***x*** - parameter set $\[\theta, \mu, \nu\]$. Note that real parameter (that is used in calculations) is transformed in suitable interval using *copula.transform()* function.

Goodness of fit for copula using Rosenblatt transform:
```python
from pyscarcopula.stattests import gof_test

gof_test(copula, moex_returns, fit_result, to_pobs=True)
```
with output
```python
CramerVonMisesResult(statistic=0.11370981269840286, pvalue=0.5219822616590395)
```

<a name="sample-copula"></a>
### 4. Sample from copula
It is possible to get sample of pseudo observations from copula
```python
copula.get_sample(N = 1000, r = 1.2)
```
The output is array of shape = (N, dim). Copula parameter r should be set manually (array of size N is also supported).

<a name="risk-metrics"></a>
### 5. Calculate risk_metrics
```python
from pyscarcopula.metrics import risk_metrics

gamma = [0.95]
window_len = 250
latent_process_tr = 10000
MC_iterations = [int(10**5)]
M_iterations = 5

#fastest calculations
# method = 'mle'
# marginals_method = 'normal'

#More precise calculations. For appropiate choice of latent_process_tr and M_iterations see README.md
method = 'scar-p-ou' # or use 'scar-m-ou'
marginals_method = 'hyperbolic'

count_instruments = len(tickers)
portfolio_weight = np.ones(count_instruments) / count_instruments
result = risk_metrics(copula,
                      returns,
                      window_len,
                      gamma,
                      MC_iterations,
                      marginals_params_method = marginals_method,
                      latent_process_type = method,
                      latent_process_tr = latent_process_tr,
                      optimize_portfolio = False,
                      portfolio_weight = portfolio_weight,
                      seed = 111,
                      M_iterations = M_iterations,
                      save_logs = False
                      )
```

Function *risk_metrics* description 
1. ***copula*** -- object of class ArchimedianCopula (and inherited classes). Type *ArchimedianCopula*. Required parameter.
2. ***data*** -- log-return dataset. Type *Numpy Array*. Required parameter.
3. ***window_len*** -- window len. Type *int*. Required parameter. To use all available data use length of ***data***.  
4. ***gamma*** -- significance level. Type *float* or *Numpy Array*. Required parameter. If *Numpy Array* then calculations is made for every element of array. Made for minimizaion of repeated calculations. Usually one set, for example, $0.95$, $0.97$, $0.99$ or array $[0.95, 0.97, 0.99]$. 
5. ***latent_process_type*** -- type of stochastic process that used as copula parameter. Type *Literal*. Available methods: *mle*, *scar-p-ou*, *scar-m-ou*, *scar-p-ld*. Required parameter.
6. ***latent_process_tr*** -- number of Monte-Carlo iterations that used for copula fit. Type *int*. Required parameter. 
7. ***marginals_params_method*** -- method of marginal distribution fit. Type *Literal*. Available methods: *normal*, *hyperbolic*, *stable*. Required parameter.
8. ***MC_iterations*** -- number of Monte-Carlo iterations that used for risk metrics calculations. Type *int* or *Numpy Array*. Required parameter. If *Numpy Array* then calculations is made for every element of array. Possible value, for example, $10^4$, $10^5$, $10^6$ and so on.
9. ***optimize_portfolio*** -- parameter responsible for the need to search for optimal CVaR weights.Type *Bool*. Optional parameter. Default value $True$.
10. ***portfolio_weight*** -- portfolio weight. Type *Numpy Array*. Optional parameter. If ***optimize_portfolio*** = True this value is ignored. Default value -- equal weighted investment portfolio.
11. ***save_logs*** -- parameter responsible for the need to save logs of copula parameters search. If ***save_logs*** = True then function would create folder logs in the current directory and save copula parameters in csv file. Optional parameter. Default value $False$.
    
Calculated values could be extracted as follows
```python
var = result[0.95][1000000]['var']
cvar = result[0.95][1000000]['cvar']
portfolio_weight = result[0.95][1000000]['weight']
```

Plot the CVaR metrics:
```python
from pyscarcopula.metrics import cvar_emp_window
from matplotlib import pyplot as plt
import matplotlib.ticker as plticker

pd_var_95 = pd.Series(data = -result[gamma[0]][MC_iterations[0]]['var'], index=returns_pd.index).shift(1)
pd_cvar_95 = pd.Series(data = -result[gamma[0]][MC_iterations[0]]['cvar'], index=returns_pd.index).shift(1)


weight = result[gamma[0]][MC_iterations[0]]['weight']

n = 1
m = 1
i1 = 250
i2 = 499

fig,ax = plt.subplots(n,m,figsize=(10,6))
loc = plticker.MultipleLocator(base=27.0)

daily_returns = ((np.exp(returns_pd) - 1) * weight).sum(axis=1)
cvar_emp = cvar_emp_window(daily_returns.values, 1 - gamma[0], window_len)

ax.plot(np.clip(daily_returns, -0.2, 0.2)[i1:i2], label = 'Portfolio log return')
ax.plot(cvar_emp[i1:i2], label = 'Emperical CVaR', linestyle='dashed', color = 'gray')

ax.plot(pd_cvar_95[i1:i2], label= f'{method} {marginals_method} CVaR 95%')

ax.set_title(f'Daily returns', fontsize = 14)

ax.xaxis.set_major_locator(loc)
ax.set_xlabel('Date', fontsize = 12, loc = 'center')
ax.set_ylabel('Log return', fontsize = 12, loc = 'center')
ax.tick_params(axis='x', labelrotation = 15, labelsize = 12)
ax.tick_params(axis='y', labelsize = 12)
ax.grid(True)
ax.legend(fontsize=12, loc = 'upper right')
```
![output](https://github.com/user-attachments/assets/673f5c1d-ede3-41ac-bdc0-be0e57c49585)

Goodness of fit for found CVaR using Kupiec test. 
```python
from pyscarcopula.stattests import Kupiec_POF

POF = Kupiec_POF(daily_returns.values[i1:i2], pd_cvar_95.values[i1:i2].flatten(), 1 - gamma[0])
```
and the output:
```python
N = 249, Expected drawdowns = 12.45, Observed drawdowns = 7, x/N = 0.0281, p = 0.05
critical_value = 3.8415, estimated_statistics = 2.9633, accept = True
```

Examples of usage of this code coulde be found in example.ipynb notebook.