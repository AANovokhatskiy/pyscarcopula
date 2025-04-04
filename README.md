- [About project](#about-project)
- [Install](#install)
- [Mathematical basis](#mathematical-basis)
  * [Joint distribution](#joint-distribution)
  * [Goodness of fit](#goodness-of-fit)
  * [Risk metrics](#risk-metrics)
- [Examples](#examples)
  * [1. Read dataset and transform to log-returns](#read-dataset)
  * [2. Initialize copula object](#initialize-copula)
  * [3. Fit copula](#fit-copula)
  * [4. Sample from copula](#sample-copula)
  * [5. Calculation of risk metrics](#risk-metrics)

# About project
This project is made to fit multivaiate distribution using copula approach. To consider complex dependencies between the random vaiables we extend the classical model with constant parameters to stochastic model, where parameter considered as Ornstein-Uhlenbeck process. The idea based on discrete model suggested in Liesenfeld and Richard, Univariate and multivariate stochastic volatility models (2003).

For that model we introduce a likelihood function as an expectation over all random process states and use Monte Carlo estimates for that expectation while solving parameter estimation problem. The approach discussed in detail in our article (link will be availble when it will published). For brief description and code examples see below. Frank, Gumbel, Clayton and Joe copulas were implemented in the code. Goodness-of-fit (GoF) metrics based on Rosenblatt transform is also available.

The stochastic models is used here to calculate metrics VaR and CVaR and build a CVaR-optimized portfolio. The calculation based on paper Rockafellar, Uryasev Optimization of conditional value-at-risk (2000).

This project is made during the MSc program "Financial mathematics and financial technologies" in Sirius University, Sochi, Russia. 2023-2024.

# Install
Installation could be made using pip:
```
git clone https://github.com/AANovokhatskiy/pyscarcopula
cd pyscarcopula
pip install .
```

# Mathematical basis
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
| Frank | $-\log{\left(\frac{\exp{(-\theta t)} - 1}{\exp{(-\theta)} - 1}\right)}$ | $-\frac{1}{\theta} \log{\left(1 + \exp{(-t)} \cdot (\exp{(-\theta)} - 1)\right)}$ | $[0, \infty)$  |
| Joe | $-\log{\left( 1 - (1-t)^\theta \right)}$ | $1 - (1 - \exp{(-t)})^{(1 / \theta)}$ | $[1, \infty)$ |
| Clayton | $\frac{1}{\theta}\left(t^{-\theta} - 1 \right)$ | $\left( 1 + t \theta\right)^{-1/\theta}$ |  $[0, \infty)$ |

Classical approach implies that $\theta$ is constant parameter. We can estimate it using Maximum likelihood method. Now consider that $\theta = \Psi (x_t)$
where $x_t$ is Ornstein-Uhlenbeck process (this process cannot be observed and considered as latent process)
```math
dx_t = \theta \left( \mu - x_t \right) dt + \nu dW_t
```
and $\Psi(x)$ - a suitable transformation that converts an $x_t$ values to appropriate copula parameter range. In such stochastic case we can write an expression for likelihood as an expectation and then use Monte-Carlo estimates for it. We don't give this precise expressions here for the simplicity. But we can solve such maximum likelihood problem and also implement some techniques for Monte-Carlo variance decreaseing. Again we don't give a precise expressions.

This approach for discrete case was proposed by Liesenfield and Richard, Univariate and Multivariate Stochastic Volatility Models: Estimation and Diagnostics (2003) and used in Hafner and Manner, Dynamic stochastic copula models: estimation, inference and applications (2012). Here we develop this approach in continious case.

## Goodness of fit
### GoF Copula
For copula GoF we use Rosenblatt transform. For given multivariate pseudo observations dataset $U$ and copula $C$ could be constructed a new dataset $U'$ using Rosenblatt transform $R$. It can be shown that if copula parameters are found correctly then random variable $Y$
```math
Y = F_{\chi^2_d}\left( \sum\limits_{i = 1}^{d} \Phi^{-1}(u'_i)^2 \right)
```
is uniformly disturbed (see details in Hering and Hofert, Goodness-of-fit Tests for Archimedean
Copulas in High Dimensions (2015)). This fact could be checked using various methods. We use a scipy implementation of Cramer-Von-Mises test.

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
Here we solve this minimizations problems numerically using Monte-Carlo estimates of function $F$. This calculations could be made in a runnig window.

# Examples
<a name="read-dataset"></a>
### 1. Read dataset and transform to log-returns
```python
import pandas as pd
import numpy as np

crypto_prices = pd.read_csv("data/crypto_prices.csv", index_col=0, sep = ';')
```

<a name="initialize-copula"></a>
### 2. Initialize copula object
```python
from pyscarcopula import GumbelCopula, FrankCopula, JoeCopula, ClaytonCopula
from pyscarcopula import VineCopula, GaussianCopula, StudentCopula

copula = GumbelCopula(dim = 2)
```
Also available `GumbelCopula`, `JoeCopula`, `ClaytonCopula`, `FrankCopula`, 'VineCopula', 'GaussianCopula', `StudentCopula`

For initialized copula it is possible to show a cdf as a sympy expression (only for Archimedian):
```python
copula.sp_cdf()
```
with output
```math
e^{- \left(\left(- \log{\left(u_{0} \right)}\right)^{r} + \left(- \log{\left(u_{1} \right)}\right)^{r} + \left(- \log{\left(u_{2} \right)}\right)^{r} + \left(- \log{\left(u_{3} \right)}\right)^{r}\right)^{\frac{1}{r}}}
```

It is also possible to call *sp_pdf()* to show pdf expression. But the anwser would be much more complex.

<a name="fit-copula"></a>
### 3. Fit copula (bivariate example)
Stochastic methods also available for VineCopula() with specified parameter method (mle by default)
```python
tickers = ['BTC-USD', 'ETH-USD']

returns_pd = np.log(crypto_prices[tickers] / crypto_prices[tickers].shift(1))[1:501]
returns = returns_pd.values

fit_result = copula.fit(data = returns, method = 'scar-m-ou', seed = 333, to_pobs = True)
fit_result
```

Function *fit* description:
1. ***data*** -- initial dataset: log-returns or pseudo observations (see ***to_pobs*** description). Type *Numpy Array*. Required parameter.
2. ***method*** -- calculation method. Type *Literal*. Available methods: *mle*, *scar-p-ou*, *scar-m-ou*, *scar-p-ld*. Required parameter.
* mle - classic method with constant parameter
* scar-p-ou - stochastic model with Ornstein-Uhlenbeck process as a parameter. ***latent_process_tr*** = 10000 is good choice here
* scar-m-ou - stochastic model with Ornstein-Uhlenbeck process as a parameter and implemented importance sampling Monte-Carlo techniques. ***latent_process_tr*** = 500 is good choice here
* scar-p-ld - stochastic model with process with logistic distribution transition density (experimental). ***latent_process_tr*** = 10000 is good choice here
3. ***alpha0*** -- starting point for optimization problem. Type *Numpy Array*. Optional parameter.
4. ***tol*** -- stop criteria (gradient norm). Type *float*. Optional parameter. Default value: $10^{-5}$ for *mle*, $10^{-2}$ for other methods.
5. ***to_pobs*** -- transform ***data*** to pseudo observations. Type *bool*. Optional parameter. Default value *False*.
6. ***latent_process_tr*** -- number of latent process trajectories (for *mle* is ignored). Type *int*. Optional parameter. Default value $500$.
7. ***M_iterations*** -- number of importance sampling steps (only for *scar-m-ou*). Type *int*. Optional parameter. Default value $3$.
8. ***seed*** -- random state. Type *int*. Optional parameter. Default value *None*. If *None* then every run of program would unique. Parameter is ignored if ***dwt*** is set explicitly.
9. ***dwt*** -- Wiener process values. Type *Numpy Array*. Optional parameter. Default value *None*. If *None* then function generate dataset automatically.
10. ***stationary*** -- stochastic model with stationary transition density. Type *bool*. Default value *False*. Optional parameter.
10. ***print_path*** -- If True shows all output of optimization process (useful for debugging). Type *bool*. Default value *False*. Optional parameter.

fit result is mostly scipy.minimize output
```python
           message: CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL
           success: True
            status: 0
               fun: 219.33843671302438
                 x: [ 9.619e-01  2.346e+00  1.451e+00]
               nit: 6
               jac: [-2.045e-03  5.006e-03  2.361e-03]
              nfev: 28
              njev: 7
          hess_inv: <3x3 LbfgsInvHessProduct with dtype=float64>
              name: Gumbel copula
            method: scar-m-ou
 latent_process_tr: 500
        stationary: False
      M_iterations: 3
```
Where ***fun*** is log likelihood and ***x*** - parameter set $\[\theta, \mu, \nu\]$. For the MLE method here is real estimated parameter.

Goodness of fit for copula using Rosenblatt transform:
```python
from pyscarcopula.stattests import gof_test

gof_test(copula, moex_returns, fit_result, to_pobs=True)
```
with output
```python
CramerVonMisesResult(statistic=0.19260862172251336, pvalue=0.28221824469771717)
```

<a name="sample-copula"></a>
### 4. Sample from copula
It is possible to get sample of pseudo observations from copula
```python
#sampling from copula with constant parameter

copula.get_sample(size = 1000, r = 1.2)
```
The output is array of shape = (N, dim). Copula parameter r should be set manually (array of size N is also supported for sampling time dependent parameter).

Sampling from random process state also available
```python
#sampling from copula with time-dependent parameter

from pyscarcopula.sampler.sampler_ou import stationary_state_ou

size = 2000
random_process_state = copula.transform(stationary_state_ou(fit_result.x, size))

copula.get_sample(size = size, r = random_process_state)
```

## 5. Calculation of risk metrics
```python
#consider multivariate case

tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'XRP-USD', 'DOGE-USD']

returns_pd = np.log(crypto_prices[tickers] / crypto_prices[tickers].shift(1))[1:]
returns = returns_pd.values
```

```python
copula = StudentCopula()
```

```python
from pyscarcopula.metrics import risk_metrics

gamma = [0.95]
window_len = 250
latent_process_tr = 500
MC_iterations = [int(10**5)]
M_iterations = 5

method = 'mle'
marginals_method = 'johnsonsu'

count_instruments = len(tickers)
portfolio_weight = np.ones(count_instruments) / count_instruments
result = risk_metrics(copula,
                      returns,
                      window_len,
                      gamma,
                      MC_iterations,
                      marginals_method = marginals_method,
                      latent_process_type = method,
                      optimize_portfolio = False,
                      portfolio_weight = portfolio_weight,

                      latent_process_tr = latent_process_tr,
                      M_iterations = M_iterations,
                      )
```

Function *risk_metrics* description 
1. ***copula*** -- object of class ArchimedianCopula (and inherited classes). Type *ArchimedianCopula*. Required parameter.
2. ***data*** -- log-return dataset. Type *Numpy Array*. Required parameter.
3. ***window_len*** -- window len. Type *int*. Required parameter. To use all available data use length of ***data***.  
4. ***gamma*** -- significance level. Type *float* or *Numpy Array*. Required parameter. If *Numpy Array* then calculations is made for every element of array. Made for minimizaion of repeated calculations. Usually one set, for example, $0.95$, $0.97$, $0.99$ or array $[0.95, 0.97, 0.99]$.  Default value $0.95$. Optional parameter.
5. ***MC_iterations*** -- number of Monte-Carlo iterations that used for risk metrics calculations. Type *int* or *Numpy Array*. Required parameter. If *Numpy Array* then calculations is made for every element of array. Possible value, for example, $10^4$, $10^5$, $10^6$ and so on. Default value $10^5$. Optional parameter.
6. ***marginals_params_method*** -- method of marginal distribution fit. Type *Literal*. Available methods: *normal*, *hyperbolic*, *stable*, *logistic*, *johnsonsu*, *laplace*. Default value *johnsonsu*. Optional parameter.
7. ***latent_process_type*** -- type of stochastic process that used as copula parameter. Type *Literal*. Available methods: *mle*, *scar-p-ou*, *scar-m-ou*, *scar-p-ld*. Default value *mle*. Optional parameter.
8. ***optimize_portfolio*** -- parameter responsible for the need to search for optimal CVaR weights.Type *Bool*. Optional parameter. Default value $True$.
9. ***portfolio_weight*** -- portfolio weight. Type *Numpy Array*. Optional parameter. If ***optimize_portfolio*** = True, this value is ignored. Default value -- equal weighted investment portfolio.
10. ***kwargs*** -- keyworded arguments for copula fit (see fit method desctiption). Type *int*. Optional parameter. 


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
i1 = window_len
i2 = len(returns) - 1

fig,ax = plt.subplots(n,m,figsize=(10,6))
loc = plticker.MultipleLocator(base=127.0)

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

Examples of usage of this code coulde be found in example.ipynb notebook.