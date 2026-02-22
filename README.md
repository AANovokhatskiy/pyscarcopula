- [About project](#about-project)
- [Install](#install)
- [Mathematical basis](#mathematical-basis)
  * [Joint distribution](#joint-distribution)
  * [Goodness of fit](#goodness-of-fit)
  * [Risk metrics](#risk-metrics)
- [Examples](#examples)
  * [1. Read dataset and transform to log-returns](#read-dataset)
  * [2. Initialize copula object](#initialize-copula)
  * [3. Fit bivariate copula](#fit-copula_2dim)
  * [4. Sample from copula](#sample-copula)
  * [5. Fit ndim copula](#fit-copula_ndim)
  * [6. Calculation of risk metrics](#risk-metrics)

# About project
This project is made to fit multivaiate distribution using copula approach. To consider complex dependencies between the random vaiables we extend the classical model with constant parameters to stochastic model, where parameter considered as Ornstein-Uhlenbeck process. The idea based on discrete model suggested in Liesenfeld and Richard, Univariate and multivariate stochastic volatility models (2003).

For that model we introduce a likelihood function as an expectation over all random process states and use Transfer matrix approach and Monte Carlo estimates for that expectation while solving parameter estimation problem. The approach discussed in detail in our article (link will be availble when it will published). For brief description and code examples see below. Frank, Gumbel, Clayton and Joe copulas were implemented in the code. Goodness-of-fit (GoF) metrics based on Rosenblatt transform is also available.

The stochastic models is used here to calculate metrics VaR and CVaR and build a CVaR-optimized portfolio. The calculation based on paper Rockafellar, Uryasev Optimization of conditional value-at-risk (2000).

This project is made during the program "Financial mathematics and financial technologies" in Sirius University, Sochi, Russia.

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
Copulas in High Dimensions (2015)).

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
from pyscarcopula.utils import pobs
from pyscarcopula import (GumbelCopula, FrankCopula, JoeCopula, ClaytonCopula,
                          BivariateGaussianCopula, GaussianCopula, StudentCopula,
                          CVineCopula)
from pyscarcopula.stattests import gof_test

crypto_prices = pd.read_csv("data/crypto_prices.csv", index_col=0, sep = ';')
tickers = ['BTC-USD', 'ETH-USD']

returns_pd = np.log(crypto_prices[tickers] / crypto_prices[tickers].shift(1))[1:]
returns = returns_pd.values
u = pobs(returns)
```

<a name="initialize-copula"></a>
### 2. Initialize copula object
```python
copula_mle = GumbelCopula(rotate=180)
copula_tm = GumbelCopula(rotate=180)
```
Available rotations = [0, 90, 180, 270]

<a name="fit-copula_2dim"></a>
### 3. Fit bivariate copula and GoF
```python
fit_result_mle = copula_mle.fit(data = returns, method = 'mle', to_pobs=True)
gof_result_mle = gof_test(copula_mle, returns, to_pobs=True)

print(f"MLE:")
print(f"log_likelihood = {fit_result_mle.log_likelihood:.4f}, copula paramter = {fit_result_mle.copula_param:.4f}")
print(f"GoF statistic={gof_result_mle.statistic:.4f}, p-value={gof_result_mle.pvalue:.4f}")

fit_result_tm = copula_tm.fit(data = returns, method = 'scar-tm-ou', to_pobs=True)
gof_result_tm = gof_test(copula_tm, returns, to_pobs=True)

print(f"Stochastic model via Trasfer matrix:")
print(f"log_likelihood = {fit_result_tm.log_likelihood:.4f}, copula paramter = {fit_result_tm.alpha}")
print(f"GoF statistic={gof_result_tm.statistic:.4f}, p-value={gof_result_tm.pvalue:.4f}")
```
Output
```python
MLE:
log_likelihood = 955.6275, copula paramter = 2.8318
GoF statistic=0.7536, p-value=0.0094

Stochastic model via Trasfer matrix:
log_likelihood = 1045.4997, copula paramter = [58.99857924  1.48765678  4.53265176]
GoF statistic=0.0718, p-value=0.7404
```
Where ***fun*** is log likelihood and ***x*** - parameter set $\[\theta, \mu, \nu\]$. For the MLE method here is real estimated parameter.

<a name="sample-copula"></a>
### 4. Sample from copula
Sampling from copula with constant parameter
```python
copula_mle.sample(n = 1000, r = 1.1)
```

Sampling next state from stochastic copula
```python
#sampling from copula with time-dependent parameter
copula_tm.predict(n = 1000)
```

<a name="fit-copula_ndim"></a>
### 5. Fit ndim copula
Implemented C-vine (mle and scar-tm-ou), Student and Gaussian (mle only)

```python
tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'XRP-USD', 'DOGE-USD']

returns_pd = np.log(crypto_prices[tickers] / crypto_prices[tickers].shift(1))[1:251]
returns = returns_pd.values
u = pobs(returns)
```

```python
vine = CVineCopula()
copula_s = StudentCopula()
copula_g = GaussianCopula()

vine.fit(u, method='scar-tm-ou')
vine.summary()
ll = vine.log_likelihood(u)
samples = vine.sample(1000, u_train=u)  # for SCAR
predictions = vine.predict(1000)


copula_s.fit(u)
copula_g.fit(u)

gof_result_s = gof_test(copula_s, u, to_pobs=False)
gof_result_g = gof_test(copula_g, u, to_pobs=False)
gof_result_vine = gof_test(vine, u, to_pobs=False)

print('Stochastic C-vine')
print(f"log_likelihood = {vine.log_likelihood(u)}")
print(f"statistic={gof_result_vine.statistic:.4f}, p-value={gof_result_vine.pvalue:.4f}\n")

print('Student')
print(f"log_likelihood = {copula_s.log_likelihood(u)}")
print(f"statistic={gof_result_s.statistic:.4f}, p-value={gof_result_s.pvalue:.4f}\n")

print('Gaussian')
print(f"log_likelihood = {copula_g.log_likelihood(u)}")
print(f"statistic={gof_result_g.statistic:.4f}, p-value={gof_result_g.pvalue:.4f}")
```

Output
```python
Stochastic C-vine
log_likelihood = 890.299064222026
statistic=0.0686, p-value=0.7609

Student
log_likelihood = 764.4232783937387
statistic=1.5673, p-value=0.0001

Gaussian
log_likelihood = 761.0026811425082
statistic=2.5324, p-value=0.0000
```

## 6. Calculation of risk metrics
```python
from pyscarcopula.metrics import risk_metrics

gamma = [0.95]
window_len = 100
MC_iterations = [int(10**5)]

method = 'mle'
marginals_method = 'johnsonsu'


d = len(tickers)
eq_weight = np.ones(d) / d

result = risk_metrics(vine,
                      returns,
                      window_len,
                      gamma,
                      MC_iterations,
                      marginals_method = marginals_method,
                      method = method,
                      optimize_portfolio = False,
                      portfolio_weight = eq_weight,
                      )
```

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

if weight.ndim == 2:
    eqw = np.ones(d) / d
    for k in range(0, window_len - 1):
        weight[k] = eqw

n = 1
m = 1
i1 = window_len
i2 = len(returns) - 1

fig,ax = plt.subplots(n,m,figsize=(8,4))
loc = plticker.MultipleLocator(base=20.0)

daily_returns = ((np.exp(returns_pd) - 1) * weight).sum(axis=1)
cvar_emp = cvar_emp_window(daily_returns.values, 1 - gamma[0], window_len)

ax.plot(np.clip(daily_returns, -0.1, 0.1)[i1:i2], label = 'Portfolio log return')
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