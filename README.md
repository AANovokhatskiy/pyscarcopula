- [About project](#about-project)
- [Install](#install)
- [Mathematical basis](#mathematical-basis)
  * [Joint distribution](#joint-distribution)
  * [Goodness of fit](#goodness-of-fit)
- [Examples](#examples)
  * [1. Read dataset and transform to log-returns](#read-dataset)
  * [2. Initialize copula object](#initialize-copula)
  * [3. Fit copula](#fit-copula)
  * [4. Sample from copula](#sample-copula)
  
# About project
This project is made to fit multivaiate distribution using copula approach. To consider complex dependencies between the random vaiables we extend the classical model with constant parameters to stochastic model, where parameter considered as Ornstein-Uhlenbeck process. For that model we introduce a likelihood function as an expectation over all random process states and use Monte Carlo estimates for that expectation while solving parameter estimation problem. The approach discussed in detail in our article (link will be availble when it will published). For brief description and code examples see below. Frank, Gumbel, Clayton and Joe copulas were implemented in the code. Goodness-of-fit (GoF) metrics based on Rosenblatt transform is also available.

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
fit_result = copula.fit(data = returns, method = 'scar-m-ou', seed = 333)
fit_result
```

Function *fit* description:
1. ***data*** -- initial dataset: log-returns or pseudo observations (see ***to_pobs*** description). Type *Numpy Array*. Required parameter.
2. ***method*** -- calculation method. Type *Literal*. Available methods: *mle*, *scar-p-ou*, *scar-m-ou*, *scar-p-ld*. Required parameter.
* mle - classic method with constant parameter
* scar-p-ou - stochastic model with Ornstein-Uhlenbeck process as a parameter. ***latent_process_tr*** = 10000 is good choice here
* scar-m-ou - stochastic model with Ornstein-Uhlenbeck process as a parameter and implemented importance sampling Monte-Carlo techniques. ***latent_process_tr*** = 500 is good choice here
* scar-p-ld - stochastic model with process with logistic distribution transition density (experimental). ***latent_process_tr*** = 10000 is good choice here
3. ***tol*** -- stop criteria (gradient norm). Type *float*. Optional parameter. Default value: $10^{-5}$ for *mle*, $10^{-2}$ for other methods.
4. ***latent_process_tr*** -- number of latent process trajectories (for *mle* is ignored). Type *int*. Optional parameter. Default value $500$.
5. ***M_iterations*** -- number of importance sampling steps (only for *scar-m-ou*). Type *int*. Optional parameter. Default value $5$.
6. ***to_pobs*** -- transform ***data*** to pseudo observations. Type *bool*. Optional parameter. Default value *True*.
7. ***dwt*** -- Wiener process values. Type *Numpy Array*. Optional parameter. Default value *None*. If *None* then function generate dataset automatically.
8. ***seed*** -- random state. Type *int*. Optional parameter. Default value *None*. If *None* then every run of program would unique. Parameter is ignored if ***dwt*** is set explicitly.
9. ***alpha0*** -- starting point for optimization problem. Type *Numpy Array*. Optional parameter.
10. ***init_state*** -- initial state of latent process. Type *Numpy Array*. Optional parameter.
11. ***stationary*** -- stochastic model with stationary transition density. Type *bool*. Default value *False*. Optional parameter.

fit result is mostly scipy.minimize output
```python
message: CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL
           success: True
            status: 0
               fun: 184.1481158495324
                 x: [ 1.636e+01  5.775e-01  1.355e+00]
               nit: 15
               jac: [-8.664e-05 -1.237e-03  4.949e-04]
              nfev: 80
              njev: 20
          hess_inv: <3x3 LbfgsInvHessProduct with dtype=float64>
              name: Gumbel copula
            method: scar-m-ou
 latent_process_tr: 500
        stationary: False
      M_iterations: 5
```
Where ***fun*** is log likelihood and ***x*** - parameter set $\[\theta, \mu, \nu\]$. Note that real parameter (that is used in calculations) is transformed in suitable interval using *copula.transform()* function.

Goodness of fit for copula using Rosenblatt transform:
```python
from pyscarcopula.stattests import gof_test

gof_test(copula, moex_returns, fit_result, to_pobs=True)
```
with output
```python
CramerVonMisesResult(statistic=0.12957361510979126, pvalue=0.45838687740863493)
```

<a name="sample-copula"></a>
### 4. Sample from copula
It is possible to get sample of pseudo observations from copula
```python
copula.get_sample(N = 1000, r = 1.2)
```
The output is array of shape = (N, dim). Copula parameter r should be set manually (array of size N is also supported).

Sampling from random process state also available
```python
from pyscarcopula.sampler.sampler_ou import stationary_state_ou

size = 2000
random_process_state = copula.transform(stationary_state_ou(fit_result.x, size))

copula.get_sample(N = size, r = random_process_state)
```