- [What is it](#what-is-it)
- [Mathematical basis](#mathematical-basis)
  * [Marginal distributions](#marginal-distributions)
  * [Joint distribution](#joint-distribution)
  * [Risk metrics](#risk-metrics)
  * [Goodness of fit](#goodness-of-fit)
- [Examples](#examples)


# What is it
This project is made to calculate to calculate portfolio risk metrics Value-at-risk (VaR) and Conditional Value-at-risk (CVaR) using stochastic multivariate copula approach.
The main idea of calculations is the following. Consider a multivariate financial time-series (stock prices for example) and transform them to log-returns. Then we do
* Fit marginal distributions. Implemented: normal (best perfomance), Levy stable (heavy-tailed), generalized hyperbolic (heavy-tailed).
* Fit joint distribution. We use multivariate Archimedian copla approach. Implemented Frank, Gumbel, Clayton and Joe copulas. For this copulas we consider a classical model with constant parameter (MLE) and time-dependend model where Ornstein-Uhlenbeck process is used as a copula parameter. This extension of model could improve joint distribution fit about 15-20% more for log-likelihood.
* Calculate risk metrics. WE calculate metrics VaR and CVaR using famous paper Rockafellar, Uryasev Optimization of conditional value-at-risk (2000). Authors proved two theorems that provides one to calculate metrics VaR and CVaR by minimazing special function and also build a CVaR optimized portfolio.
* Assess goodness-of-fit (GoF) metrics. For copulate GoF we use Rosenblatt transform approach and calculate corresponding p-value; for risk metrics -- Kupiec likelihood ratio test and critical statistics.

More information see below in Mathematical basis section.

This project is made during the MSc program "Financial mathematics and financial technologies" in Sirius University, Sochi, Russia. 2023-2024.

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
| Frank | $-\log{\left(\frac{\exp{(-\theta t)} - 1}{\exp{(\theta)} - 1}\right)}$ | $-\frac{1}{\theta} \log{\left(1 + \exp{(-t)} \right)}$ | $(-\infty, \infty)\setminus\lbrace 0 \rbrace$  |
| Joe | $-\log{\left( 1 - (1-t)^\theta \right)}$ | $1 - (1 - \exp{(-t)})^{(1 / \theta}$ | $[1, \infty)$ |
| Clayton | $\frac{1}{\theta}\left(t^{-\theta} - 1 \right)$ | $\left( 1 + t \theta\right)^{-1/\theta}$ |  $[0, \infty)$ |

Classical approach implies that $\theta$ is constant parameter. We can estimate it using Maximul likelihood method. Now consider that $\theta = \Lambda (x_t)$
where $x_t$ is Ornstein-Uhlenbeck process
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
Examples of using this code coulde be found in example.ipynb notebook.

