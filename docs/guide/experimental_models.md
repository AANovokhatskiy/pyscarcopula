# Experimental Models

!!! warning "Experimental API"
    The models in `pyscarcopula.copula.experimental` are under active
    development. Their API may change in future releases.

## Overview

The `experimental` module contains $d$-dimensional copula models that extend
the SCAR framework beyond bivariate Archimedean families. All models use a
**single scalar latent OU process**, so the existing transfer matrix
infrastructure works unchanged.

| Model | Class | Latent parameter | Description |
|-------|-------|-----------------|-------------|
| Equicorrelation Gaussian | `EquicorrGaussianCopula` | $\rho(t)$ | Single dynamic correlation for d assets |
| Stochastic Student-t | `StochasticStudentCopula` | $\nu(t)$ | Fixed correlation, OU-driven degrees of freedom |
| Stochastic Student-t DCC | `StochasticStudentDCCCopula` | $\nu(t)$ | DCC-driven correlation + OU-driven df |

## Mathematical Contract

The experimental models are multivariate copula emission models with one
time-varying scalar parameter. For pseudo-observations
$u_t=(u_{t1},\ldots,u_{td})\in(0,1)^d$, each model supplies a density

$$c(u_t; r_t), \qquad r_t = \Psi(g_t).$$

The scalar state can be estimated in three ways:

- **MLE**: $r_t=r$ is constant and estimated by likelihood maximization.
- **GAS**: $g_t$ follows the observation-driven recursion
  $$g_{t+1} = \omega + \beta g_t + \gamma s_t,$$
  where $s_t$ is the score of $\log c(u_t;\Psi(g_t))$ with respect to $g_t$.
- **SCAR-TM-OU**: $g_t$ is a latent OU process. The likelihood integrates over
  the latent state using the transfer-matrix filter.

This is the main implementation contract: the strategy layer does not need to
know the full multivariate model, only how to evaluate row-wise densities and,
for GAS, row-wise score derivatives.

## Equicorrelation Gaussian Copula

For $d$ assets, the standard Gaussian copula has $d(d-1)/2$ static
correlation parameters. The equicorrelation model uses a single dynamic
correlation:

$$R(t) = (1-\rho(t)) \cdot I + \rho(t) \cdot \mathbf{1}\mathbf{1}^\top$$

All pairwise correlations equal $\rho(t)$. The Gaussian copula density is

$$c(u_t;\rho_t) =
\frac{\phi_d(z_t;0,R(\rho_t))}
     {\prod_{j=1}^d \phi(z_{tj})}, \qquad
z_{tj}=\Phi^{-1}(u_{tj}).$$

The valid range is
$\rho_t\in(-1/(d-1),1)$, enforced by the parameter transform. In SCAR,
$\rho_t=\Psi(x_t)$ and $x_t$ is the scalar OU state. This gives 3 dynamic
parameters instead of $d(d-1)/2$ static correlations.

The density is analytical and $O(d)$ per evaluation, with no matrix inversion required.

### Usage

```python
from pyscarcopula.copula.experimental import EquicorrGaussianCopula

cop = EquicorrGaussianCopula(d=6)

# MLE (constant rho)
cop.fit(u, method='mle')

# GAS (score-driven rho)
cop.fit(u, method='gas')

# SCAR (time-varying rho)
cop.fit(u, method='scar-tm-ou')
```

### When to use

Equicorrelation SCAR is a good fit when:

- All pairwise correlations move together, common in equity and crypto markets
- You need fast estimation for large $d$, with $O(d)$ density evaluation
- You want a compact, interpretable model with 3 parameters

For heterogeneous dependence, use C-vine or R-vine instead.

## Stochastic Student-t Copula

A $d$-dimensional t-copula where the correlation matrix $R$ is estimated once
via Kendall's $\tau$ and fixed, while the degrees-of-freedom parameter
$\nu(t)$ follows a latent OU process:

$$\nu(t) = 2 + \mathrm{softplus}(x(t)), \qquad x(t) \sim \text{OU}(\theta, \mu, \sigma)$$

The transform maps $\mathbb{R} \to (2, \infty)$, ensuring finite variance.
The copula density is

$$c(u_t;R,\nu_t)=
\frac{t_d(q_t;0,R,\nu_t)}
     {\prod_{j=1}^d t_1(q_{tj};\nu_t)}, \qquad
q_{tj}=T_{\nu_t}^{-1}(u_{tj}).$$

$R$ is treated as a nuisance dependence estimate: it is obtained from
Kendall's tau before the dynamic fit and then held fixed. The latent/dynamic
part of the model controls only tail thickness. Smaller $\nu_t$ means heavier
joint tails; larger $\nu_t$ moves the copula toward the Gaussian copula.

Implementation notes:

- GAS uses the derivative of the Student copula log-density with respect to
  $\nu_t$ and then applies the chain rule through
  $\nu_t=2+\mathrm{softplus}(g_t)$.
- SCAR-TM-OU evaluates the Student copula density on a latent grid and filters
  the OU state by transfer matrix.
- The Student quantile and log-density formulas are evaluated in vectorized
  numerical kernels, but the mathematical model is the standard t-copula above.

### Usage

```python
from pyscarcopula.copula.experimental import StochasticStudentCopula

cop = StochasticStudentCopula(d=6)

# Fit (R estimated automatically via Kendall tau)
result = cop.fit(returns, method='scar-tm-ou', to_pobs=True)

# GAS is also supported
gas_result = cop.fit(returns, method='gas', to_pobs=True)

# Predictive df(t) path
df_t = cop.predictive_mean()

# Predict with time-varying df
pred = cop.predict(10000)

# GoF
from pyscarcopula.stattests import gof_test
gof = gof_test(cop, returns, to_pobs=True)
```

### When to use

- When tail dependence varies over time
- When the correlation structure is relatively stable but tail thickness changes
- As an alternative to vine copulas for moderate dimensions

## Stochastic Student-t DCC Copula

Extends the Stochastic Student-t copula with a time-varying correlation matrix driven by a DCC(1,1) filter:

$$U_t \mid x_t, R_t \sim \text{t-copula}(R_t, \nu_t)$$

where $\nu_t = 2 + \mathrm{softplus}(x_t)$ is OU-driven and $R_t$ is a deterministic DCC path.

The density is the same Student copula density, but with $R$ replaced by
$R_t$:

$$c(u_t;R_t,\nu_t)=
\frac{t_d(q_t;0,R_t,\nu_t)}
     {\prod_{j=1}^d t_1(q_{tj};\nu_t)}.$$

The DCC layer is fitted separately on standardized residuals. In the current
implementation, the correlation path is deterministic conditional on those
residuals:

$$Q_t=(1-a-b)\bar Q + a z_{t-1}z_{t-1}^\top + bQ_{t-1},$$

$$R_t=\operatorname{diag}(Q_t)^{-1/2} Q_t
      \operatorname{diag}(Q_t)^{-1/2}.$$

The latent state remains scalar, so GAS and SCAR-TM-OU only estimate the
degrees-of-freedom dynamics. The model is therefore a two-layer construction:

- DCC supplies the deterministic time-varying correlation path $R_t$.
- GAS or SCAR supplies the time-varying tail thickness $\nu_t$.

This separation is intentional. It keeps the latent SCAR filter one-dimensional
while still allowing changing correlation structure.

### Usage

```python
from pyscarcopula.copula.experimental import StochasticStudentDCCCopula

cop = StochasticStudentDCCCopula(d=6)

# Estimate the DCC correlation path first.
cop.fit_R_t(returns=returns)

# Fit the df dynamics on pseudo-observations.
result = cop.fit(returns, method='scar-tm-ou', to_pobs=True)

# GAS df dynamics uses the same cached DCC path.
gas_result = cop.fit(returns, method='gas', to_pobs=True)

# Predict (uses last R_T or DCC forecast)
pred = cop.predict(10000)
```

### When to use

- When both correlation structure and tail heaviness change over time
- For capturing joint dynamics in volatile markets
- When a static correlation matrix is too restrictive but a full vine is too complex

## Common API

All experimental models support the same core operations:

```python
# Goodness of fit
from pyscarcopula.stattests import gof_test
gof = gof_test(cop, u, to_pobs=False)

# Sampling (fixed parameter)
samples = cop.sample(n=10000)

# Prediction (conditional on data)
pred = cop.predict(n=10000)

# Predictive mean parameter path
params_t = cop.predictive_mean()
```
