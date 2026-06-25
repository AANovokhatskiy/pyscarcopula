# Multivariate Models

## Overview

The `multivariate` module contains $d$-dimensional copula models that extend
the SCAR framework beyond bivariate families. Dynamic models use a
**single scalar latent OU process**, so the existing transfer matrix
infrastructure works unchanged.

| Model | Class | Latent parameter | Description |
|-------|-------|-----------------|-------------|
| Equicorrelation Gaussian | `EquicorrGaussianCopula` | $\rho(t)$ | Single dynamic correlation for d assets |
| Stochastic Student-t | `StochasticStudentCopula` | $\nu(t)$ | Fixed correlation, OU-driven degrees of freedom |

## Mathematical Contract

The dynamic multivariate models are copula emission models with one
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

The shared formulas for scalar dynamic states, parameter links, SCAR filters,
and dynamic Rosenblatt GoF are summarized in
[Mathematical Contracts](mathematical-contracts.md).

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

The density and its scalar score are analytical and can be evaluated without a
generic dense matrix inversion because the equicorrelation determinant and
inverse have closed forms.

### Usage

```python
from pyscarcopula.copula.multivariate import EquicorrGaussianCopula

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

A $d$-dimensional t-copula where the degrees-of-freedom parameter $\nu(t)$
follows a latent OU process and the static correlation matrix $R$ is fixed or
estimated jointly:

$$\nu(t) = 2 + 10^{-6} + \mathrm{softplus}(x(t)), \qquad x(t) \sim \text{OU}(\theta, \mu, \sigma)$$

The transform maps $\mathbb{R} \to (2 + 10^{-6}, \infty)$, ensuring finite
variance.
The copula density is

$$c(u_t;R,\nu_t)=
\frac{t_d(q_t;0,R,\nu_t)}
     {\prod_{j=1}^d t_1(q_{tj};\nu_t)}, \qquad
q_{tj}=T_{\nu_t}^{-1}(u_{tj}).$$

The latent/dynamic part of the model controls only tail thickness. Smaller
$\nu_t$ means heavier joint tails; larger $\nu_t$ moves the copula toward the
Gaussian copula.

### Stochastic Student copula with estimated static correlation

Static correlation can be handled in three modes:

```python
# fixed correlation, current default
cop = StochasticStudentCopula(d=5, R=R, corr_mode="fixed")

# one-parameter shrinkage toward identity
cop = StochasticStudentCopula(d=5, corr_mode="shrinkage")

# full static correlation for small dimensions
cop = StochasticStudentCopula(d=5, corr_mode="cholesky")
```

`shrinkage` and `cholesky` estimate static correlation jointly with constant-df
MLE or with the three OU parameters in SCAR-TM-OU. `cholesky` mode estimates
`d(d-1)/2` additional static parameters and is intended for low-dimensional
problems. Their initialization/base matrix uses `corr_base` when supplied,
otherwise `R`, and otherwise a Kendall estimate from the fit data. Python owns
the joint parameterization and L-BFGS-B optimizer, while all SCAR-TM-OU
likelihood evaluations use the native engine. GAS keeps fixed-correlation
semantics.

The C++ layer receives the current static correlation matrix for each
objective evaluation. It does not optimize correlation parameters internally
but matrix and local transition backends expose analytical derivatives with
respect to its unique off-diagonal entries. Python applies the parameterization
chain rule and remains the owner of L-BFGS-B. The spectral transition path
also exposes native correlation derivatives. Diagnostics report
`correlation_gradient='analytical'` and `joint_gradient='analytical'` when
that route is used; a numerical correlation fallback is retained only for a
future evaluator that does not expose those derivatives.

Static and stochastic Student models share the same Kendall preprocessing.
Each pair uses
$R_{ij}=\sin(\pi\tau_{ij}/2)$. If a pairwise Kendall statistic is unavailable,
for example because one column is constant, that pair is initialized with zero
dependence. The resulting matrix is projected to an SPD correlation matrix
when necessary. Fit diagnostics report:

- `corr_initialization_source`;
- `corr_projection_applied`;
- `corr_min_eigenvalue_before` and `corr_min_eigenvalue_after`;
- `corr_nonfinite_kendall_pairs`.

Gaussian score-space correlation fitting is intentionally separate because it
uses a different estimator and therefore does not share this Kendall contract.

Implementation notes:

- GAS uses the derivative of the Student copula log-density with respect to
  $\nu_t$ and then applies the chain rule through
  $\nu_t=2+10^{-6}+\mathrm{softplus}(g_t)$.
- Fixed and Python-parameterized static-correlation SCAR-TM-OU can evaluate the
  Student copula density in the C++ OU backend. Joint parameterization and
  optimization remain in Python. With `analytical_grad=True`, the joint
  optimizer uses analytical OU and static-correlation derivatives for C++
  matrix/local transitions. The native spectral evaluator uses analytical OU
  and static-correlation derivatives. `analytical_grad=False` retains a fully
  numerical optimizer gradient.
- The Student quantile table is built once per pseudo-observation array and
  reused by GAS, TM, and Hermite-TM block evaluations. Block calls pass a row
  offset into this full-sample cache instead of rebuilding quantiles for
  temporary slices. The model uses a transient `StudentPPFCache`. Replacing
  `R` keeps this table and refreshes only
  correlation-specific state. C++ cache initialization accepts contiguous
  NumPy buffers, validates them once, and performs one owning copy without
  Python-list conversion. Python and C++ use cubic Hermite interpolation
  inside the node range and exact Student quantiles outside it.
- GAS filtering, likelihood, state updates, and gradients use the mandatory
  native evaluator.
- The Student log-density formulas are evaluated in vectorized numerical
  kernels, but the mathematical model is the standard t-copula above.
- Conditional Gaussian and Student sampling use native linear-algebra kernels.
  Python retains input validation, pseudo-observation clipping, quantile/CDF
  conversion, and ownership of `numpy.random.Generator`. It generates normal
  and chi-square innovations before calling C++, so public seed reproducibility
  is preserved. Native factorization errors are reported to Python and are not
  masked by a Python fallback.

### Usage

```python
from pyscarcopula.copula.multivariate import StochasticStudentCopula

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

## Common API

Static Gaussian, static Student, equicorrelation Gaussian MLE, and stochastic
Student MLE return the same `MultivariateMLEResult` contract:

```python
from pyscarcopula import GaussianCopula

cop = GaussianCopula()
result = cop.fit(u)

assert result is cop.fit_result
correlation = result.correlation_matrix
parameters = result.model_parameters
print(result.log_likelihood, result.n_params, result.aic, result.bic)
```

The result stores natural model parameters, observation count, an explicit
correlation matrix, optimizer status, diagnostics, and the common parameter
count used by AIC/BIC. Static `fit()` methods no longer return a raw matrix or
`(correlation, df)` tuple.

All multivariate models support the following core operations where applicable:

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
