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

Native row and emission work can be parallelized explicitly for a large fit:

```python
from pyscarcopula import NumericalConfig

result = cop.fit(
    u,
    method="scar-tm-ou",
    config=NumericalConfig(n_threads=4),
)
```

If `NumericalConfig` is omitted, the fit always uses one native thread. See
[CPU Parallelism](parallelism.md) for deterministic execution, thread-safety,
and nested process/thread guidance.

### When to use

Equicorrelation SCAR is a good fit when:

- All pairwise correlations move together, common in equity and crypto markets
- You need fast estimation for large $d$, with $O(d)$ density evaluation
- You want a compact, interpretable model with 3 parameters

For heterogeneous dependence, use C-vine or R-vine instead.

The equicorrelation density hot path is linear in `d` and does not construct a
dense correlation matrix. Output arrays and input storage still determine the
end-to-end memory footprint, so large sampling jobs should be batched.

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

For `method='scar-tm-ou'`, the public OU parameters remain
`(kappa, mu, nu)`, where `nu` is the diffusion coefficient. Internally,
`StochasticStudentCopula` is optimized in
`(log(kappa), mu, log(sigma_x))`, with stationary latent standard deviation

$$\sigma_x=\frac{\nu}{\sqrt{2\kappa}}.$$

This parameterization keeps the two positive scale parameters unconstrained
and avoids the poorly conditioned combination of very small mean reversion
and very large stationary variance. `alpha0` and the fitted result still use
the public `(kappa, mu, nu)` representation; no conversion is required in
user code. The fit diagnostics expose the internal choice as
`optimizer_parameterization='log_kappa_mu_log_stationary_sigma'`.

Dynamic emission evaluation uses a Student inverse-CDF table from the model
boundary at `df = 2 + 1e-6` through `df = 1000`. Outside the table, the native
gradient obtains the quantile derivative analytically from the Student CDF;
it does not finite-difference the complete emission likelihood. Above the
final table node, dynamic fits use a third-order Cornish-Fisher expansion
toward the normal quantile, together with its analytical `df` derivative.
Static Student likelihoods retain exact quantiles.

### Stochastic Student copula with estimated static correlation

Static correlation can be handled in three modes:

```python
# fixed correlation
cop = StochasticStudentCopula(d=5, R=R, corr_mode="fixed")

# one-parameter shrinkage toward identity
cop = StochasticStudentCopula(d=5, corr_mode="shrinkage")

# full static correlation for small dimensions
cop = StochasticStudentCopula(d=5, corr_mode="cholesky")

# compact fixed factor correlation for large dimensions
cop = StochasticStudentCopula(
    d=u.shape[1],
    corr_mode="factor",
    factor_rank=8,
)
cop.initialize_factor(u)
```

`shrinkage` and `cholesky` estimate static correlation jointly with constant-df
MLE or with the three OU parameters in SCAR-TM-OU. `cholesky` mode estimates
`d(d-1)/2` additional static parameters and is intended for low-dimensional
problems. Their initialization/base matrix uses `corr_base` when supplied,
otherwise `R`, and otherwise a Kendall estimate from the fit data. GAS keeps
fixed-correlation semantics.

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
uses a different estimator.

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

The `fixed`, `shrinkage`, and `cholesky` modes retain a dense `O(d^2)`
correlation representation. The phase-9.4 `factor` mode instead stores
`O(d*k + k^2)` state, can accept validated loadings or estimate them with a
deterministic tiled randomized SVD, and exposes static row/grid evaluation.
It never creates dense `R` implicitly. Phase 9.5 supports static MLE, native
GAS, SCAR-TM-OU matrix/local/spectral fitting and native SCAR-MC trajectory
likelihood without changing that representation. Phase 9.6 adds structural
factor Student generation, bounded `sample_at_parameter_batches`,
fitted-model `sample_batches`/`predict_batches`, and exact conditional
sampling without a dense Schur complement.

Static MLE can also refine `df` and the factor loadings jointly when the
identifiable parameter count is deliberately bounded:

```python
joint = StochasticStudentCopula(
    d=u.shape[1],
    corr_mode="factor",
    factor_rank=4,
    factor_estimation="joint",
    factor_joint_max_params=100_000,
)
joint_result = joint.fit(
    u, method="mle", config=NumericalConfig(n_threads=4))
```

The joint path uses analytical matrix-free gradients and an identified
pivoted lower-triangular loading convention. Diagnostics report the anchor
rows, initial/final penalized objective, terminal gradient and acceptance
threshold, condition estimate, and native reduction workspace. `success=True`
requires both optimizer success and the terminal-gradient gate. Joint
loadings for GAS and SCAR are rejected: those methods require additional
derivatives through their sequential recursions. Use the production
`two-stage` policy for dynamic models and large `d`.

```python
factor_samples = cop.sample_at_parameter(
    50_000, r=6.0, rng=np.random.default_rng(2026), n_threads=4)

factor_conditional = cop.sample_conditional(
    50_000,
    r=6.0,
    given={0: 0.25, 3: 0.8},
    rng=np.random.default_rng(2027),
    n_threads=4,
)
```

For fixed coordinates `G`, conditioning factorizes only
`I + B_G.T @ D_G^-1 @ B_G`, a `k*k` matrix. Free coordinates retain a
diagonal-plus-low-rank covariance, so memory remains linear in `d*k` plus
the requested output. Use the batch APIs and `memory_budget_bytes` when
`n*d` itself is large.

## Common API

Static Gaussian, static Student, equicorrelation Gaussian MLE, and stochastic
Student MLE return the same result type:

```python
from pyscarcopula import GaussianCopula

cop = GaussianCopula()
result = cop.fit(u)

assert result is cop.fit_result
correlation = result.correlation_matrix
parameters = result.model_parameters
print(result.log_likelihood, result.n_params, result.aic, result.bic)
```

The result stores model parameters, observation count, an explicit correlation
matrix, optimizer status, diagnostics, and the parameter count used by AIC/BIC.
For `GaussianCopula(corr_mode="factor")` and factor Student models, the result
intentionally stores `correlation_matrix=None` and keeps compact loadings and
uniqueness in `model_parameters`.

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

Static Gaussian and Student models also support exact conditional generation
in pseudo-observation space:

```python
import numpy as np

from pyscarcopula import GaussianCopula, StudentCopula

gaussian = GaussianCopula()
gaussian.fit(u, method='mle')
gaussian_conditional = gaussian.sample_conditional(
    n=10_000,
    given={0: 0.25, 2: 0.8},
    rng=np.random.default_rng(2026),
    n_threads=4,
)

student = StudentCopula()
student.fit(u, method='mle')
student_conditional = student.sample_conditional(
    n=10_000,
    given={1: 0.6},
    rng=np.random.default_rng(2026),
    n_threads=4,
)
```

`given` maps zero-based variable indices to values strictly inside `(0, 1)`.
The supplied columns remain fixed and the remaining coordinates are drawn from
the fitted conditional copula. If every variable is supplied, both APIs return
constant rows. Static `GaussianCopula` and `StudentCopula` accept only
`method='mle'`.

The conditional kernels generate random draws in Python before entering the
native implementation. Reusing the same seed produces the same tested output
for `n_threads=1` and parallel thread counts.

For a static Gaussian model with large `d`, select factor correlation
explicitly:

```python
from pyscarcopula import NumericalConfig

factor_gaussian = GaussianCopula(
    d=u.shape[1],
    corr_mode="factor",
    factor_rank=8,
    factor_tile_size=16_384,
)
factor_result = factor_gaussian.fit(
    u,
    method="mle",
    config=NumericalConfig(n_threads=4),
)

for block in factor_gaussian.sample_batches(
    1_000_000,
    batch_rows=128,
    given={0: 0.25},
    n_threads=4,
):
    consume(block)
```

The Gaussian adapter reuses `FactorCorrelation` for likelihood, sampling,
conditioning, persistence, and rolling-window reconstruction. Two-stage
normal-score estimation is tiled and does not build a `d*d` covariance.
Likelihood and sampling remain `O(d*k)` per row; conditional generation uses
only a `k*k` factor system. The factor Rosenblatt transform used by
`gof_test` keeps `O(T*k + k^2)` workspace.
