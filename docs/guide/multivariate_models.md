# Multivariate Models

## Overview

The `multivariate` module contains $d$-dimensional copula models that extend
the SCAR framework beyond bivariate families. Dynamic models vary one scalar parameter using GAS or a latent OU process.
The OU fits use the same transfer-matrix machinery as bivariate SCAR.

| Model | Class | Latent parameter | Description |
|-------|-------|-----------------|-------------|
| Static Gaussian | `GaussianCopula` | None | Static Gaussian dependence with configurable correlation policy |
| Static Student-t | `StudentCopula` | None | Static Student dependence with fitted `df` and configurable correlation policy |
| Equicorrelation Gaussian | `EquicorrGaussianCopula` | $\rho(t)$ | Single dynamic correlation for d assets |
| Stochastic Student-t | `StochasticStudentCopula` | $\operatorname{df}(t)$ | Fixed correlation, OU-driven degrees of freedom |

## Example data

Examples use pseudo-observations with observations in rows. The static and
factor sections use the five-column `u` below. The dynamic usage examples
either reuse `u` or declare a separate input explicitly. Each fit replaces the model's
fitted state; inspect `result.success` before using a candidate.

```python
import numpy as np
from pyscarcopula import NumericalConfig

u = np.random.default_rng(2026).uniform(0.01, 0.99, size=(200, 5))
```

## Static Gaussian and Student Models

For `GaussianCopula` and `StudentCopula`, `method="mle"` denotes a static
model. It is not a promise that correlation was optimized jointly. Select the
correlation procedure with `corr_mode` and inspect `corr_estimator` in the fit
result for the procedure that actually ran.

```python
from pyscarcopula import GaussianCopula, StudentCopula

# Fast compatibility path: estimate a plug-in correlation from u.
gaussian = GaussianCopula(corr_mode="fixed")
student = StudentCopula(corr_mode="fixed")

gaussian_result = gaussian.fit(u, method="mle")
student_result = student.fit(u, method="mle")

# Joint static correlation optimization.
gaussian_joint = GaussianCopula(corr_mode="cholesky")
student_joint = StudentCopula(corr_mode="shrinkage")
```

The canonical modes are:

| Mode | Meaning |
|---|---|
| `fixed` | Keep supplied `R`, or estimate a Gaussian-score/Kendall plug-in correlation when `R` is absent |
| `shrinkage` | Jointly estimate one shrinkage weight around `corr_base`, supplied `R`, or a plug-in start |
| `cholesky` | Jointly estimate a full SPD correlation; intended for small `d` |
| `factor` | Store `O(d*k + k^2)` compact state; two-stage by default, with joint loadings available for static Student |

The default `fixed` mode preserves the previous quick behaviour. A plug-in
correlation is counted in AIC/BIC even though it is absent from the optimizer
vector. Conversely, constructor-supplied fixed `R` is not counted: a Gaussian
result then has `n_params == 0`, while Student still estimates one `df`
parameter.

Full Cholesky uses `d*(d-1)/2` correlation parameters and is guarded by
`cholesky_d_max=10` unless `allow_large_cholesky=True` is chosen explicitly.
For large dimensions, use factor mode when a low-rank correlation is a valid
model assumption:

```python
factor_student = StudentCopula(
    d=u.shape[1],
    corr_mode="factor",
    factor_rank=min(8, u.shape[1] - 1),
    factor_estimation="two-stage",  # or "joint" for static Student
)
factor_result = factor_student.fit(u, method="mle")
assert factor_result.correlation_matrix is None
```

## Dynamic Multivariate Mathematical Contract

The dynamic multivariate models are copula emission models with one
time-varying scalar parameter. For pseudo-observations
$u_t=(u_{t1},\ldots,u_{td})\in(0,1)^d$, each model supplies a density

$$c(u_t; r_t), \qquad r_t = \Psi(g_t).$$

The scalar state can be estimated in three ways:

- **MLE**: $r_t=r$ is constant and estimated by likelihood maximization.
- **GAS**: $g_t$ follows the observation-driven recursion
  $g_{t+1} = \omega + \beta g_t + \gamma s_t$,
  where $s_t$ is the score of $\log c(u_t;\Psi(g_t))$ with respect to $g_t$.
- **SCAR-TM-OU**: $g_t$ is a latent OU process. The likelihood integrates over
  the latent state using the transfer-matrix filter.

The shared formulas for scalar dynamic states, parameter links, SCAR filters,
and dynamic Rosenblatt GoF are summarized in
[Mathematical Contracts](mathematical-contracts.md).

Factor correlation, static and joint MLE, dynamic GAS/SCAR examples, and
standalone operator usage are covered on the dedicated
[Factor Models](factor-models.md) page.

For both dynamic model classes, `cop.log_likelihood(u)` evaluates the fitted
strategy: the GAS filtered likelihood or the integrated SCAR likelihood.
Passing an explicit `r` evaluates the static emission likelihood at that
parameter instead. With nonempty `given` and no explicit `r`,
`cop.sample_conditional(n, given=...)` uses the next predictive distribution,
as does `cop.predict(n, given=...)`. With no fixed coordinates it retains
the model-reproduction semantics of `cop.sample(n)`.

Dynamic fits retain an owned snapshot of their training observations;
subsequent changes to the caller's array do not change prediction history.
If fitting raises, the previous fitted result, correlation, history, and
caches are restored. A normally returned dynamic candidate still carries
its optimizer `success` flag, which callers should inspect.

Each Student dynamic fit recomputes data-derived correlation from the new
observations, including when `gamma0` or `alpha0` is supplied. Fixed `R`
and factor loadings supplied to the constructor remain fixed. An explicit
`initialize_factor(data)` also retains those loadings for later fits.
For estimated dense modes, `corr_base` and `R` remain initialization policy;
they do not freeze jointly estimated correlation parameters.

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

cop = EquicorrGaussianCopula(d=u.shape[1])

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
dense correlation matrix. Conditional sampling uses the closed-form
equicorrelation conditional mean and covariance decomposition and likewise
does not construct `R` or a dense Schur complement. Static MLE results retain
only scalar `rho`, with `correlation_matrix=None`. Output arrays and input
storage still determine the end-to-end memory footprint, so large sampling
jobs should be batched.

## Stochastic Student-t Copula

A $d$-dimensional t-copula where the degrees-of-freedom parameter $\operatorname{df}(t)$
follows a latent OU process and the static correlation matrix $R$ is fixed or
estimated jointly:

$$\operatorname{df}(t) = 2 + 10^{-6} + \mathrm{softplus}(x(t)), \qquad x(t) \sim \text{OU}(\kappa, \mu, \nu)$$

The transform maps $\mathbb{R} \to (2 + 10^{-6}, \infty)$, giving the underlying Student distribution finite variance. Copula uniforms
are bounded regardless of this condition.
The copula density is

$$c(u_t;R,\operatorname{df}_t)=
\frac{t_d(q_t;0,R,\operatorname{df}_t)}
     {\prod_{j=1}^d t_1(q_{tj};\operatorname{df}_t)}, \qquad
q_{tj}=T_{\operatorname{df}_t}^{-1}(u_{tj}).$$

The latent/dynamic part of the model controls only tail thickness. Smaller
$\operatorname{df}_t$ means heavier joint tails; larger $\operatorname{df}_t$ moves the copula toward the
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
Static Student likelihoods use native CDF inversion below `df=1000` and
a refined sixth-order normal-limit expansion at and above that threshold.
See [Student Quantiles](../reference/student-quantiles.md) for the shared
cache, inversion, and asymptotic contract.

### Stochastic Student copula with estimated static correlation

Static correlation can be handled in four modes:

```python
import numpy as np
from pyscarcopula import StochasticStudentCopula

u = np.random.default_rng(2026).uniform(0.01, 0.99, size=(200, 5))
R = np.full((5, 5), 0.25)
np.fill_diagonal(R, 1.0)

# fixed correlation
fixed = StochasticStudentCopula(d=5, R=R, corr_mode="fixed")

# one-parameter shrinkage toward identity
shrinkage = StochasticStudentCopula(d=5, corr_mode="shrinkage")

# full static correlation for small dimensions
cholesky = StochasticStudentCopula(d=5, corr_mode="cholesky")

# compact fixed factor correlation for large dimensions
factor = StochasticStudentCopula(
    d=5,
    corr_mode="factor",
    factor_rank=3,
)
factor.initialize_factor(u)
```

`shrinkage` and `cholesky` estimate static correlation jointly with constant-df
MLE or with the three OU parameters in SCAR-TM-OU. `cholesky` mode estimates
`d(d-1)/2` additional static parameters and is intended for low-dimensional
problems. Their initialization/base matrix uses `corr_base` when supplied,
otherwise `R`, and otherwise a Kendall estimate from the fit data. GAS
supports `fixed`, `shrinkage`, and `factor` correlation modes; `cholesky` is
restricted to MLE and SCAR-TM-OU.

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

import numpy as np

returns = np.random.default_rng(20260728).standard_normal((60, 6))
cop = StochasticStudentCopula(d=returns.shape[1])

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
correlation representation. The `factor` mode instead stores
`O(d*k + k^2)` state, accepts validated loadings or estimates them with a
deterministic tiled randomized SVD, and never creates dense `R` implicitly.
It supports static MLE, GAS, SCAR-TM-OU, row/grid evaluation, bounded batch
generation, and exact conditional sampling without a dense Schur complement.

Static MLE can also refine `df` and the factor loadings jointly when the
identifiable parameter count is deliberately bounded:

```python
joint = StochasticStudentCopula(
    d=u.shape[1],
    corr_mode="factor",
    factor_rank=2,  # for the five-column example: d >= 2*k + 1
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
loadings for GAS and SCAR are rejected because those methods require loading
derivatives through their sequential recursions. Dynamic factor models
therefore require `factor_estimation="two-stage"`.

```python
factor_samples = factor.sample_at_parameter(
    50_000, r=6.0, rng=np.random.default_rng(2026), n_threads=4)

factor_conditional = factor.sample_conditional(
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

The result stores model parameters, observation count, correlation state,
optimizer status, diagnostics, and the parameter count used by AIC/BIC.
For `GaussianCopula(corr_mode="factor")` and factor Student models, the result
intentionally stores `correlation_matrix=None` and keeps compact loadings and
uniqueness in `model_parameters`.

Static `GaussianCopula` and `StudentCopula` expose fitting, likelihood,
sampling, prediction, conditional sampling, and GoF. They do not expose a
time-varying parameter path:

| Model | Fit methods | Conditional sampling | `predictive_mean` |
|---|---|---|---|
| `GaussianCopula` | MLE | Exact | No |
| `StudentCopula` | MLE | Exact | No |
| `EquicorrGaussianCopula` | MLE, GAS, SCAR-TM-OU | Exact | Yes |
| `StochasticStudentCopula` | MLE, GAS, SCAR-TM-OU | Exact | Yes |

```python
# Goodness of fit
from pyscarcopula.stattests import gof_test
gof = gof_test(cop, u, to_pobs=False)

# Parametric-bootstrap calibration
gof_bootstrap = gof_test(
    cop,
    u,
    to_pobs=False,
    bootstrap=True,
    n_bootstrap=499,
    n_jobs=-1,
    rng=20260730,
)

# Sampling (fixed parameter)
samples = cop.sample(n=10000)

# Prediction (conditional on data)
pred = cop.predict(n=10000)
```

Parametric-bootstrap calibration supports static Gaussian and Student models
in dense and compact factor modes,
plus `EquicorrGaussianCopula` and `StochasticStudentCopula` fitted by MLE, GAS,
or SCAR-TM-OU. Stochastic Student correlation policy is retained for fixed,
shrinkage, Cholesky, and supported two-stage factor modes. Factor Rosenblatt
statistics use rank-dimensional conditioning with `O(T*k + k^2)` workspace
and do not materialize a dense correlation matrix for MLE, GAS, or
SCAR-TM-OU. Replicas use independent process-owned models and random streams.
With a fixed seed, bootstrap
statistics are invariant to `n_jobs`; parallel workers default to one native
thread each. For dynamic stochastic Student fits with an estimated
correlation, call `gof_test` on the fitted model: a standalone GAS/SCAR result
does not contain enough correlation state to reconstruct an unfitted
prototype.

Dynamic scalar-parameter models additionally expose `predictive_mean`. For
example, after fitting an `EquicorrGaussianCopula`:

```python
from pyscarcopula import EquicorrGaussianCopula

dynamic = EquicorrGaussianCopula(d=u.shape[1])
dynamic.fit(u, method="gas")
rho_t = dynamic.predictive_mean(u)
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
    factor_rank=min(8, u.shape[1] - 1),
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

Static factor Student uses the same compact persistence, worker
reconstruction, sampling, conditional sampling, and Rosenblatt contracts.
`factor_estimation="joint"` additionally optimizes `df` and identified
loadings together; the two-stage mode counts its data-derived loadings as
plug-in parameters.
