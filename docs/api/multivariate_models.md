# Multivariate Models API

## Static Gaussian and Student Copulas

Static multivariate MLE returns `MultivariateMLEResult` for Gaussian, Student,
equicorrelation Gaussian, and stochastic Student models. The returned object
is also stored as `copula.fit_result`.

```python
from pyscarcopula import GaussianCopula

cop = GaussianCopula()
result = cop.fit(u)

result.correlation_matrix
result.model_parameters
result.log_likelihood
result.n_params
result.aic
result.bic
```

::: pyscarcopula.MultivariateMLEResult

Static `GaussianCopula` and `StudentCopula` accept only `method='mle'`. Both
provide exact conditional generation in pseudo-observation space through
`sample_conditional(n, given, rng=None, *, n_threads=1)` and
`predict(n, given=..., rng=...)`:

```python
import numpy as np

conditional = cop.sample_conditional(
    n=10_000,
    given={0: 0.25, 2: 0.8},
    rng=np.random.default_rng(2026),
    n_threads=4,
)
```

The supplied columns remain fixed. Supplying every variable returns constant
rows equal to `given`.

Here `MLE` is the static model label. Correlation estimation is controlled by
`corr_mode`, and only modes marked joint below put correlation parameters in
the likelihood optimizer vector:

| `corr_mode` | Correlation procedure | Gaussian count | Student count |
|---|---|---:|---:|
| `fixed`, supplied `R` | held fixed | `0` | `1` for `df` |
| `fixed`, no `R` | Gaussian-score/Kendall plug-in | `d*(d-1)/2` | `1 + d*(d-1)/2` |
| `shrinkage` | joint one-parameter shrinkage | `1` | `2` |
| `cholesky` | joint full correlation | `d*(d-1)/2` | `1 + d*(d-1)/2` |
| `factor`, two-stage | compact plug-in loadings | identifiable loading count | `1 +` that count |
| `factor`, joint | Student only | unavailable | `1 +` identifiable loading count |

The default is `fixed`, preserving the previous fast behaviour. Use
`result.diagnostics["corr_estimator"]` to distinguish `supplied`,
`gaussian_score`, `kendall_plugin`, `joint_mle`, `factor_two_stage`, and
`factor_joint`. Plug-in counts are included in AIC/BIC. A supplied fixed
Gaussian has zero fitted parameters, which is valid for
`MultivariateMLEResult`.

```python
from pyscarcopula import GaussianCopula, StudentCopula

gaussian_fast = GaussianCopula(corr_mode="fixed")
gaussian_joint = GaussianCopula(corr_mode="cholesky")
student_fast = StudentCopula(corr_mode="fixed")
student_joint = StudentCopula(corr_mode="shrinkage")
```

Full Cholesky is guarded by `cholesky_d_max=10` by default and is intended for
small dimensions. Factor mode is the scalable choice when the low-rank
assumption is appropriate.

### GaussianCopula

::: pyscarcopula.copula.multivariate.gaussian.GaussianCopula
    options:
      show_bases: false
      members: false

### StudentCopula

::: pyscarcopula.copula.multivariate.student.StudentCopula
    options:
      show_bases: false
      members: false

All multivariate APIs that expose `n_threads` use a literal default of `1`.
No environment variable changes that default. Fit-level native parallelism is
enabled with `NumericalConfig(n_threads=N)`.

### Gaussian factor correlation

For large dimensions, static Gaussian models can compose the same independent
factor operator described below:

```python
from pyscarcopula import GaussianCopula, NumericalConfig

gaussian = GaussianCopula(
    d=u.shape[1],
    corr_mode="factor",
    factor_rank=min(8, u.shape[1] - 1),
    factor_tile_size=16_384,
)
result = gaussian.fit(
    u,
    method="mle",
    config=NumericalConfig(n_threads=4),
)
```

If `factor_loadings` is omitted, fitting uses a fixed-seed, tiled normal-score
randomized SVD without constructing a dense covariance matrix. Loadings may
also be supplied to the constructor. The fitted result keeps
`correlation_matrix=None` and stores compact loadings and uniqueness in
`model_parameters`; its parameter count is
`d*k - k*(k-1)/2`.

`log_likelihood`, `log_pdf_rows`, `sample`, `sample_batches`,
`sample_conditional`, `predict`, and `predict_batches` accept a literal
`n_threads=1` default. Conditional generation factorizes only
`I + B_G.T @ D_G^-1 @ B_G` for the fixed coordinates. Goodness-of-fit uses a
sequential rank-dimensional factor update with `O(T*k + k^2)` workspace.
Persistence and rolling-window worker reconstruction retain the compact
constructor policy.

### Static Student factor correlation

`StudentCopula` exposes the same compact representation and adds optional
joint static loading estimation:

```python
from pyscarcopula import StudentCopula

student_factor = StudentCopula(
    d=u.shape[1],
    corr_mode="factor",
    factor_rank=min(8, u.shape[1] - 1),
    factor_estimation="two-stage",  # use "joint" for joint df/loadings MLE
)
student_result = student_factor.fit(u, method="mle")
```

Two-stage loadings are counted as plug-in parameters. Joint mode uses an
identified loading parameterization and analytical matrix-free gradients.
Sampling, conditional sampling, GoF, bootstrap, persistence, and worker
reconstruction retain the factor operator without implicit dense
materialization.

## Factor correlation operator

`FactorCorrelation` stores a correlation matrix as
$R=D+BB^\top$ and prepares a matrix-free Woodbury operator.
`FactorStudentEvaluator` combines that operator with Student copula
likelihoods. `StochasticStudentCopula(corr_mode="factor")` exposes the same
representation through the model API.

```python
import numpy as np
from pyscarcopula import FactorCorrelation, FactorStudentEvaluator

rng = np.random.default_rng(2026)
B = rng.normal(scale=0.05, size=(20, 3))
u = rng.uniform(0.01, 0.99, size=(50, 20))
factor = FactorCorrelation(B, uniqueness_min=1e-8)
operator = factor.prepare()
evaluation = FactorStudentEvaluator(operator, u).evaluate(
    df=7.0,
    n_threads=4,
)
```

For construction, fitting, batching, conditional sampling, persistence,
complexity, and memory limits, see
[Factor Models](../guide/factor-models.md). The generated reference below is
the canonical source for method signatures and defaults.

### API

::: pyscarcopula.copula.multivariate.factor_correlation.FactorCorrelation

::: pyscarcopula.copula.multivariate.factor_correlation.PreparedFactorCorrelation

::: pyscarcopula.copula.multivariate.factor_student.FactorStudentEvaluator

::: pyscarcopula.copula.multivariate.factor_student.FactorStudentEvaluation

::: pyscarcopula.copula.multivariate.factor_student.FactorStudentGridEvaluation

::: pyscarcopula.copula.multivariate.factor_student.FactorStudentJointEvaluation

## Equicorrelation Gaussian Copula

The equicorrelation model, its parameter range, and selection guidance are
described in the [Multivariate Models guide](../guide/multivariate_models.md).

### Usage

```python
from pyscarcopula.copula.multivariate import EquicorrGaussianCopula

cop = EquicorrGaussianCopula(d=6)

# MLE (constant rho)
cop.fit(u, method='mle')

# SCAR (time-varying rho)
cop.fit(u, method='scar-tm-ou')

# GAS (observation-driven rho)
cop.fit(u, method='gas')
```

### High-dimensional preparation

For data that is already expressed as pseudo-observations, prepare the two
equicorrelation sufficient statistics without materializing a dense
correlation matrix:

```python
from pyscarcopula import EquicorrGaussianCopula

cop = EquicorrGaussianCopula(d=100_000)
prepared = cop.prepare_sufficient_statistics(
    u_batches,                 # ndarray, memmap, or iterable of 2D blocks
    batch_rows=256,
    dimension_tile=16_384,
    n_threads=4,               # omit for the unconditional one-thread default
)

prepared.save_npz("equicorr-statistics.npz")
prepared.save_mmap("equicorr-statistics-mmap")

mle_result = cop.fit(prepared, method="MLE")
gas_result = cop.fit(prepared, method="GAS")
scar_result = cop.fit(prepared, method="scar-tm-ou")
```

`prepared` is an immutable `EquicorrPreparedData` object containing only
`sum_z` and `sum_z2`, two `float64` vectors of length `T`. It does not retain
the input matrix. The native reduction clips values with the library's
pseudo-observation policy, evaluates each normal quantile once, and merges
fixed dimension tiles in deterministic order. Results are therefore identical
across supported thread counts for a fixed `dimension_tile`.

The mmap directory format stores each vector as a read-only `.npy` mapping;
the `.npz` format is the compact portable option. Diagnostics report clipping,
block/tile counts, the selected parallel axes, and peak temporary scalar
storage.

This preparation API expects pseudo-observations and does not perform global
ranking. MLE, GAS, and SCAR-TM-OU consume the prepared vectors directly;
static row likelihood and emission-grid methods do the same. Existing ndarray
inputs remain supported. Every Equicorr MLE result represents correlation by
its scalar `equicorrelation_rho` and leaves `correlation_matrix=None`, avoiding
an otherwise prohibitive `d * d` result allocation for both dense and prepared
inputs.

Grid output can also be bounded explicitly. Both the density and gradient
arrays count toward `memory_budget_bytes`:

```python
for pdf_block, grad_block in cop.pdf_and_grad_on_grid_batches(
    prepared,
    x_grid,
    batch_rows=64,
    memory_budget_bytes=2 * 64 * len(x_grid) * 8,
    n_threads=4,
):
    consume(pdf_block, grad_block)
```

### Goodness of fit

```python
from pyscarcopula.stattests import gof_test
gof = gof_test(cop, u, to_pobs=False)

gof_bootstrap = gof_test(
    cop,
    u,
    to_pobs=False,
    bootstrap=True,
    n_bootstrap=499,
    n_jobs=-1,
    rng=20260730,
)
```

Parametric-bootstrap calibration is supported for static `GaussianCopula`
and `StudentCopula` (dense and factor correlation representations),
`EquicorrGaussianCopula`, and `StochasticStudentCopula`. Dynamic models retain
their fitted MLE/GAS/SCAR-TM-OU strategy settings and correlation policy.
Stochastic Student supports fixed, shrinkage, Cholesky, and two-stage factor
correlation. Its factor Rosenblatt transform uses compact rank-dimensional
conditioning for MLE, GAS, and SCAR-TM-OU without materializing a dense
correlation matrix. Replicas use independent process-owned models and
deterministic per-replication random streams. GAS/SCAR stochastic Student
results with estimated correlation must be paired with their fitted model,
because the result object alone does not store the full fitted correlation.

### Sampling

```python
samples = cop.predict(n=10000)
samples = cop.sample(n=10000)
parameter_samples = cop.sample_at_parameter(n=10000, r=0.5)

# Bounded-memory unconditional generation, including negative rho:
for block in cop.sample_at_parameter_batches(
    n=1_000_000, r=-1e-6, batch_rows=128
):
    consume(block)

# Bounded output from a fitted MLE, GAS, or SCAR-TM-OU model:
for block in cop.sample_batches(
    n=1_000_000,
    batch_rows=128,
    memory_budget_bytes=128 * cop.d * 8,
):
    consume(block)

for block in cop.predict_batches(
    n=1_000_000,
    batch_rows=128,
    memory_budget_bytes=128 * cop.d * 8,
):
    consume(block)
```

The structural sampler costs `O(n*d)` and does not construct a dense
correlation matrix. It supports the full admissible open interval
`-1/(d-1) < rho < 1`. Fitted batching preserves model semantics: GAS updates
its state after every generated row, while SCAR uses one OU path for
unconditional sampling and one frozen posterior state for prediction.
Monolithic `sample`, `predict`, `sample_conditional`, and
`sample_at_parameter` also accept `memory_budget_bytes` and fail before
allocating an oversized output.

### API

::: pyscarcopula.copula.multivariate.equicorr.EquicorrGaussianCopula
    options:
      members:
        - fit
        - prepare_sufficient_statistics
        - sample
        - sample_batches
        - sample_conditional
        - predict
        - predict_batches
        - predictive_mean
        - xT_distribution
        - log_likelihood
        - log_pdf_rows
        - dlog_pdf_dr_rows
        - log_pdf_and_dlog_dr_rows
        - pdf_on_grid
        - pdf_and_grad_on_grid
        - pdf_and_grad_on_grid_batch
        - pdf_and_grad_on_grid_batches
        - sample_at_parameter
        - sample_at_parameter_batches
        - transform
        - inv_transform
        - dtransform

## StochasticStudentCopula

A multivariate Student copula with dynamic degrees of freedom and either fixed
or jointly estimated static correlation. The dynamic scalar parameter is

$$\nu_t = 2 + 10^{-6} + \mathrm{softplus}(g_t),$$

and the row density is the standard Student copula density

$$c(u_t;R,\nu_t)=
\frac{t_d(T_{\nu_t}^{-1}(u_t);0,R,\nu_t)}
     {\prod_j t_1(T_{\nu_t}^{-1}(u_{tj});\nu_t)}.$$

`method='mle'` estimates a constant $\nu$, `method='gas'` estimates a
score-driven recursion for $g_t$, and `method='scar-tm-ou'` treats $g_t$ as a
latent OU process integrated by transfer matrix.

SCAR-TM-OU accepts and returns the physical OU parameters
`(kappa, mu, nu)`. For this model only, optimization is performed internally
in `(log(kappa), mu, log(sigma_x))`, where
`sigma_x = nu / sqrt(2 * kappa)`. The representation used by the optimizer is
reported in `result.diagnostics['optimizer_parameterization']`; it does not
change `alpha0` or the fitted parameter values exposed by the API.

### Stochastic Student copula with estimated static correlation

Static correlation modes are selected with `corr_mode`:

```python
StochasticStudentCopula(d=5, R=R, corr_mode="fixed")
StochasticStudentCopula(d=5, corr_mode="shrinkage")
StochasticStudentCopula(d=5, corr_mode="cholesky")
StochasticStudentCopula(
    d=100_000, corr_mode="factor", factor_rank=8)
```

`shrinkage` estimates one additional static parameter. `cholesky` estimates
`d(d-1)/2` static parameters and is intended for low-dimensional problems.
For estimated modes, the initialization/base matrix is selected in this order:
an explicit `corr_base`, then `R`, then a Kendall estimate from the fit data.
Both estimated-correlation modes are available for MLE and SCAR-TM-OU.
GAS supports fixed correlation and the one-parameter `shrinkage` mode; GAS
with `corr_mode="cholesky"` is rejected before fitting. Setting
`analytical_grad=False` retains a fully numerical optimizer gradient. See the
multivariate guide for fitting details and diagnostic fields.

`factor` stores `O(d*k + k^2)` state and supports explicit
`initialize_factor`, static row likelihood, tiled latent-grid evaluation,
MLE, GAS, SCAR-TM-OU, bounded batch sampling, and exact conditional sampling.
It forbids `R` and `corr_base`, and its `R` property never silently allocates
a dense matrix. Factor sampling and conditioning retain the compact
representation.

Dynamic emission densities normally use an interpolated, precomputed Student
quantile (PPF) table of shape `(n_df_nodes, T, d)`, covering the model boundary
through `df = 1000`. Its size is capped at `DEFAULT_MAX_TABLE_BYTES`
(256 MiB). If the values table is skipped, native evaluation keeps the node
range metadata, uses exact quantiles through the final node, and switches to
a controlled third-order normal-quantile asymptotic above it. Quantile
derivatives with respect to `df` are analytical both in the exact and
large-`df` paths, so an out-of-cache gradient does not repeat the expensive
quantile inversion through finite differences. See the performance guide for
accuracy and memory details.

Pass `NumericalConfig(n_threads=N)` to `fit` to parallelize eligible native
emission, row-likelihood, and Monte Carlo work. Methods with a direct
`n_threads` parameter, including conditional sampling and row/grid evaluation,
can opt in per call. Omitting it always selects one native thread.

::: pyscarcopula.copula.multivariate.stochastic_student.StochasticStudentCopula
    options:
      members:
        - fit
        - sample_at_parameter
        - sample_at_parameter_batches
        - sample
        - sample_batches
        - sample_conditional
        - predict
        - predict_batches
        - predictive_mean
        - xT_distribution
        - log_likelihood
        - log_pdf_rows
        - log_pdf_and_dlog_dr_rows
        - pdf_on_grid
        - pdf_and_grad_on_grid
        - transform
        - inv_transform
        - dtransform
