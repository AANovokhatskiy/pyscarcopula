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

## Factor correlation operator

For an end-to-end guide covering standalone operators, Gaussian and Student
MLE, dynamic GAS/SCAR, sampling, and persistence, see
[Factor Models](../guide/factor-models.md).

`FactorCorrelation` is a reusable correlation representation independent of
any copula family:

$$
R = D + BB^\top,\qquad
D_{ii}=1-\lVert B_{i,:}\rVert^2.
$$

The loadings have shape `(d, k)`, with `1 <= k < d`. Positive uniqueness
values make `R` positive definite and the definition of `D` gives it an exact
unit diagonal. Preparing the object builds a compact native Woodbury
workspace; neither construction materializes a `d * d` matrix.

```python
import numpy as np
from pyscarcopula import FactorCorrelation

rng = np.random.default_rng(2026)
B = rng.normal(scale=0.02, size=(100_000, 8))

factor = FactorCorrelation(B, uniqueness_min=1e-8)
operator = factor.prepare()

x = rng.normal(size=(32, factor.dimension))
rx = operator.matvec(x, n_threads=4)
solution = operator.solve(x, n_threads=4)
quadratic = operator.quadratic_forms(x, n_threads=4)
log_det = operator.logdet
```

Preparation costs `O(d*k^2 + k^3)` and stores `O(d*k + k^2)` values.
Matrix products, solves, quadratic forms, and normal generation cost
`O(n*d*k + n*k^2)` for `n` rows. All methods with `n_threads` have the
literal default `1`; environment variables cannot enable parallel execution.
Results are deterministic across supported thread counts, and one prepared
operator can serve concurrent read-only calls.

Normal output can be generated in bounded batches:

```python
for block in operator.sample_normal_batches(
    1_000_000,
    batch_rows=128,
    rng=np.random.default_rng(7),
    n_threads=4,
    memory_budget_bytes=128 * (factor.dimension + factor.rank) * 8,
):
    consume(block)
```

Use `save_npz` for a compact portable representation or `save_mmap` for a
directory whose loadings can be reopened read-only:

```python
factor.save_npz("factor-correlation.npz")
factor.save_mmap("factor-correlation-mmap")

portable = FactorCorrelation.load_npz("factor-correlation.npz")
mapped = FactorCorrelation.load_mmap("factor-correlation-mmap")
```

`to_dense()` is an explicit diagnostic operation protected by
`max_dimension` and `memory_budget_bytes`. It should not be used for large
dimensions.

The independent Student likelihood adapter composes this operator without
adding Student state to `FactorCorrelation`:

```python
from pyscarcopula import FactorStudentEvaluator

student = FactorStudentEvaluator(operator, u)
evaluation = student.evaluate(df=7.0, n_threads=4)

row_log_pdf = evaluation.log_pdf
row_df_gradient = evaluation.dlog_ddf
log_likelihood = evaluation.log_likelihood
df_gradient = evaluation.dlog_likelihood_ddf

# Direct L-BFGS-B-compatible negative objective:
value, gradient = student.objective_and_gradient(7.0, n_threads=4)
```

`df` may be one common scalar or an array with one value per observation.
The aggregate `dlog_likelihood_ddf` is available only for a common scalar;
row-specific evaluations retain their individual derivatives. The evaluator
owns a read-only observation copy, shares the immutable prepared correlation,
and is safe for concurrent read-only calls.

The native adapter evaluates exact Student quantiles and analytical
quantile derivatives, then applies the Woodbury solve. Model storage remains
`O(d*k + k^2)`, with `O(d)` workspace per active row worker and no dense
correlation, Cholesky, or precision matrix.

For emission over a grid of degrees of freedom, use the tiled API:

```python
df_grid = np.linspace(2.01, 40.0, 64)
grid_result = student.evaluate_grid(
    df_grid,
    dimension_tile=16_384,
    n_threads=4,
    memory_budget_bytes=64 * 1024**2,
)

log_pdf = grid_result.log_pdf       # shape (T, K)
dlog_ddf = grid_result.dlog_ddf
density, d_density_ddf = grid_result.pdf_and_gradient()
```

The kernel never stores Student quantiles with shape `(T, K, d)`. Each fixed
dimension tile contributes diagonal quadratic terms, factor projections,
their `df` derivatives, and marginal log-density summaries. Tiles are merged
in a fixed order before the small `k * k` solve.

When there are enough `(row, df)` cells, those cells are the parallel axis.
For a small grid with very large `d`, dimension tiles become the parallel
axis. Both paths preserve identical results for a fixed `dimension_tile`
across supported thread counts.

Bound the `(T, K)` output and native workspace with row batches:

```python
for block in student.evaluate_grid_batches(
    df_grid,
    batch_rows=128,
    dimension_tile=16_384,
    n_threads=4,
    memory_budget_bytes=64 * 1024**2,
):
    consume(block.log_pdf, block.dlog_ddf)
```

The budget accounts conservatively for native result vectors, their NumPy
copies, active worker accumulators, and deterministic tile partials.

The factor operator is available through `StochasticStudentCopula`:

```python
from pyscarcopula import NumericalConfig, StochasticStudentCopula

copula = StochasticStudentCopula(
    d=u.shape[1],
    corr_mode="factor",
    factor_rank=min(8, u.shape[1] - 1),
    factor_loadings=B,       # optional
    factor_tile_size=16_384,
)

if copula.factor_loadings_ is None:
    copula.initialize_factor(u)

rows = copula.log_pdf_rows(u, r=7.0, n_threads=4)
density, gradient = copula.pdf_and_grad_on_grid_batch(
    u, x_grid, n_threads=4)
```

When loadings are omitted, `initialize_factor` uses normal-score tiles and a
fixed-seed randomized SVD without constructing a `d*d` Kendall or covariance
matrix. `R` never materializes implicitly in factor mode. Use
`correlation_operator_` for matrix-free operations or the guarded
`to_correlation_matrix()` diagnostic for a deliberately small problem.
Model persistence stores the compact loadings and rebuilds the native
operator on load.

The same compact operator is used during fitting:

```python
mle = copula.fit(u, method="mle", config=NumericalConfig(n_threads=4))
gas = copula.fit(u, method="gas", config=NumericalConfig(n_threads=4))
scar = copula.fit(
    u,
    method="scar-tm-ou",
    config=NumericalConfig(n_threads=4),
)
```

MLE results keep `correlation_matrix=None` and persist loadings/uniqueness
instead. GAS uses the native analytical Student score inside its sequential
time recursion. SCAR-TM-OU matrix, local, and spectral paths share the
immutable factor operator and call the tiled exact-emission kernel directly;
SCAR-P/M native trajectory density uses the same matrix-free Student kernel.
Estimated factor parameters are included in effective parameter counts even
though two-stage loadings remain fixed during the main optimizer.

Factor Student models provide unconditional, fitted, predictive, batch, and
conditional sampling:

```python
draws = copula.sample_at_parameter(
    100_000,
    r=6.0,
    rng=np.random.default_rng(17),
    n_threads=4,
)

conditional = copula.sample_conditional(
    100_000,
    r=6.0,
    given={0: 0.25, 7: 0.8},
    rng=np.random.default_rng(18),
    n_threads=4,
)

for block in copula.sample_at_parameter_batches(
    10_000_000,
    r=6.0,
    batch_rows=128,
    rng=np.random.default_rng(19),
    n_threads=4,
):
    consume(block)
```

Unconditional draws use `B f + sqrt(D) epsilon` followed by the common
Student radial scale. Conditional draws solve and factorize only a `k*k`
system for the fixed coordinates; no dense correlation or Schur complement
is formed. `sample_batches` and `predict_batches` retain MLE, sequential GAS,
and SCAR parameter-path semantics while bounding materialized rows.
`memory_budget_bytes` is a conservative pre-allocation guard. Identical seeds
produce identical tested results across thread counts.

Static Student MLE can jointly estimate guarded factor loadings:

```python
joint = StochasticStudentCopula(
    d=u.shape[1],
    corr_mode="factor",
    factor_rank=4,
    factor_estimation="joint",
    factor_joint_max_params=100_000,
    factor_joint_penalty=1e-6,
    factor_joint_condition_max=1e12,
)
result = joint.fit(
    u,
    method="mle",
    config=NumericalConfig(n_threads=4),
)
```

The optimizer combines natural `df` with
`d*k-k*(k-1)/2` identifiable loading coordinates. Pivot-selected anchor rows
form a lower-triangular block with positive diagonal, eliminating factor
rotation ambiguity. Every likelihood evaluation uses analytical,
matrix-free `df` and loading gradients; finite differences over `d*k` are not
used. A smooth row transform preserves the uniqueness floor, while an L2
penalty and maximum Woodbury-core condition estimate protect weakly identified
trials. An optimizer success flag is rejected when the terminal gradient does
not pass the recorded convergence gate.

Omitted loadings use tiled randomized SVD only as the joint starting point.
The result remains compact and stores its anchor rows and convergence
diagnostics. Joint loading estimation is currently limited to static MLE.
`method="gas"` and SCAR methods reject this policy explicitly because their
sequential filters do not yet expose loading derivatives.

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
```

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
MLE, GAS, SCAR-TM-OU, native SCAR-MC trajectory likelihood, bounded batch
sampling, and exact conditional sampling. It forbids `R` and `corr_base`, and
its `R` property never silently allocates a dense matrix. Factor sampling and
conditioning retain the compact representation.

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
