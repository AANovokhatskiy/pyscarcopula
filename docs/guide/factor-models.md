# Factor Models

## Overview

Factor correlation is the scalable correlation representation used by the
multivariate Gaussian and stochastic Student models:

$$
R = D + BB^\top,
\qquad
D_{ii}=1-\lVert B_{i,:}\rVert^2.
$$

Here `B` has shape `(d, k)`, normally with `k << d`. The definition of `D`
gives `diag(R) = 1`; a positive uniqueness floor makes the correlation
positive definite.

The implementation has three separate layers:

| Layer | Public type | Responsibility |
|---|---|---|
| Correlation value | `FactorCorrelation` | Validate and persist `B` and `D` |
| Prepared operator | `PreparedFactorCorrelation` | Woodbury linear algebra and normal generation |
| Model adapter | `GaussianCopula`, `StochasticStudentCopula`, `FactorStudentEvaluator` | Marginal transforms, likelihood, fitting, dynamics, and sampling |

`FactorCorrelation` is independent of Student, Gaussian, GAS, SCAR, and
optimizer state. The same read-only prepared operator can therefore be
composed with different model adapters.

### Complexity

For `n` rows:

| Operation | Time | Stored model state |
|---|---:|---:|
| Prepare operator | `O(d*k^2 + k^3)` | `O(d*k + k^2)` |
| Matrix product or solve | `O(n*d*k + n*k^2)` | `O(d*k + k^2)` |
| Gaussian or Student row likelihood | `O(n*d*k)` | `O(d*k + k^2)` |
| Structural sampling | `O(n*d*k)` | `O(d*k + k^2)` plus output |

No factor path implicitly constructs a `d*d` correlation, Cholesky, precision
matrix, or Schur complement.

## Standalone factor correlation

### Construct and prepare

```python
import numpy as np

from pyscarcopula import FactorCorrelation

rng = np.random.default_rng(2026)
d = 100_000
k = 8

B = rng.normal(scale=0.01, size=(d, k))
factor = FactorCorrelation(
    B,
    uniqueness_min=1e-8,
)
operator = factor.prepare()

print(factor.dimension, factor.rank)
print(factor.uniqueness.min())
print(operator.logdet)
print(operator.diagnostics)
```

The value object owns read-only loadings and uniqueness arrays. The prepared
operator owns immutable native Woodbury state and is safe for concurrent
read-only calls.

### Matrix-free linear algebra

```python
x = rng.standard_normal((32, d))

rx = operator.matvec(x, n_threads=4)
precision_x = operator.solve(x, n_threads=4)
quadratic = operator.quadratic_forms(x, n_threads=4)

single_quadratic = operator.quadratic_form(x[0], n_threads=1)
```

Every public `n_threads` argument has a literal default of `1`. Omitting it
always selects the sequential path, independently of environment variables.
Parallel execution is per call:

```python
sequential = operator.solve(x)                 # exactly one thread
parallel = operator.solve(x, n_threads=4)      # explicit opt-in
```

### Normal generation and bounded output

```python
normal_rows = operator.sample_normal(
    128,
    rng=np.random.default_rng(7),
    n_threads=4,
)

for block in operator.sample_normal_batches(
    1_000_000,
    batch_rows=128,
    rng=np.random.default_rng(8),
    n_threads=4,
    memory_budget_bytes=128 * (d + k + 4) * 8,
):
    consume(block)
```

Batching bounds temporary rows. It cannot remove the memory required by a
single returned block, so choose `batch_rows` from the application memory
budget.

### Persistence and explicit dense diagnostics

```python
factor.save_npz("correlation.npz")
factor.save_mmap("correlation-mmap")

portable = FactorCorrelation.load_npz("correlation.npz")
mapped = FactorCorrelation.load_mmap("correlation-mmap")
```

`load_mmap` is useful when large read-only loadings should be shared by
application processes according to the operating system's file-mapping
semantics.

Dense materialization is diagnostic and guarded:

```python
small_factor = FactorCorrelation(B[:256])
small_R = small_factor.to_dense(
    max_dimension=2_048,
    memory_budget_bytes=2_048**2 * 8,
)
```

For a large `d`, use the prepared operator instead of raising these guards.

## Static Gaussian factor model

`GaussianCopula` composes the independent operator for static Gaussian MLE,
row likelihood, sampling, conditioning, persistence, and goodness-of-fit.

### Estimate loadings from data

```python
from pyscarcopula import GaussianCopula, NumericalConfig

gaussian = GaussianCopula(
    d=u.shape[1],
    corr_mode="factor",
    factor_rank=8,
    factor_tile_size=16_384,
    factor_seed=2026,
)

result = gaussian.fit(
    u,
    method="mle",
    config=NumericalConfig(n_threads=4),
)

assert result.correlation_matrix is None
print(result.model_parameters["factor_loadings"].shape)
print(gaussian.factor_diagnostics())
```

When loadings are omitted, the model uses a tiled normal-score randomized SVD.
It does not build a dense sample covariance. The seed makes initialization
repeatable.

### Use supplied loadings

```python
gaussian = GaussianCopula(
    d=B.shape[0],
    corr_mode="factor",
    factor_rank=B.shape[1],
    factor_loadings=B,
)

rows = gaussian.log_pdf_rows(u, n_threads=4)
total = gaussian.log_likelihood(u, n_threads=4)
draws = gaussian.sample(
    10_000,
    rng=np.random.default_rng(9),
    n_threads=4,
)
```

Conditional generation fixes values in pseudo-observation space:

```python
conditional = gaussian.sample_conditional(
    10_000,
    given={0: 0.25, 7: 0.80},
    rng=np.random.default_rng(10),
    n_threads=4,
)
```

Only a `k*k` system for the fixed coordinates is factorized.

## Stochastic Student factor model

`StochasticStudentCopula` combines factor correlation with Student tail
dependence. The factor correlation is static; the Student degrees of freedom
can be constant or dynamic:

| Fit method | `df` behavior | Loading policy |
|---|---|---|
| MLE + `two-stage` | One fitted constant `df` | Estimated first, then fixed |
| MLE + `joint` | One fitted constant `df` | Optimized jointly with `df` |
| GAS | Score-driven `df_t` | Supplied or two-stage, then fixed |
| SCAR-TM-OU | Latent OU-driven `df_t` | Supplied or two-stage, then fixed |

### Static MLE with two-stage loadings

This is the default factor policy. It keeps the main optimizer independent of
the `d*k` loading coordinates:

```python
from pyscarcopula import NumericalConfig, StochasticStudentCopula

student = StochasticStudentCopula(
    d=u.shape[1],
    corr_mode="factor",
    factor_rank=8,
    factor_estimation="two-stage",
    factor_tile_size=16_384,
    factor_seed=2026,
)

mle = student.fit(
    u,
    method="mle",
    config=NumericalConfig(n_threads=4),
)

assert mle.correlation_matrix is None
print(mle.copula_param)  # constant df
print(student.factor_diagnostics())
```

The sequence is:

1. Estimate `B` with tiled randomized SVD.
2. Prepare the immutable factor operator.
3. Hold `B` fixed and optimize constant `df`.
4. Count the generic factor-correlation dimension in AIC/BIC, capped at
   `d*(d-1)/2` when the requested rank saturates the correlation space.

Supplied loadings skip step 1:

```python
student = StochasticStudentCopula(
    d=B.shape[0],
    corr_mode="factor",
    factor_rank=B.shape[1],
    factor_loadings=B,
    factor_estimation="two-stage",
)
```

### Static MLE with joint loadings

For deliberately bounded problems, static MLE can optimize constant `df` and
the factor loadings together:

```python
joint_student = StochasticStudentCopula(
    d=u.shape[1],
    corr_mode="factor",
    factor_rank=4,
    factor_estimation="joint",
    factor_joint_max_params=100_000,
    factor_joint_penalty=1e-6,
    factor_joint_condition_max=1e12,
)

joint_result = joint_student.fit(
    u,
    method="mle",
    config=NumericalConfig(n_threads=4),
)

print(joint_result.success)
print(joint_result.diagnostics["joint_gradient_inf_norm"])
print(joint_result.diagnostics["joint_gradient_gate"])
```

Joint fitting requires `d >= 2*k + 1`, a sufficient regime for generic
identifiability. Higher ranks remain available for two-stage fitting. This
guard does not certify singular or rank-deficient loading configurations.
The row-deletion criterion behind this restriction is described in
[the factor-identification literature](https://www.mdpi.com/2225-1146/11/4/26).

The optimizer uses `d*k-k*(k-1)/2` rotation-anchored coordinates:
pivot-selected anchor rows form a lower-triangular block with positive
diagonal. `factor_joint_max_params` additionally bounds the optimization
size (the `StochasticStudentCopula` guard conservatively uses `d*k`). Native
analytical gradients are used for both `df` and every loading. A reported optimizer success is
accepted only when uniqueness, Woodbury condition, finite objective, and
terminal-gradient gates all pass.

For very large `d`, prefer `two-stage`. Joint optimization retains compact
correlation storage but is still a high-dimensional nonlinear optimization
problem.

## Dynamic Student models

### GAS

GAS makes `df_t` observation-driven while keeping factor loadings fixed:

```python
from pyscarcopula.api import predictive_mean

gas_student = StochasticStudentCopula(
    d=u.shape[1],
    corr_mode="factor",
    factor_rank=8,
    factor_estimation="two-stage",
    factor_tile_size=16_384,
)

gas_result = gas_student.fit(
    u,
    method="gas",
    config=NumericalConfig(n_threads=4),
)

df_path = predictive_mean(gas_student, u, gas_result)
gas_draws = gas_student.predict(
    10_000,
    u=u,
    rng=np.random.default_rng(11),
    n_threads=4,
)
```

The GAS time recursion remains sequential because each state depends on the
previous state. Native threads accelerate independent work inside Student
emissions and sampling.

### SCAR

SCAR-TM-OU treats the transformed `df_t` as a latent OU process:

```python
scar_student = StochasticStudentCopula(
    d=u.shape[1],
    corr_mode="factor",
    factor_rank=8,
    factor_estimation="two-stage",
    factor_tile_size=16_384,
)

scar_result = scar_student.fit(
    u,
    method="scar-tm-ou",
    config=NumericalConfig(n_threads=4),
)

scar_draws = scar_student.predict(
    10_000,
    u=u,
    rng=np.random.default_rng(12),
    n_threads=4,
)
```

The matrix, local, and spectral SCAR-TM backends consume the same compact
factor operator. Forward/backward filtering remains sequential in time;
emission rows, grid cells, and dimension tiles are parallelized where their
workload passes the native thresholds.

### Estimation-mode compatibility

`factor_estimation="joint"` is available only with static MLE. Use
`factor_estimation="two-stage"` for GAS and SCAR-TM-OU:

```python
dynamic_student = StochasticStudentCopula(
    d=u.shape[1],
    corr_mode="factor",
    factor_rank=4,
    factor_estimation="two-stage",
)

gas_result = dynamic_student.fit(u, method="gas")
```

In dynamic fits the estimated loadings remain fixed while the degrees of
freedom follow the selected GAS or SCAR process.

## Student likelihood without a model

`FactorStudentEvaluator` is useful when an application already owns the
factor correlation and needs a static Student likelihood adapter:

```python
from pyscarcopula import FactorStudentEvaluator

evaluator = FactorStudentEvaluator(operator, u)

evaluation = evaluator.evaluate(df=7.0, n_threads=4)
print(evaluation.log_likelihood)
print(evaluation.dlog_likelihood_ddf)

joint = evaluator.joint_likelihood_and_gradient(
    df=7.0,
    n_threads=4,
)
print(joint.dlog_likelihood_dloadings.shape)
```

Grid evaluation is tiled and avoids an `O(T*K*d)` Student quantile cache:

```python
df_grid = np.linspace(2.01, 40.0, 64)

for block in evaluator.evaluate_grid_batches(
    df_grid,
    batch_rows=128,
    dimension_tile=16_384,
    n_threads=4,
    memory_budget_bytes=64 * 1024**2,
):
    consume(block.log_pdf, block.dlog_ddf)
```

## Sampling fitted Student models

Use batches whenever `n*d` output is itself large:

```python
for block in student.sample_batches(
    1_000_000,
    u=u,
    batch_rows=128,
    rng=np.random.default_rng(13),
    n_threads=4,
    memory_budget_bytes=128 * (2 * student.d + student.factor_rank + 8) * 8,
):
    consume(block)
```

Conditional sampling keeps fixed pseudo-observations exact:

```python
conditional = student.sample_conditional(
    10_000,
    r=mle.copula_param,
    given={0: 0.25, 3: 0.80},
    rng=np.random.default_rng(14),
    n_threads=4,
)
```

For dynamic fitted models, `sample_batches` and `predict_batches` preserve the
GAS or SCAR parameter-path semantics instead of replacing them with one
constant `df`.

## Scope of the factor representation

`FactorCorrelation` is a compact representation of a correlation in the
original `d`-dimensional space. It is not a dimension-reduction model and
does not map observed returns or pseudo-observations to `k`-dimensional
returns or pseudo-observations.

The decomposition can be given a latent-factor interpretation, but recovering
factor scores from observations requires an additional, family-specific
posterior estimator. Such scores depend on the marginal transform, are
rotation-dependent, and are neither returns nor uniform pseudo-observations.
Converting them to pseudo-observations would introduce another estimation
step and a separate statistical contract. The package therefore does not
present factor scores as output of this API.

The supported compositions are the first-party adapters
`GaussianCopula(corr_mode="factor")`,
`StochasticStudentCopula(corr_mode="factor")`, and
`FactorStudentEvaluator`.

## Choosing a mode

| Goal | Recommended API |
|---|---|
| Reusable matrix-free correlation | `FactorCorrelation.prepare()` |
| Static Gaussian copula at large `d` | `GaussianCopula(corr_mode="factor")` |
| Constant Student `df`, scalable safe default | Student factor + `two-stage` MLE |
| Constant Student `df`, bounded joint refinement | Student factor + `joint` MLE |
| Dynamic observation-driven tails | Student factor + GAS + `two-stage` |
| Dynamic latent tails | Student factor + SCAR + `two-stage` |

For CPU threading, reproducibility, nested parallelism, and rolling-window
guidance, see [CPU Parallelism](parallelism.md). For the broader multivariate
model contract, see [Multivariate Models](multivariate_models.md).
