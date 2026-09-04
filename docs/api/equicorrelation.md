# Equicorrelation Gaussian API

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

All Equicorr sampling and prediction methods shown above, including their
batch variants and `sample_conditional`, accept keyword-only `n_threads=1`.
The requested count reaches the native observation sampler for explicit
parameters and fitted MLE/GAS/SCAR predictions. Thread counts must be integers
in `[1, 256]`, including when all coordinates are fixed or the output is empty.
Small blocks and the one-observation-at-a-time GAS model trajectory retain
their sequential execution policy.

SCAR `sample_batches` generates OU states one block at a time, with
`dt=1/(n-1)` based on the full requested length (`dt=1` for a single row).
Its temporary path storage is `O(batch_rows)`, not `O(n)`. Observation draws
are interleaved with OU blocks, so reproduce a draw sequence with the same
seed **and** `batch_rows`; changing the block size may change the sequence.

For both `EquicorrGaussianCopula` and `StochasticStudentCopula`, omitting
`r` in `log_likelihood(u)` evaluates the fitted MLE/GAS/SCAR strategy.
An explicit `r` requests a static emission likelihood. Omitting `r` in
`sample_conditional(n, given=...)` with fixed coordinates uses the next
predictive distribution. Pass `r` to condition at a fixed parameter, or
use `sample_batches(..., given=...)` for a conditional model trajectory.

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
