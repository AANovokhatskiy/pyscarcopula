# Multivariate Models API

## Static MLE result

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

All multivariate APIs that expose `n_threads` use a literal default of `1`.
No environment variable changes that default. Fit-level native parallelism is
enabled with `NumericalConfig(n_threads=N)`.

## Equicorrelation Gaussian Copula

For $d$ assets, the standard Gaussian copula has $d(d-1)/2$ static
correlation parameters. The equicorrelation model uses a single dynamic
correlation:

$$R(t) = (1-\rho(t)) \cdot I + \rho(t) \cdot \mathbf{1}\mathbf{1}^\top$$

All pairwise correlations equal $\rho(t)$, which follows an OU process via
SCAR. This gives 3 parameters instead of $d(d-1)/2$.

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
inputs remain supported. A prepared MLE result represents correlation by its
scalar `equicorrelation_rho` and leaves `correlation_matrix=None`, avoiding an
otherwise prohibitive `d * d` result allocation.

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
```

The structural sampler costs `O(n*d)` and does not construct a dense
correlation matrix. It supports the full admissible open interval
`-1/(d-1) < rho < 1`.

### When to use

Equicorrelation SCAR is a good fit when:

- All pairwise correlations move together, common in equity and crypto markets
- You need fast estimation for large $d$, with $O(d)$ density evaluation
- You want a compact, interpretable model with 3 parameters

For heterogeneous dependence, use a C-vine or R-vine instead.

### API

::: pyscarcopula.copula.multivariate.equicorr.EquicorrGaussianCopula
    options:
      members:
        - fit
        - prepare_sufficient_statistics
        - sample
        - sample_conditional
        - predict
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
        - sample
        - sample_conditional
        - predict
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
