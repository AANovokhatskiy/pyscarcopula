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
`sample_conditional(n, given, rng=None)` and `predict(n, given=..., rng=...)`:

```python
import numpy as np

conditional = cop.sample_conditional(
    n=10_000,
    given={0: 0.25, 2: 0.8},
    rng=np.random.default_rng(2026),
)
```

The supplied columns remain fixed. Supplying every variable returns constant
rows equal to `given`.

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
```

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
        - sample
        - predict
        - predictive_mean
        - xT_distribution
        - log_likelihood
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

::: pyscarcopula.copula.multivariate.stochastic_student.StochasticStudentCopula
    options:
      members:
        - fit
        - sample
        - predict
        - predictive_mean
        - xT_distribution
        - log_likelihood
        - transform
        - inv_transform
        - dtransform
