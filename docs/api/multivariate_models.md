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

::: pyscarcopula._types.MultivariateMLEResult

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
These estimated-correlation modes are available for MLE and SCAR-TM-OU.
Joint SCAR-TM-OU uses a Python-owned optimizer and correlation parameterization
with an analytical joint Jacobian. C++ differentiates with respect to the
current static correlation matrix for matrix, local, and spectral transitions;
Python applies the chain rule to the `shrinkage` or `cholesky` raw parameters.
Setting `analytical_grad=False` retains the fully numerical optimizer gradient.
Likelihood evaluations always use the native engine. GAS remains
fixed-correlation only.

The C++ backend does not parameterize or optimize the static correlation
matrix. Parameterization and L-BFGS-B ownership remain in Python.

The implementation caches the full-sample Student quantile table for repeated
emission evaluations. The transient PPF cache is independent of correlation
state: changing `R`
refreshes only Cholesky/log-determinant state and the C++ copula spec. The PPF
table crosses pybind as contiguous NumPy buffers and is copied once into
owning C++ storage without intermediate Python lists. Buffer shape, finite
values, and strictly increasing nodes are validated at assignment. Python
and C++ use cubic Hermite interpolation inside the node range and exact
Student quantiles outside it. The cache is rebuilt after loading a
persisted model. GAS filtering and likelihood evaluation use the mandatory
native evaluator.

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
