# Experimental Models API

!!! warning "Experimental API"
    These classes are under active development and their API may change.

## Equicorrelation Gaussian Copula

For $d$ assets, the standard Gaussian copula has $d(d-1)/2$ static
correlation parameters. The equicorrelation model uses a single dynamic
correlation:

$$R(t) = (1-\rho(t)) \cdot I + \rho(t) \cdot \mathbf{1}\mathbf{1}^\top$$

All pairwise correlations equal $\rho(t)$, which follows an OU process via
SCAR. This gives 3 parameters instead of $d(d-1)/2$.

### Usage

```python
from pyscarcopula.copula.experimental.equicorr import EquicorrGaussianCopula

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
samples = cop.sample(n=10000, r=0.5)
```

### When to use

Equicorrelation SCAR is a good fit when:

- All pairwise correlations move together, common in equity and crypto markets
- You need fast estimation for large $d$, with $O(d)$ density evaluation
- You want a compact, interpretable model with 3 parameters

For heterogeneous dependence, use a C-vine or R-vine instead.

### API

::: pyscarcopula.copula.experimental.equicorr.EquicorrGaussianCopula
    options:
      members:
        - fit
        - sample
        - predict
        - predictive_mean
        - smoothed_params
        - xT_distribution
        - log_likelihood
        - transform
        - inv_transform
        - dtransform

## StochasticStudentCopula

A fixed-correlation multivariate Student copula with dynamic degrees of
freedom. The correlation matrix is estimated once from Kendall's tau and held
fixed. The dynamic scalar parameter is

$$\nu_t = 2 + \mathrm{softplus}(g_t),$$

and the row density is the standard Student copula density

$$c(u_t;R,\nu_t)=
\frac{t_d(T_{\nu_t}^{-1}(u_t);0,R,\nu_t)}
     {\prod_j t_1(T_{\nu_t}^{-1}(u_{tj});\nu_t)}.$$

`method='mle'` estimates a constant $\nu$, `method='gas'` estimates a
score-driven recursion for $g_t$, and `method='scar-tm-ou'` treats $g_t$ as a
latent OU process integrated by transfer matrix.

::: pyscarcopula.copula.experimental.stochastic_student.StochasticStudentCopula
    options:
      members:
        - fit
        - sample
        - predict
        - predictive_mean
        - smoothed_params
        - xT_distribution
        - log_likelihood
        - transform
        - inv_transform
        - dtransform

## StochasticStudentDCCCopula

A multivariate Student copula with two sources of time variation:

- $R_t$ is a deterministic DCC(1,1) correlation path fitted from standardized
  residuals.
- $\nu_t = 2 + \mathrm{softplus}(g_t)$ controls tail thickness and is fitted by
  MLE, GAS, or SCAR-TM-OU.

Conditional on the cached DCC path, the emission density is

$$c(u_t;R_t,\nu_t)=
\frac{t_d(T_{\nu_t}^{-1}(u_t);0,R_t,\nu_t)}
     {\prod_j t_1(T_{\nu_t}^{-1}(u_{tj});\nu_t)}.$$

The DCC step is deliberately separate from the scalar latent-state fit:
`fit_R_t()` builds and caches $R_t$, then `fit(..., method=...)` estimates the
degrees-of-freedom dynamics against that path.

::: pyscarcopula.copula.experimental.stochastic_student_dcc.StochasticStudentDCCCopula
    options:
      members:
        - fit_R_t
        - fit
        - sample
        - predict
        - predictive_mean
        - smoothed_params
        - log_likelihood
        - transform
        - inv_transform
        - dtransform
