# Experimental Models API

!!! warning "Experimental API"
    These classes are under active development and their API may change.

## Equicorrelation Gaussian Copula

For `d` assets, the standard Gaussian copula has `d(d-1)/2` static
correlation parameters. The equicorrelation model uses a single dynamic
correlation:

$$R(t) = (1-\rho(t)) \cdot I + \rho(t) \cdot \mathbf{1}\mathbf{1}^\top$$

All pairwise correlations equal `rho(t)`, which follows an OU process via
SCAR. This gives 3 parameters instead of `d(d-1)/2`.

### Usage

```python
from pyscarcopula.copula.experimental.equicorr import EquicorrGaussianCopula

cop = EquicorrGaussianCopula(d=6)

# MLE (constant rho)
cop.fit(u, method='mle')

# SCAR (time-varying rho)
cop.fit(u, method='scar-tm-ou')
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
- You need fast estimation for large `d`, with `O(d)` density evaluation
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

::: pyscarcopula.copula.experimental.stochastic_student_dcc.StochasticStudentDCCCopula
    options:
      members:
        - fit
        - sample
        - predict
        - predictive_mean
        - smoothed_params
        - log_likelihood
        - transform
        - inv_transform
        - dtransform
