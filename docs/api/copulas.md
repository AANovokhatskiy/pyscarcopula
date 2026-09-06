# Copulas API

`predict(...)` also supports:

- `given={0: 0.4}` for conditional generation in pseudo-observation space
- `horizon='current'|'next'` for SCAR-TM predictive mixtures
- `predict_config=PredictConfig(...)` for explicit prediction options
- `rng=np.random.default_rng(seed)` for reproducible Monte Carlo output

See [Prediction Semantics](../guide/prediction-semantics.md) for the
mathematical meaning of these options.

## Top-level API

::: pyscarcopula.api.fit

::: pyscarcopula.api.sample

::: pyscarcopula.api.predict

::: pyscarcopula.api.predictive_mean

::: pyscarcopula.api.log_likelihood

::: pyscarcopula.api.mixture_h

The following complete example evaluates both fitted likelihood and the
conditional CDF used by pair-copula and vine calculations:

```python
import numpy as np
from pyscarcopula import GumbelCopula
from pyscarcopula.api import fit, log_likelihood, mixture_h

rng = np.random.default_rng(2026)
source = GumbelCopula(rotate=180)
u = source.sample_at_parameter(200, np.full(200, 1.7), rng=rng)

copula = GumbelCopula(rotate=180)
result = fit(copula, u, method="mle")
fitted_log_likelihood = log_likelihood(copula, u, result)
conditional_cdf = mixture_h(copula, u, result)
```

## BivariateCopula (base class)

`BivariateCopula.predict(...)` mirrors the top-level API and accepts
`given`, `horizon`, and `predict_config`.

`BivariateCopula.sample(n, u=None, ...)` reproduces the fitted model, matching
the multivariate and vine APIs. Use
`BivariateCopula.sample_at_parameter(n, r, ...)` for generation at an
explicit copula parameter.

Clayton, Gumbel, Frank and Joe sampling uses native conditional inversion.
Their conditional CDFs and inverses use logarithmic formulas in the tails;
Gumbel and Joe use safeguarded monotone root solving. Reflected inverse
functions preserve small probabilities without first forming `1 - q`.
These corrections can change fixed-seed results relative to 0.20.1.

Conditional CDFs and inverse CDFs can reach the mathematical endpoints 0 and
1. Unconditional sampling keeps its output in the open unit interval: if an
interior result rounds to an endpoint, only that endpoint is moved to the
nearest interior float64 value. There is no artificial `1e-10` sampling floor.
A failed inverse is reported as an error rather than accepted as a clipped
finite sample. This does not guarantee an arbitrarily small CDF residual
when the exact inverse lies between adjacent representable float64 values.

`pobs(data)` computes ordinal ranks in C++ using the standard library:
rows are sorted by value and then by their original row index. Equal values
receive successive ranks in input order; the ranks are divided by `n + 1`.
Integer comparisons retain their original precision, and NaNs sort last.
The optional `ties_method="ordinal"` selects the same behavior explicitly.
Use `ties_method="legacy"` for the historical 0.20.1 ordering within ties.
This mode uses a separate native implementation of the historical ordering
rules; it does not require Numba. Equal values still receive different ranks,
but their input order is not preserved. Compare versions using the same
precomputed pseudo-observations when exact input agreement matters.

All built-in bivariate families share the same fitting surface:

```python
from pyscarcopula import (
    BivariateGaussianCopula,
    ClaytonCopula,
    FrankCopula,
    IndependentCopula,
    JoeCopula,
)

models = [
    ClaytonCopula(),
    FrankCopula(),
    JoeCopula(),
    BivariateGaussianCopula(),
    IndependentCopula(),
]
results = [model.fit(u, method="mle") for model in models]
```

Kendall-tau dynamic fitting with `method='scar-tm-jacobi'` requires
`tau_to_param` and `param_to_tau`. These mappings are implemented for
`GumbelCopula`, `ClaytonCopula`, `FrankCopula`, `JoeCopula`, and
`BivariateGaussianCopula`.

::: pyscarcopula.copula.base.BivariateCopula
    options:
      members:
        - pdf
        - log_pdf
        - h
        - h_inverse
        - sample
        - sample_at_parameter
        - predict
        - tau_to_param
        - param_to_tau
        - transform
        - inv_transform

## GumbelCopula

::: pyscarcopula.copula.gumbel.GumbelCopula
    options:
      show_bases: false
      members: false

## ClaytonCopula

::: pyscarcopula.copula.clayton.ClaytonCopula
    options:
      show_bases: false
      members: false

## FrankCopula

::: pyscarcopula.copula.frank.FrankCopula
    options:
      show_bases: false
      members: false

## JoeCopula

::: pyscarcopula.copula.joe.JoeCopula
    options:
      show_bases: false
      members: false

## IndependentCopula

::: pyscarcopula.copula.independent.IndependentCopula
    options:
      show_bases: false
      members: false

## BivariateGaussianCopula

::: pyscarcopula.copula.elliptical.BivariateGaussianCopula
    options:
      show_bases: false
      members: false
