# Copulas API

`predict(...)` also supports:

- `given={0: 0.4}` for conditional generation in pseudo-observation space
- `horizon='current'|'next'` for SCAR-TM predictive mixtures
- `predict_config=PredictConfig(...)` for explicit prediction options
- `rng=np.random.default_rng(seed)` for reproducible Monte Carlo output

See [Prediction Semantics](../guide/prediction-semantics.md) for the
mathematical meaning of these options.

## Top-level API

All API functions are stateless: they accept a copula object, data, and a
result, and return new values without mutation.

::: pyscarcopula.api.fit

::: pyscarcopula.api.sample

::: pyscarcopula.api.predict

::: pyscarcopula.api.predictive_mean

::: pyscarcopula.api.mixture_h

## BivariateCopula (base class)

`BivariateCopula.predict(...)` mirrors the top-level API and accepts
`given`, `horizon`, and `predict_config`.

`BivariateCopula.sample(n, u=None, ...)` reproduces the fitted model, matching
the multivariate and vine APIs. Use
`BivariateCopula.sample_at_parameter(n, r, ...)` for generation at an
explicit copula parameter.

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

## GaussianCopula

::: pyscarcopula.copula.multivariate.gaussian.GaussianCopula
    options:
      show_bases: false
      members: false

## StudentCopula

::: pyscarcopula.copula.multivariate.student.StudentCopula
    options:
      show_bases: false
      members: false

## StochasticStudentCopula

::: pyscarcopula.copula.multivariate.stochastic_student.StochasticStudentCopula
    options:
      members:
        - fit
        - sample
        - predict
        - predictive_mean
        - transform
        - inv_transform
