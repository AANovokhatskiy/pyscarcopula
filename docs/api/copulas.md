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

::: pyscarcopula.api.smoothed_params

::: pyscarcopula.api.mixture_h

## BivariateCopula (base class)

`BivariateCopula.predict(...)` mirrors the top-level API and accepts
`given`, `horizon`, and `predict_config`.

::: pyscarcopula.copula.base.BivariateCopula
    options:
      members:
        - pdf
        - log_pdf
        - h
        - h_inverse
        - sample
        - predict
        - sample_model
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

## GaussianCopula

::: pyscarcopula.copula.elliptical.GaussianCopula
    options:
      show_bases: false
      members: false

## StudentCopula

::: pyscarcopula.copula.elliptical.StudentCopula
    options:
      show_bases: false
      members: false

## StochasticStudentCopula

::: pyscarcopula.copula.experimental.stochastic_student.StochasticStudentCopula
    options:
      members:
        - fit
        - sample
        - predict
        - smoothed_params
        - transform
        - inv_transform
