# Copulas API

## Top-level API

All API functions are stateless — they accept a copula object, data, and a result, and return new values without mutation.

::: pyscarcopula.api.fit

::: pyscarcopula.api.sample

::: pyscarcopula.api.predict

::: pyscarcopula.api.smoothed_params

::: pyscarcopula.api.mixture_h

## BivariateCopula (base class)

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

::: pyscarcopula.copula.stochastic_student.StochasticStudentCopula
    options:
      members:
        - fit
        - sample
        - predict
        - smoothed_params
        - transform
        - inv_transform