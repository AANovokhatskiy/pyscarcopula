# Configuration and Fit Results

Use `NumericalConfig` for fitting and `PredictConfig` for prediction defaults.
`LBFGSBConfig` supplies optimizer overrides; `None` fields inherit the library
defaults. Per-call fit keywords override the corresponding configured values.
See [Optimizer Controls](../reference/optimizers.md) for examples and
[Estimation Methods](../guide/estimation-methods.md) for model compatibility.

```python
from pyscarcopula import LBFGSBConfig, NumericalConfig, PredictConfig

config = NumericalConfig(mle_optimizer=LBFGSBConfig(gtol=1e-6))
prediction = PredictConfig()
```

## Fitting configuration

::: pyscarcopula.NumericalConfig
    options:
      show_root_heading: true
      show_root_full_path: false
      merge_init_into_class: true
      separate_signature: true
      members: false

## Optimizer options

::: pyscarcopula.LBFGSBConfig
    options:
      show_root_heading: true
      show_root_full_path: false
      merge_init_into_class: true
      separate_signature: true
      members: [merged, options]

## Prediction configuration

::: pyscarcopula.PredictConfig
    options:
      show_root_heading: true
      show_root_full_path: false
      merge_init_into_class: true
      separate_signature: true
      members: [validated, replace]

## Fit results

Scalar model `fit` calls return a typed result. Inspect `success`, `message`,
and `log_likelihood` before using an optimizer candidate. `n_params` includes
estimated static parameters where applicable; it need not equal the length of
the dynamic process vector. Configuration and `MultivariateMLEResult` are
exported at package level. The other result classes below are defined in
`pyscarcopula._types`; their fields are described here because fitted models
return them.

| Fit | Result type | Parameter fields |
|---|---|---|
| Scalar MLE | `MLEResult` | `copula_param`, `n_params`, `diagnostics` |
| Multivariate static MLE | `MultivariateMLEResult` | `model_parameters`, `correlation_matrix`, `aic`, `bic`; `copula_param` is `None` for Gaussian |
| GAS | `GASResult` | `params.omega`, `params.gamma`, `params.beta`, `scaling`, `score_eps`, `r_last` |
| SCAR-TM-OU | `LatentResult` | `params.kappa`, `params.mu`, `params.nu`, solver metadata |
| SCAR-TM-JACOBI | `LatentResult` | `params.kappa`, `params.m`, `params.xi`, solver and sampling metadata |
| Independence | `IndependentResult` | `copula_param == 0`, `n_params == 0`, zero log likelihood |

GAS is observation-driven and has its own result type. It is not a
`LatentResult`. `LatentProcessParams.names`, `.values`, and `.to_dict()` allow
generic inspection without assuming OU parameter names. `GASResult.r_last`
contains the one-step-ahead copula parameter.

`VineCopula.fit` returns the fitted vine itself. Its `fit_result` is a
`scipy.optimize.OptimizeResult` summary; individual edges carry the typed
results above. See [Vine API](vine.md).

## Common fields

| Field | Meaning |
|---|---|
| `log_likelihood` | Fitted log likelihood |
| `method` | Estimation method label |
| `copula_name` | Model name |
| `success` | Optimizer/candidate acceptance status |
| `nfev` | Reported objective evaluation count |
| `message` | Fit status or termination explanation |

::: pyscarcopula._types.FitResultBase
    options:
      show_root_heading: true
      show_root_full_path: false
      merge_init_into_class: true
      separate_signature: true
      members: false

## Scalar MLE result

::: pyscarcopula._types.MLEResult
    options:
      show_root_heading: true
      show_root_full_path: false
      merge_init_into_class: true
      separate_signature: true
      members: [n_params]

## Multivariate MLE result

Factor results store compact loadings in `model_parameters`; their
`correlation_matrix` is `None`. A supplied fixed Gaussian correlation has
zero fitted parameters. Other correlation counts follow
[Mathematical Contracts](../guide/mathematical-contracts.md#static-elliptical-correlation-estimation).

::: pyscarcopula.MultivariateMLEResult
    options:
      show_root_heading: true
      show_root_full_path: false
      merge_init_into_class: true
      separate_signature: true
      inherited_members: [n_params]
      members: [n_params, aic, bic]

## GAS result

::: pyscarcopula._types.GASResult
    options:
      show_root_heading: true
      show_root_full_path: false
      merge_init_into_class: true
      separate_signature: true
      members: [omega, gamma, beta, n_params]

## SCAR result

::: pyscarcopula._types.LatentResult
    options:
      show_root_heading: true
      show_root_full_path: false
      merge_init_into_class: true
      separate_signature: true
      members: [n_params]

## Independence result

::: pyscarcopula._types.IndependentResult
    options:
      show_root_heading: true
      show_root_full_path: false
      merge_init_into_class: true
      separate_signature: true
      members: [n_params]

## Named process parameters

::: pyscarcopula._types.LatentProcessParams
    options:
      show_root_heading: true
      show_root_full_path: false
      merge_init_into_class: true
      separate_signature: true
      members: [n_params, to_dict, replace]
