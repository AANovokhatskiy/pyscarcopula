# Static Multivariate Models API

## Static Gaussian and Student Copulas

Static multivariate MLE returns `MultivariateMLEResult` for Gaussian, Student,
equicorrelation Gaussian, and stochastic Student models. The returned object
is also stored as `copula.fit_result`.

```python
import numpy as np
from pyscarcopula import GaussianCopula

u = np.random.default_rng(2026).uniform(0.01, 0.99, size=(80, 5))
cop = GaussianCopula()
result = cop.fit(u)

result.correlation_matrix
result.model_parameters
result.log_likelihood
result.n_params
result.aic
result.bic
```

See [fit-result fields](configuration.md#fit-results).

Static `GaussianCopula` and `StudentCopula` accept only `method='mle'`. Both
provide exact conditional generation in pseudo-observation space through
`sample_conditional(n, given, rng=None, *, n_threads=1)` and
`predict(n, given=..., rng=..., n_threads=1)`. Both models also accept
`n_threads=1` in `sample`; an empty `given` preserves the requested thread
count when delegating to unconditional sampling:

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

Here `MLE` is the static model label. Correlation estimation is controlled by
`corr_mode`, and only modes marked joint below put correlation parameters in
the likelihood optimizer vector:

| `corr_mode` | Correlation procedure | Gaussian count | Student count |
|---|---|---:|---:|
| `fixed`, supplied `R` | held fixed | `0` | `1` for `df` |
| `fixed`, no `R` | Gaussian-score/Kendall plug-in | `d*(d-1)/2` | `1 + d*(d-1)/2` |
| `shrinkage` | joint one-parameter shrinkage | `1` | `2` |
| `cholesky` | joint full correlation | `d*(d-1)/2` | `1 + d*(d-1)/2` |
| `factor`, two-stage | compact plug-in loadings | identifiable loading count | `1 +` that count |
| `factor`, joint | Student only | unavailable | `1 +` identifiable loading count |

The default is `fixed`, preserving the previous fast behaviour. Use
`result.diagnostics["corr_estimator"]` to distinguish `supplied`,
`gaussian_score`, `kendall_plugin`, `joint_mle`, `factor_two_stage`, and
`factor_joint`. Plug-in counts are included in AIC/BIC. A supplied fixed
Gaussian has zero fitted parameters, which is valid for
`MultivariateMLEResult`.

```python
from pyscarcopula import GaussianCopula, StudentCopula

gaussian_fast = GaussianCopula(corr_mode="fixed")
gaussian_joint = GaussianCopula(corr_mode="cholesky")
student_fast = StudentCopula(corr_mode="fixed")
student_joint = StudentCopula(corr_mode="shrinkage")
```

Full Cholesky is guarded by `cholesky_d_max=10` by default and is intended for
small dimensions. Factor mode is the scalable choice when the low-rank
assumption is appropriate.

## Gaussian operations

::: pyscarcopula.copula.multivariate.gaussian.GaussianCopula
    options:
      show_root_heading: true
      show_root_full_path: false
      merge_init_into_class: true
      separate_signature: true
      show_bases: false
      members:
        - fit
        - log_likelihood
        - log_pdf_rows
        - sample
        - sample_conditional
        - predict
        - to_correlation_matrix
        - sample_batches
        - predict_batches
        - initialize_factor
        - factor_diagnostics

## Student operations

::: pyscarcopula.copula.multivariate.student.StudentCopula
    options:
      show_root_heading: true
      show_root_full_path: false
      merge_init_into_class: true
      separate_signature: true
      show_bases: false
      members:
        - fit
        - log_likelihood
        - log_pdf_rows
        - sample
        - sample_conditional
        - predict
        - to_correlation_matrix

All multivariate APIs that expose `n_threads` use a literal default of `1`.
No environment variable changes that default. Fit-level native parallelism is
enabled with `NumericalConfig(n_threads=N)`.

## Gaussian factor correlation

For large dimensions, static Gaussian models can compose the same independent
[factor operator](factor.md):

```python
from pyscarcopula import GaussianCopula, NumericalConfig

gaussian = GaussianCopula(
    d=u.shape[1],
    corr_mode="factor",
    factor_rank=min(8, u.shape[1] - 1),
    factor_tile_size=16_384,
)
result = gaussian.fit(
    u,
    method="mle",
    config=NumericalConfig(n_threads=4),
)
```

If `factor_loadings` is omitted, fitting uses a fixed-seed, tiled normal-score
randomized SVD without constructing a dense covariance matrix. Loadings may
also be supplied to the constructor. The fitted result keeps
`correlation_matrix=None` and stores compact loadings and uniqueness in
`model_parameters`. Data-estimated loadings count
`min(d*k - k*(k-1)/2, d*(d-1)/2)` correlation parameters.
See the [parameter-accounting contract](../guide/mathematical-contracts.md#static-elliptical-correlation-estimation)
for optimizer, plug-in, and effective counts.

`log_likelihood`, `log_pdf_rows`, `sample`, `sample_batches`,
`sample_conditional`, `predict`, and `predict_batches` accept a literal
`n_threads=1` default. Conditional generation factorizes only
`I + B_G.T @ D_G^-1 @ B_G` for the fixed coordinates. Goodness-of-fit uses a
sequential rank-dimensional factor update with `O(T*k + k^2)` workspace.
Persistence and rolling-window worker reconstruction retain the compact
constructor policy.

## Static Student factor correlation

`StudentCopula` exposes the same compact representation and adds optional
joint static loading estimation:

```python
from pyscarcopula import StudentCopula

student_factor = StudentCopula(
    d=u.shape[1],
    corr_mode="factor",
    factor_rank=min(8, u.shape[1] - 1),
    factor_estimation="two-stage",  # use "joint" for joint df/loadings MLE
)
student_result = student_factor.fit(u, method="mle")
```

Two-stage loadings are counted as plug-in parameters. Joint mode uses an
identified loading parameterization and analytical matrix-free gradients.
Sampling, conditional sampling, GoF, bootstrap, persistence, and worker
reconstruction retain the factor operator without implicit dense
materialization.
