# Native Core And Multivariate Migration

This release completes the migration of built-in numerical production paths
to the required C++ extension and promotes multivariate models out of the
experimental namespace.

## Import Changes

The old experimental module paths were removed:

```python
# Removed
from pyscarcopula.copula.experimental.equicorr import EquicorrGaussianCopula
from pyscarcopula.copula.experimental.stochastic_student import StochasticStudentCopula
```

Use the multivariate namespace:

```python
from pyscarcopula.copula.multivariate import (
    EquicorrGaussianCopula,
    StochasticStudentCopula,
)
```

Top-level imports are unchanged:

```python
from pyscarcopula import EquicorrGaussianCopula, StochasticStudentCopula
```

`StochasticStudentDCCCopula` was removed rather than promoted.

## Native Runtime

The C++ extension is mandatory. Built-in copula point operations, static
likelihoods, GAS, and SCAR-TM-OU use native numerical implementations.

The `backend` options previously accepted by GAS and SCAR-TM-OU were removed.
There is no Python fallback and unsupported custom copulas fail before fitting.
Custom Python strategies and non-native Python workflows remain supported.

SCAR-TM-JACOBI and SCAR-MC/EIS retain Python orchestration. Numba remains a
dependency for retained utilities, GoF helpers, MC/EIS kernels, and contrib
analytics.

## Persistence Compatibility

Versioned JSON loading remains backward compatible:

- persisted experimental equicorrelation paths migrate to
  `pyscarcopula.copula.multivariate.equicorr.EquicorrGaussianCopula`;
- persisted experimental stochastic Student paths migrate to
  `pyscarcopula.copula.multivariate.stochastic_student.StochasticStudentCopula`;
- removed legacy `backend` fields in GAS and SCAR results are ignored.

Newly saved models use the multivariate class paths.

## Application Changes

No change is required for users importing supported models from
`pyscarcopula`. Code importing from `copula.experimental` must switch to
`copula.multivariate`.

Remove `backend=...` from GAS and SCAR-TM-OU calls:

```python
result = fit(copula, u, method="gas")
result = fit(copula, u, method="scar-tm-ou")
```
# Compatibility cleanup

The development API no longer includes the following compatibility surfaces:

- `LatentResult.alpha`; use `result.params.values`;
- `legacy_smart_initial_point` and `legacy_smart_init`;
- `pyscarcopula.numerical.auto_tm`;
- `pyscarcopula.numerical.tm_gradient`;
- `tm_functions._forward_loglik`;
- Jacobi transition aliases `matrix` and `spectral`; use
  `spectral_matrix` or `spectral_coeff` explicitly;
- `spectral_basis_order="adaptive"`; use `"auto"`;
- `CVineCopula.sample_model`;
- ignored `u_train` and extra keywords in `CVineCopula.sample`;
- compatibility Student density functions from
  `stochastic_student`; use model methods or native numerical adapters.

`CVineCopula.sample(n, u=None, rng=...)` reproduces the fitted model.

`RVineCopula` now uses the same explicit sampling/prediction split:

- `sample(n, u=None, rng=None)` reproduces the fitted model; `u` is accepted
  uniformly across model types and is not needed by fitted vine edges;
- `predict(n, u=..., ...)` uses `u` as the sole public history argument and
  falls back to the observations stored by `fit` when `u` is omitted;
- the former public `u_train` keyword is no longer accepted.

Documentation now uses the runtime contracts consistently:

- MLE `alpha0` is documented in natural parameter units;
- the gradient capability matrix distinguishes native model scores from
  optimizer Jacobians;
- numerical safety boundaries are named by purpose;
- fitted-model `sample(n, u=None)`, low-level
  `sample_at_parameter(n, r)`, and predictive `predict(n, u=...)` are
  documented as separate operations across model types;
- removed aliases are rejected by executable documentation contract tests.

The validation-remediation plan completed its final acceptance on
June 20, 2026. The forced native rebuild, full non-benchmark suite,
distributional validation, persistence, documentation, and standard
performance gates passed. See
`validation/validation_remediation_final.md` for the recorded matrix and
performance measurements.
