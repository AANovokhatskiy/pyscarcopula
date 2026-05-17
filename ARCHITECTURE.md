# Architecture

## Module Map

```text
pyscarcopula/
├── __init__.py              # Public re-exports and BLAS thread policy
├── api.py                   # Stateless top-level API: fit(), predict(), sample()
├── _types.py                # FitResult types, PredictConfig, NumericalConfig
├── _utils.py                # pobs(), broadcast(), clip_unit(), linear algebra helpers
├── io.py                    # Model persistence
├── stattests.py             # Goodness-of-fit diagnostics
│
├── copula/                  # Pair-copula math and multivariate copula models
│   ├── _protocol.py         # Bivariate copula protocol
│   ├── base.py              # BivariateCopula base and OO convenience wrapper
│   ├── gumbel.py, frank.py, joe.py, clayton.py
│   ├── independent.py       # Zero-parameter null model
│   ├── elliptical.py        # Gaussian and Student-t copulas
│   └── experimental/
│       ├── equicorr.py
│       ├── stochastic_student.py
│       └── stochastic_student_dcc.py
│
├── strategy/                # Estimation methods via registry
│   ├── _base.py             # FitStrategy protocol, @register_strategy
│   ├── mle.py               # Constant-parameter MLE
│   ├── scar_tm.py           # Transfer matrix + analytical gradient
│   ├── scar_mc.py           # Monte Carlo SCAR variants
│   ├── gas.py               # Score-driven GAS model
│   ├── initial_point.py
│   └── predict_helpers.py
│
├── numerical/               # Computational kernels
│   ├── ou_kernels.py
│   ├── tm_grid.py
│   ├── tm_functions.py
│   ├── tm_gradient.py
│   ├── predictive_tm.py
│   ├── mc_samplers.py
│   └── gas_filter.py
│
├── vine/                    # C-vine and R-vine models
│   ├── cvine.py
│   ├── rvine.py
│   ├── _pair_copula.py
│   ├── _edge_adapter.py
│   ├── _dynamic_conditioning.py
│   ├── _selection.py
│   ├── _structure.py
│   ├── _rvine_*.py
│   └── _conditional_*.py
│
└── contrib/                 # Optional analytics: risk metrics, marginals
```

## Dependency Flow

The functional core is intentionally layered:

```text
api.py -> strategy/ -> numerical/
                 \-> copula/
vine/ -> strategy/ + copula/ + numerical helpers
contrib/ -> public API + optional model helpers
```

`strategy/_base.py` is the central method registry. New estimation methods
register with `@register_strategy("METHOD")`; callers obtain them through
`get_strategy()`.

## State Model

The functional API is stateless:

```python
from pyscarcopula.api import fit, predict

result = fit(copula, u, method="scar-tm-ou")
samples = predict(copula, u, result, n=1000)
```

`BivariateCopula.fit()`, `predict()`, and `sample_model()` are stateful
convenience wrappers. They delegate to the functional API, store `fit_result`
and the last fitting data, and exist for backward-compatible object-oriented
usage.

## BLAS Thread Policy

Transfer-matrix likelihood evaluation performs many small matrix-vector
operations. Multi-threaded BLAS usually adds overhead and competes with useful
outer-level parallelism in `contrib.risk_metrics` and future vine edge work.

Package import therefore forces common BLAS backends to one thread by default.
Set `PYSCA_BLAS_THREADS` before importing `pyscarcopula` to override this
policy intentionally.

## Key Types

`NumericalConfig` centralizes numerical constants such as grid size, optimizer
tolerances, clipping thresholds, and GAS/MC defaults.

`PredictConfig` carries prediction-time options shared by API, bivariate
copulas, vines, and strategies.

`FitResult` is a union of immutable dataclasses:

```text
FitResultBase
├── MLEResult
├── LatentResult
├── GASResult
└── IndependentResult
```

`LatentProcessParams` stores named latent-process parameters, for example
`ou_params(kappa, mu, nu)` or `gas_params(omega, gamma, beta)`.

## Adding An Estimation Method

Create a strategy module and register the class:

```python
from pyscarcopula.strategy._base import register_strategy

@register_strategy("MY-METHOD")
class MyStrategy:
    def __init__(self, config=None, **kwargs): ...
    def fit(self, copula, u, **kwargs): ...
    def predictive_mean(self, copula, u, result): ...
    def mixture_h(self, copula, u, result): ...
```

After import registration, `fit(copula, data, method="my-method")` works
through the same dispatch path as the built-in methods.

## Vine Refactor Backlog

The current vine layer intentionally keeps the small compatibility fixes
separate from larger structural work. Remaining large-scope items:

- Keep CVine and RVine edge runtime behavior behind `_edge_adapter.py`,
  `_rvine_edges.py`, and strategy-owned state methods. `PairCopula` is the
  shared edge container, while `EdgeView` and edge accessor helpers provide a
  common read-only contract for compatibility code.
- Keep dynamic conditioning strategy-generic. New dynamic methods should expose
  prediction, conditioning, and model-sampling state through the strategy
  contract instead of adding method-specific branches in `rvine.py`.
- Keep `vine.__all__` limited to public objects; internal helpers should stay
  importable only from their private modules.
- Cache repeated edge fits during conditional R-vine beam search.
- Add optional per-tree parallel edge fitting once the single-edge contract is
  stable.
