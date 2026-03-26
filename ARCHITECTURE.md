# Architecture

## Module map

```
pyscarcopula/
├── __init__.py              # Re-exports
├── api.py                   # Top-level: fit(), smoothed_params(), mixture_h()
├── _types.py                # FitResult types, LatentProcessParams, NumericalConfig
├── _utils.py                # pobs(), broadcast(), clip_unit()
│
├── copula/                  # Pure math: PDF, h-functions, transforms
│   ├── _protocol.py         # CopulaProtocol interface
│   ├── base.py              # BivariateCopula base class
│   ├── gumbel.py, frank.py, joe.py, clayton.py
│   ├── independent.py       # Zero-parameter null model
│   ├── elliptical.py        # Gaussian, Student-t
│   ├── equicorr.py          # Equicorrelation Gaussian (d-dimensional)
│   └── vine.py              # CVineCopula
│
├── strategy/                # Estimation methods (Strategy pattern)
│   ├── _base.py             # FitStrategy protocol, @register_strategy
│   ├── mle.py               # Constant parameter
│   ├── scar_tm.py           # Transfer matrix + analytical gradient
│   ├── scar_mc.py           # Monte Carlo (p-sampler, m-sampler with EIS)
│   └── gas.py               # Score-driven (unit / fisher scaling)
│
├── numerical/               # Computational kernels
│   ├── ou_kernels.py        # Numba: OU path generation
│   ├── tm_grid.py           # TMGrid: adaptive grid, dense/sparse operator
│   ├── tm_functions.py      # loglik, smoothed, rosenblatt, mixture_h
│   ├── tm_gradient.py       # Analytical gradient in xi-coordinates
│   └── mc_samplers.py       # p-sampler, m-sampler, EIS regression
│
├── vine/                    # Vine constructions
│   └── forward_pass.py      # vine_forward_iter(), vine_rosenblatt()
│
├── stattests.py             # GoF tests (Rosenblatt + CvM)
│
└── contrib/                 # Optional: risk metrics, marginals
    ├── risk_metrics.py
    ├── marginal.py
    └── empirical.py
```

## Dependency flow

```
api.py  →  strategy/  →  numerical/  →  _types.py, _utils.py
                ↓
           copula/ (stateless math)
```

Dependencies go strictly downward — no circular imports.

## Key types

### LatentProcessParams

Generic container for latent process parameters with named access:

```python
from pyscarcopula._types import ou_params

p = ou_params(theta=49.97, mu=2.42, nu=10.65)
p.theta     # 49.97
p.n_params  # 3
p.to_dict() # {'theta': 49.97, 'mu': 2.42, 'nu': 10.65}
```

### FitResult types

```
FitResultBase (log_likelihood, method, copula_name, success)
├── MLEResult (copula_param)
├── LatentResult (params: LatentProcessParams, K, grid_range, ...)
├── GASResult (params: LatentProcessParams, scaling)
└── IndependentResult ()
```

### NumericalConfig

All numerical constants in one place:

```python
from pyscarcopula._types import NumericalConfig

cfg = NumericalConfig(default_K=500, default_tol_scar=5e-2)
result = fit(copula, u, method='scar-tm-ou', config=cfg)
```

## Adding a new estimation method

Create a file in `strategy/` with the `@register_strategy` decorator:

```python
from pyscarcopula.strategy._base import register_strategy

@register_strategy('MY-METHOD')
class MyStrategy:
    def __init__(self, config=None, **kwargs): ...
    def fit(self, copula, u, **kwargs): ...
    def smoothed_params(self, copula, u, result): ...
    def mixture_h(self, copula, u, result): ...
```

After this, `fit(copula, data, method='my-method')` works automatically.
