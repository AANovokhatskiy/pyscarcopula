# Architecture

This document describes the internal structure of `pyscarcopula` after the v0.3 refactoring.

## Principles

1. **Copulas are stateless.** A copula object defines the mathematical function (PDF, h-function, transform). It does not hold fit results or mutable state.
2. **Estimation via strategies.** Each estimation method (MLE, SCAR-TM, GAS) is a separate class implementing the `FitStrategy` protocol. Adding a new method means adding a file — no existing code changes.
3. **Typed, immutable results.** Fit results are frozen dataclasses (`MLEResult`, `LatentResult`, `GASResult`), not dynamically patched `OptimizeResult`. No `hasattr()` in consumer code.
4. **Variable-parameter latent processes.** The OU process has 3 parameters `(theta, mu, nu)`, but the `LatentProcessParams` container supports any number of named parameters — ready for future Lévy or fBm processes.
5. **Single source of truth.** Shared functions (`broadcast`, `pobs`, clip) live in `_utils.py`. Vine tree traversal lives in `vine/forward_pass.py`. Magic numbers live in `NumericalConfig`.

## Module map

```
pyscarcopula/
├── __init__.py              # Re-exports (no side effects)
├── api.py                   # Stateless top-level: fit(), smoothed_params(), mixture_h()
├── _types.py                # FitResult hierarchy, LatentProcessParams, NumericalConfig
├── _utils.py                # pobs(), broadcast(), clip_unit()
│
├── copula/                  # Layer 1: pure math (stateless)
│   ├── _protocol.py         # CopulaProtocol interface
│   ├── base.py              # BivariateCopula base (rotation, h-inversion, sampling)
│   ├── gumbel.py            # Numba kernels + class
│   ├── frank.py
│   ├── joe.py
│   ├── clayton.py
│   ├── independent.py       # Zero-parameter null model
│   ├── elliptical.py        # BivariateGaussian, Gaussian, Student-t
│   ├── equicorr.py          # Equicorrelation Gaussian (d-dimensional, 1 scalar param)
│   └── vine.py              # CVineCopula (legacy API, delegates to vine/)
│
├── strategy/                # Layer 2: estimation methods
│   ├── _base.py             # FitStrategy protocol, @register_strategy, get_strategy()
│   ├── mle.py               # MLEStrategy — constant parameter
│   ├── scar_tm.py           # SCARTMStrategy — transfer matrix + analytical gradient
│   ├── scar_mc.py           # SCARPStrategy, SCARMStrategy — Monte Carlo
│   └── gas.py               # GASStrategy — score-driven (preserves scaling='unit'/'fisher')
│
├── numerical/               # Layer 3: computational kernels
│   ├── ou_kernels.py        # @njit: OU path generation, EIS normalizing factors
│   ├── tm_grid.py           # TMGrid: adaptive grid, dense/sparse transfer operator
│   ├── tm_functions.py      # loglik, smoothed, rosenblatt, mixture_h, xT_distribution
│   ├── tm_gradient.py       # Analytical gradient in xi-coordinates
│   └── mc_samplers.py       # p-sampler, m-sampler, EIS auxiliary regression
│
├── vine/                    # Layer 4: vine constructions
│   └── forward_pass.py      # vine_forward_iter(), vine_rosenblatt()
│
├── latent/                  # Legacy (original code, still functional)
│   ├── ou_process.py
│   ├── gas_process.py
│   └── initial_point.py
│
├── stattests.py             # GoF tests (Rosenblatt + CvM)
│
└── contrib/                 # Layer 5: optional applied pipelines
    ├── risk_metrics.py      # VaR, CVaR, portfolio optimization
    ├── marginal.py          # Marginal distribution models (johnsonsu, etc.)
    └── empirical.py         # Empirical CVaR
```

## Dependency flow

Dependencies go **strictly downward** — no circular imports.

```
api.py
  ↓
strategy/ ← uses → copula/ (stateless)
  ↓
numerical/ (pure computation, no copula state)
  ↓
_types.py, _utils.py (shared foundations)
```

## Key types

### LatentProcessParams

Generic container for latent process parameters with named access:

```python
from pyscarcopula._types import ou_params, LatentProcessParams

# OU process: 3 parameters
p = ou_params(theta=49.97, mu=2.42, nu=10.65)
p.theta  # -> 49.97
p.n_params  # -> 3

# Future Lévy process: 4 parameters (same type!)
p = LatentProcessParams(
    process_type='levy',
    names=('alpha', 'beta', 'mu', 'sigma'),
    values=[1.5, 0.3, 0.0, 1.0],
)
p.alpha  # -> 1.5
p.n_params  # -> 4
```

### FitResult hierarchy

```
FitResultBase (log_likelihood, method, copula_name, success)
├── MLEResult (copula_param: float)
├── LatentResult (params: LatentProcessParams, K, grid_range, ...)
├── GASResult (params: LatentProcessParams, scaling: str)
└── IndependentResult ()
```

All are `@dataclass(frozen=True)` — immutable after creation.

### NumericalConfig

All magic numbers in one place:

```python
from pyscarcopula._types import NumericalConfig

# Defaults
cfg = NumericalConfig()  # K=300, grid_range=5.0, pts_per_sigma=4, ...

# Override for speed
cfg = NumericalConfig(default_K=150, default_tol_scar=5e-2)
```

## Strategy pattern

Adding a new estimation method:

```python
# strategy/scar_levy.py
from pyscarcopula.strategy._base import register_strategy
from pyscarcopula._types import LatentResult, LatentProcessParams

@register_strategy('SCAR-TM-LEVY')
class SCARLevyStrategy:
    def __init__(self, config=None, **kwargs):
        ...

    def fit(self, copula, u, **kwargs):
        # ... your Lévy-specific TM code ...
        params = LatentProcessParams(
            process_type='levy',
            names=('alpha', 'beta', 'mu', 'sigma'),
            values=fitted_values,
        )
        return LatentResult(
            log_likelihood=..., method='SCAR-TM-LEVY',
            copula_name=copula.name, success=True,
            params=params, K=K,
        )

    def smoothed_params(self, copula, u, result):
        ...
    def mixture_h(self, copula, u, result):
        ...
```

After this, `fit(copula, data, method='scar-tm-levy')` works automatically.

## Vine forward pass

The C-vine tree traversal is defined once in `vine/forward_pass.py`:

```python
from pyscarcopula.vine import vine_forward_iter

for step in vine_forward_iter(u, edges, h_func):
    # step.tree, step.edge_idx, step.u1, step.u2, step.u_pair
    edge = edges[step.tree][step.edge_idx]
    total_ll += compute_ll(edge, step.u_pair)
```

This replaces the 4 copies of the nested loop that existed in `fit()`, `log_likelihood()`, `sample()`, and `vine_rosenblatt_transform()`.

## Migration status

| Component | Old location | New location | Status |
|-----------|-------------|-------------|--------|
| Fit results | Dynamic attrs on OptimizeResult | `_types.py` dataclasses | ✅ Added |
| Strategy dispatch | `if/elif` in `base.py` | `strategy/` modules | ✅ Added |
| OU kernels | `ou_process.py` lines 28-122 | `numerical/ou_kernels.py` | ✅ Extracted |
| TM grid | `ou_process.py` lines 278-554 | `numerical/tm_grid.py` | ✅ Extracted |
| TM gradient | `ou_process.py` lines 735-979 | `numerical/tm_gradient.py` | ✅ Extracted |
| MC samplers | `ou_process.py` lines 124-275 | `numerical/mc_samplers.py` | ✅ Extracted |
| Vine traversal | 4 copies in vine.py/stattests.py | `vine/forward_pass.py` | ✅ Added |
| Top-level API | `copula.fit()` (mutating) | `api.fit()` (stateless) | ✅ Added |
| Old `BivariateCopula.fit()` | `copula/base.py` | Unchanged (backward compat) | 🔄 Legacy |
| Old `OULatentProcess` | `latent/ou_process.py` | Unchanged (backward compat) | 🔄 Legacy |

Legacy code remains functional. The new modules are purely additive and can be used alongside the old API.
