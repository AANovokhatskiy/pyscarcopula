# Performance Tuning

This page summarizes the knobs that affect fitting speed and optimizer
stability. The same strategy options are used by standalone bivariate copulas
and by pair-copula edges inside C-vines and R-vines.

## Bivariate Models

All bivariate fits go through the strategy registry:

```python
from pyscarcopula import JoeCopula
from pyscarcopula.api import fit

copula = JoeCopula(rotate=180)

# Default GAS fit.
result = fit(copula, u, method='gas')

# Tighten or relax optimizer controls for this run.
result = fit(
    copula,
    u,
    method='gas',
    gamma_bound=30.0,
    beta_bound=0.995,
    gtol=1e-4,
    ftol=1e-12,
    maxfun=3000,
)
```

The object method is equivalent:

```python
result = copula.fit(u, method='scar-tm-ou', K=500, gtol=5e-3)
```

### MLE

MLE estimates one constant copula parameter.

| Parameter | Where | Default | Effect |
|-----------|-------|---------|--------|
| `alpha0` | fit kwarg | auto | Initial point in transformed parameter space. |
| `gtol` | fit kwarg / `mle_optimizer.gtol` | `1e-3` | L-BFGS-B projected-gradient tolerance. |
| `maxls` | fit kwarg / `mle_optimizer.maxls` | `20` | Maximum L-BFGS-B line-search steps per iteration. |

```python
from pyscarcopula._types import LBFGSBConfig, NumericalConfig

cfg = NumericalConfig(mle_optimizer=LBFGSBConfig(gtol=1e-6))
result = fit(copula, u, method='mle', config=cfg)
```

### GAS

GAS estimates an observation-driven recursion
`g_t = omega + beta * g_{t-1} + gamma * score_{t-1}`.

| Parameter | Where | Default | Effect |
|-----------|-------|---------|--------|
| `gamma0` | fit kwarg | MLE-based | Initial `[omega, gamma, beta]`. |
| `gtol` | fit kwarg / `gas_optimizer.gtol` | `1e-3` | L-BFGS-B projected-gradient tolerance. |
| `ftol` | fit kwarg / `gas_optimizer.ftol` | `1e-12` | Relative objective decrease tolerance. Use a tight value to avoid premature FACTR convergence. |
| `maxfun` | fit kwarg / `gas_optimizer.maxfun` | `1000` | Maximum function evaluations. |
| `maxiter` | fit kwarg / `gas_optimizer.maxiter` | `1000` | Maximum optimizer iterations. |
| `maxls` | fit kwarg / `gas_optimizer.maxls` | `20` | Maximum L-BFGS-B line-search steps per iteration. |
| `eps` | fit kwarg / `gas_optimizer.eps` | `1e-5` | L-BFGS-B finite-difference step. |
| `score_eps` | fit kwarg / `gas_score_eps` | `1e-4` | Finite-difference step for score calculations where needed. |
| `gamma_bound` | fit kwarg / `gas_gamma_bound` | `20.0` | Bounds score sensitivity to `[-gamma_bound, gamma_bound]`. |
| `beta_bound` | fit kwarg / `gas_beta_bound` | `0.999` | Bounds persistence to `[-beta_bound, beta_bound]`; must be in `(0, 1)`. |
| `scaling` | strategy kwarg | `'unit'` | Score scaling mode; `'fisher'` is available but less stable. |

```python
result = fit(
    JoeCopula(rotate=180),
    u,
    method='gas',
    scaling='unit',
    ftol=1e-12,
    maxfun=3000,
)
```

`ftol` matters for GAS because L-BFGS-B can otherwise report `success=True`
after a small relative objective decrease even when the gradient is still
large. If a GAS result looks sensitive to `gamma_bound` even though the fitted
`gamma` is far from the bound, rerun with tighter `ftol` and larger `maxfun`.

### SCAR-TM-OU

SCAR-TM-OU uses a deterministic transfer-matrix likelihood for an OU latent
state. It is usually the best stochastic model when reproducible likelihoods
and predictive paths are needed.

| Parameter | Where | Default | Effect |
|-----------|-------|---------|--------|
| `alpha0` | fit kwarg | smart/MLE-based | Initial `[kappa, mu, nu]`. |
| `gtol` | fit kwarg / `scar_optimizer.gtol` | `1e-3` | L-BFGS-B projected-gradient tolerance. Larger values are faster but less precise. |
| `maxfun` | fit kwarg / `scar_optimizer.maxfun` | `100` | Maximum function evaluations. |
| `maxiter` | fit kwarg / `scar_optimizer.maxiter` | `100` | Maximum optimizer iterations. |
| `maxls` | fit kwarg / `scar_optimizer.maxls` | `20` | Maximum L-BFGS-B line-search steps per iteration. |
| `eps` | fit kwarg / `scar_optimizer.eps` | `1e-4` | L-BFGS-B finite-difference step for numerical-gradient fits. |
| `K` | strategy kwarg / `default_K` | `300` | Minimum latent grid size. May be increased by the adaptive rule. |
| `grid_range` | strategy kwarg / `default_grid_range` | `5.0` | Grid spans `[-grid_range*sigma, +grid_range*sigma]`. |
| `grid_method` | strategy kwarg / `default_grid_method` | `'auto'` | `'auto'`, `'dense'`, or `'sparse'`. Use sparse for large grids. |
| `adaptive` | strategy kwarg / `default_adaptive` | `True` | Enlarges `K` when the OU transition kernel needs more resolution. |
| `pts_per_sigma` | strategy kwarg / `default_pts_per_sigma` | `4` | Minimum grid points per conditional standard deviation. |
| `analytical_grad` | strategy kwarg | `True` | Uses analytical gradient and parameter rescaling. Usually much faster. |
| `smart_init` | strategy kwarg | `True` | Uses a heuristic initial point before falling back to MLE-based init. |

```python
result = fit(
    copula,
    u,
    method='scar-tm-ou',
    K=500,
    grid_method='sparse',
    gtol=5e-3,
    analytical_grad=True,
)
```

When `adaptive=True`, the grid is enlarged so the OU transition kernel is
resolved with at least `pts_per_sigma` points per conditional standard
deviation. For slow mean reversion this can produce large grids. If that is too
expensive, use `grid_method='sparse'`, reduce `pts_per_sigma`, or set
`adaptive=False` with an explicit `K`.

### SCAR-MC

The Monte Carlo SCAR strategies are stochastic likelihood estimators.

| Parameter | Where | Default | Effect |
|-----------|-------|---------|--------|
| `n_tr` | strategy kwarg / `default_n_tr` | `500` | Number of Monte Carlo trajectories. |
| `M_iterations` | strategy kwarg | `3` | EIS iterations for `scar-m-ou`. |
| `stationary` | strategy kwarg | `True` | Initializes the OU process in stationarity. |
| `seed` / `dwt` | fit kwarg | random | Controls Wiener increments for reproducibility. |
| `gtol` | fit kwarg / `scar_optimizer.gtol` | `1e-3` | L-BFGS-B projected-gradient tolerance. |
| `maxfun` | fit kwarg / `scar_optimizer.maxfun` | `100` | Maximum function evaluations. |
| `maxiter` | fit kwarg / `scar_optimizer.maxiter` | `100` | Maximum optimizer iterations. |
| `maxls` | fit kwarg / `scar_optimizer.maxls` | `20` | Maximum L-BFGS-B line-search steps per iteration. |
| `eps` | fit kwarg / `scar_optimizer.eps` | `1e-4` | L-BFGS-B finite-difference step. |

```python
result = fit(
    copula,
    u,
    method='scar-m-ou',
    n_tr=1000,
    M_iterations=5,
    seed=123,
)
```

## NumericalConfig

Use `NumericalConfig` when a setting should apply to many fits:

```python
from pyscarcopula._types import LBFGSBConfig, NumericalConfig

cfg = NumericalConfig(
    gas_optimizer=LBFGSBConfig(
        gtol=1e-4,
        ftol=1e-12,
        maxfun=3000,
        maxiter=3000,
        maxls=50,
    ),
    scar_optimizer=LBFGSBConfig(
        gtol=1e-4,
        maxls=50,
    ),
    default_K=500,
)

result = fit(copula, u, method='gas', config=cfg)
```

Per-call keyword arguments override the config values for that fit.

## C-Vines

`CVineCopula.fit` selects a family for each edge using an MLE screening/refine
step, then optionally refits selected edges with the requested dynamic method.
All remaining keyword arguments are forwarded to the pair-copula strategy for
the dynamic refit.

```python
from pyscarcopula import CVineCopula
from pyscarcopula._types import LBFGSBConfig, NumericalConfig

cfg = NumericalConfig(
    gas_optimizer=LBFGSBConfig(ftol=1e-12, maxfun=3000, maxiter=3000))

vine = CVineCopula()
vine.fit(
    u,
    method='gas',
    config=cfg,
    gamma_bound=30.0,
    ftol=1e-12,
    truncation_level=2,
    min_edge_logL=10.0,
)
```

For `method='mle'`, the edge stays at the MLE selection result. For dynamic
methods such as `'gas'` or `'scar-tm-ou'`, these controls are forwarded to each
dynamic edge fit:

- GAS: `gamma0`, `gtol`, `ftol`, `maxfun`, `maxiter`, `maxls`, `eps`, `score_eps`,
  `gamma_bound`, `beta_bound`, `scaling`, `verbose`
- SCAR-TM-OU: `alpha0`, `gtol`, `ftol`, `maxfun`, `maxiter`, `maxls`, `eps`, `K`, `grid_range`, `grid_method`, `adaptive`,
  `pts_per_sigma`, `analytical_grad`, `smart_init`, `verbose`
- SCAR-MC: `alpha0`, `gtol`, `ftol`, `maxfun`, `maxiter`, `maxls`, `eps`, `n_tr`, `M_iterations`, `stationary`, `seed`,
  `dwt`, `smart_init`, `verbose`

Vine-level pruning controls reduce the number of dynamic edge fits:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `truncation_level` | `None` | Tree levels `>= truncation_level` stay MLE. Useful for high-dimensional vines. |
| `min_edge_logL` | `None` | Edges with MLE log-likelihood below the threshold stay MLE. |
| `transform_type` | `'softplus'` | Parameter transform used for Archimedean candidate copulas. |

In C-vines, `config` is accepted through `**kwargs` and is passed to the
dynamic edge refit. The initial MLE family-selection stage uses its own fast
MLE path.

## R-Vines

`RVineCopula.fit` has an explicit `config` argument and forwards strategy
options to edge fits through the Dissmann selector.

```python
from pyscarcopula import RVineCopula
from pyscarcopula._types import LBFGSBConfig, NumericalConfig

cfg = NumericalConfig(
    gas_optimizer=LBFGSBConfig(ftol=1e-12, maxfun=3000, maxiter=3000))

vine = RVineCopula(
    truncation_level=2,
    truncation_fill='independent',
    threshold=0.02,
    min_edge_logL=5.0,
)

vine.fit(
    u,
    method='gas',
    config=cfg,
    gamma_bound=30.0,
    ftol=1e-12,
)
```

The same strategy options listed for C-vines are forwarded to every
non-independent, non-truncated edge selected for dynamic fitting. R-vine
structure controls are:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `truncation_level` | instance value / `None` | Tree levels `>= truncation_level` are truncated. |
| `truncation_fill` | `'independent'` | Truncated trees become independent edges or MLE-only edges (`'mle'`). |
| `threshold` | `0.0` | Edges with `abs(Kendall tau) < threshold` are made independent before fitting. |
| `min_edge_logL` | `None` | Fitted weak edges below the threshold are replaced by independence. |
| `structure_search` | `'beam'` | Conditional-structure search mode when `given_vars` is used. |
| `beam_width` | `4` | Number of partial candidate structures retained by beam search. |
| `transform_type` | instance value / `'softplus'` | Parameter transform used for Archimedean candidate copulas. |

As with C-vines, automatic family selection is MLE-based. `gtol`, `ftol`,
`gamma_bound`, `K`, and similar strategy controls affect the dynamic edge refit
after a family has been selected. If `method='gas'`, a too-loose `ftol` can make
some edges stop early with `success=True`; set `ftol=1e-12` and increase
`maxfun` for difficult edges.

## BLAS Threads

`pyscarcopula` sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and
`OPENBLAS_NUM_THREADS` to `1` during package import to avoid oversubscription
in transfer-matrix workloads. This only affects BLAS libraries that have not
been initialized yet. If NumPy/SciPy/OpenBLAS was imported before
`pyscarcopula`, set the variables before starting Python:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python script.py
```

On Windows PowerShell:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
python script.py
```

If an application imports NumPy first and still needs to change BLAS threads at
runtime, use a runtime thread limiter such as `threadpoolctl`.
