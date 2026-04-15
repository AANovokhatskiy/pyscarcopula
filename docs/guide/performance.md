# Performance Tuning

## Bivariate copula

```python
from pyscarcopula import GumbelCopula
from pyscarcopula.api import fit

copula = GumbelCopula(rotate=180)
result = fit(copula, u, method='scar-tm-ou')

# Relaxed tolerance (faster)
result = fit(copula, u, method='scar-tm-ou', tol=5e-2)
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `analytical_grad` | `True` | Uses analytical gradient. About 3-4x fewer function evaluations. |
| `smart_init` | `True` | Heuristic initial point. Up to 5x speedup on long series. |
| `tol` | `1e-2` | L-BFGS-B gradient tolerance. `5e-2` is about 2x faster with negligible logL loss. |
| `K` | `300` | Minimum grid size. May be auto-increased for adequate resolution. |
| `pts_per_sigma` | `2` | Grid density. Increase to 4 for very peaked transition kernels. |
| `transform_type` | `'xtanh'` | `'softplus'` often gives better logL on financial data. |

**Adaptive grid size.** When `adaptive=True` (the default), the grid is automatically enlarged so that the OU transition kernel is resolved with at least `pts_per_sigma` points per conditional standard deviation. The resulting grid size is

`K_min = O(1 / sqrt(theta * dt))`, where `dt = 1/(n-1)`.

For small `theta` (slow mean-reversion) this can produce very large grids. In dense mode the cost is `O(K^2)`; in sparse mode it is `O(K * b)`, where `b` is the kernel bandwidth, which is usually manageable. Workarounds for excessive grid sizes:

- Use `grid_method='sparse'` or leave `'auto'`, which prefers sparse for large grids
- Set `adaptive=False` and specify an explicit `K`
- Increase `theta` or decrease `n` (fewer observations -> larger `dt` -> wider conditional kernel -> smaller grid)

## Vine copula

```python
from pyscarcopula import CVineCopula

vine = CVineCopula()
vine.fit(u, method='scar-tm-ou',
         truncation_level=2,
         min_edge_logL=10,
         tol=5e-2)
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `truncation_level` | `None` | Trees at level >= this stay MLE. Recommended: 2-3 for d > 10. |
| `min_edge_logL` | `None` | Edges with MLE logL below threshold stay MLE. Recommended: 5-10. |

**d=6 crypto, T=250:**

| Configuration | Time | logL | GoF p-value |
|---|---|---|---|
| Full SCAR (15 edges) | ~30s | ~928 | ~0.90 |
| Truncated SCAR (level=2, logL>=10) | ~13s | ~922 | ~0.90 |
| MLE only | ~0.6s | ~869 | ~0.21 |

## NumericalConfig

Override default numerical parameters:

```python
from pyscarcopula._types import NumericalConfig

cfg = NumericalConfig(default_K=500, default_tol_scar=5e-2)
result = fit(copula, u, method='scar-tm-ou', config=cfg)
```
