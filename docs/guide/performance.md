# Performance Tuning

## Bivariate copula

```python
copula.fit(u, method='scar-tm-ou',
           analytical_grad=True,   # default
           smart_init=True,        # default
           tol=1e-2,               # gradient tolerance
           K=300,                  # minimum grid size
           pts_per_sigma=2,        # grid density
           transform_type='softplus',  # or 'xtanh'
)
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `analytical_grad` | `True` | Uses analytical gradient. ~3-4x fewer function evaluations. |
| `smart_init` | `True` | Heuristic initial point. Up to 5x speedup on long series. |
| `tol` | `1e-2` | L-BFGS-B gradient tolerance. `5e-2` is ~2x faster with negligible logL loss. |
| `K` | `300` | Minimum grid size. May be auto-increased for adequate resolution. |
| `pts_per_sigma` | `2` | Grid density. Increase to 4 for very peaked transition kernels. |
| `transform_type` | `'xtanh'` | `'softplus'` often gives better logL on financial data. |

## Vine copula

```python
vine.fit(u, method='scar-tm-ou',
         truncation_level=2,    # trees >= 2 stay MLE
         min_edge_logL=10,      # weak edges stay MLE
         tol=5e-2,              # relaxed tolerance for speed
         transform_type='softplus',
)
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `truncation_level` | `None` | Trees at level >= this stay MLE. Recommended: 2-3 for d > 10. |
| `min_edge_logL` | `None` | Edges with MLE logL below threshold stay MLE. Recommended: 5-10. |

**d=6 crypto, T=250:**

| Configuration | Time | logL | GoF p-value |
|---|---|---|---|
| Full SCAR (15 edges) | ~30s | ~928 | ~0.90 |
| Truncated SCAR (level=2, logL≥10) | ~13s | ~922 | ~0.90 |
| MLE only | ~0.6s | ~869 | ~0.21 |

## BLAS thread control

The package sets `OMP_NUM_THREADS=1` at import to prevent thread oversubscription. Override before importing if needed:

```python
import os
os.environ['OMP_NUM_THREADS'] = '4'
import pyscarcopula
```
