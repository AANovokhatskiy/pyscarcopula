# Performance Tuning

## Bivariate copula

The default settings (`analytical_grad=True`, `smart_init=True`) are optimized for most use cases. For fine-tuning:

```python
copula.fit(u, method='scar-tm-ou',
           analytical_grad=True,   # analytical gradient (default)
           smart_init=True,        # heuristic initial point (default)
           tol=1e-2,               # gradient tolerance
           K=300,                  # minimum grid size
           pts_per_sigma=2,        # grid density
)
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `analytical_grad` | `True` | Analytical gradient of the TM log-likelihood. ~3-4x fewer function evaluations than numerical gradient. |
| `smart_init` | `True` | Initial point from MLE heuristic + L-BFGS-B parameter rescaling. Up to 5x speedup on long series. |
| `tol` | `1e-2` | L-BFGS-B gradient tolerance. `5e-2` gives ~2x speedup with negligible logL loss — recommended for vine. |
| `K` | `300` | Minimum grid size. The adaptive rule may increase it to resolve the transition kernel. |
| `pts_per_sigma` | `2` | Grid points per conditional $\sigma_c$. The adaptive rule sets $K_\text{eff} = \max(K, \lceil 2R\sigma / (\sigma_c / \text{pts\_per\_sigma}) \rceil)$. Increase to 4 for very peaked kernels (large $\theta$, small $\nu$). |

## Vine copula

For vine models with many edges, truncation is the most effective optimization:

```python
vine.fit(u, method='scar-tm-ou',
         truncation_level=2,    # trees ≥ 2 stay MLE
         min_edge_logL=10,      # weak edges stay MLE
         tol=5e-2,              # relaxed tolerance
)
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `truncation_level` | `None` | Trees at level ≥ this stay MLE. Recommended: 2-3 for $d > 10$. |
| `min_edge_logL` | `None` | Edges with MLE logL below threshold stay MLE. Recommended: 5-10. |

Truncated edges contribute MLE log-likelihood and use constant-parameter h-functions. The GoF test handles mixed models correctly.

**d=6 crypto, T=250:**

| Configuration | Time | logL | GoF p-value |
|---|---|---|---|
| `truncation_level=2, min_edge_logL=10` | ~13s | ~922 | ~0.90 |
| Full SCAR (15 edges) | ~30s | ~891 | ~0.90 |
| MLE only | ~0.6s | ~869 | ~0.21 |

!!! note
    Truncation can sometimes *improve* logL compared to full SCAR by preventing overfitting on weak edges.

## BLAS thread control

The package sets `OMP_NUM_THREADS=1` at import to prevent thread oversubscription when using parallel computation. Override before importing if needed:

```python
import os
os.environ['OMP_NUM_THREADS'] = '4'
import pyscarcopula
```
