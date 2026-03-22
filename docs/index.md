# pyscarcopula

**Stochastic copula models with Ornstein-Uhlenbeck latent process in Python.**

pyscarcopula fits multivariate distributions using the copula approach with time-varying dependence. The copula parameter follows a latent Ornstein-Uhlenbeck process, estimated via a deterministic transfer matrix method — no Monte Carlo bias.

## Key Features

- **Archimedean copulas**: Gumbel, Frank, Clayton, Joe (with rotations)
- **Elliptical copulas**: Gaussian, Student-t
- **C-vine copulas**: automatic family selection, truncation, mixed SCAR/MLE
- **Transfer matrix method**: exact likelihood, analytical gradient, $O(TKb)$ complexity
- **Diagnostics**: GoF via mixture Rosenblatt transform, smoothed parameters

## Quick Example

```python
from pyscarcopula import GumbelCopula
from pyscarcopula.stattests import gof_test
from pyscarcopula.utils import pobs
import numpy as np

u = pobs(returns)  # pseudo-observations from log-returns

copula = GumbelCopula(rotate=180)
copula.fit(u, method='scar-tm-ou')

print(f"logL = {copula.fit_result.log_likelihood:.2f}")
print(f"alpha = {copula.fit_result.alpha}")

gof = gof_test(copula, u, to_pobs=False)
print(f"GoF p-value = {gof.pvalue:.4f}")
```

## Comparison on BTC-ETH daily data (T=1460)

| Model | logL | GoF p-value |
|-------|------|-------------|
| MLE (constant) | 955.63 | 0.009 |
| GAS (score-driven) | 1031.42 | 0.528 |
| **SCAR-TM (this package)** | **1042.47** | **0.620** |
