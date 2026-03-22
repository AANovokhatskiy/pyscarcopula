# pyscarcopula

**Stochastic copula models with Ornstein-Uhlenbeck latent process.**

pyscarcopula models time-varying dependence between financial assets. The copula parameter follows a latent stochastic process, estimated via a deterministic transfer matrix method — no Monte Carlo required.

## Key Features

- **Archimedean copulas**: Gumbel, Frank, Clayton, Joe (with rotations)
- **Elliptical copulas**: Gaussian, Student-t
- **Equicorrelation Gaussian copula**: single dynamic correlation for d assets
- **C-vine copulas**: automatic family selection, truncation, mixed SCAR/MLE
- **Estimation**: MLE, GAS, SCAR-TM-OU (transfer matrix with analytical gradient)
- **Transform functions**: `xtanh` (default), `softplus` (asymmetric)
- **Diagnostics**: GoF test, smoothed parameters, predictive distribution

## Quick Example

```python
from pyscarcopula import GumbelCopula
from pyscarcopula.stattests import gof_test
from pyscarcopula.utils import pobs

u = pobs(returns)

copula = GumbelCopula(rotate=180, transform_type='softplus')
copula.fit(u, method='scar-tm-ou')

print(f"logL = {copula.fit_result.log_likelihood:.2f}")
gof = gof_test(copula, u, to_pobs=False)
print(f"GoF p-value = {gof.pvalue:.4f}")
```

## Comparison on BTC-ETH daily data (T=1460)

| Model | logL | GoF p-value |
|-------|------|-------------|
| MLE (constant) | 955.63 | 0.009 |
| GAS (score-driven) | 1031.42 | 0.528 |
| SCAR-TM xtanh | 1042.47 | 0.620 |
| **SCAR-TM softplus** | **1045.78** | **0.737** |
