# pyscarcopula: copula Python library

**A Python library for bivariate, multivariate, vine, and stochastic copula models.**

pyscarcopula models dependence between variables in Python for financial time
series, risk analytics, and experiments with dynamic dependence. Alongside
classical constant-parameter copulas, it supports SCAR models where the copula
parameter follows a latent Ornstein-Uhlenbeck stochastic process estimated via
a deterministic transfer matrix method.

## Key Features

- **Archimedean copulas**: Gumbel, Frank, Clayton, Joe (with rotations)
- **Elliptical copulas**: Gaussian, Student-t
- **Experimental models**: equicorrelation Gaussian, stochastic Student-t, stochastic Student-t DCC
- **C-vine copulas**: automatic family selection, truncation, mixed SCAR/MLE
- **R-vine conditional sampling**: exact suffix/rebuild path plus arbitrary
  runtime-DAG + MCMC fallback
- **Estimation**: MLE, GAS, SCAR-TM-OU (transfer matrix with analytical gradient)
- **Prediction controls**: `PredictConfig`, diagnostics, dynamic conditioning,
  reproducible `rng`
- **Transform functions**: `xtanh` (default), `softplus` (asymmetric)
- **Diagnostics**: GoF test, smoothed parameters

## Quick Example

```python
from pyscarcopula import GumbelCopula
from pyscarcopula.api import fit, smoothed_params
from pyscarcopula.stattests import gof_test
from pyscarcopula._utils import pobs

u = pobs(returns)
copula = GumbelCopula(rotate=180)

result = fit(copula, u, method='scar-tm-ou')
print(f"logL = {result.log_likelihood:.2f}")

gof = gof_test(copula, u, fit_result=result, to_pobs=False)
print(f"GoF p-value = {gof.pvalue:.4f}")

r_t = smoothed_params(copula, u, result)
```

## Comparison on BTC-ETH daily data (T=1460)

| Model | logL | GoF p-value |
|-------|------|-------------|
| MLE (constant) | 955.63 | 0.0105 |
| GAS (score-driven) | 1031.42 | 0.5187 |
| **SCAR-TM** | **1042.47** | **0.6544** |
