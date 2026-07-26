# pyscarcopula: copula Python library

**A Python library for bivariate, multivariate, vine, and stochastic copula models.**

pyscarcopula models dependence between variables in Python for financial time
series, risk analytics, and experiments with dynamic dependence. Alongside
classical constant-parameter copulas, it supports SCAR models where the copula
parameter follows an Ornstein-Uhlenbeck latent process or where Kendall's tau
follows a bounded Jacobi diffusion estimated via a deterministic transfer
matrix method.

## Key Features

- **Archimedean copulas**: Gumbel, Frank, Clayton, Joe (with rotations)
- **Elliptical copulas**: Gaussian, Student-t
- **Multivariate models**: Gaussian, Student-t, equicorrelation Gaussian,
  and stochastic Student-t
- **C-vine copulas**: automatic family selection, truncation, mixed SCAR/MLE
- **R-vine conditional sampling**: exact and approximate conditional
  prediction modes
- **Explicit CPU parallelism**: native threads for eligible multivariate
  kernels and process workers for independent fits, with an absolute
  one-thread default
- **Estimation**: MLE, GAS, SCAR-TM-OU, SCAR-TM-JACOBI
- **Compiled numerical engine** included in official wheels
- **Prediction controls**: `PredictConfig`, diagnostics, dynamic conditioning,
  reproducible `rng`
- **Transform functions**: `softplus` (default), `xtanh` (symmetric)
- **Diagnostics**: GoF test, predictive mean parameter paths

For native threading, rolling-window safety, external process workers, and
large-dimension limits, see [CPU Parallelism](guide/parallelism.md).

## Quick Example

```python
from pyscarcopula import GumbelCopula
from pyscarcopula.api import fit, predictive_mean
from pyscarcopula.stattests import gof_test
import numpy as np

source = GumbelCopula(rotate=180)
u = source.sample_at_parameter(
    400,
    r=np.full(400, 1.8),
    rng=np.random.default_rng(2026),
)
copula = GumbelCopula(rotate=180)

result = fit(copula, u, method='scar-tm-ou')
print(f"logL = {result.log_likelihood:.2f}")

gof = gof_test(copula, u, fit_result=result, to_pobs=False)
print(f"GoF p-value = {gof.pvalue:.4f}")

r_t = predictive_mean(copula, u, result)
```

## Where to Go Next

- [Install pyscarcopula](getting-started/installation.md) and run the
  [end-to-end quick start](getting-started/quickstart.md).
- Use [Choosing a Model](getting-started/choosing-a-model.md) to select between
  bivariate, multivariate, factor, and vine models.
- Read [Estimation Methods](guide/estimation-methods.md) before comparing MLE,
  GAS, and SCAR fits.
- Read [Prediction Semantics](guide/prediction-semantics.md) before using
  conditional or dynamic forecasts.
- Go directly to the [API Reference](api/copulas.md) when you already know the
  model and operation you need.
