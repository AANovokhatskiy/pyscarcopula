# Contrib API

Optional modules for risk metrics and marginal models. Not part of the core copula API.

```python
from pyscarcopula.contrib.risk_metrics import risk_metrics
from pyscarcopula.contrib.marginal import MarginalModel
```

`risk_metrics(..., n_jobs=...)` is used for both rolling marginal fits and
rolling copula/risk windows. `n_jobs=-1` uses all available workers where the
selected marginal model supports parallel fitting.

## risk_metrics

::: pyscarcopula.contrib.risk_metrics.risk_metrics

## MarginalModel

::: pyscarcopula.contrib.marginal.MarginalModel
    options:
      members:
        - create
        - fit_rolling
        - ppf
