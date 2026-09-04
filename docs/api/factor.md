# Factor Operators API

## Factor correlation operator

`FactorCorrelation` stores a correlation matrix as
$R=D+BB^\top$ and prepares a matrix-free Woodbury operator.
`FactorStudentEvaluator` combines that operator with Student copula
likelihoods. `StochasticStudentCopula(corr_mode="factor")` exposes the same
representation through the model API.

Prepared operators and evaluators require real-valued observations, loadings,
parameters, and draws; complex arrays are rejected before conversion.
Dimensions, ranks, row indices, and factor-initialization integer options
require Python or NumPy integers, excluding booleans. Conditional samplers
validate the supplied correlation, Student degrees of freedom, and thread
count even when every coordinate is fixed by `given`.

```python
import numpy as np
from pyscarcopula import FactorCorrelation, FactorStudentEvaluator

rng = np.random.default_rng(2026)
B = rng.normal(scale=0.05, size=(20, 3))
u = rng.uniform(0.01, 0.99, size=(50, 20))
factor = FactorCorrelation(B, uniqueness_min=1e-8)
operator = factor.prepare()
evaluation = FactorStudentEvaluator(operator, u).evaluate(
    df=7.0,
    n_threads=4,
)
```

For construction, fitting, batching, conditional sampling, persistence,
complexity, and memory limits, see
[Factor Models](../guide/factor-models.md). The generated reference below is
the canonical source for method signatures and defaults.

### API

::: pyscarcopula.copula.multivariate.factor_correlation.FactorCorrelation

::: pyscarcopula.copula.multivariate.factor_correlation.PreparedFactorCorrelation

::: pyscarcopula.copula.multivariate.factor_student.FactorStudentEvaluator

::: pyscarcopula.copula.multivariate.factor_student.FactorStudentEvaluation

::: pyscarcopula.copula.multivariate.factor_student.FactorStudentGridEvaluation

::: pyscarcopula.copula.multivariate.factor_student.FactorStudentJointEvaluation
