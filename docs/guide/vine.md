# Vine Copulas

## C-vine structure

A C-vine decomposes a $d$-dimensional copula into $d(d-1)/2$ bivariate edge copulas arranged in $d-1$ trees:

- **Tree 0**: pairs $(1,2), (1,3), \ldots, (1,d)$ — unconditional
- **Tree 1**: pairs $(2,3|1), (2,4|1), \ldots$ — conditional on variable 1
- **Tree $j$**: conditional on variables $1, \ldots, j$

Each edge copula can be from a different family, selected automatically via AIC.

## Basic usage

```python
from pyscarcopula import CVineCopula
from pyscarcopula.utils import pobs

u = pobs(returns)  # (T, d) pseudo-observations

vine = CVineCopula()
vine.fit(u, method='scar-tm-ou')
vine.summary()

print(vine.fit_result.log_likelihood)
print(vine.fit_result.name)  # "C-vine (6d, 15 edges)"
```

## Truncation

For large $d$, not all edges benefit from dynamic parameters. Two truncation mechanisms reduce computation:

### Tree-level truncation

Edges at tree level $\geq k$ stay MLE (constant parameter):

```python
vine.fit(u, method='scar-tm-ou', truncation_level=2)
# Trees 0-1: SCAR-TM-OU
# Trees 2+: MLE
```

### Edge-level truncation

Edges with MLE log-likelihood below a threshold stay MLE:

```python
vine.fit(u, method='scar-tm-ou', min_edge_logL=10)
# Strong edges: SCAR-TM-OU
# Weak edges (logL < 10): MLE
```

Both can be combined:

```python
vine.fit(u, method='scar-tm-ou', 
         truncation_level=2, min_edge_logL=10)
```

### Independence copula

Edges where no parametric copula beats independence by AIC are automatically set to `IndependentCopula` ($c(u_1,u_2)=1$, $h(u_2|u_1)=u_2$). These edges have zero computational cost.

## Goodness of fit

The GoF test works on mixed SCAR/MLE vines. Each edge uses its own h-function: mixture h for SCAR edges, constant-parameter h for MLE edges.

```python
from pyscarcopula.stattests import gof_test

gof = gof_test(vine, u, to_pobs=False)
print(f"p-value = {gof.pvalue:.4f}")
```

## Results on 6-crypto data (T=250)

| Model | logL | GoF p-value | Fit time |
|-------|------|-------------|----------|
| **C-vine SCAR-TM** | **921.9** | **0.90** | 13s |
| C-vine MLE | 869.2 | 0.21 | 0.6s |
| Student-t | 764.4 | 0.0001 | — |
| Gaussian | 761.0 | 0.0000 | — |

## Sampling

```python
# Predict: sample from x_T distribution
samples = vine.predict(n=10000)

# Sample with training data (for SCAR: uses last smoothed params)
samples = vine.sample(n=10000, u_train=u)
```
