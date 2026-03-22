# Vine Copulas

## C-vine structure

A C-vine decomposes a d-dimensional copula into d(d-1)/2 bivariate copulas:

- **Tree 0**: unconditional pairs
- **Tree 1+**: conditional pairs

Each edge copula is selected automatically from all available families via AIC.

## Usage

```python
from pyscarcopula import CVineCopula

vine = CVineCopula()
vine.fit(u, method='scar-tm-ou',
         truncation_level=2,
         min_edge_logL=10,
         transform_type='softplus')

print(vine.fit_result.log_likelihood)
print(vine.fit_result.name)  # "C-vine (6d, 15 edges)"
vine.summary()
```

## Truncation

For large d, not all edges benefit from dynamic parameters:

```python
# Trees 0-1: SCAR, trees 2+: MLE
vine.fit(u, method='scar-tm-ou', truncation_level=2)

# Edges with weak MLE dependence stay MLE
vine.fit(u, method='scar-tm-ou', min_edge_logL=10)

# Both
vine.fit(u, method='scar-tm-ou',
         truncation_level=2, min_edge_logL=10)
```

Edges where no parametric copula beats independence by AIC are set to `IndependentCopula` automatically.

## Goodness of fit

```python
from pyscarcopula.stattests import gof_test
gof = gof_test(vine, u, to_pobs=False)
```

The GoF test handles mixed SCAR/MLE vines correctly.

## Sampling

```python
samples = vine.predict(n=10000)
samples = vine.sample(n=10000, u_train=u)
```

## Results on 6-crypto data (T=250)

| Model | logL | GoF p-value | Time |
|-------|------|-------------|------|
| **C-vine SCAR-TM** | **921.9** | **0.89** | 13s |
| C-vine MLE | 869.2 | 0.21 | 0.6s |
| Student-t | 764.4 | 0.0001 | — |
| Gaussian | 761.0 | 0.0000 | — |
