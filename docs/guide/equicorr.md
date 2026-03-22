# Equicorrelation Model

## Idea

For d assets, the standard Gaussian copula has d(d-1)/2 correlation parameters — all static. The equicorrelation model uses a single dynamic correlation:

$$R(t) = (1-\rho(t)) \cdot I + \rho(t) \cdot \mathbf{1}\mathbf{1}^\top$$

All pairwise correlations equal ρ(t), which follows an OU process via SCAR. This gives 3 parameters instead of d(d-1)/2, while capturing the dominant risk-on/risk-off dynamics.

## Usage

```python
from pyscarcopula.copula.equicorr import EquicorrGaussianCopula

cop = EquicorrGaussianCopula(d=6)

# MLE (constant rho)
cop.fit(u, method='mle')

# SCAR (time-varying rho)
cop.fit(u, method='scar-tm-ou')

print(cop.fit_result.log_likelihood)
print(cop.fit_result.alpha)  # (theta, mu, nu)
```

## Goodness of fit

```python
from pyscarcopula.stattests import gof_test
gof = gof_test(cop, u, to_pobs=False)
```

The GoF test uses the mixture Rosenblatt transform with sequential conditioning for the equicorrelation structure.

## Sampling

```python
samples = cop.predict(n=10000)      # from stationary OU
samples = cop.sample(n=10000, r=0.5)  # at fixed rho
```

## When to use

Equicorrelation SCAR is a good fit when:

- All pairwise correlations move together (common in equity/crypto markets)
- You need fast estimation for large d (O(d) per density evaluation)
- You want a simple interpretable model (3 parameters)

For heterogeneous dependence (different pairs have different dynamics), use C-vine instead.

## Comparison on 6-crypto data (T=250)

| Model | logL | GoF p-value | Parameters |
|-------|------|-------------|------------|
| C-vine SCAR-TM | 921.9 | 0.89 | 15×3 |
| **Equicorr SCAR** | **814.0** | **0.04** | **3** |
| Student-t | 764.4 | 0.00 | 16 |
| Gaussian | 761.0 | 0.00 | 15 |
