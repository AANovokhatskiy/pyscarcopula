# Installation

## From PyPI

```bash
pip install pyscarcopula
```

## From source (for development)

```bash
git clone https://github.com/AANovokhatskiy/pyscarcopula
cd pyscarcopula
pip install -e ".[test]"
```

## Run tests

```bash
pytest tests/
```

Tests require the `data/` directory, which is included in the git repository but not in the PyPI package.

## Dependencies

| Package | Min version | Purpose |
|---------|-------------|---------|
| numpy | 1.22 | Arrays, linear algebra |
| numba | 0.56 | JIT-compiled copula kernels |
| scipy | 1.9 | Optimization (L-BFGS-B), sparse matrices |
| joblib | 1.0 | Parallel computation |
| tqdm | 4.0 | Progress bars |

## Python version

Python 3.9 or newer is required. Tested on 3.9-3.13.
