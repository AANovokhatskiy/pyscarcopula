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

Source installs build the optional SCAR-TM-OU C++ extension. You need a C++17
compiler available to `setuptools`/`pybind11`: MSVC Build Tools on Windows,
Xcode Command Line Tools on macOS, or GCC/Clang on Linux.

The extension is used automatically by the default `backend='auto'` for
supported SCAR-TM-OU fits. If the extension or model combination is
unavailable, `backend='auto'` falls back to Python/Numba. Use
`backend='python'` to force that path explicitly.

## Run tests

```bash
pytest tests/
```

Tests require the `data/` directory, which is included in the git repository but
not in the PyPI package. C++ backend tests are skipped automatically when the
compiled extension is not available.

For a source-tree C++ check, build the extension in place first:

```bash
python setup.py build_ext --inplace
pytest tests/test_cpp.py
```

## Dependencies

| Package | Min version | Purpose |
|---------|-------------|---------|
| numpy | 1.22 | Arrays, linear algebra |
| numba | 0.56 | JIT-compiled copula kernels |
| scipy | 1.9 | Optimization (L-BFGS-B), sparse matrices |
| joblib | 1.0 | Parallel computation |
| tqdm | 4.0 | Progress bars |

## Python version

Python 3.10 or newer is required. Tested on 3.10-3.13.
