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

Official wheels contain the required native extension. It provides built-in
copula kernels, static likelihoods, GAS, and SCAR-TM-OU evaluation.

Source installs build this extension and fail if it cannot be compiled. You
need a C++17 compiler available to `setuptools`/`pybind11`: MSVC Build Tools on
Windows, Xcode Command Line Tools on macOS, or GCC/Clang on Linux.

SCAR-TM-OU and GAS always use the native evaluator. They have no backend
selector or Python fallback. Custom Python copulas may still be used by custom
Python strategies and utilities, but unsupported classes are rejected by
built-in native strategies.

Verify the installed wheel or source build:

```bash
python -m pyscarcopula._native_smoke
```

## Run tests

```bash
pytest tests/
```

Tests require the `data/` directory, which is included in the git repository
but not in the PyPI package. Native tests require a successful extension
build.

For a source-tree C++ check, build the extension in place first:

```bash
python setup.py build_ext --inplace
pytest tests/test_cpp.py
```

## Dependencies

| Package | Min version | Purpose |
|---------|-------------|---------|
| numpy | 1.22 | Arrays, linear algebra |
| numba | 0.56 | Retained Python analytics, GoF, and MC/EIS helpers |
| scipy | 1.9 | Optimization (L-BFGS-B), sparse matrices |
| joblib | 1.0 | Parallel computation |
| tqdm | 4.0 | Progress bars |

## Python version

Python 3.10 or newer is required. Tested on 3.10-3.14.
