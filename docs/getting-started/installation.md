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

Official wheels contain the compiled extension used for built-in copula
families, static likelihoods, GAS, and SCAR-TM-OU evaluation.

Source installs build this extension and fail if it cannot be compiled. You
need a C++17 compiler: MSVC Build Tools on Windows, Xcode Command Line Tools
on macOS, or GCC/Clang on Linux.

SCAR-TM-OU and GAS require compiled support for the selected family. Custom
Python copulas may still be used by custom Python strategies and utilities.

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
| numba | 0.56 | Python analytics, GoF, and MC/EIS helpers |
| scipy | 1.9 | Optimization (L-BFGS-B), sparse matrices |
| joblib | 1.0 | Parallel computation |
| tqdm | 4.0 | Progress bars |

## Python version

Python 3.10 or newer is required. Tested on 3.10-3.14.
