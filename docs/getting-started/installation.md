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

The extension's parallel runtime and portable linear-algebra kernels use only
the C++17 standard library. Wheels do not require Eigen, BLAS, or OpenMP and do
not create an additional third-party thread pool. Native threads remain
disabled unless `n_threads` is passed explicitly.

Source installs build this extension and fail if it cannot be compiled. You
need a C++17 compiler: MSVC Build Tools or MinGW-w64 GCC on Windows, Xcode
Command Line Tools on macOS, or GCC/Clang on Linux. MSVC remains the default
Windows toolchain. To opt into MinGW-w64 explicitly (for example from an MSYS2
`ucrt64` shell), use:

```bash
PYSCA_CPP_COMPILER=mingw32 pip install .
# or, from the source tree:
python setup.py build_ext --compiler=mingw32 --inplace
```

MinGW builds link the GCC and winpthreads runtimes statically, so the resulting
extension does not require MSYS2 runtime DLLs when imported.

### C++ build parallelism

C++ source compilation is sequential by default (`1` build job). For a source
or editable install, opt into an explicit positive number of jobs with:

```bash
PYSCA_CPP_BUILD_JOBS=4 pip install .
# or, from the source tree:
python setup.py build_ext --parallel 4 --inplace
```

This uses the existing pybind11 build helper and does not require CMake, Ninja,
OpenMP, or another package. It changes compilation only: linking remains
sequential and the installed extension's runtime thread policy is unaffected.

SCAR-TM-OU and GAS require compiled support for the selected built-in family.
Unknown Python subclasses are rejected by exact-type native dispatch.

Verify the installed wheel or source build:

```bash
python -m pyscarcopula._native.smoke
```

## Run tests

```bash
pytest tests/
```

Tests require the `data/` directory, which is included in the git repository
but not in the PyPI package. Native tests require a successful extension
build.

The default is sequential and uses one pytest worker. After installing the
`test` extra, independent test modules can be distributed over an explicit
number of CPU cores:

```bash
pytest tests/ -n 4
```

Tests from the same module stay on one worker (`--dist=loadscope`) because
some modules intentionally share local runtime state. Timing gates use only
relative comparisons; absolute seconds are report-only. CPU placement remains
under operating-system control. Baseline and candidate timings are collected
in paired, interleaved rounds to avoid a systematic split across different
classes of CPU core.

The same `-n N` mode supports `benchmark`, `validation`, `external`,
`high_dimensional`, `data`, sanitizer, and native-runtime marker selections.
Relative alternatives within one benchmark remain sequential by design, and
module-scoped report writers remain on one worker.

## Run the notebooks

Clone the repository so the example datasets are available, then install the
optional notebook dependencies:

```bash
pip install -e ".[examples,contrib]"
jupyter lab examples/
```

The comparison notebook requires the optional `pyvinecopulib` dependency,
which is not installed with pyscarcopula. Install the pinned comparison dependency before running
`06_pyvinecopulib_comparison.ipynb`:

```bash
pip install -e ".[examples,contrib,external]"
```

The `contrib` extra supplies Numba for the marginal and risk helpers used by
`05_risk_metrics.ipynb`; `examples` alone does not install it.

## Build the documentation

Install the documentation dependencies and run a strict build:

```bash
pip install -e ".[docs]"
mkdocs build --strict
```

The strict build treats unresolved references, invalid navigation entries, and
other MkDocs warnings as errors.

For a source-tree C++ check, build the extension in place first:

```bash
python setup.py build_ext --inplace
pytest tests/test_cpp.py
```

The standalone Python-free C++ boundary check uses the same default and
environment variable, and also accepts a command-line override:

```bash
python tools/build_cpp_tests.py --build-jobs 4
```

## Dependencies

| Package | Min version | Purpose |
|---------|-------------|---------|
| numpy | 1.22 | Arrays, linear algebra |
| scipy | 1.9 | Optimization (L-BFGS-B), sparse matrices |
| joblib | 1.0 | Parallel computation |
| tqdm | 4.0 | Progress bars |

Numba is only used by the optional `contrib` helpers. Install them with
`pip install "pyscarcopula[contrib]"`. The core package, including the C++
pseudo-observation rank transform, does not require Numba.

## Python version

Python 3.10 or newer is required. Tested on 3.10-3.14.
