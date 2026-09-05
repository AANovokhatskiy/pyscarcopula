"""Rank-transform behavior and the Numba-free core import contract."""

import ast
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.stats import rankdata

from pyscarcopula._utils import pobs


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64,
                                  np.uint64, np.bool_])
@pytest.mark.parametrize("layout", ["C", "F", "slice", "reverse"])
def test_ordinal_ranks_match_scipy(dtype, layout):
    data = np.random.default_rng(72).integers(0, 30, size=(101, 6)).astype(dtype)
    if layout == "F":
        data = np.asfortranarray(data)
    elif layout == "slice":
        data = data[::2, ::2]
    elif layout == "reverse":
        data = data[::-1, ::-1]
    before = data.copy()
    expected = rankdata(data, axis=0, method="ordinal").astype(np.float64)
    expected /= len(data) + 1
    result = pobs(data)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(data, before)
    assert result.dtype == np.float64
    assert not np.shares_memory(result, data)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_large_integers_keep_distinct_ranks(dtype):
    maximum = np.iinfo(dtype).max
    data = np.array([[maximum], [maximum - 2], [maximum - 1]], dtype=dtype)
    np.testing.assert_array_equal(pobs(data), [[0.75], [0.25], [0.5]])


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (4, 0), (1, 3)])
def test_empty_and_single_row_arrays(shape):
    result = pobs(np.zeros(shape))
    assert result.shape == shape
    np.testing.assert_array_equal(result, np.full(shape, 0.5))


def test_nonfinite_values_and_ties():
    data = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, np.nan])[:, None]
    expected = np.array([5, 4, 1, 2, 3, 6])[:, None] / 7
    np.testing.assert_array_equal(pobs(data), expected)


@pytest.mark.parametrize("n", [12, 15, 16, 17, 100, 1000])
def test_equal_values_receive_ranks_in_original_row_order(n):
    data = np.ones((n, 3))
    expected = np.broadcast_to(np.arange(1, n + 1)[:, None] / (n + 1), data.shape)
    np.testing.assert_array_equal(pobs(data), expected)
    np.testing.assert_array_equal(pobs(data, ties_method="ordinal"), expected)


def test_interleaved_ties_are_ranked_per_column():
    data = np.array([[2, 1], [1, 2], [2, 1], [1, 2]])
    expected = np.array([[3, 1], [1, 3], [4, 2], [2, 4]]) / 5
    np.testing.assert_array_equal(pobs(data), expected)


@pytest.mark.parametrize("method", ["legacy", "average", "invalid"])
def test_unsupported_tie_methods_are_explicitly_rejected(method):
    with pytest.raises(ValueError, match="ties_method must be 'ordinal'"):
        pobs([[1], [1]], ties_method=method)


@pytest.mark.parametrize("data", [1.0, [], [1, 2], np.zeros((2, 3, 4))])
def test_invalid_shape(data):
    with pytest.raises(ValueError, match="shape"):
        pobs(data)


@pytest.mark.parametrize("data", [np.ones((2, 2), dtype=complex),
                                  [["a"], ["b"]], np.array([[object()]])])
def test_non_real_numeric_inputs(data):
    with pytest.raises(TypeError, match="real numeric"):
        pobs(data)


def test_readonly_non_native_endian_input():
    data = np.array([[3, 2], [1, 3], [2, 1]], dtype=">i8")
    data.flags.writeable = False
    np.testing.assert_array_equal(pobs(data), [[.75, .5], [.25, .75], [.5, .25]])


def test_package_has_no_numba_imports_outside_contrib():
    package = Path(__file__).resolve().parents[1] / "pyscarcopula"
    for path in package.rglob("*.py"):
        if "contrib" in path.relative_to(package).parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            assert not any(name.split(".")[0] == "numba" for name in names), path


def test_numba_is_only_an_optional_dependency():
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(path.read_text(encoding="utf-8"))["project"]
    assert not any("numba" in name.lower() for name in project["dependencies"])
    assert "numba>=0.56" in project["optional-dependencies"]["contrib"]


def test_core_imports_and_pobs_work_with_numba_blocked():
    source = '''
import importlib
import importlib.abc
from pathlib import Path
import sys

class RejectNumba(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "numba":
            raise AssertionError("Unexpected Numba import: " + fullname)
        return None

sys.meta_path.insert(0, RejectNumba())
import pyscarcopula
for path in Path(pyscarcopula.__file__).parent.rglob("*.py"):
    parts = path.relative_to(Path(pyscarcopula.__file__).parent).with_suffix("").parts
    if any(part in {"contrib", "_cpp", "__pycache__"} for part in parts):
        continue
    if parts[-1] == "__init__":
        parts = parts[:-1]
    importlib.import_module(".".join(("pyscarcopula", *parts)))

import numpy as np
from pyscarcopula._utils import pobs
from pyscarcopula import ClaytonCopula
data = np.random.default_rng(73).normal(size=(40, 2))
u = pobs(data)
assert np.all((u > 0) & (u < 1))
ClaytonCopula().fit(data, to_pobs=True, method="MLE")
assert not any(name.split(".")[0] == "numba" for name in sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", source], capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
