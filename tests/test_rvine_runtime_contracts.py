"""Oracle and dispatch contracts for the R-vine runtime."""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import pyscarcopula
from pyscarcopula import (
    BivariateGaussianCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    IndependentCopula,
    JoeCopula,
)
from pyscarcopula._constants import PSEUDO_OBS_EPS
from pyscarcopula._native import _extension as _cpp_extension
from pyscarcopula._native.errors import (
    NativeUnavailable,
    NativeUnsupported,
)
from pyscarcopula.stattests import (
    rvine_rosenblatt_transform,
    student_rosenblatt_transform,
)
from pyscarcopula.vine._rvine_dag import (
    build_runtime_rvine_dag,
    plan_conditional_sample,
)
from pyscarcopula.vine.vine import VineCopula

from rvine_runtime_cases import (
    configured_mixed_gas_vine,
    configured_mixed_family_vine,
    configured_static_dvine,
    fitted_pair,
    scalar_parameters,
)


def _assert_rng_equal(left, right):
    np.testing.assert_array_equal(left.random(16), right.random(16))


def _native_symbol_available(name):
    try:
        module = _cpp_extension.load()
    except NativeUnavailable:
        return False
    return callable(getattr(module, name, None))


def test_missing_base_extension_is_native_unavailable(monkeypatch):
    def unavailable():
        raise NativeUnavailable("compiled extension missing")

    monkeypatch.setattr(_cpp_extension, "load", unavailable)
    with pytest.raises(NativeUnavailable, match="compiled extension missing"):
        configured_static_dvine(3).sample(2, rng=np.random.default_rng(7))


def test_extension_loader_reports_real_import_failure(monkeypatch):
    monkeypatch.setattr(_cpp_extension, "_MODULE", None)
    monkeypatch.setattr(_cpp_extension, "_MODULE_ERROR", None)
    original_import = _cpp_extension.importlib.import_module

    def fail_extension_import(name):
        if name == "pyscarcopula._native._scar_cpp":
            raise ImportError("synthetic missing _scar_cpp")
        return original_import(name)

    monkeypatch.setattr(
        _cpp_extension.importlib, "import_module", fail_extension_import)
    with pytest.raises(NativeUnavailable, match="synthetic missing _scar_cpp"):
        _cpp_extension.load()


def test_sampling_rng_state_and_output_match_across_batch_sizes(monkeypatch):
    vine = configured_mixed_family_vine()
    one_rng = np.random.default_rng(20260814)
    many_rng = np.random.default_rng(20260814)
    one_batch = vine.sample(97, rng=one_rng, batch_rows=97)
    many_batches = vine.sample(97, rng=many_rng, batch_rows=7)
    np.testing.assert_array_equal(one_batch, many_batches)
    _assert_rng_equal(one_rng, many_rng)


@pytest.mark.rvine_native
def test_existing_gas_sampling_is_reproducible_on_native_path():
    if not _native_symbol_available("gas_rvine_sample"):
        pytest.skip("native gas_rvine_sample is unavailable")
    vine = configured_mixed_gas_vine()
    expected_rng = np.random.default_rng(202608141)
    actual_rng = np.random.default_rng(202608141)
    expected = vine.sample(64, rng=expected_rng)
    actual = vine.sample(64, rng=actual_rng)
    np.testing.assert_array_equal(actual, expected)
    _assert_rng_equal(actual_rng, expected_rng)


def test_memory_budget_rejection_precedes_rng_consumption(monkeypatch):
    vine = configured_mixed_family_vine()
    actual_rng = np.random.default_rng(118)
    expected_rng = np.random.default_rng(118)
    with pytest.raises(MemoryError, match="memory_budget_bytes"):
        vine.sample(
            100,
            rng=actual_rng,
            batch_rows=10,
            memory_budget_bytes=1,
        )
    _assert_rng_equal(actual_rng, expected_rng)


def test_public_predict_diagnostics_contract(monkeypatch):
    vine = configured_mixed_family_vine()
    given = {2: 0.42}
    samples, diagnostics = vine.predict(
        5,
        given=given,
        rng=np.random.default_rng(44),
        return_diagnostics=True,
    )
    assert samples.shape == (5, 3)
    np.testing.assert_array_equal(samples[:, 2], np.full(5, 0.42))
    required = {
        "given",
        "dynamic_conditioning",
        "suffix_start_col",
        "matrix_rebuilt",
        "conditional_method",
        "updated_edges",
        "skipped_edges",
        "timings_ms",
    }
    assert required <= diagnostics.keys()
    assert diagnostics["conditional_method"] == "suffix"
    assert all(
        isinstance(value, float) and value >= 0.0
        for value in diagnostics["timings_ms"].values()
    )
