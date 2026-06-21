"""Contracts for the validation-only real-data optimization harness."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = (
    _ROOT / "numerical experiments"
    / "wp10_real_data_optimization_validation.py"
)
_SPEC = importlib.util.spec_from_file_location("real_data_validation", _SCRIPT)
validation_harness = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = validation_harness
_SPEC.loader.exec_module(validation_harness)


def test_validation_dataset_definitions_match_historical_script():
    crypto = validation_harness.load_crypto6()
    daily = validation_harness.load_daily()
    hf = validation_harness.load_hf()

    assert crypto.name == "crypto6"
    assert crypto.u.shape == (250, 6)
    assert daily.name == "daily"
    assert daily.u.shape[1] == 2
    assert hf.name == "hf"
    assert hf.u.shape == (12_000, 2)
    np.testing.assert_allclose(
        validation_harness.bivariate_view(crypto).u,
        crypto.u[:, :2],
    )


def test_removed_reference_engine_is_rejected():
    with pytest.raises(RuntimeError, match="removed in WP11"):
        validation_harness.engine_context("python")


def test_cpp_run_records_objective_and_gradient_checkpoints():
    rng = np.random.default_rng(20260618)
    u = rng.uniform(0.05, 0.95, size=(18, 2))
    dates = pd.date_range("2026-01-01", periods=len(u))
    returns = pd.DataFrame(
        rng.normal(size=(len(u), 2)),
        index=dates,
        columns=["x", "y"],
    )
    spec = validation_harness.DatasetSpec(
        "synthetic",
        ("x", "y"),
        returns,
        u,
        _SCRIPT,
    )
    options = validation_harness.NumericalOptions(
        transition_method="matrix",
        K=12,
        adaptive=False,
        max_K=12,
        maxiter=1,
        maxfun=8,
    )
    alpha0 = np.array([1.0, 0.0, 1.0])

    row = validation_harness.run_record(
        "cpp",
        "gaussian",
        spec,
        options,
        alpha0,
        None,
        None,
        0,
    )

    assert row["status"] == "ok"
    assert row["checkpoint_status"] == "ok"
    checkpoints = json.loads(row["gradient_checkpoints_json"])
    assert np.isfinite(checkpoints["initial_cpp"]["objective"])
    assert np.all(np.isfinite(checkpoints["initial_cpp"]["gradient"]))
