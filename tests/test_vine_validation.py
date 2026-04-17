"""Optional distributional validation checks for RVine conditional sampling."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _skip_unless_enabled():
    if os.environ.get("PYSCA_RUN_VALIDATION") != "1":
        pytest.skip("set PYSCA_RUN_VALIDATION=1 to run validation checks")


def _run_script(*args):
    cmd = [sys.executable, *args]
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, (
        "command failed:\n"
        f"{' '.join(cmd)}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "status=PASS" in result.stdout


@pytest.mark.validation
def test_gaussian_block_oracle_graph_validation():
    _skip_unless_enabled()
    _run_script(
        "scripts/validate_rvine_conditional_gaussian.py",
        "--d", "6",
        "--corr", "block",
        "--fit-n", "6000",
        "--sample-n", "900",
        "--given", "{0: 0.2}",
        "--conditional-method", "graph",
        "--structure-mode", "dissmann",
        "--check",
    )


@pytest.mark.validation
def test_dynamic_gaussian_sampler_gas_process_validation():
    _skip_unless_enabled()
    _run_script(
        "scripts/validate_rvine_conditional_dynamic_gaussian.py",
        "--rho-process", "gas",
        "--sample-n", "1000",
        "--given", "{0: 0.2}",
        "--conditional-method", "graph",
        "--check",
    )


@pytest.mark.validation
def test_dynamic_fit_gas_validation():
    _skip_unless_enabled()
    _run_script(
        "scripts/validate_rvine_conditional_dynamic_fit.py",
        "--method", "gas",
        "--fit-n", "350",
        "--sample-n", "500",
        "--given", "{0: 0.2}",
        "--conditional-method", "graph",
        "--check",
    )


@pytest.mark.validation
def test_dynamic_fit_scar_ou_validation():
    _skip_unless_enabled()
    _run_script(
        "scripts/validate_rvine_conditional_dynamic_fit.py",
        "--method", "scar-tm-ou",
        "--rho-process", "ou",
        "--fit-n", "220",
        "--sample-n", "300",
        "--given", "{0: 0.2}",
        "--conditional-method", "graph",
        "--K", "30",
        "--tol", "0.8",
        "--max-mean-error", "0.6",
        "--max-cov-error", "0.8",
        "--max-corr-error", "0.8",
        "--min-ks-pvalue", "1e-6",
        "--check",
    )


@pytest.mark.validation
def test_dynamic_fit_gas_multi_given_grid_validation():
    _skip_unless_enabled()
    _run_script(
        "scripts/validate_rvine_conditional_dynamic_fit.py",
        "--d", "4",
        "--method", "gas",
        "--fit-n", "360",
        "--sample-n", "500",
        "--given", "{0: 0.2, 2: 0.8}",
        "--conditional-method", "grid",
        "--K", "40",
        "--tol", "0.5",
        "--max-mean-error", "0.5",
        "--max-cov-error", "0.7",
        "--max-corr-error", "0.7",
        "--min-ks-pvalue", "1e-7",
        "--check",
    )
