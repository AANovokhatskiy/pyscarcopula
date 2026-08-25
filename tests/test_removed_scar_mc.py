"""Removal contracts for the discontinued SCAR Monte Carlo strategies."""

from dataclasses import fields
import importlib.util
import json

import numpy as np
import pytest

from pyscarcopula import CVineCopula, GumbelCopula, VineCopula, load_model
from pyscarcopula._native import _extension
from pyscarcopula._native.registry import STRATEGY_REQUIREMENTS
from pyscarcopula._types import LatentResult, NumericalConfig, ou_params
from pyscarcopula.api import fit
from pyscarcopula.copula.base import CopulaCapabilities
from pyscarcopula.io import MODEL_FORMAT, _from_jsonable, _to_jsonable
from pyscarcopula.strategy._base import (
    ensure_strategy_supported,
    get_strategy,
    list_methods,
)


REMOVED_METHOD_ALIASES = (
    "scar-p-ou",
    "SCAR-P-OU",
    "scar_p_ou",
    "Scar_P_Ou",
    "scarpou",
    "scar-m-ou",
    "SCAR-M-OU",
    "scar_m_ou",
    "Scar_M_Ou",
    "scarmou",
)


def test_removed_methods_are_absent_from_all_public_and_native_registries():
    assert list_methods() == ["GAS", "MLE", "SCAR-TM-JACOBI", "SCAR-TM-OU"]
    assert set(STRATEGY_REQUIREMENTS) == set(list_methods())

    cpp = _extension.load()
    assert not hasattr(cpp.DynamicsKind, "ScarPOu")
    assert not hasattr(cpp.DynamicsKind, "ScarMOu")
    assert not hasattr(cpp, "copula_log_pdf_trajectory_grid")


def test_removed_modules_and_public_fields_do_not_exist():
    assert importlib.util.find_spec("pyscarcopula.strategy.scar_mc") is None
    assert importlib.util.find_spec("pyscarcopula.numerical.mc_samplers") is None
    assert importlib.util.find_spec("pyscarcopula.numerical.mc_native") is None

    assert "supports_scar_mc" not in {
        field.name for field in fields(CopulaCapabilities)
    }
    assert {"default_n_tr", "default_M_iterations"}.isdisjoint(
        field.name for field in fields(NumericalConfig)
    )
    assert {"n_tr", "M_iterations"}.isdisjoint(
        field.name for field in fields(LatentResult)
    )


@pytest.mark.parametrize("method", REMOVED_METHOD_ALIASES)
def test_removed_aliases_are_rejected_without_strategy_fallback(method):
    observations = np.random.default_rng(20260825).uniform(
        0.1, 0.9, size=(12, 2)
    )

    with pytest.raises(ValueError, match="Unknown method"):
        get_strategy(method)
    with pytest.raises(ValueError, match="Unknown method"):
        ensure_strategy_supported(GumbelCopula(), method)
    with pytest.raises(ValueError, match="Unknown method"):
        fit(GumbelCopula(), observations, method=method)


@pytest.mark.parametrize("method", ("scar-p-ou", "scar_m_ou"))
@pytest.mark.parametrize("factory", (
    lambda: VineCopula.cvine(d=3, order=[0, 1, 2]),
    CVineCopula,
))
def test_vines_reject_removed_methods_before_edge_selection(factory, method):
    observations = np.random.default_rng(20260826).uniform(
        0.1, 0.9, size=(12, 3)
    )

    with pytest.raises(ValueError, match="Unknown method"):
        factory().fit(observations, method=method)


@pytest.mark.parametrize("method", ("SCAR-P-OU", "scar_m_ou"))
def test_legacy_result_payloads_fail_with_explicit_unsupported_error(method):
    result = LatentResult(
        log_likelihood=0.0,
        method=method,
        copula_name="Gumbel",
        success=True,
        params=ou_params(1.0, 0.0, 0.5),
    )

    with pytest.raises(ValueError, match="Unsupported persisted model method"):
        _from_jsonable(_to_jsonable(result))


def test_legacy_model_file_is_rejected_before_class_reconstruction(tmp_path):
    path = tmp_path / "legacy-scar-mc.json"
    path.write_text(json.dumps({
        "format": MODEL_FORMAT,
        "class": "pyscarcopula.copula.gumbel.GumbelCopula",
        "include_data": False,
        "state": {"method": "SCAR-P-OU"},
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="no migration execution path"):
        load_model(path)
