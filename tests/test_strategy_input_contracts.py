"""Strategy support and unsupported persisted-state contracts."""
import json
import numpy as np
import pytest
from pyscarcopula import ClaytonCopula, GumbelCopula, VineCopula, load_model
from pyscarcopula._native.errors import NativeUnsupported
from pyscarcopula._native.registry import STRATEGY_REQUIREMENTS
from pyscarcopula._types import LatentResult, ou_params
from pyscarcopula.api import fit
from pyscarcopula.io import MODEL_FORMAT, _from_jsonable, _to_jsonable
from pyscarcopula.strategy._base import ensure_strategy_supported, get_strategy, list_methods
from pyscarcopula.strategy.scar_jacobi import SCARJacobiStrategy
from pyscarcopula.strategy.scar_tm import SCARTMStrategy

UNSUPPORTED_METHODS = ("unknown-method", "SCAR-P-OU", "scar_m_ou")

def test_builtin_methods_have_native_requirements():
    assert set(list_methods()) == {"MLE", "GAS", "SCAR-TM-OU", "SCAR-TM-JACOBI"}
    assert set(STRATEGY_REQUIREMENTS) == set(list_methods())

def test_unknown_copula_subclass_is_rejected_before_callback_execution():
    calls = []

    class CallbackCopula(ClaytonCopula):
        def pdf(self, *args, **kwargs):
            calls.append("pdf")
            return super().pdf(*args, **kwargs)

    data = np.random.default_rng(86).uniform(0.01, 0.99, size=(20, 2))
    with pytest.raises(NativeUnsupported, match="exact registered"):
        fit(CallbackCopula(), data, method="mle")
    with pytest.raises(NativeUnsupported, match="exact registered"):
        VineCopula.cvine(2, candidates=[CallbackCopula])
    assert calls == []


@pytest.mark.parametrize("alias", ["matrix", "spectral"])
def test_jacobi_transition_aliases_are_rejected(alias):
    with pytest.raises(ValueError, match="transition_method"):
        SCARJacobiStrategy(transition_method=alias)


def test_adaptive_spectral_basis_alias_is_rejected():
    with pytest.raises(ValueError, match="spectral_basis_order"):
        SCARTMStrategy(spectral_basis_order="adaptive")


@pytest.mark.parametrize("method", UNSUPPORTED_METHODS)
def test_unknown_methods_are_rejected_without_strategy_fallback(method):
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
def test_vines_reject_unknown_methods_before_edge_selection(method):
    observations = np.random.default_rng(20260826).uniform(
        0.1, 0.9, size=(12, 3)
    )

    with pytest.raises(ValueError, match="Unknown method"):
        VineCopula.cvine(d=3, order=[0, 1, 2]).fit(
            observations, method=method)


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
