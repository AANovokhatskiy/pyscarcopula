"""Contracts for compatibility names intentionally removed in task 15."""

import importlib.util
import inspect

import numpy as np
import pytest

from pyscarcopula import GumbelCopula
from pyscarcopula._types import LatentResult, ou_params
from pyscarcopula.copula.multivariate import StochasticStudentCopula
from pyscarcopula.strategy.scar_jacobi import SCARJacobiStrategy
from pyscarcopula.strategy.scar_tm import SCARTMStrategy
from pyscarcopula.vine.cvine import CVineCopula
from pyscarcopula.vine.rvine import RVineCopula


def test_removed_numerical_facade_modules_are_absent():
    assert importlib.util.find_spec("pyscarcopula.numerical.auto_tm") is None
    assert importlib.util.find_spec(
        "pyscarcopula.numerical.tm_gradient") is None


def test_latent_result_has_no_alpha_alias():
    result = LatentResult(
        log_likelihood=0.0,
        method="SCAR-TM-OU",
        copula_name="Gumbel",
        success=True,
        params=ou_params(1.0, 0.0, 0.5),
    )
    assert not hasattr(result, "alpha")
    np.testing.assert_array_equal(result.params.values, [1.0, 0.0, 0.5])


def test_initial_point_legacy_names_are_absent():
    from pyscarcopula.strategy import initial_point

    assert not hasattr(initial_point, "legacy_smart_initial_point")
    assert not hasattr(initial_point, "legacy_smart_init")


@pytest.mark.parametrize("alias", ["matrix", "spectral"])
def test_jacobi_transition_aliases_are_rejected(alias):
    with pytest.raises(ValueError, match="transition_method"):
        SCARJacobiStrategy(transition_method=alias)


def test_adaptive_spectral_basis_alias_is_rejected():
    with pytest.raises(ValueError, match="spectral_basis_order"):
        SCARTMStrategy(spectral_basis_order="adaptive")


def test_cvine_sample_has_no_legacy_arguments_or_alias():
    signature = inspect.signature(CVineCopula.sample)
    assert tuple(signature.parameters) == ("self", "n", "u", "rng")
    assert not hasattr(CVineCopula, "sample_model")


def test_rvine_sample_and_predict_have_canonical_signatures():
    sample_signature = inspect.signature(RVineCopula.sample)
    predict_signature = inspect.signature(RVineCopula.predict)

    assert tuple(sample_signature.parameters) == ("self", "n", "u", "rng")
    assert "u" in predict_signature.parameters
    assert "u_train" not in predict_signature.parameters


def test_student_object_creating_compatibility_functions_are_absent():
    from pyscarcopula.copula.multivariate import stochastic_student

    for name in (
        "_student_copula_logpdf",
        "_student_copula_dlogpdf_ddf",
        "_student_copula_logpdf_and_dlogpdf_ddf",
    ):
        assert not hasattr(stochastic_student, name)

    model = StochasticStudentCopula(d=2, R=np.eye(2))
    values = model.log_pdf_rows(
        np.array([[0.25, 0.75]], dtype=np.float64), 5.0)
    assert np.all(np.isfinite(values))


def test_forward_loglik_compatibility_oracle_is_absent():
    from pyscarcopula.numerical import tm_functions

    assert not hasattr(tm_functions, "_forward_loglik")
