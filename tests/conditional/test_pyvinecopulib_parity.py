"""Optional Stage-3 edge-level parity with pyvinecopulib 0.7.5."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
import pytest


os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "pyscarcopula-matplotlib-cache"),
)
pv = pytest.importorskip("pyvinecopulib")

from pyscarcopula import (  # noqa: E402
    BivariateGaussianCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    IndependentCopula,
    JoeCopula,
)
from pyscarcopula.strategy.predict_helpers import (  # noqa: E402
    conditional_sample_bivariate,
)

from ._bivariate_cases import (  # noqa: E402
    CASES,
    MEDIUM_CASES,
    BivariateCase,
    transposed_rotation,
)
from ._bivariate_oracles import (  # noqa: E402
    conditional_cdf,
    rotated_cdf,
)


pytestmark = pytest.mark.external

FAMILY_CLASSES = {
    "independent": IndependentCopula,
    "gaussian": BivariateGaussianCopula,
    "clayton": ClaytonCopula,
    "gumbel": GumbelCopula,
    "frank": FrankCopula,
    "joe": JoeCopula,
}

PV_FAMILIES = {
    "independent": pv.BicopFamily.indep,
    "gaussian": pv.BicopFamily.gaussian,
    "clayton": pv.BicopFamily.clayton,
    "gumbel": pv.BicopFamily.gumbel,
    "frank": pv.BicopFamily.frank,
    "joe": pv.BicopFamily.joe,
}


def _production(case: BivariateCase):
    return FAMILY_CLASSES[case.family](rotate=case.rotation)


def _production_transposed(case: BivariateCase):
    return FAMILY_CLASSES[case.family](
        rotate=transposed_rotation(case.rotation)
    )


def _reference(case: BivariateCase):
    kwargs = {
        "family": PV_FAMILIES[case.family],
        "rotation": case.rotation,
    }
    if case.family != "independent":
        kwargs["parameters"] = np.array([[case.parameter]], dtype=np.float64)
    return pv.Bicop(**kwargs)


def _f_matrix(first, second) -> np.ndarray:
    return np.asfortranarray(np.column_stack((first, second)))


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_edge_density_h_inverse_and_cdf_match_pyvinecopulib(case):
    copula = _production(case)
    transposed = _production_transposed(case)
    reference = _reference(case)
    u1 = np.array([1e-6, 0.007, 0.08, 0.31, 0.57, 0.86, 0.993, 1 - 1e-6])
    u2 = np.array([0.42, 0.98, 0.17, 0.69, 0.04, 0.77, 0.26, 0.91])
    parameter = np.full(len(u1), case.parameter)
    points = _f_matrix(u1, u2)

    expected_pdf = reference.pdf(points)
    observed_pdf = copula.pdf(u1, u2, parameter)
    pdf_rtol = 6e-9 if case.family == "gaussian" else 3e-10
    np.testing.assert_allclose(
        observed_pdf, expected_pdf, rtol=pdf_rtol, atol=2e-10
    )
    np.testing.assert_allclose(
        copula.log_pdf(u1, u2, parameter),
        np.log(expected_pdf),
        rtol=0.0,
        atol=1.5e-7 if case.family == "gaussian" else 2e-9,
    )

    expected_first = reference.hfunc2(points)
    expected_second = reference.hfunc1(points)
    observed_first = copula.h(u1, u2, parameter)
    observed_second = transposed.h(u2, u1, parameter)
    h_tolerance = 7e-9 if case.family == "gaussian" else 3e-10
    np.testing.assert_allclose(
        observed_first, expected_first, rtol=0.0, atol=h_tolerance
    )
    np.testing.assert_allclose(
        observed_second, expected_second, rtol=0.0, atol=h_tolerance
    )
    np.testing.assert_allclose(
        conditional_cdf(
            u1, u2, parameter, case.family, case.rotation, free_index=0
        ),
        expected_first,
        rtol=0.0,
        atol=5e-10,
    )
    np.testing.assert_allclose(
        conditional_cdf(
            u2, u1, parameter, case.family, case.rotation, free_index=1
        ),
        expected_second,
        rtol=0.0,
        atol=5e-10,
    )

    q = np.array([1e-4, 0.01, 0.09, 0.33, 0.64, 0.93, 1.0 - 1e-4])
    given = np.array([0.03, 0.91, 0.22, 0.48, 0.79, 0.97, 0.37])
    inverse_parameter = np.full(len(q), case.parameter)
    np.testing.assert_allclose(
        copula.h_inverse(q, given, inverse_parameter),
        reference.hinv2(_f_matrix(q, given)),
        rtol=0.0,
        atol=8e-8,
    )
    np.testing.assert_allclose(
        transposed.h_inverse(q, given, inverse_parameter),
        reference.hinv1(_f_matrix(given, q)),
        rtol=0.0,
        atol=8e-8,
    )

    expected_cdf = reference.cdf(points)
    analytical_cdf = rotated_cdf(
        u1, u2, case.parameter, case.family, case.rotation
    )
    cdf_tolerance = 2e-5 if case.family == "gaussian" else 3e-12
    np.testing.assert_allclose(
        analytical_cdf, expected_cdf, rtol=0.0, atol=cdf_tolerance
    )


@pytest.mark.parametrize("case", MEDIUM_CASES, ids=lambda case: case.id)
@pytest.mark.parametrize("given_index", [0, 1], ids=["given-u1", "given-u2"])
def test_conditional_sampler_matches_pyvine_inverse_for_same_uniforms(
        case, given_index):
    n = 257
    given = 0.83
    seed = 20261600 + MEDIUM_CASES.index(case) * 2 + given_index
    expected_rng = np.random.default_rng(seed)
    q = expected_rng.uniform(0.0, 1.0, size=n)
    reference = _reference(case)
    if given_index == 0:
        expected_free = reference.hinv1(
            _f_matrix(np.full(n, given), q)
        )
        free_index = 1
    else:
        expected_free = reference.hinv2(
            _f_matrix(q, np.full(n, given))
        )
        free_index = 0

    samples = conditional_sample_bivariate(
        _production(case),
        n,
        case.parameter,
        given={given_index: given},
        rng=np.random.default_rng(seed),
    )
    np.testing.assert_array_equal(samples[:, given_index], given)
    np.testing.assert_allclose(
        samples[:, free_index], expected_free, rtol=0.0, atol=8e-8
    )
