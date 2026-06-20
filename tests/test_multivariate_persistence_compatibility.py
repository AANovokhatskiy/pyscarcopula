"""Compatibility with persisted multivariate models from version 2."""

import json
from pathlib import Path

import numpy as np
import pytest

from pyscarcopula import (
    EquicorrGaussianCopula,
    GaussianCopula,
    StochasticStudentCopula,
    StudentCopula,
    load_model,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "persistence"


@pytest.mark.parametrize(
    ("filename", "old_path", "expected_type"),
    [
        (
            "v2_gaussian_old_path.json",
            "pyscarcopula.copula.elliptical.GaussianCopula",
            GaussianCopula,
        ),
        (
            "v2_student_old_path.json",
            "pyscarcopula.copula.elliptical.StudentCopula",
            StudentCopula,
        ),
        (
            "v2_equicorr_old_path.json",
            "pyscarcopula.copula.experimental.equicorr.EquicorrGaussianCopula",
            EquicorrGaussianCopula,
        ),
        (
            "v2_stochastic_student_old_path.json",
            (
                "pyscarcopula.copula.experimental.stochastic_student."
                "StochasticStudentCopula"
            ),
            StochasticStudentCopula,
        ),
    ],
)
def test_historical_multivariate_class_paths_remain_loadable(
    filename, old_path, expected_type
):
    path = FIXTURE_DIR / filename
    envelope = json.loads(path.read_text(encoding="utf-8"))

    assert envelope["format_version"] == 2
    assert envelope["class"] == old_path
    assert envelope["state"]["class"] == old_path

    loaded = load_model(path)

    assert isinstance(loaded, expected_type)


def test_historical_gaussian_and_student_payload_state_is_preserved():
    gaussian = load_model(FIXTURE_DIR / "v2_gaussian_old_path.json")
    student = load_model(FIXTURE_DIR / "v2_student_old_path.json")

    np.testing.assert_allclose(gaussian.corr, [[1.0, 0.35], [0.35, 1.0]])
    assert gaussian.fit_result["log_likelihood"] == 1.25
    np.testing.assert_allclose(student.shape, [[1.0, -0.2], [-0.2, 1.0]])
    assert student.df == 6.5
    assert student.fit_result["log_likelihood"] == 0.75


def test_historical_dynamic_multivariate_payload_state_is_preserved():
    equicorr = load_model(FIXTURE_DIR / "v2_equicorr_old_path.json")
    student = load_model(
        FIXTURE_DIR / "v2_stochastic_student_old_path.json"
    )

    assert equicorr.d == 3
    assert equicorr.fit_result.copula_param == 0.3
    assert student.d == 3
    assert student.fit_result.copula_param == 7.0
    assert student._ppf_cache is None
    np.testing.assert_allclose(np.diag(student.R), 1.0)
