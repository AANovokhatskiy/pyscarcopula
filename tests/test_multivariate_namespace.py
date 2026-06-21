"""Namespace and persistence contracts for multivariate models."""

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


@pytest.mark.parametrize(
    ("cls", "module"),
    (
        (GaussianCopula, "pyscarcopula.copula.multivariate.gaussian"),
        (StudentCopula, "pyscarcopula.copula.multivariate.student"),
        (
            EquicorrGaussianCopula,
            "pyscarcopula.copula.multivariate.equicorr",
        ),
        (
            StochasticStudentCopula,
            "pyscarcopula.copula.multivariate.stochastic_student",
        ),
    ),
)
def test_public_multivariate_classes_use_canonical_qualified_paths(cls, module):
    assert cls.__module__ == module


def test_experimental_multivariate_package_has_no_source_files():
    package_dir = (
        Path(__file__).parents[1]
        / "pyscarcopula"
        / "copula"
        / "experimental"
    )
    assert not list(package_dir.glob("*.py"))


@pytest.mark.parametrize(
    "factory",
    (
        GaussianCopula,
        StudentCopula,
        lambda: EquicorrGaussianCopula(d=3),
        lambda: StochasticStudentCopula(d=3, R=np.eye(3)),
    ),
)
def test_serialized_models_only_contain_canonical_multivariate_class_paths(factory):
    model = factory()
    path = Path.cwd() / f".multivariate-namespace-{type(model).__name__}.json"
    try:
        model.save(path, include_data=False)
        text = path.read_text(encoding="utf-8")
        envelope = json.loads(text)

        assert envelope["format_version"] == 2
        assert envelope["class"].startswith(
            "pyscarcopula.copula.multivariate.")
        assert "copula.experimental" not in text
        assert isinstance(load_model(path), type(model))
    finally:
        path.unlink(missing_ok=True)
