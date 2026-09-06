"""Namespace and persistence contracts for multivariate models."""

import json

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
    "factory",
    (
        GaussianCopula,
        StudentCopula,
        lambda: EquicorrGaussianCopula(d=3),
        lambda: StochasticStudentCopula(d=3, R=np.eye(3)),
    ),
)
def test_serialized_models_only_contain_canonical_multivariate_class_paths(factory, tmp_path):
    model = factory()
    path = tmp_path / "model.json"
    model.save(path, include_data=False)
    envelope = json.loads(path.read_text(encoding="utf-8"))

    assert set(envelope) == {"class", "format", "include_data", "state"}
    assert envelope["class"].startswith("pyscarcopula.copula.multivariate.")
    assert isinstance(load_model(path), type(model))
