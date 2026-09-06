"""Prepared Equicorr history survives JSON or is explicitly omitted."""
import json

import numpy as np
import pytest

from pyscarcopula import EquicorrGaussianCopula, EquicorrPreparedData, load_model
from pyscarcopula.io import _from_jsonable, _to_jsonable


@pytest.fixture(scope="module", params=[
    "MLE", "GAS-unit", "GAS-fisher", "OU-matrix", "OU-local",
    "OU-spectral", "OU-auto",
])
def prepared_fit(request):
    observations = np.random.default_rng(6077).uniform(0.1, 0.9, (37, 3))
    model = EquicorrGaussianCopula(3)
    prepared = model.prepare_sufficient_statistics(
        observations, batch_rows=11, dimension_tile=2)
    options = {"maxiter": 30}
    method = request.param
    if method.startswith("GAS"):
        options.update(scaling=method.split("-")[1], gamma0=[0.03, 0.02, 0.7])
        method = "GAS"
    elif method.startswith("OU"):
        options.update(
            transition_method=method.split("-")[1], alpha0=[1.4, 0.1, 0.6],
            K=32, max_K=64, adaptive=False,
            spectral_basis_order=16, spectral_quad_order=32)
        method = "SCAR-TM-OU"
    result = model.fit(prepared, method=method, **options)
    assert model.fit_result is result
    return model, observations, prepared


@pytest.mark.parametrize("include_data", ["default", True, False])
def test_prepared_fitted_model_json_roundtrip(prepared_fit, include_data, tmp_path):
    model, observations, prepared = prepared_fit
    result = model.fit_result
    expected_loglik = model.log_likelihood(prepared)
    expected_prediction = model.predict(9, rng=np.random.default_rng(441))
    path = tmp_path / "prepared_model.json"

    if include_data == "default":
        model.save(str(path))
    else:
        model.save(path, include_data=include_data)
    restored = EquicorrGaussianCopula.load(path)

    assert model._last_prepared is prepared
    assert model._last_u is None
    assert model.fit_result is result
    assert not prepared.sum_z.flags.writeable
    assert not prepared.sum_z2.flags.writeable
    assert restored.fit_result.method == result.method
    assert restored.fit_result.success == result.success
    assert restored._last_u is None
    assert restored.log_likelihood(prepared) == pytest.approx(expected_loglik, abs=1e-10)
    np.testing.assert_array_equal(
        restored.predict(9, u=prepared, rng=np.random.default_rng(441)),
        expected_prediction)
    assert restored.log_likelihood(observations) == pytest.approx(
        model.log_likelihood(observations), abs=1e-10)

    envelope = json.loads(path.read_text(encoding="utf-8"))
    retained = include_data != False
    assert envelope["include_data"] is retained
    if retained:
        saved = restored._last_prepared
        assert isinstance(saved, EquicorrPreparedData)
        assert saved is not prepared
        np.testing.assert_array_equal(saved.sum_z, prepared.sum_z)
        np.testing.assert_array_equal(saved.sum_z2, prepared.sum_z2)
        assert saved.dimension == prepared.dimension
        assert dict(saved.diagnostics) == dict(prepared.diagnostics)
        assert not saved.sum_z.flags.writeable
        assert not saved.sum_z2.flags.writeable
        with pytest.raises(TypeError):
            saved.diagnostics["changed"] = True
        np.testing.assert_array_equal(
            restored.predict(9, rng=np.random.default_rng(441)), expected_prediction)
    else:
        assert restored._last_prepared is None
        assert "EquicorrPreparedData" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("invalid", ["format_version", "sum_z2"])
def test_prepared_json_load_revalidates_statistics(invalid):
    model = EquicorrGaussianCopula(3)
    prepared = model.prepare_sufficient_statistics(np.full((2, 3), 0.6))
    payload = _to_jsonable(prepared)
    if invalid == "format_version":
        payload["fields"]["format_version"] = -1
    else:
        payload["fields"]["sum_z2"] = _to_jsonable(np.full(2, -1.0))
    with pytest.raises(ValueError):
        _from_jsonable(payload)


@pytest.fixture(scope="module")
def prepared_model_json(tmp_path_factory):
    model = EquicorrGaussianCopula(4)
    observations = np.random.default_rng(735).uniform(0.1, 0.9, (19, 4))
    prepared = model.prepare_sufficient_statistics(observations)
    model.fit(prepared, method="MLE")
    path = tmp_path_factory.mktemp("prepared_json") / "model.json"
    model.save(path, include_data=True)
    return path.read_text(encoding="utf-8")


@pytest.fixture(params=["model", "prepared"])
def prepared_document(prepared_model_json, request, tmp_path):
    envelope = json.loads(prepared_model_json)
    record = envelope["state"]["state"]["_last_prepared"]
    if request.param == "prepared":
        envelope["class"] = record["class"]
        envelope["state"] = record
    path = tmp_path / "model.json"

    def load():
        path.write_text(json.dumps(envelope, allow_nan=False), encoding="utf-8")
        if request.param == "model":
            return EquicorrGaussianCopula.load(path)._last_prepared
        return load_model(path, expected_type=EquicorrPreparedData)

    return record, load


@pytest.mark.parametrize("tag", ["dataclass", "object"])
@pytest.mark.parametrize("invalid", [
    "format_version", "n_obs", "dimension", "negative_square", "nonfinite",
    "shape", "cauchy_bound", "clipping_epsilon",
])
def test_public_loader_rejects_invalid_prepared_state(
        prepared_document, tag, invalid):
    record, load = prepared_document
    values = record["fields"]
    if invalid == "format_version":
        values["format_version"] = -1
    elif invalid == "n_obs":
        values["n_obs"] += 1
    elif invalid == "dimension":
        values["dimension"] = 1
    elif invalid == "negative_square":
        values["sum_z2"]["data"][0] = -1.0
    elif invalid == "nonfinite":
        values["sum_z"]["data"][0] = {
            "__pyscarcopula_type__": "float", "value": "nan"}
    elif invalid == "shape":
        values["sum_z2"]["data"].pop()
        values["sum_z2"]["shape"][0] -= 1
    elif invalid == "cauchy_bound":
        values["sum_z"]["data"][0] = 1000.0
        values["sum_z2"]["data"][0] = 0.0
    else:
        values["clipping_epsilon"] = 0.5
    if tag == "object":
        record["__pyscarcopula_type__"] = tag
        record["state"] = record.pop("fields")

    with pytest.raises(ValueError):
        load()


def test_public_loader_rejects_object_tag_even_with_valid_prepared_fields(
        prepared_document):
    record, load = prepared_document
    record["__pyscarcopula_type__"] = "object"
    record["state"] = record.pop("fields")

    with pytest.raises(ValueError, match="requires the 'dataclass' tag"):
        load()


def test_public_loader_restores_immutable_prepared_state(prepared_document):
    record, load = prepared_document
    restored = load()

    for name in ("sum_z", "sum_z2"):
        values = getattr(restored, name)
        np.testing.assert_array_equal(values, record["fields"][name]["data"])
        with pytest.raises(ValueError):
            values[0] = 0.0
    with pytest.raises(TypeError):
        restored.diagnostics["changed"] = True
