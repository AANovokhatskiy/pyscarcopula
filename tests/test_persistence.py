"""Model persistence tests."""

import json

import numpy as np
import pytest
from scipy.stats import norm

from pyscarcopula import (
    BivariateGaussianCopula,
    CVineCopula,
    GaussianCopula,
    GumbelCopula,
    RVineCopula,
    load_model,
)
from pyscarcopula._utils import pobs
from pyscarcopula._types import (
    GASResult,
    LatentResult,
    gas_params,
    jacobi_params,
    ou_params,
)
from pyscarcopula.api import log_likelihood, predict, predictive_mean, sample
from pyscarcopula.io import _from_jsonable, _to_jsonable


def test_bivariate_save_load_roundtrip(tmp_path, random_u2):
    cop = GumbelCopula(rotate=180)
    cop.fit(random_u2, method="mle")

    path = tmp_path / "gumbel.json"
    cop.save(path)
    loaded = GumbelCopula.load(path)

    envelope = json.loads(path.read_text(encoding="utf-8"))
    assert envelope["format"] == "pyscarcopula-model"
    assert set(envelope) == {
        "class", "format", "include_data", "state",
    }
    assert envelope["class"] == "pyscarcopula.copula.gumbel.GumbelCopula"

    assert isinstance(loaded, GumbelCopula)
    assert loaded.rotate == cop.rotate
    assert loaded.fit_result == cop.fit_result
    np.testing.assert_allclose(loaded._last_u, cop._last_u)

    samples = loaded.predict(8, rng=np.random.default_rng(0))
    assert samples.shape == (8, 2)
    assert np.all((samples > 0.0) & (samples < 1.0))


def test_legacy_gas_backend_field_is_ignored_on_load():
    result = GASResult(
        log_likelihood=1.5,
        method="GAS",
        copula_name="Gaussian",
        success=True,
        params=gas_params(0.1, 0.2, 0.7),
    )
    payload = _to_jsonable(result)
    assert "backend" not in payload["fields"]
    payload["fields"]["backend"] = "python"

    loaded = _from_jsonable(payload)

    assert loaded.log_likelihood == result.log_likelihood
    assert loaded.method == result.method
    assert loaded.copula_name == result.copula_name
    np.testing.assert_allclose(loaded.params.values, result.params.values)
    assert not hasattr(loaded, "backend")


def test_legacy_scar_backend_field_is_ignored_on_load():
    result = LatentResult(
        log_likelihood=2.5,
        method="SCAR-TM-OU",
        copula_name="Gumbel",
        success=True,
        params=ou_params(1.0, 0.2, 0.8),
    )
    payload = _to_jsonable(result)
    assert "backend" not in payload["fields"]
    payload["fields"]["backend"] = "python"

    loaded = _from_jsonable(payload)

    assert loaded.log_likelihood == result.log_likelihood
    np.testing.assert_allclose(loaded.params.values, result.params.values)
    assert not hasattr(loaded, "backend")


def test_initialization_diagnostics_json_roundtrip():
    diagnostics = {
        "requested_method": "strength_aware",
        "selected_method": "heuristic",
        "alpha0": [1.0, 0.2, 0.8],
        "attempts": [
            {
                "method": "strength_aware",
                "success": False,
                "error_type": "ValueError",
                "error_message": "failed",
            },
            {"method": "heuristic", "success": True},
        ],
        "success": True,
    }
    result = LatentResult(
        log_likelihood=2.5,
        method="SCAR-TM-OU",
        copula_name="Gumbel",
        success=True,
        params=ou_params(1.0, 0.2, 0.8),
        diagnostics={"initialization": diagnostics},
    )

    loaded = _from_jsonable(_to_jsonable(result))

    assert loaded.diagnostics["initialization"] == diagnostics


def test_legacy_jacobi_result_without_semantic_options_uses_defaults():
    result = LatentResult(
        log_likelihood=1.25,
        method="SCAR-TM-JACOBI",
        copula_name="Gumbel",
        success=True,
        params=jacobi_params(1.2, 0.4, 0.3),
        transition_method="auto",
        spectral_basis_order=8,
        spectral_quad_order=32,
    )
    payload = _to_jsonable(result)
    for name in (
            "tau_eps",
            "theta_cap",
            "clip_negative",
            "negative_mass_tol",
            "stationary_shape_max",
            "transition_storage",
            "stationarity_correction",
            "sampling_method",
            "lamperti_substeps",
            "lamperti_boundary",
            "lamperti_eps",
            "lamperti_engine",
            "lamperti_chunk_observations",
            "memory_budget_bytes"):
        payload["fields"].pop(name)

    loaded = _from_jsonable(payload)

    assert loaded.tau_eps == pytest.approx(1e-6)
    assert loaded.theta_cap is None
    assert not loaded.clip_negative
    assert loaded.negative_mass_tol == pytest.approx(1e-5)
    assert loaded.stationary_shape_max == pytest.approx(500.0)
    assert loaded.transition_storage == "dense"
    assert loaded.stationarity_correction == "none"
    assert loaded.sampling_method == "tm_grid"
    assert loaded.lamperti_substeps == 8
    assert loaded.lamperti_boundary == "reflect"
    assert loaded.lamperti_eps == pytest.approx(1e-10)
    assert loaded.lamperti_engine == "numba"
    assert loaded.lamperti_chunk_observations == 4096
    assert loaded.memory_budget_bytes is None


def test_jacobi_semantic_options_model_roundtrip(tmp_path):
    u = np.random.default_rng(123).uniform(size=(30, 2))
    copula = GumbelCopula()
    result = copula.fit(
        u,
        method="scar-tm-jacobi",
        transition_method="local_fixed",
        basis_order=8,
        quad_order=32,
        gh_order=7,
        tau_eps=2e-5,
        theta_cap=1.25,
        clip_negative=True,
        negative_mass_tol=2e-7,
        stationary_shape_max=None,
        sampling_method="lamperti_euler",
        lamperti_substeps=4,
        lamperti_boundary="clip",
        lamperti_eps=2e-9,
        lamperti_engine="python",
        lamperti_chunk_observations=3,
        memory_budget_bytes=2_000_000,
        smart_init=False,
        maxiter=2,
        maxfun=20,
    )
    path = tmp_path / "gumbel-jacobi-options.json"
    copula.save(path, include_data=True)

    loaded = GumbelCopula.load(path)
    loaded_result = loaded.fit_result

    assert loaded_result.transition_method == "local_fixed"
    assert loaded_result.gh_order == 7
    assert loaded_result.spectral_basis_order == 8
    assert loaded_result.spectral_quad_order == 32
    assert loaded_result.tau_eps == pytest.approx(2e-5)
    assert loaded_result.theta_cap == pytest.approx(1.25)
    assert loaded_result.clip_negative
    assert loaded_result.negative_mass_tol == pytest.approx(2e-7)
    assert loaded_result.stationary_shape_max is None
    assert loaded_result.sampling_method == "lamperti_euler"
    assert loaded_result.lamperti_substeps == 4
    assert loaded_result.lamperti_boundary == "clip"
    assert loaded_result.lamperti_eps == pytest.approx(2e-9)
    assert loaded_result.lamperti_engine == "python"
    assert loaded_result.lamperti_chunk_observations == 3
    assert loaded_result.memory_budget_bytes == 2_000_000
    assert log_likelihood(loaded, u, loaded_result) == pytest.approx(
        result.log_likelihood, rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(
        predictive_mean(loaded, u, loaded_result),
        predictive_mean(copula, u, result),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        predict(
            loaded,
            u,
            loaded_result,
            25,
            rng=np.random.default_rng(77),
        ),
        predict(
            copula,
            u,
            result,
            25,
            rng=np.random.default_rng(77),
        ),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        sample(
            loaded,
            u,
            loaded_result,
            50,
            rng=np.random.default_rng(78),
        ),
        sample(
            copula,
            u,
            result,
            50,
            rng=np.random.default_rng(78),
        ),
    )


def test_include_data_false_drops_cached_training_data(tmp_path, random_u2):
    cop = GumbelCopula(rotate=180)
    cop.fit(random_u2, method="mle")

    path = tmp_path / "gumbel-no-data.json"
    cop.save(path, include_data=False)
    loaded = GumbelCopula.load(path)

    assert loaded.fit_result == cop.fit_result
    assert loaded._last_u is None
    with pytest.raises(ValueError, match="No data for predict"):
        loaded.predict(4)

    samples = loaded.predict(4, u=random_u2, rng=np.random.default_rng(1))
    assert samples.shape == (4, 2)


def test_latent_result_json_roundtrip_uses_strict_json(tmp_path, random_u2):
    cop = GumbelCopula(rotate=180)
    cop.fit(random_u2, method="scar-tm-ou")

    path = tmp_path / "gumbel-scar.json"
    cop.save(path)
    text = path.read_text(encoding="utf-8")
    assert "Infinity" not in text
    assert "NaN" not in text

    loaded = GumbelCopula.load(path)
    assert loaded.fit_result.method == cop.fit_result.method
    assert loaded.fit_result.log_likelihood == cop.fit_result.log_likelihood
    np.testing.assert_allclose(
        loaded.fit_result.params.values,
        cop.fit_result.params.values,
    )
    np.testing.assert_allclose(
        loaded.fit_result.params.bounds_lower,
        cop.fit_result.params.bounds_lower,
    )
    np.testing.assert_allclose(
        loaded.fit_result.params.bounds_upper,
        cop.fit_result.params.bounds_upper,
    )
    np.testing.assert_allclose(
        loaded.predict(4, rng=np.random.default_rng(2)),
        cop.predict(4, rng=np.random.default_rng(2)),
    )


def test_top_level_load_rejects_wrong_expected_type(tmp_path, random_u2):
    cop = GumbelCopula(rotate=180)
    cop.fit(random_u2, method="mle")

    path = tmp_path / "gumbel.json"
    cop.save(path)

    with pytest.raises(TypeError, match="Expected CVineCopula"):
        CVineCopula.load(path)
    assert isinstance(load_model(path), GumbelCopula)


def test_gaussian_copula_save_load_roundtrip(tmp_path):
    u = pobs(np.random.default_rng(6).standard_normal((120, 3)))
    cop = GaussianCopula()
    cop.fit(u)

    path = tmp_path / "gaussian.json"
    cop.save(path)
    loaded = GaussianCopula.load(path)

    np.testing.assert_allclose(loaded.corr, cop.corr)
    np.testing.assert_allclose(
        loaded.sample(5, rng=np.random.default_rng(7)),
        cop.sample(5, rng=np.random.default_rng(7)),
    )


def test_cvine_save_load_roundtrip(tmp_path):
    u = pobs(np.random.default_rng(2).standard_normal((120, 4)))
    vine = CVineCopula().fit(u, method="mle")

    path = tmp_path / "cvine.json"
    vine.save(path)
    loaded = CVineCopula.load(path)

    assert loaded.d == vine.d
    assert loaded.method == vine.method
    assert loaded.fit_result.log_likelihood == vine.fit_result.log_likelihood
    assert len(loaded.edges) == len(vine.edges)
    assert [
        [type(edge.copula).__name__ for edge in level]
        for level in loaded.edges
    ] == [
        [type(edge.copula).__name__ for edge in level]
        for level in vine.edges
    ]
    np.testing.assert_allclose(
        loaded.sample(5, rng=np.random.default_rng(3)),
        vine.sample(5, rng=np.random.default_rng(3)),
    )


def test_rvine_save_load_preserves_conditional_structure(tmp_path):
    rng = np.random.default_rng(4)
    sigma = np.array([
        [1.0, 0.7, 0.2, 0.1],
        [0.7, 1.0, 0.5, 0.2],
        [0.2, 0.5, 1.0, 0.6],
        [0.1, 0.2, 0.6, 1.0],
    ])
    u = norm.cdf(rng.multivariate_normal(np.zeros(4), sigma, size=240))
    vine = RVineCopula(candidates=[BivariateGaussianCopula]).fit(
        u,
        method="mle",
        given_vars=[3],
    )

    path = tmp_path / "rvine.json"
    vine.save(path)
    loaded = RVineCopula.load(path)

    np.testing.assert_array_equal(loaded.matrix, vine.matrix)
    assert loaded.trees == vine.trees
    assert loaded._edge_map == vine._edge_map
    assert loaded.candidates == vine.candidates
    assert loaded._target_given_vars == (3,)
    assert loaded._conditional_mode == "suffix"
    assert loaded._conditional_fit_supported is True
    assert loaded.fit_diagnostics == vine.fit_diagnostics
    assert loaded.log_likelihood() == vine.log_likelihood()

    samples, diagnostics = loaded.predict(
        10,
        given={3: 0.4},
        rng=np.random.default_rng(5),
        return_diagnostics=True,
    )
    assert samples.shape == (10, 4)
    assert diagnostics["conditional_method"] == "suffix"
    np.testing.assert_allclose(samples[:, 3], 0.4)
