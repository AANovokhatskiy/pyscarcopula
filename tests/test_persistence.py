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


def test_bivariate_save_load_roundtrip(tmp_path, random_u2):
    cop = GumbelCopula(rotate=180)
    cop.fit(random_u2, method="mle")

    path = tmp_path / "gumbel.json"
    cop.save(path)
    loaded = GumbelCopula.load(path)

    envelope = json.loads(path.read_text(encoding="utf-8"))
    assert envelope["format"] == "pyscarcopula-model"
    assert envelope["format_version"] == 2
    assert envelope["class"] == "pyscarcopula.copula.gumbel.GumbelCopula"

    assert isinstance(loaded, GumbelCopula)
    assert loaded.rotate == cop.rotate
    assert loaded.fit_result == cop.fit_result
    np.testing.assert_allclose(loaded._last_u, cop._last_u)

    samples = loaded.predict(8, rng=np.random.default_rng(0))
    assert samples.shape == (8, 2)
    assert np.all((samples > 0.0) & (samples < 1.0))


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
    cop.fit(random_u2, method="scar-tm-ou", K=20, tol=0.5)

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
