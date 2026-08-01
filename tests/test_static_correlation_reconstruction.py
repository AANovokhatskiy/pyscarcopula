from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import norm

from pyscarcopula import GaussianCopula, StudentCopula
from pyscarcopula._parallel import create_worker_model, get_copula_constructor


CORRELATION = np.array([
    [1.0, 0.42, 0.08],
    [0.42, 1.0, -0.18],
    [0.08, -0.18, 1.0],
])
LOADINGS = np.array([[0.48], [0.31], [-0.24]])


def _observations(seed=1201, rows=100):
    latent = np.random.default_rng(seed).multivariate_normal(
        np.zeros(3), CORRELATION, size=rows)
    return norm.cdf(latent)


@pytest.mark.parametrize("model_type", [GaussianCopula, StudentCopula])
@pytest.mark.parametrize("corr_mode", ["fixed", "shrinkage", "cholesky"])
def test_worker_reconstruction_preserves_dense_constructor_policy(
        model_type, corr_mode):
    kwargs = {
        "d": 3,
        "R": CORRELATION,
        "corr_mode": corr_mode,
        "corr_shrinkage_init": 0.63,
        "cholesky_d_max": 4,
        "allow_large_cholesky": True,
    }
    if corr_mode != "fixed":
        kwargs["corr_base"] = np.eye(3)
    source = model_type(**kwargs)
    fit_kwargs = (
        {"maxiter": 200}
        if model_type is StudentCopula or corr_mode != "fixed" else {})
    source.fit(_observations(), **fit_kwargs)

    rebuilt = create_worker_model(source)

    assert type(rebuilt) is model_type
    assert rebuilt.corr_mode == corr_mode
    assert rebuilt.fit_result is None
    assert rebuilt._corr_shrinkage_init == pytest.approx(0.63)
    assert rebuilt._cholesky_d_max == 4
    assert rebuilt._allow_large_cholesky is True
    np.testing.assert_allclose(
        rebuilt._constructor_R, CORRELATION, atol=3e-16, rtol=0.0)
    if corr_mode != "fixed":
        np.testing.assert_array_equal(
            rebuilt._constructor_corr_base, np.eye(3))


@pytest.mark.parametrize("model_type", [GaussianCopula, StudentCopula])
def test_worker_reconstruction_preserves_compact_factor_policy(model_type):
    source = model_type(
        3,
        corr_mode="factor",
        factor_rank=1,
        factor_loadings=LOADINGS,
        factor_tile_size=7,
        factor_uniqueness_min=1e-7,
        factor_seed=13,
        factor_oversampling=5,
    )
    source.fit(_observations())

    rebuilt = create_worker_model(source)

    assert type(rebuilt) is model_type
    assert rebuilt.corr_mode == "factor"
    assert rebuilt.factor_rank == 1
    assert rebuilt.factor_estimation == "two-stage"
    assert rebuilt.fit_result is None
    assert rebuilt._factor_operator is not None
    np.testing.assert_array_equal(rebuilt.factor_loadings_, LOADINGS)
    assert rebuilt.to_correlation_matrix().shape == (3, 3)


@pytest.mark.parametrize("model_type", [GaussianCopula, StudentCopula])
@pytest.mark.parametrize(
    "model_kwargs",
    [
        {"R": CORRELATION},
        {"corr_mode": "shrinkage", "corr_base": CORRELATION},
        {"corr_mode": "cholesky", "corr_base": CORRELATION},
        {
            "d": 3,
            "corr_mode": "factor",
            "factor_rank": 1,
            "factor_loadings": LOADINGS,
        },
    ],
)
def test_json_roundtrip_preserves_static_policy_and_fitted_state(
        model_type, model_kwargs, tmp_path):
    model = model_type(**model_kwargs)
    fit_kwargs = (
        {"maxiter": 250}
        if model_type is StudentCopula
        or model.corr_mode in {"shrinkage", "cholesky"}
        else {})
    result = model.fit(_observations(seed=1202), **fit_kwargs)
    path = tmp_path / f"{model_type.__name__}-{model.corr_mode}.json"

    model.save(path)
    loaded = model_type.load(path)

    assert loaded.corr_mode == model.corr_mode
    assert loaded.corr_estimator_ == model.corr_estimator_
    assert loaded.fit_result.log_likelihood == pytest.approx(
        result.log_likelihood)
    assert loaded.fit_result.parameter_count == result.parameter_count
    np.testing.assert_array_equal(
        loaded.fit_result.model_parameters["corr_params_raw"],
        result.model_parameters["corr_params_raw"],
    )
    if model.corr_mode == "factor":
        assert loaded.fit_result.correlation_matrix is None
        assert loaded._factor_operator is not None
        np.testing.assert_array_equal(
            loaded.factor_loadings_, model.factor_loadings_)
    else:
        np.testing.assert_array_equal(
            loaded.to_correlation_matrix(), model.to_correlation_matrix())


def test_legacy_gaussian_state_normalizes_dense_alias_and_metadata():
    model = GaussianCopula(R=CORRELATION)
    model.fit(_observations(seed=1203))
    state = model.__getstate__()
    state["_corr_mode"] = "dense"
    state["fit_result"] = replace(
        model.fit_result, diagnostics={}, model_parameters={})
    for key in ("_corr_params_raw", "_corr_alpha"):
        state.pop(key, None)

    restored = GaussianCopula.__new__(GaussianCopula)
    restored.__setstate__(state)

    assert restored.corr_mode == "fixed"
    assert restored.fit_result.diagnostics["corr_mode"] == "fixed"
    assert restored.fit_result.diagnostics["corr_mode_migrated_from"] == (
        "dense")
    assert "final_validation_passed" in restored.fit_result.diagnostics
    assert restored.fit_result.model_parameters["corr_estimator"] == (
        "supplied")


def test_legacy_student_fitted_shape_is_kendall_plugin_not_supplied():
    model = StudentCopula()
    model.fit(_observations(seed=1204))
    state = model.__getstate__()
    state["shape"] = state.pop("_correlation")
    state["fit_result"] = replace(
        model.fit_result,
        diagnostics={},
        model_parameters={"df": model.df},
    )
    for key in (
            "_corr_mode", "_corr_estimator", "_constructor_R",
            "_constructor_corr_base", "_corr_params_raw", "_corr_alpha"):
        state.pop(key, None)

    restored = StudentCopula.__new__(StudentCopula)
    restored.__setstate__(state)

    assert restored.corr_mode == "fixed"
    assert restored.corr_estimator_ == "kendall_plugin"
    assert restored._constructor_R is None
    assert restored.fit_result.diagnostics["corr_plugin_n_params"] == 3
    assert restored.fit_result.diagnostics["corr_policy_migration"] == (
        "legacy_fixed_kendall_plugin")
    assert restored.fit_result.model_parameters["corr_estimator"] == (
        "kendall_plugin")


def test_student_joint_factor_worker_preserves_identification_settings():
    source = StudentCopula(
        3,
        corr_mode="factor",
        factor_rank=1,
        factor_loadings=LOADINGS,
        factor_estimation="joint",
        factor_joint_max_params=9,
        factor_joint_penalty=2e-5,
        factor_joint_condition_max=4e8,
    )
    model_type, kwargs = get_copula_constructor(source)
    rebuilt = model_type(**kwargs)

    assert rebuilt.factor_estimation == "joint"
    assert rebuilt._factor_joint_max_params == 9
    assert rebuilt._factor_joint_penalty == pytest.approx(2e-5)
    assert rebuilt._factor_joint_condition_max == pytest.approx(4e8)
