"""Contracts for the shared native bivariate operation path."""

from pathlib import Path

import numpy as np
import pytest

from pyscarcopula import (
    BivariateGaussianCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    IndependentCopula,
    JoeCopula,
)
from pyscarcopula.copula.base import (
    _inv_xtanh_transform,
    _softplus_dtransform,
    _softplus_inv_transform,
    _softplus_transform,
    _xtanh_dtransform,
    _xtanh_transform,
)
from pyscarcopula.numerical import _cpp_copula, _cpp_extension, copula_native


_FAMILIES = [
    (lambda: IndependentCopula(), 0.0),
    (lambda: ClaytonCopula(rotate=90), 0.8),
    (lambda: GumbelCopula(rotate=180), 1.6),
    (lambda: FrankCopula(), 2.0),
    (lambda: JoeCopula(rotate=270), 1.7),
    (lambda: BivariateGaussianCopula(), 0.35),
]


_ROTATED_ARCHIMEDEAN_FAMILIES = [
    (ClaytonCopula, 1.4),
    (GumbelCopula, 1.8),
    (JoeCopula, 2.0),
]


def _rotated_coordinates(first, second, rotation):
    rotated_first = np.asarray(first, dtype=np.float64).copy()
    rotated_second = np.asarray(second, dtype=np.float64).copy()
    if rotation in (90, 180):
        rotated_first = 1.0 - rotated_first
    if rotation in (180, 270):
        rotated_second = 1.0 - rotated_second
    return rotated_first, rotated_second


@pytest.mark.parametrize(
    "factory,param",
    _ROTATED_ARCHIMEDEAN_FAMILIES,
)
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_rotated_h_uses_explicit_rotation_identity(
        factory, param, rotation):
    base = factory(rotate=0)
    rotated = factory(rotate=rotation)
    u = np.array([1e-8, 0.2, 0.55, 0.8, 1.0 - 1e-8])
    v = np.array([0.31, 1e-8, 0.65, 1.0 - 1e-8, 0.72])
    transformed_u, transformed_v = _rotated_coordinates(u, v, rotation)

    expected = base.h(
        transformed_u,
        transformed_v,
        np.full(len(u), param),
    )
    if rotation in (90, 180):
        expected = 1.0 - expected

    actual = rotated.h(u, v, np.full(len(u), param))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-12)

    h_uv, h_vu = rotated.h_pair(u, v, np.full(len(u), param))
    np.testing.assert_allclose(h_uv, actual, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        h_vu,
        rotated.h(v, u, np.full(len(u), param)),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "factory,param",
    _ROTATED_ARCHIMEDEAN_FAMILIES,
)
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_rotated_h_inverse_identity_and_roundtrip(
        factory, param, rotation):
    base = factory(rotate=0)
    rotated = factory(rotate=rotation)
    u = np.array([0.08, 0.2, 0.55, 0.8, 0.92])
    given = np.array([0.31, 0.15, 0.65, 0.87, 0.72])
    parameter = np.full(len(u), param)
    q = rotated.h(u, given, parameter)
    transformed_q, transformed_given = _rotated_coordinates(
        q, given, rotation)

    expected = base.h_inverse(
        transformed_q,
        transformed_given,
        parameter,
    )
    if rotation in (90, 180):
        expected = 1.0 - expected

    actual = rotated.h_inverse(q, given, parameter)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-12)
    np.testing.assert_allclose(actual, u, rtol=0.0, atol=3e-8)


@pytest.mark.parametrize(
    "factory,param",
    _ROTATED_ARCHIMEDEAN_FAMILIES,
)
def test_native_h_rejects_invalid_rotation_instead_of_using_270(
        factory, param):
    module = _cpp_extension.load()
    spec = _cpp_copula.make_copula_ops_spec(module, factory())
    spec.rotation = module.Rotation(45)
    observations = np.array([[0.27, 0.63]], dtype=np.float64)
    parameter = np.array([param], dtype=np.float64)

    h_value = np.asarray(
        module.copula_h(spec, observations, parameter),
        dtype=np.float64,
    )
    inverse_value = np.asarray(
        module.copula_h_inverse(spec, observations, parameter),
        dtype=np.float64,
    )

    assert np.isnan(h_value[0])
    assert np.isnan(inverse_value[0])


def test_pybind_exports_complete_bivariate_operation_surface():
    module = _cpp_extension.load()
    expected = {
        "copula_transform",
        "copula_inverse_transform",
        "copula_dtransform",
        "copula_pdf",
        "copula_log_pdf",
        "copula_dlog_pdf_dr",
        "copula_h",
        "copula_h_pair",
        "copula_h_inverse",
        "copula_pdf_grid",
        "copula_pdf_and_grad_grid",
    }
    assert expected <= set(dir(module))


@pytest.mark.parametrize("factory,param", _FAMILIES)
def test_direct_family_operations_use_shared_native_adapter(factory, param):
    copula = factory()
    u = np.array([0.2, 0.55, 0.8])
    v = np.array([0.3, 0.65, 0.7])
    r = np.full(3, param)
    x = np.array([-0.5, 0.0, 0.5])
    observations = np.column_stack((u, v))

    assert copula.transform(x).shape == (3,)
    assert copula.inv_transform(r).shape == (3,)
    assert copula.dtransform(x).shape == (3,)
    assert copula.pdf(u, v, r).shape == (3,)
    assert copula.log_pdf(u, v, r).shape == (3,)
    assert copula.dlog_pdf_dr(u, v, r).shape == (3,)
    assert copula.h(u, v, r).shape == (3,)
    h_uv, h_vu = copula.h_pair(u, v, r)
    np.testing.assert_allclose(h_uv, copula.h(u, v, r))
    np.testing.assert_allclose(h_vu, copula.h(v, u, r))
    np.testing.assert_allclose(copula.h_inverse(h_uv, v, r), u, atol=2e-8)

    grid = copula.copula_grid_batch(observations, x)
    grid_pdf, grid_grad = copula.pdf_and_grad_on_grid_batch(
        observations, x)
    assert grid.shape == grid_pdf.shape == grid_grad.shape == (3, 3)
    np.testing.assert_allclose(grid, grid_pdf)


@pytest.mark.parametrize(
    "factory,offset",
    [
        (ClaytonCopula, 0.0001),
        (FrankCopula, 0.0001),
        (GumbelCopula, 1.0001),
        (JoeCopula, 1.0001),
    ],
)
@pytest.mark.parametrize("transform_type", ["softplus", "xtanh"])
def test_native_transforms_preserve_python_contract(
        factory, offset, transform_type):
    copula = factory(transform_type=transform_type)
    x = np.array([-30.0, -1.0, 0.0, 1.0, 30.0])
    if transform_type == "softplus":
        expected_r = _softplus_transform(x, offset)
        expected_d = _softplus_dtransform(x)
        expected_x = _softplus_inv_transform(expected_r, offset)
    else:
        expected_r = _xtanh_transform(x, offset)
        expected_d = _xtanh_dtransform(x)
        expected_x = _inv_xtanh_transform(expected_r, offset)

    np.testing.assert_allclose(copula.transform(x), expected_r)
    np.testing.assert_allclose(copula.dtransform(x), expected_d)
    np.testing.assert_allclose(copula.inv_transform(expected_r), expected_x)


@pytest.mark.parametrize(
    "factory",
    [ClaytonCopula, FrankCopula, GumbelCopula, JoeCopula],
)
def test_native_xtanh_inverse_is_modulus_approximation_not_roundtrip(factory):
    copula = factory(transform_type="xtanh")
    parameter = np.asarray(copula.bounds[0][0] + np.array([0.2, 1.0, 4.0]))
    latent = copula.inv_transform(parameter)

    np.testing.assert_allclose(
        latent,
        np.abs(parameter) + copula.bounds[0][0],
    )
    assert not np.allclose(
        copula.transform(latent),
        parameter,
        rtol=1e-10,
        atol=1e-10,
    )


def test_rvine_has_no_separate_cpp_pair_operation_router():
    root = Path(__file__).resolve().parents[1] / "pyscarcopula" / "vine"
    source = (root / "_rvine_edges.py").read_text(encoding="utf-8")
    assert "_cpp_scar_ou" not in source
    assert "_try_cpp_h" not in source


def test_family_python_kernels_are_removed():
    import pyscarcopula.copula.clayton as clayton_module

    for name in (
        "_clayton_pdf",
        "_clayton_log_pdf",
        "_clayton_dlogc_dr",
        "_clayton_h",
        "_clayton_h_pair",
        "_clayton_h_inv",
        "_clayton_pdf_and_grad_batch",
    ):
        assert not hasattr(clayton_module, name)


def test_adapter_is_the_base_operation_surface(monkeypatch):
    sentinel = np.array([0.125])
    monkeypatch.setattr(
        copula_native,
        "pdf",
        lambda copula, u1, u2, r, unrotated=False: sentinel,
    )
    assert ClaytonCopula().pdf_unrotated(0.2, 0.3, 0.8) is sentinel
