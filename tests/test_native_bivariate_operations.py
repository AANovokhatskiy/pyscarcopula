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
from pyscarcopula._native import _descriptors as _cpp_copula, _extension as _cpp_extension, pair as copula_native


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


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(
            lambda copula: copula.log_pdf(0.2 + 7.0j, 0.4, 0.8),
            id="log-pdf-observation",
        ),
        pytest.param(
            lambda copula: copula.transform([0.2 + 7.0j]),
            id="transform",
        ),
        pytest.param(
            lambda copula: copula.tau_to_param([0.4 + 7.0j]),
            id="tau-to-param",
        ),
        pytest.param(
            lambda copula: copula.param_to_tau([0.8 + 7.0j]),
            id="param-to-tau",
        ),
        pytest.param(
            lambda copula: copula.sample_at_parameter(2, 0.8 + 7.0j),
            id="sampling-parameter",
        ),
        pytest.param(
            lambda copula: copula.pdf_on_grid(
                [0.2 + 7.0j, 0.4], [-1.0, 1.0]),
            id="grid-observation",
        ),
    ],
)
def test_bivariate_public_operations_reject_complex_inputs(operation):
    with pytest.raises(TypeError, match="real values"):
        operation(ClaytonCopula())


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
    transposed_rotation = {
        0: 0,
        90: 270,
        180: 180,
        270: 90,
    }[rotation]
    np.testing.assert_allclose(
        h_vu,
        factory(rotate=transposed_rotation).h(
            v,
            u,
            np.full(len(u), param),
        ),
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
        "copula_sample_from_uniforms",
        "copula_sample_from_rng_draws",
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
    rotation = int(getattr(copula, "rotate", 0))
    transposed = (
        type(copula)(rotate=360 - rotation)
        if rotation in (90, 270)
        else copula
    )
    np.testing.assert_allclose(h_vu, transposed.h(v, u, r))
    np.testing.assert_allclose(copula.h_inverse(h_uv, v, r), u, atol=2e-8)

    uniforms = np.array([[0.17, 0.31], [0.43, 0.59], [0.83, 0.71]])
    samples = copula_native.sample_from_uniforms(copula, uniforms, r)

    transposed_rotation = {
        0: 0,
        90: 270,
        180: 180,
        270: 90,
    }[rotation]
    transposed = (
        type(copula)(rotate=transposed_rotation)
        if transposed_rotation != rotation else copula
    )
    np.testing.assert_array_equal(samples[:, 0], uniforms[:, 0])
    # Preserve the established family-kernel precision contract.
    np.testing.assert_allclose(
        transposed.h(samples[:, 1], uniforms[:, 0], r),
        uniforms[:, 1],
        rtol=0.0,
        atol=2e-8,
    )

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
@pytest.mark.parametrize(
    "transform_type", ["softplus", "xtanh", "exp", "logistic"])
def test_native_transforms_match_public_formula_contract(
        factory, offset, transform_type):
    copula = factory(transform_type=transform_type)
    x = np.array([-30.0, -1.0, 0.0, 1.0, 30.0])
    if transform_type == "softplus":
        expected_r = np.logaddexp(0.0, x) + offset
        expected_d = 1.0 / (1.0 + np.exp(-x))
        shifted = expected_r - offset
        expected_x = np.where(
            shifted > 20.0,
            shifted,
            np.log(np.expm1(shifted)),
        )
    elif transform_type == "xtanh":
        tanh_x = np.tanh(x)
        expected_r = x * tanh_x + offset
        expected_d = tanh_x + x * (1.0 - tanh_x * tanh_x)
        expected_x = np.abs(expected_r) + offset
    elif transform_type == "exp":
        expected_r = np.exp(x) + offset
        expected_d = np.exp(x)
        expected_x = np.log(expected_r - offset)
    else:
        probability = 1.0 / (1.0 + np.exp(-x / 2.0))
        expected_r = offset + 20.0 * probability
        expected_d = 10.0 * probability * (1.0 - probability)
        expected_x = 2.0 * (
            np.log(probability) - np.log1p(-probability))

    np.testing.assert_allclose(copula.transform(x), expected_r)
    np.testing.assert_allclose(copula.dtransform(x), expected_d)
    np.testing.assert_allclose(
        copula.inv_transform(expected_r),
        expected_x,
        atol=4.0 * np.finfo(np.float64).eps,
    )


@pytest.mark.parametrize(
    "factory,offset",
    [
        (ClaytonCopula, 0.0001),
        (FrankCopula, 0.0001),
        (GumbelCopula, 1.0001),
        (JoeCopula, 1.0001),
    ],
)
@pytest.mark.parametrize("transform_type", ["exp", "logistic"])
def test_bounded_inverse_transforms_reject_parameters_outside_domain(
        factory, offset, transform_type):
    copula = factory(transform_type=transform_type)
    upper = offset + 20.0 if transform_type == "logistic" else None

    with pytest.raises(ValueError, match="inverse-transform parameter"):
        copula.inv_transform([offset - 1e-12])
    if upper is not None:
        with pytest.raises(ValueError, match="inverse-transform parameter"):
            copula.inv_transform([upper + 1e-12])


@pytest.mark.parametrize(
    "factory,offset",
    [
        (ClaytonCopula, 0.0001),
        (FrankCopula, 0.0001),
        (GumbelCopula, 1.0001),
        (JoeCopula, 1.0001),
    ],
)
@pytest.mark.parametrize("transform_type", ["exp", "logistic"])
def test_bounded_inverse_transforms_define_finite_boundary_initializers(
        factory, offset, transform_type):
    copula = factory(transform_type=transform_type)
    parameters = [offset]
    if transform_type == "logistic":
        parameters.append(offset + 20.0)

    latent = copula.inv_transform(parameters)
    assert np.all(np.isfinite(latent))


@pytest.mark.parametrize(
    "factory",
    [ClaytonCopula, FrankCopula, GumbelCopula, JoeCopula],
)
def test_archimedean_default_transform_remains_softplus(factory):
    copula = factory()
    module = _cpp_extension.load()
    spec = _cpp_copula.make_copula_ops_spec(module, copula)

    assert copula._transform_type == "softplus"
    assert spec.transform == module.Transform.Softplus


@pytest.mark.parametrize(
    "factory",
    [ClaytonCopula, FrankCopula, GumbelCopula, JoeCopula],
)
@pytest.mark.parametrize(
    "transform_type", ["softplus", "xtanh", "exp", "logistic"])
def test_archimedean_fast_path_gradient_uses_selected_transform(
        factory, transform_type):
    copula = factory(transform_type=transform_type)
    observations = np.array([[0.23, 0.71], [0.64, 0.38]])
    x = np.array([-1.2, 0.4, 1.3])
    step = 1e-6

    pdf, gradient = copula.pdf_and_grad_on_grid_batch(observations, x)
    plus = copula.copula_grid_batch(observations, x + step)
    minus = copula.copula_grid_batch(observations, x - step)
    finite_difference = (plus - minus) / (2.0 * step)

    assert np.all(np.isfinite(pdf))
    assert np.all(np.isfinite(gradient))
    np.testing.assert_allclose(
        gradient, finite_difference, rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize(
    "factory,expected_upper",
    [
        (ClaytonCopula, 20.0001),
        (FrankCopula, 20.0001),
        (GumbelCopula, 21.0001),
        (JoeCopula, 21.0001),
    ],
)
def test_logistic_transform_exposes_bounded_parameter_domain(
        factory, expected_upper):
    copula = factory(transform_type="logistic")
    assert copula.bounds[0][1] == pytest.approx(expected_upper)


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


def test_pair_sampling_formulas_are_removed_from_production_python():
    from pyscarcopula.copula import base, frank, gumbel, joe

    assert not hasattr(base.BivariateCopula, "psi")
    assert not hasattr(base.BivariateCopula, "V")
    assert not hasattr(frank, "_frank_bivariate_sample_from_uniforms")
    assert not hasattr(gumbel, "_generate_levy_stable_from_uniforms")
    assert not hasattr(joe, "_joe_v_from_uniforms")


def test_pair_h_inverse_has_one_native_implementation():
    cpp_root = Path(__file__).parents[1] / "pyscarcopula" / "_cpp"
    production = "\n".join(
        path.read_text(encoding="utf-8")
        for directory in (cpp_root / "include", cpp_root / "src")
        for path in directory.rglob("*")
        if path.suffix in {".cpp", ".hpp"}
    )
    assert "sample_inverse_h" not in production


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_joe_h_inverse_preserves_legacy_sampling_accuracy(rotation):
    copula = JoeCopula(rotate=rotation)
    quantiles = np.array([0.031, 0.173, 0.421, 0.793, 0.941])
    given = np.array([0.887, 0.509, 0.257, 0.619, 0.113])
    parameter = np.full(len(quantiles), 1.72)
    inverse = copula.h_inverse(quantiles, given, parameter)
    np.testing.assert_allclose(
        copula.h(inverse, given, parameter),
        quantiles,
        rtol=0.0,
        atol=1e-10,
    )


def test_gaussian_h_inverse_preserves_legacy_sampling_accuracy():
    from scipy.stats import norm

    copula = BivariateGaussianCopula()
    quantiles = np.array([0.031, 0.173, 0.421, 0.793, 0.941])
    given = np.array([0.887, 0.509, 0.257, 0.619, 0.113])
    parameter = np.array([-0.73, -0.21, 0.0, 0.47, 0.82])
    expected = norm.cdf(
        norm.ppf(quantiles) * np.sqrt(1.0 - parameter**2)
        + parameter * norm.ppf(given)
    )
    np.testing.assert_allclose(
        copula.h_inverse(quantiles, given, parameter),
        expected,
        rtol=0.0,
        atol=5e-9,
    )


def test_adapter_is_the_base_operation_surface(monkeypatch):
    sentinel = np.array([0.125])
    monkeypatch.setattr(
        copula_native,
        "pdf",
        lambda copula, u1, u2, r, unrotated=False: sentinel,
    )
    assert ClaytonCopula().pdf_unrotated(0.2, 0.3, 0.8) is sentinel
