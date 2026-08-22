"""Validation for the generic vine exact suffix subsystem."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from pyscarcopula import (
    BivariateGaussianCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    IndependentCopula,
    JoeCopula,
    VineCopula,
)
from pyscarcopula.vine import dvine_structure
from pyscarcopula.vine._rvine_matrix_builder import (
    build_rvine_matrix_with_edge_map,
)

from ._analytical_oracles import gaussian_conditional_parameters
from ._bivariate_oracles import conditional_cdf
from ._statistical_assertions import (
    assert_covariance_with_whitening,
    assert_mean_with_mc_error,
    assert_uniform_pit,
)
from ._vine_fixtures import (
    EdgeSpec,
    exact_given,
    exact_structure_cases,
    fitted_static_vine,
    gaussian_vine,
    homogeneous_vine,
    mixed_vine,
    peel_order,
    truncated_gaussian_vine,
)
from ._vine_oracles import (
    gaussian_vine_correlation,
    gaussian_vine_correlation_path,
    one_free_gaussian_pit,
)


STRUCTURES = exact_structure_cases()
STRUCTURE_IDS = tuple(STRUCTURES)
PATHS = ("direct", "rebuilt")

FAMILY_ROTATIONS = (
    [(IndependentCopula, 0), (BivariateGaussianCopula, 0), (FrankCopula, 0)]
    + [
        (family, rotation)
        for family in (ClaytonCopula, GumbelCopula, JoeCopula)
        for rotation in (0, 90, 180, 270)
    ]
)


def _assert_exact_contract(samples, diagnostics, given, *, rebuilt):
    assert samples.ndim == 2
    assert np.all(np.isfinite(samples))
    assert np.all((samples > 0.0) & (samples < 1.0))
    for variable, value in given.items():
        np.testing.assert_array_equal(
            samples[:, variable], np.full(len(samples), value)
        )
    assert diagnostics["conditional_method"] == "suffix"
    assert diagnostics["suffix_start_col"] is not None
    assert diagnostics["matrix_rebuilt"] is rebuilt
    assert diagnostics["given"] == given
    assert "mcmc" not in diagnostics


def _shallow_runtime(vine):
    clone = object.__new__(type(vine))
    clone.__dict__ = vine.__dict__.copy()
    return clone


@pytest.mark.parametrize("structure_id", STRUCTURE_IDS)
@pytest.mark.parametrize("path", PATHS)
def test_c_d_r_direct_and_rebuilt_suffix_contract(structure_id, path):
    vine = gaussian_vine(STRUCTURES[structure_id])
    given = exact_given(vine, path)

    samples, diagnostics = vine.predict(
        73,
        given=given,
        return_diagnostics=True,
        rng=np.random.default_rng(20262000),
    )

    _assert_exact_contract(
        samples, diagnostics, given, rebuilt=(path == "rebuilt")
    )


@pytest.mark.parametrize(
    ("family", "rotation"),
    FAMILY_ROTATIONS,
    ids=lambda value: getattr(value, "__name__", f"r{value}"),
)
def test_all_family_rotation_cells_execute_on_exact_suffix(family, rotation):
    index = FAMILY_ROTATIONS.index((family, rotation))
    structure_id = STRUCTURE_IDS[index % len(STRUCTURE_IDS)]
    path = PATHS[index % len(PATHS)]
    vine = homogeneous_vine(STRUCTURES[structure_id], family, rotation)
    given = exact_given(vine, path)

    samples, diagnostics = vine.predict(
        31,
        given=given,
        return_diagnostics=True,
        rng=np.random.default_rng(20262100 + index),
    )

    _assert_exact_contract(
        samples, diagnostics, given, rebuilt=(path == "rebuilt")
    )


@pytest.mark.parametrize("structure_id", STRUCTURE_IDS)
@pytest.mark.parametrize("path", PATHS)
def test_mixed_family_rotation_vines_remain_exact(structure_id, path):
    vine = mixed_vine(STRUCTURES[structure_id])
    given = exact_given(vine, path)

    samples, diagnostics = vine.predict(
        41,
        given=given,
        return_diagnostics=True,
        rng=np.random.default_rng(20262200),
    )

    _assert_exact_contract(
        samples, diagnostics, given, rebuilt=(path == "rebuilt")
    )


@pytest.mark.parametrize("path", PATHS)
def test_truncated_vine_with_independent_upper_trees_remains_exact(path):
    vine = truncated_gaussian_vine(STRUCTURES["r-vine"])
    assert all(
        isinstance(pair.copula, IndependentCopula)
        for (tree, _column), pair in vine.pair_copulas.items()
        if tree > 0
    )
    given = exact_given(vine, path)

    samples, diagnostics = vine.predict(
        43,
        given=given,
        return_diagnostics=True,
        rng=np.random.default_rng(20262300),
    )

    _assert_exact_contract(
        samples, diagnostics, given, rebuilt=(path == "rebuilt")
    )


def test_all_given_shortcut_reports_exact_suffix_diagnostics():
    vine = mixed_vine(STRUCTURES["r-vine"])
    given = {index: 0.12 + 0.15 * index for index in range(vine.d)}

    samples, diagnostics = vine.predict(
        9, given=given, return_diagnostics=True
    )

    expected = np.array([[given[index] for index in range(vine.d)]] * 9)
    np.testing.assert_array_equal(samples, expected)
    assert diagnostics == {
        "given": given,
        "dynamic_conditioning": "ignore",
        "suffix_start_col": 0,
        "matrix_rebuilt": False,
        "conditional_method": "suffix",
        "updated_edges": [],
        "skipped_edges": [],
        "all_variables_given": True,
        "timings_ms": diagnostics["timings_ms"],
    }


def test_rebuilt_representation_is_pointwise_equivalent_for_rotated_edges():
    vine = mixed_vine(STRUCTURES["d-vine"])
    given = exact_given(vine, "rebuilt")
    state = vine._suffix_sampling_state(given)
    assert state is not None
    start, matrix, edge_map, pair_copulas = state
    assert start == vine.d - len(given)
    assert not np.array_equal(matrix, vine.natural_order_matrix)

    equivalent = _shallow_runtime(vine)
    equivalent._natural_order_matrix = matrix.copy()
    equivalent._edge_map = dict(edge_map)
    equivalent._orig_edge_key = {
        (tree, original): (tree, column)
        for (tree, column), original in edge_map.items()
    }
    equivalent.pair_copulas = dict(pair_copulas)
    equivalent._suffix_state_cache = {}
    equivalent._predict_history_cache = {}

    rebuilt, rebuilt_diag = vine.predict(
        257,
        given=given,
        return_diagnostics=True,
        rng=np.random.default_rng(20262400),
    )
    direct, direct_diag = equivalent.predict(
        257,
        given=given,
        return_diagnostics=True,
        rng=np.random.default_rng(20262400),
    )

    assert rebuilt_diag["matrix_rebuilt"] is True
    assert direct_diag["matrix_rebuilt"] is False
    np.testing.assert_allclose(rebuilt, direct, rtol=0.0, atol=2e-12)


@pytest.mark.parametrize(
    ("family", "family_name"),
    [
        (ClaytonCopula, "clayton"),
        (GumbelCopula, "gumbel"),
        (JoeCopula, "joe"),
    ],
)
@pytest.mark.parametrize("rotation", [90, 270])
@pytest.mark.parametrize("given_index", [0, 1])
def test_asymmetric_rotation_suffix_uses_canonical_edge_orientation(
    family, family_name, rotation, given_index
):
    vine = homogeneous_vine(dvine_structure(2), family, rotation)
    given = {given_index: 0.37}
    free_index = 1 - given_index
    n = 257
    seed = 20262450 + 10 * rotation + given_index
    uniforms = np.random.default_rng(seed).uniform(0.0, 1.0, size=(n, 2))

    samples = vine.predict(
        n, given=given, rng=np.random.default_rng(seed)
    )
    pit = conditional_cdf(
        samples[:, free_index],
        given[given_index],
        vine.pair_copulas[(0, 0)].param,
        family_name,
        rotation,
        free_index,
    )

    np.testing.assert_array_equal(samples[:, given_index], given[given_index])
    np.testing.assert_allclose(pit, uniforms[:, 0], rtol=0.0, atol=2e-8)


@pytest.mark.parametrize("structure_id", STRUCTURE_IDS)
def test_gaussian_vine_correlation_oracle_reconstructs_known_matrix(
    structure_id,
):
    structure = STRUCTURES[structure_id]
    d = structure.d
    expected = np.fromfunction(
        lambda row, column: 0.48 ** np.abs(row - column),
        (d, d),
    )
    trees = structure.to_trees()

    def spec(tree, edge):
        conditioned, conditioning = trees[tree][edge]
        first, second = sorted(conditioned)
        fixed = np.array(sorted(conditioning), dtype=np.int64)
        if not len(fixed):
            partial = expected[first, second]
        else:
            r_dd = expected[np.ix_(fixed, fixed)]
            first_d = expected[first, fixed]
            second_d = expected[second, fixed]
            numerator = (
                expected[first, second]
                - first_d @ np.linalg.solve(r_dd, second_d)
            )
            first_variance = 1.0 - first_d @ np.linalg.solve(r_dd, first_d)
            second_variance = 1.0 - second_d @ np.linalg.solve(
                r_dd, second_d
            )
            partial = numerator / np.sqrt(first_variance * second_variance)
        return EdgeSpec(BivariateGaussianCopula, 0, float(partial))

    vine = fitted_static_vine(structure, spec)
    observed = gaussian_vine_correlation(vine)

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=2e-15)


@pytest.mark.parametrize("structure_id", STRUCTURE_IDS)
def test_fit_time_given_vars_strict_contract_for_fixed_structures(structure_id):
    structure = STRUCTURES[structure_id]
    trees = structure.to_trees()
    matrix, _edge_map = build_rvine_matrix_with_edge_map(
        structure.d, trees, validate=True
    )
    target = int(matrix[0, structure.d - 1])
    data = np.random.default_rng(20262500).uniform(
        0.03, 0.97, size=(37, structure.d)
    )
    specs = [
        [(IndependentCopula, 0) for _edge in level]
        for level in trees
    ]

    vine = VineCopula(structure=structure).fit(
        data,
        copulas=specs,
        given_vars=[target],
        conditional_strict=True,
    )
    samples, diagnostics = vine.predict(
        17,
        given={target: 0.61},
        return_diagnostics=True,
        rng=np.random.default_rng(20262501),
    )

    assert vine.fit_diagnostics["conditional_fit_supported"] is True
    assert vine.fit_diagnostics["target_given_vars"] == (target,)
    _assert_exact_contract(samples, diagnostics, {target: 0.61}, rebuilt=False)


def test_auto_selected_rvine_exposes_exact_suffix_path():
    data = np.random.default_rng(20262600).uniform(0.02, 0.98, size=(53, 5))
    vine = VineCopula.rvine(candidates=[IndependentCopula]).fit(data)
    target = peel_order(vine)[-1]

    samples, diagnostics = vine.predict(
        19,
        given={target: 0.44},
        return_diagnostics=True,
        rng=np.random.default_rng(20262601),
    )

    assert vine.structure_source == "auto"
    _assert_exact_contract(samples, diagnostics, {target: 0.44}, rebuilt=False)


@pytest.mark.validation
@pytest.mark.parametrize("structure_id", STRUCTURE_IDS)
@pytest.mark.parametrize("path", PATHS)
def test_gaussian_exact_suffix_matches_full_mvn_oracle(structure_id, path):
    vine = gaussian_vine(STRUCTURES[structure_id])
    given = exact_given(vine, path)
    correlation = gaussian_vine_correlation(vine)
    oracle = gaussian_conditional_parameters(correlation, given)

    samples, diagnostics = vine.predict(
        9000,
        given=given,
        return_diagnostics=True,
        rng=np.random.default_rng(20262700),
    )
    latent = norm.ppf(np.clip(
        samples[:, oracle.free_indices], 1e-12, 1.0 - 1e-12
    ))

    _assert_exact_contract(
        samples, diagnostics, given, rebuilt=(path == "rebuilt")
    )
    assert_mean_with_mc_error(latent, oracle.mean, oracle.covariance)
    assert_covariance_with_whitening(
        latent, oracle.mean, oracle.covariance
    )


@pytest.mark.validation
@pytest.mark.parametrize("structure_id", STRUCTURE_IDS)
@pytest.mark.parametrize("path", PATHS)
def test_independent_exact_suffix_has_joint_uniform_free_coordinates(
    structure_id, path
):
    vine = homogeneous_vine(STRUCTURES[structure_id], IndependentCopula)
    given = exact_given(vine, path)
    samples = vine.predict(
        5000,
        given=given,
        rng=np.random.default_rng(20262800),
    )
    free = [index for index in range(vine.d) if index not in given]

    for index in free:
        assert_uniform_pit(samples[:, index])
    if len(free) > 1:
        dependence = np.corrcoef(samples[:, free], rowvar=False)
        off_diagonal = dependence - np.eye(len(free))
        assert np.max(np.abs(off_diagonal)) < 0.055


@pytest.mark.validation
@pytest.mark.parametrize("path", PATHS)
def test_truncated_gaussian_suffix_matches_implied_mvn(path):
    vine = truncated_gaussian_vine(STRUCTURES["r-vine"])
    given = exact_given(vine, path)
    oracle = gaussian_conditional_parameters(
        gaussian_vine_correlation(vine), given
    )

    samples = vine.predict(
        9000,
        given=given,
        rng=np.random.default_rng(20262900),
    )
    latent = norm.ppf(np.clip(
        samples[:, oracle.free_indices], 1e-12, 1.0 - 1e-12
    ))

    assert_mean_with_mc_error(latent, oracle.mean, oracle.covariance)
    assert_covariance_with_whitening(
        latent, oracle.mean, oracle.covariance
    )


@pytest.mark.validation
@pytest.mark.parametrize(
    ("path", "free_variable"),
    [("direct", 1), ("rebuilt", 3)],
)
def test_rowwise_dynamic_edge_parameters_follow_component_oracle(
    monkeypatch, path, free_variable
):
    vine = gaussian_vine(STRUCTURES["c-vine"])
    given = {
        index: 0.18 + 0.14 * index
        for index in range(vine.d)
        if index != free_variable
    }
    state = vine._suffix_sampling_state(given)
    assert state is not None
    assert np.array_equal(state[1], vine.natural_order_matrix) is (
        path == "direct"
    )
    runtime = _shallow_runtime(vine)
    runtime._natural_order_matrix = state[1]
    runtime._edge_map = state[2]
    runtime.pair_copulas = state[3]
    captured = {}

    def deterministic_paths(keys, pair_copulas, _edge_map, n, *_args, **_kwargs):
        grid = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        result = {}
        for offset, key in enumerate(keys):
            base = float(pair_copulas[key].param)
            result[key] = base + 0.055 * np.sin(grid + 0.31 * offset)
        captured.update(result)
        return result

    monkeypatch.setattr(vine, "_predict_r_for_edges", deterministic_paths)
    n = 10000
    samples = vine.predict(
        n,
        given=given,
        rng=np.random.default_rng(20263000),
    )
    correlations = gaussian_vine_correlation_path(runtime, captured, n)
    pit = one_free_gaussian_pit(samples, correlations, given)

    assert_uniform_pit(pit, numerical_floor=0.006)
