"""Independent analytical and pyvine references for exact vine sampling."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.stats import norm

from pyscarcopula import IndependentCopula


def gaussian_vine_correlation(
    vine,
    parameter_paths: Mapping[tuple[int, int], object] | None = None,
    *,
    row: int | None = None,
) -> np.ndarray:
    """Recover a Gaussian correlation matrix from vine partial correlations.

    The recursion uses only the mathematical partial-correlation identity and
    semantic edge sets.  It does not import a production sampling helper.
    """

    d = int(vine.d)
    correlation = np.eye(d, dtype=np.float64)
    original_to_key = {
        (tree, original): key
        for key, original in vine._edge_map.items()
        for tree in (key[0],)
    }

    for tree, level in enumerate(vine.trees):
        for original, (conditioned, conditioning) in enumerate(level):
            first, second = sorted(conditioned)
            key = original_to_key[(tree, original)]
            pair = vine.pair_copulas[key]
            if isinstance(pair.copula, IndependentCopula):
                partial = 0.0
            elif parameter_paths is None:
                partial = float(pair.param)
            else:
                value = np.asarray(parameter_paths[key], dtype=np.float64)
                partial = float(value if value.ndim == 0 else value[row])
            if not np.isfinite(partial) or abs(partial) >= 1.0:
                raise ValueError(f"invalid Gaussian partial correlation {partial}")

            fixed = np.array(sorted(conditioning), dtype=np.int64)
            if not len(fixed):
                unconditional = partial
            else:
                r_dd = correlation[np.ix_(fixed, fixed)]
                first_d = correlation[first, fixed]
                second_d = correlation[second, fixed]
                solve_first = np.linalg.solve(r_dd, first_d)
                solve_second = np.linalg.solve(r_dd, second_d)
                first_variance = 1.0 - first_d @ solve_first
                second_variance = 1.0 - second_d @ solve_second
                conditional_location = first_d @ solve_second
                unconditional = (
                    conditional_location
                    + partial * np.sqrt(first_variance * second_variance)
                )
            correlation[first, second] = unconditional
            correlation[second, first] = unconditional

    correlation = 0.5 * (correlation + correlation.T)
    np.linalg.cholesky(correlation)
    return correlation


def gaussian_vine_correlation_path(vine, parameter_paths, n: int) -> np.ndarray:
    return np.stack([
        gaussian_vine_correlation(vine, parameter_paths, row=row)
        for row in range(n)
    ])


def one_free_gaussian_pit(samples, correlations, given) -> np.ndarray:
    """Conditional Gaussian PIT for row-wise correlation matrices."""

    values = np.asarray(samples, dtype=np.float64)
    matrices = np.asarray(correlations, dtype=np.float64)
    if matrices.ndim == 2:
        matrices = np.broadcast_to(matrices, (len(values), *matrices.shape))
    given_indices = np.array(sorted(given), dtype=np.int64)
    free_indices = [
        index for index in range(values.shape[1])
        if index not in given
    ]
    if len(free_indices) != 1:
        raise ValueError("one_free_gaussian_pit requires exactly one free variable")
    free = free_indices[0]
    z_given = norm.ppf(np.array([given[int(i)] for i in given_indices]))
    z_free = norm.ppf(np.clip(values[:, free], 1e-12, 1.0 - 1e-12))
    pit = np.empty(len(values), dtype=np.float64)
    for row, matrix in enumerate(matrices):
        r_gg = matrix[np.ix_(given_indices, given_indices)]
        r_fg = matrix[free, given_indices]
        solved = np.linalg.solve(r_gg, r_fg)
        mean = float(r_fg @ np.linalg.solve(r_gg, z_given))
        variance = float(1.0 - r_fg @ solved)
        pit[row] = norm.cdf((z_free[row] - mean) / np.sqrt(variance))
    return pit


def pyvine_matrix(natural_matrix) -> np.ndarray:
    """Convert the production natural-order layout to pyvine's layout."""

    source = np.asarray(natural_matrix, dtype=np.int64)
    d = source.shape[0]
    target = np.zeros_like(source, dtype=np.uint64)
    for column in range(d):
        length = d - column
        target[:length - 1, column] = (
            source[:length - 1, column][::-1] + 1
        )
        target[length - 1, column] = source[length - 1, column] + 1
    return np.asfortranarray(target)


def _pyvine_family(pv, copula):
    by_name = {
        "IndependentCopula": pv.BicopFamily.indep,
        "BivariateGaussianCopula": pv.BicopFamily.gaussian,
        "ClaytonCopula": pv.BicopFamily.clayton,
        "GumbelCopula": pv.BicopFamily.gumbel,
        "FrankCopula": pv.BicopFamily.frank,
        "JoeCopula": pv.BicopFamily.joe,
    }
    return by_name[type(copula).__name__]


def pyvine_model_from_exact_state(vine, state, pv):
    """Build the same fixed model in pyvine's matrix-edge orientation."""

    _start, matrix, _edge_map, pair_copulas = state
    d = int(vine.d)
    levels = []
    for tree in range(d - 1):
        level = []
        for column in range(d - 1 - tree):
            pair = pair_copulas[(tree, column)]
            leaf = int(matrix[d - 1 - column, column])
            partner = int(matrix[d - 2 - column - tree, column])
            rotation = int(pair.copula.rotate)
            if leaf > partner and rotation in (90, 270):
                rotation = 360 - rotation
            kwargs = {
                "family": _pyvine_family(pv, pair.copula),
                "rotation": rotation,
            }
            if not isinstance(pair.copula, IndependentCopula):
                kwargs["parameters"] = np.array(
                    [[float(pair.param)]], dtype=np.float64
                )
            level.append(pv.Bicop(**kwargs))
        levels.append(level)
    return pv.Vinecop.from_structure(
        matrix=pyvine_matrix(matrix), pair_copulas=levels
    )


def pyvine_exact_suffix_sample(vine, n: int, given, seed: int, pv):
    """Independent exact suffix draw through pyvine's Rosenblatt API."""

    state = vine._suffix_sampling_state(given)
    if state is None:
        raise ValueError("given set is not exact-suffix supported")
    _start, matrix, _edge_map, _pairs = state
    reference = pyvine_model_from_exact_state(vine, state, pv)

    # For an exact suffix, transformed fixed coordinates cannot depend on the
    # arbitrary placeholder values used for free variables.
    transformed = []
    for placeholder in (0.31, 0.79):
        anchor = np.full((1, vine.d), placeholder, dtype=np.float64)
        for variable, value in given.items():
            anchor[0, variable] = value
        transformed.append(reference.rosenblatt(
            np.asfortranarray(anchor), randomize_discrete=False
        )[0])
    for variable in given:
        np.testing.assert_allclose(
            transformed[0][variable],
            transformed[1][variable],
            rtol=0.0,
            atol=3e-12,
        )

    uniforms = np.random.default_rng(seed).uniform(
        0.0, 1.0, size=(n, vine.d)
    )
    base = np.empty_like(uniforms)
    for column in range(vine.d):
        variable = int(matrix[vine.d - 1 - column, column])
        base[:, variable] = uniforms[:, column]
    for variable in given:
        base[:, variable] = transformed[0][variable]
    return reference.inverse_rosenblatt(np.asfortranarray(base)), reference


__all__ = [
    "gaussian_vine_correlation",
    "gaussian_vine_correlation_path",
    "one_free_gaussian_pit",
    "pyvine_exact_suffix_sample",
    "pyvine_matrix",
    "pyvine_model_from_exact_state",
]
