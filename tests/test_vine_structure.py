"""Permanent tests for the public regular-vine structure representation."""

from copy import deepcopy

import numpy as np
import pytest

from pyscarcopula.vine import (
    RVineMatrix,
    cvine_structure,
    dvine_structure,
)
from pyscarcopula.vine._rvine_matrix_builder import build_rvine_matrix


def _edge_sets(trees):
    return [
        {
            (frozenset(conditioned), frozenset(conditioning))
            for conditioned, conditioning in level
        }
        for level in trees
    ]


def _cvine_trees(order):
    return [
        [
            (
                frozenset((order[tree], order[index])),
                frozenset(order[:tree]),
            )
            for index in range(tree + 1, len(order))
        ]
        for tree in range(len(order) - 1)
    ]


def _dvine_trees(order):
    return [
        [
            (
                frozenset((order[start], order[start + tree + 1])),
                frozenset(order[start + 1:start + tree + 1]),
            )
            for start in range(len(order) - tree - 1)
        ]
        for tree in range(len(order) - 1)
    ]


def _mixed_trees():
    return [
        [
            (frozenset({0, 1}), frozenset()),
            (frozenset({1, 2}), frozenset()),
            (frozenset({1, 3}), frozenset()),
            (frozenset({3, 4}), frozenset()),
        ],
        [
            (frozenset({0, 2}), frozenset({1})),
            (frozenset({0, 3}), frozenset({1})),
            (frozenset({1, 4}), frozenset({3})),
        ],
        [
            (frozenset({2, 3}), frozenset({0, 1})),
            (frozenset({0, 4}), frozenset({1, 3})),
        ],
        [
            (frozenset({2, 4}), frozenset({0, 1, 3})),
        ],
    ]


@pytest.mark.parametrize(
    "order",
    [
        [0, 1],
        [0, 1, 2],
        [0, 1, 2, 3, 4],
        [3, 1, 4, 0, 2],
    ],
)
def test_cvine_structure_uses_successive_roots(order):
    structure = cvine_structure(len(order), order)

    assert structure.d == len(order)
    assert _edge_sets(structure.to_trees()) == _edge_sets(
        _cvine_trees(order))


@pytest.mark.parametrize(
    "order",
    [
        [0, 1],
        [0, 1, 2],
        [0, 1, 2, 3, 4, 5],
        [2, 0, 3, 1],
    ],
)
def test_dvine_structure_uses_requested_path_order(order):
    structure = dvine_structure(len(order), order)

    assert structure.d == len(order)
    assert _edge_sets(structure.to_trees()) == _edge_sets(
        _dvine_trees(order))


@pytest.mark.parametrize("factory", [cvine_structure, dvine_structure])
def test_structure_factory_defaults_to_natural_variable_order(factory):
    structure = factory(5)
    expected = (
        _cvine_trees(list(range(5)))
        if factory is cvine_structure
        else _dvine_trees(list(range(5)))
    )
    assert _edge_sets(structure.to_trees()) == _edge_sets(expected)


@pytest.mark.parametrize("factory", [cvine_structure, dvine_structure])
def test_structure_factory_does_not_mutate_order(factory):
    order = [2, 0, 3, 1]
    original = order.copy()

    factory(4, order)

    assert order == original


@pytest.mark.parametrize(
    ("d", "error"),
    [
        (1, ValueError),
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (3.0, TypeError),
    ],
)
@pytest.mark.parametrize("factory", [cvine_structure, dvine_structure])
def test_structure_factory_rejects_invalid_dimension(factory, d, error):
    with pytest.raises(error, match="d must"):
        factory(d)


@pytest.mark.parametrize(
    ("order", "error"),
    [
        ([0, 1, 2], ValueError),
        ([0, 1, 1, 3], ValueError),
        ([0, 1, 2, 4], ValueError),
        ([0, 1, 2, -1], ValueError),
        ([0, 1, 2, True], TypeError),
        ([0, 1, 2, 3.0], TypeError),
    ],
)
@pytest.mark.parametrize("factory", [cvine_structure, dvine_structure])
def test_structure_factory_rejects_malformed_order(factory, order, error):
    with pytest.raises(error, match="order"):
        factory(4, order)


def test_from_trees_roundtrip_for_arbitrary_regular_vine():
    trees = _mixed_trees()
    original = deepcopy(trees)

    structure = RVineMatrix.from_trees(5, trees)

    assert _edge_sets(structure.to_trees()) == _edge_sets(trees)
    assert trees == original


@pytest.mark.parametrize(
    "trees_factory",
    [
        lambda: _cvine_trees([2, 0, 3, 1]),
        lambda: _dvine_trees([2, 0, 3, 1]),
        _mixed_trees,
    ],
)
def test_from_natural_order_to_trees_roundtrip(trees_factory):
    trees = trees_factory()
    d = len(trees) + 1
    natural = build_rvine_matrix(d, trees)
    original = natural.copy()

    structure = RVineMatrix.from_natural_order(natural)

    assert _edge_sets(structure.to_trees()) == _edge_sets(trees)
    np.testing.assert_array_equal(natural, original)


def test_matrix_and_tree_accessors_are_defensive():
    source = dvine_structure(4).matrix
    structure = RVineMatrix(source)
    source[0, 0] = 99

    first_matrix = structure.matrix
    first_matrix[0, 0] = 98
    assert structure.matrix[0, 0] == 0

    first_trees = structure.to_trees()
    first_trees[0].clear()
    assert len(structure.to_trees()[0]) == 3


def test_rvine_matrix_has_value_equality_and_is_unhashable():
    left = dvine_structure(4, [2, 0, 3, 1])
    right = RVineMatrix(left.matrix)
    different = cvine_structure(4, [2, 0, 3, 1])

    assert left == right
    assert left != different
    with pytest.raises(TypeError, match="unhashable"):
        hash(left)


@pytest.mark.parametrize(
    ("matrix", "error"),
    [
        (np.array([0, 1]), ValueError),
        (np.zeros((2, 3), dtype=int), ValueError),
        (np.array([[0]], dtype=int), ValueError),
        (np.array([[False, False], [True, True]]), TypeError),
        (np.array([[0.0, 0.0], [1.0, 1.0]]), TypeError),
    ],
)
def test_rvine_matrix_rejects_bad_shape_dimension_or_dtype(matrix, error):
    with pytest.raises(error):
        RVineMatrix(matrix)


def test_rvine_matrix_rejects_nonzero_upper_triangle():
    matrix = dvine_structure(4).matrix
    matrix[0, 1] = 1

    with pytest.raises(ValueError, match="above the diagonal"):
        RVineMatrix(matrix)


def test_rvine_matrix_rejects_invalid_diagonal_and_columns():
    bad_diagonal = dvine_structure(4).matrix
    bad_diagonal[1, 1] = bad_diagonal[0, 0]
    with pytest.raises(ValueError, match="Diagonal"):
        RVineMatrix(bad_diagonal)

    duplicate_column = dvine_structure(4).matrix
    duplicate_column[2, 0] = duplicate_column[1, 0]
    with pytest.raises(ValueError, match="duplicate"):
        RVineMatrix(duplicate_column)

    out_of_range = dvine_structure(4).matrix
    out_of_range[2, 0] = 8
    with pytest.raises(ValueError, match="out of range"):
        RVineMatrix(out_of_range)


def test_rvine_matrix_rejects_proximity_failure():
    matrix = dvine_structure(4).matrix
    matrix[2, 0], matrix[3, 0] = matrix[3, 0], matrix[2, 0]

    with pytest.raises(ValueError, match="proximity"):
        RVineMatrix(matrix)


@pytest.mark.parametrize(
    ("d", "trees", "error"),
    [
        (1, [], ValueError),
        (True, [], TypeError),
        (3, [[(frozenset({0, 1}), frozenset())]], ValueError),
        (
            2,
            [[(frozenset({False, 1}), frozenset())]],
            TypeError,
        ),
        (
            2,
            [[(frozenset({0, 2}), frozenset())]],
            ValueError,
        ),
    ],
)
def test_from_trees_rejects_invalid_dimension_shape_or_indices(
        d, trees, error):
    with pytest.raises(error, match="RVineMatrix.from_trees"):
        RVineMatrix.from_trees(d, trees)


def test_edge_lookup_rejects_bool_and_out_of_range_indices():
    structure = dvine_structure(4)

    with pytest.raises(TypeError, match="tree"):
        structure.edge(True, 0)
    with pytest.raises(TypeError, match="edge_idx"):
        structure.edge(0, False)
    with pytest.raises(IndexError, match="Tree index"):
        structure.edge(-1, 0)
    with pytest.raises(IndexError, match="Tree 0"):
        structure.edge(0, 3)
