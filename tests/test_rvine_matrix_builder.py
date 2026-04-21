"""Unit tests for vine._rvine_matrix_builder (natural-order convention)."""
import numpy as np
import pytest

from pyscarcopula.vine._rvine_matrix_builder import (
    build_rvine_matrix,
    build_rvine_matrix_with_edge_map,
    decode_matrix_to_trees,
    validate_natural_order_matrix,
)


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

def _edge(conditioned, conditioning=()):
    return (frozenset(conditioned), frozenset(conditioning))


def _edge_sets(trees):
    """Normalize trees to sets-of-sets for structural comparison."""
    return [
        {(c, cc) for c, cc in level}
        for level in trees
    ]


def _dvine_trees(d):
    """D-vine trees on path 0-1-2-...-(d-1)."""
    trees = []
    for t in range(d - 1):
        level = []
        for col in range(d - 1 - t):
            i = col
            j = col + t + 1
            conditioning = frozenset(range(i + 1, j))
            level.append((frozenset({i, j}), conditioning))
        trees.append(level)
    return trees


def _cvine_trees(d):
    """Canonical C-vine with pivot variable ``t`` at tree level ``t``.

    Tree ``t`` edges are ``(t, j | 0, 1, ..., t-1)`` for ``j = t+1, ..., d-1``.
    Each tree is a star rooted at variable ``t``.
    """
    trees = []
    for t in range(d - 1):
        level = []
        conditioning = frozenset(range(t))
        for j in range(t + 1, d):
            conditioned = frozenset({t, j})
            level.append((conditioned, conditioning))
        trees.append(level)
    return trees


# ═══════════════════════════════════════════════════════════
# Basic construction
# ═══════════════════════════════════════════════════════════

class TestBasic:

    def test_d1_trivial(self):
        M = build_rvine_matrix(1, [])
        assert M.shape == (1, 1)
        assert M[0, 0] == 0

    def test_d2_single_edge(self):
        trees = [[_edge({0, 1})]]
        M, emap = build_rvine_matrix_with_edge_map(2, trees)
        # Natural order, d=2: M[1, 0] = leaf, M[0, 0] = other,
        # M[0, 1] = remaining variable.
        assert M[1, 0] == 0 and M[0, 0] == 1 and M[0, 1] == 1
        assert emap == {(0, 0): 0}

    def test_dvine_d4_matrix_is_valid(self):
        trees = _dvine_trees(4)
        M = build_rvine_matrix(4, trees)
        assert validate_natural_order_matrix(M)

    def test_dvine_d4_matrix_layout(self):
        """For a D-vine on [0..3], peel deterministically picks 0 first,
        then 1, then 2, leaving 3 for the final column.
        """
        trees = _dvine_trees(4)
        M = build_rvine_matrix(4, trees)
        # Anti-diagonal = peel order = [0, 1, 2, 3].
        assert M[3, 0] == 0
        assert M[2, 1] == 1
        assert M[1, 2] == 2
        assert M[0, 3] == 3

    def test_cvine_d4_is_valid(self):
        """Small C-vine (d=4) encodes fine in natural-order convention."""
        trees = _cvine_trees(4)
        M = build_rvine_matrix(4, trees)
        assert validate_natural_order_matrix(M)

    def test_cvine_d6_is_representable(self):
        """Natural-order accepts any R-vine, including pure C-vines."""
        trees = _cvine_trees(6)
        M = build_rvine_matrix(6, trees)
        assert validate_natural_order_matrix(M)
        decoded = decode_matrix_to_trees(M)
        assert _edge_sets(decoded) == _edge_sets(trees)

    def test_matrix_passes_validation_d5_dvine(self):
        trees = _dvine_trees(5)
        M = build_rvine_matrix(5, trees)
        assert validate_natural_order_matrix(M)

    def test_mixed_structure_d5(self):
        """A mixed Dissmann-like structure: tree 0 is not a star nor a path."""
        tree_0 = [
            (frozenset({0, 1}), frozenset()),
            (frozenset({1, 2}), frozenset()),
            (frozenset({1, 3}), frozenset()),
            (frozenset({3, 4}), frozenset()),
        ]
        tree_1 = [
            (frozenset({0, 2}), frozenset({1})),
            (frozenset({0, 3}), frozenset({1})),
            (frozenset({1, 4}), frozenset({3})),
        ]
        tree_2 = [
            (frozenset({2, 3}), frozenset({0, 1})),
            (frozenset({0, 4}), frozenset({1, 3})),
        ]
        tree_3 = [
            (frozenset({2, 4}), frozenset({0, 1, 3})),
        ]
        trees = [tree_0, tree_1, tree_2, tree_3]
        M = build_rvine_matrix(5, trees)
        assert validate_natural_order_matrix(M)
        decoded = decode_matrix_to_trees(M)
        assert _edge_sets(decoded) == _edge_sets(trees)


# ═══════════════════════════════════════════════════════════
# Roundtrip: build(decode(M)) preserves edge structure
# ═══════════════════════════════════════════════════════════

class TestRoundtrip:

    @pytest.mark.parametrize("d", [3, 4, 5, 6, 8, 10])
    def test_dvine_roundtrip(self, d):
        trees = _dvine_trees(d)
        M = build_rvine_matrix(d, trees)
        decoded = decode_matrix_to_trees(M)
        assert _edge_sets(decoded) == _edge_sets(trees)

    @pytest.mark.parametrize("d", [3, 4, 5, 6, 7])
    def test_cvine_roundtrip(self, d):
        trees = _cvine_trees(d)
        M = build_rvine_matrix(d, trees)
        decoded = decode_matrix_to_trees(M)
        assert _edge_sets(decoded) == _edge_sets(trees)

    def test_rebuild_from_decoded(self):
        """Take known-valid trees, build, decode, rebuild — must match."""
        trees = _dvine_trees(6)
        M1 = build_rvine_matrix(6, trees)
        decoded = decode_matrix_to_trees(M1)
        M2 = build_rvine_matrix(6, decoded)
        # The matrix is deterministic, so rebuild must be bit-identical.
        np.testing.assert_array_equal(M1, M2)


# ═══════════════════════════════════════════════════════════
# Natural-order matrix layout invariants
# ═══════════════════════════════════════════════════════════

class TestLayout:

    def test_zero_padding_in_lower_right(self):
        d = 6
        M = build_rvine_matrix(d, _dvine_trees(d))
        for c in range(d):
            for r in range(d - c, d):
                assert M[r, c] == 0, f"M[{r},{c}] should be zero"

    def test_column_entries_are_distinct(self):
        d = 5
        M = build_rvine_matrix(d, _dvine_trees(d))
        for c in range(d):
            col = [int(M[r, c]) for r in range(d - c)]
            assert len(set(col)) == len(col)

    def test_anti_diagonal_is_permutation(self):
        d = 5
        M = build_rvine_matrix(d, _dvine_trees(d))
        leaves = {int(M[d - 1 - c, c]) for c in range(d)}
        assert leaves == set(range(d))


# ═══════════════════════════════════════════════════════════
# Edge map
# ═══════════════════════════════════════════════════════════

class TestEdgeMap:

    def test_edge_map_covers_all_positions(self):
        d = 5
        trees = _dvine_trees(d)
        M, emap = build_rvine_matrix_with_edge_map(d, trees)
        expected_keys = {
            (t, col)
            for t in range(d - 1)
            for col in range(d - 1 - t)
        }
        assert set(emap.keys()) == expected_keys

    def test_edge_map_points_to_correct_edge(self):
        d = 5
        trees = _dvine_trees(d)
        M, emap = build_rvine_matrix_with_edge_map(d, trees)

        for (t, col), edge_idx in emap.items():
            # Natural order lookup:
            conditioned_m = frozenset({
                int(M[d - 1 - col, col]),
                int(M[d - 2 - col - t, col]),
            })
            conditioning_m = frozenset(
                int(M[r, col])
                for r in range(d - 1 - col - t, d - 1 - col)
            )
            conditioned_in, conditioning_in = trees[t][edge_idx]
            assert conditioned_m == conditioned_in
            assert conditioning_m == conditioning_in

    def test_edge_map_no_duplicate_indices_per_tree(self):
        d = 6
        trees = _dvine_trees(d)
        M, emap = build_rvine_matrix_with_edge_map(d, trees)
        for t in range(d - 1):
            indices = [emap[(t, col)] for col in range(d - 1 - t)]
            assert len(set(indices)) == len(indices)
            assert set(indices) == set(range(d - 1 - t))


# ═══════════════════════════════════════════════════════════
# Input validation (clear RuntimeError on malformed input)
# ═══════════════════════════════════════════════════════════

class TestValidation:

    def test_wrong_number_of_trees(self):
        with pytest.raises(RuntimeError, match="expected 3 trees"):
            build_rvine_matrix(4, _dvine_trees(4)[:2])

    def test_wrong_edges_per_tree(self):
        trees = _dvine_trees(4)
        trees[1] = trees[1][:1]  # drop one edge
        with pytest.raises(RuntimeError, match="tree 1 must have 2 edges"):
            build_rvine_matrix(4, trees)

    def test_conditioned_set_wrong_size(self):
        trees = [[(frozenset({0, 1, 2}), frozenset())]]
        with pytest.raises(RuntimeError,
                           match="conditioned set must have exactly 2"):
            build_rvine_matrix(2, trees)

    def test_conditioning_set_wrong_size_at_tree_t(self):
        trees = _dvine_trees(4)
        c, _ = trees[1][0]
        trees[1][0] = (c, frozenset())
        with pytest.raises(RuntimeError,
                           match="conditioning set must have exactly 1"):
            build_rvine_matrix(4, trees)

    def test_conditioned_and_conditioning_overlap(self):
        bad = [
            (frozenset({0, 1}), frozenset()),
            (frozenset({0, 2}), frozenset()),
            (frozenset({1, 2}), frozenset()),
        ]
        tree1 = [
            (frozenset({0, 1}), frozenset({0})),  # overlap!
            (frozenset({1, 2}), frozenset({0})),
        ]
        trees = [bad, tree1, [(frozenset({0, 2}), frozenset({0, 1}))]]
        with pytest.raises(RuntimeError, match="overlap"):
            build_rvine_matrix(4, trees)

    def test_tree0_missing_variable(self):
        trees = [
            [
                (frozenset({0, 1}), frozenset()),
                (frozenset({1, 2}), frozenset()),
                (frozenset({0, 2}), frozenset()),
            ],
        ] + _dvine_trees(4)[1:]
        with pytest.raises(RuntimeError,
                           match="tree 0 must cover variables"):
            build_rvine_matrix(4, trees)

    def test_d_must_be_positive_int(self):
        with pytest.raises(RuntimeError, match="d must be int"):
            build_rvine_matrix(0, [])

    def test_d1_with_extra_trees(self):
        with pytest.raises(RuntimeError, match="d=1 requires 0 trees"):
            build_rvine_matrix(1, [[_edge({0, 1})]])


# ═══════════════════════════════════════════════════════════
# validate_natural_order_matrix standalone checks
# ═══════════════════════════════════════════════════════════

class TestValidator:

    def test_valid_dvine_matrix(self):
        M = build_rvine_matrix(5, _dvine_trees(5))
        assert validate_natural_order_matrix(M)

    def test_rejects_bad_anti_diagonal(self):
        """Duplicate on anti-diagonal (not a permutation)."""
        M = np.array([
            [2, 2, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],  # M[2, 0] should be 2; anti-diag = [0, 0, 1, 0]?
            [0, 0, 0, 0],  # anti-diag M[3,0]=0, M[2,1]=0 → duplicate
        ])
        assert not validate_natural_order_matrix(M)

    def test_rejects_nonzero_in_zero_triangle(self):
        M = build_rvine_matrix(4, _dvine_trees(4))
        M_bad = M.copy()
        M_bad[3, 1] = 7   # should be in zero-padded region
        assert not validate_natural_order_matrix(M_bad)
