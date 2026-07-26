"""Contracts for the canonical unconditional R-vine traversal plan."""

from dataclasses import replace

import numpy as np
import pytest

from pyscarcopula.vine._rvine_sampling_plan import (
    build_rvine_sampling_plan,
    rvine_sampling_edge_keys,
)


MATRIX = np.array(
    [
        [0, 0, 0],
        [1, 1, 0],
        [2, 0, 0],
    ],
    dtype=int,
)
TREES = (
    (
        (frozenset({1, 2}), frozenset()),
        (frozenset({0, 1}), frozenset()),
    ),
    (
        (frozenset({0, 2}), frozenset({1})),
    ),
)
EDGE_MAP = {(0, 0): 0, (0, 1): 1, (1, 0): 0}
ACTIVE_KEYS = ((0, 1), (0, 0), (1, 0))


def _plan():
    return build_rvine_sampling_plan(
        3, MATRIX, TREES, EDGE_MAP, ACTIVE_KEYS, 1)


def test_sampling_edge_keys_use_canonical_column_and_tree_order():
    assert rvine_sampling_edge_keys(4, 2) == (
        (0, 2),
        (0, 1),
        (1, 1),
        (0, 0),
        (1, 0),
        (2, 0),
    )
    assert rvine_sampling_edge_keys(4, -1) == ()


def test_sampling_plan_resolves_all_base_and_update_nodes():
    plan = _plan()

    assert plan.active_keys == ACTIVE_KEYS
    assert tuple(plan.node_keys[node] for node in plan.output_nodes) == (
        (0, frozenset()),
        (1, frozenset()),
        (2, frozenset()),
    )
    assert all(
        plan.node_keys[node][0] == variable
        for node, variable in zip(plan.update_u1_nodes, (0, 1, 0))
    )
    assert all(
        plan.node_keys[node][0] == variable
        for node, variable in zip(plan.update_u2_nodes, (1, 2, 2))
    )


def test_sampling_plan_rejects_orientation_inconsistent_with_nodes():
    plan = _plan()
    orientation = list(plan.inverse_transposed)
    orientation[0] = 1 - orientation[0]

    with pytest.raises(ValueError, match="orientation"):
        replace(plan, inverse_transposed=tuple(orientation))


def test_sampling_plan_rejects_missing_active_edge():
    with pytest.raises(ValueError, match="active edge keys"):
        build_rvine_sampling_plan(
            3,
            MATRIX,
            TREES,
            EDGE_MAP,
            ACTIVE_KEYS[:-1],
            1,
        )
