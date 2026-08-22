"""Canonical unconditional R-vine sampling traversal plan."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EdgeKey = tuple[int, int]
NodeKey = tuple[int, frozenset[int]]


def rvine_sampling_edge_keys(dimension, max_active_tree):
    """Return canonical edge order for unconditional R-vine sampling."""
    d = int(dimension)
    max_active_tree = int(max_active_tree)
    if d < 2:
        raise ValueError("R-vine sampling dimension must be at least two")
    if max_active_tree < 0:
        return ()
    keys = []
    for col in range(d - 2, -1, -1):
        active_top = min(d - 2 - col, max_active_tree)
        keys.extend((tree, col) for tree in range(active_top + 1))
    return tuple(keys)


def _offsets_are_valid(offsets, item_count, column_count):
    return (
        len(offsets) == column_count + 1
        and offsets[0] == 0
        and offsets[-1] == item_count
        and all(left <= right for left, right in zip(offsets, offsets[1:]))
    )


@dataclass(frozen=True)
class RVineTraversalPlan:
    """Model-independent node and edge program for unconditional sampling."""

    dimension: int
    active_keys: tuple[EdgeKey, ...]
    node_keys: tuple[NodeKey, ...]
    last_uniform_column: int
    last_output_node: int
    output_nodes: tuple[int, ...]
    column_uniforms: tuple[int, ...]
    inverse_offsets: tuple[int, ...]
    inverse_edges: tuple[int, ...]
    inverse_partner_nodes: tuple[int, ...]
    inverse_output_nodes: tuple[int, ...]
    inverse_transposed: tuple[int, ...]
    forward_offsets: tuple[int, ...]
    forward_edges: tuple[int, ...]
    forward_leaf_nodes: tuple[int, ...]
    forward_partner_nodes: tuple[int, ...]
    forward_leaf_output_nodes: tuple[int, ...]
    forward_partner_output_nodes: tuple[int, ...]
    forward_transposed: tuple[int, ...]
    update_u1_nodes: tuple[int, ...]
    update_u2_nodes: tuple[int, ...]

    def __post_init__(self):
        self._validate()

    def _validate(self):
        d = self.dimension
        node_count = len(self.node_keys)
        edge_count = len(self.active_keys)
        column_count = len(self.column_uniforms)
        if d < 2 or node_count < d:
            raise ValueError("R-vine traversal plan has invalid dimensions")
        if len(set(self.active_keys)) != edge_count:
            raise ValueError("R-vine traversal plan has duplicate edge keys")
        if len(set(self.node_keys)) != node_count:
            raise ValueError("R-vine traversal plan has duplicate node keys")
        if not 0 <= self.last_uniform_column < d:
            raise ValueError("R-vine traversal plan has invalid last uniform")
        if len(self.output_nodes) != d:
            raise ValueError("R-vine traversal plan has invalid output count")
        if not _offsets_are_valid(
                self.inverse_offsets, len(self.inverse_edges), column_count):
            raise ValueError("R-vine traversal plan has invalid inverse offsets")
        if not _offsets_are_valid(
                self.forward_offsets, len(self.forward_edges), column_count):
            raise ValueError("R-vine traversal plan has invalid forward offsets")

        inverse_count = len(self.inverse_edges)
        if any(len(values) != inverse_count for values in (
                self.inverse_partner_nodes,
                self.inverse_output_nodes,
                self.inverse_transposed,
        )):
            raise ValueError("R-vine traversal plan has inconsistent inverse ops")
        forward_count = len(self.forward_edges)
        if any(len(values) != forward_count for values in (
                self.forward_leaf_nodes,
                self.forward_partner_nodes,
                self.forward_leaf_output_nodes,
                self.forward_partner_output_nodes,
                self.forward_transposed,
        )):
            raise ValueError("R-vine traversal plan has inconsistent forward ops")
        if (
                len(self.update_u1_nodes) != edge_count
                or len(self.update_u2_nodes) != edge_count):
            raise ValueError("R-vine traversal plan has invalid update nodes")

        def valid_node(value):
            return 0 <= value < node_count

        def valid_edge(value):
            return 0 <= value < edge_count

        if (
                not valid_node(self.last_output_node)
                or not all(valid_node(value) for value in self.output_nodes)
                or not all(0 <= value < d for value in self.column_uniforms)
                or not all(valid_edge(value) for value in self.inverse_edges)
                or not all(valid_edge(value) for value in self.forward_edges)
                or not all(valid_node(value) for values in (
                    self.inverse_partner_nodes,
                    self.inverse_output_nodes,
                    self.forward_leaf_nodes,
                    self.forward_partner_nodes,
                    self.forward_leaf_output_nodes,
                    self.forward_partner_output_nodes,
                    self.update_u1_nodes,
                    self.update_u2_nodes,
                ) for value in values)
                or not all(value in (0, 1) for values in (
                    self.inverse_transposed,
                    self.forward_transposed,
                ) for value in values)):
            raise ValueError("R-vine traversal plan contains invalid indices")

        if edge_count == 0:
            expected_nodes = tuple(
                (variable, frozenset()) for variable in range(d))
            if (
                    self.node_keys != expected_nodes
                    or self.last_uniform_column != d - 1
                    or self.last_output_node != d - 1
                    or self.output_nodes != tuple(range(d))
                    or self.column_uniforms != tuple(range(d - 2, -1, -1))
                    or inverse_count != 0
                    or forward_count != 0):
                raise ValueError(
                    "independent R-vine traversal plan must be an identity "
                    "uniform mapping")
            return

        for index in range(inverse_count):
            target_variable = self.node_keys[
                self.inverse_output_nodes[index]][0]
            partner_variable = self.node_keys[
                self.inverse_partner_nodes[index]][0]
            if self.inverse_transposed[index] != int(
                    target_variable > partner_variable):
                raise ValueError(
                    "R-vine inverse orientation disagrees with node variables")
        for index in range(forward_count):
            leaf_variable = self.node_keys[
                self.forward_leaf_nodes[index]][0]
            partner_variable = self.node_keys[
                self.forward_partner_nodes[index]][0]
            if self.forward_transposed[index] != int(
                    leaf_variable > partner_variable):
                raise ValueError(
                    "R-vine forward orientation disagrees with node variables")

        initialized = {self.last_output_node}
        for column in range(column_count):
            for index in range(
                    self.inverse_offsets[column],
                    self.inverse_offsets[column + 1]):
                if self.inverse_partner_nodes[index] not in initialized:
                    raise ValueError(
                        "R-vine inverse operation reads an uninitialized node")
                initialized.add(self.inverse_output_nodes[index])
            for index in range(
                    self.forward_offsets[column],
                    self.forward_offsets[column + 1]):
                if (
                        self.forward_leaf_nodes[index] not in initialized
                        or self.forward_partner_nodes[index] not in initialized):
                    raise ValueError(
                        "R-vine forward operation reads an uninitialized node")
                initialized.add(self.forward_leaf_output_nodes[index])
                initialized.add(self.forward_partner_output_nodes[index])
        if not all(node in initialized for node in (
                *self.output_nodes,
                *self.update_u1_nodes,
                *self.update_u2_nodes,
        )):
            raise ValueError(
                "R-vine traversal plan leaves required nodes uninitialized")


def build_rvine_sampling_plan(
        dimension,
        matrix,
        trees,
        edge_map,
        active_keys,
        max_active_tree):
    """Compile canonical R-vine structure into an executor-neutral plan."""
    d = int(dimension)
    M = np.asarray(matrix)
    if M.shape != (d, d):
        raise ValueError(
            f"R-vine sampling matrix must have shape ({d}, {d}), "
            f"got {M.shape}")
    active_keys = tuple(
        (int(tree), int(column)) for tree, column in active_keys)
    expected_active_keys = rvine_sampling_edge_keys(d, max_active_tree)
    if active_keys != expected_active_keys:
        raise ValueError(
            "R-vine sampling active edge keys do not match the canonical "
            f"order: expected {expected_active_keys}, got {active_keys}")
    if not active_keys:
        columns = tuple(range(d - 2, -1, -1))
        offsets = (0,) * (len(columns) + 1)
        return RVineTraversalPlan(
            dimension=d,
            active_keys=(),
            node_keys=tuple(
                (variable, frozenset()) for variable in range(d)),
            last_uniform_column=d - 1,
            last_output_node=d - 1,
            output_nodes=tuple(range(d)),
            column_uniforms=columns,
            inverse_offsets=offsets,
            inverse_edges=(),
            inverse_partner_nodes=(),
            inverse_output_nodes=(),
            inverse_transposed=(),
            forward_offsets=offsets,
            forward_edges=(),
            forward_leaf_nodes=(),
            forward_partner_nodes=(),
            forward_leaf_output_nodes=(),
            forward_partner_output_nodes=(),
            forward_transposed=(),
            update_u1_nodes=(),
            update_u2_nodes=(),
        )
    edge_indices = {
        key: index for index, key in enumerate(active_keys)
    }
    nodes = {}

    def node_id(variable, conditioning):
        key = (
            int(variable),
            frozenset(int(item) for item in conditioning),
        )
        if key not in nodes:
            nodes[key] = len(nodes)
        return nodes[key]

    last_uniform_column = d - 1
    last_var = int(M[0, d - 1])
    last_output_node = node_id(last_var, ())

    column_uniforms = []
    inverse_offsets = [0]
    inverse_edges = []
    inverse_partner_nodes = []
    inverse_output_nodes = []
    inverse_transposed = []
    forward_offsets = [0]
    forward_edges = []
    forward_leaf_nodes = []
    forward_partner_nodes = []
    forward_leaf_output_nodes = []
    forward_partner_output_nodes = []
    forward_transposed = []

    for col in range(d - 2, -1, -1):
        leaf = int(M[d - 1 - col, col])
        active_top = min(d - 2 - col, int(max_active_tree))
        column_uniforms.append(col)

        for tree in range(active_top, -1, -1):
            row = d - 2 - col - tree
            partner = int(M[row, col])
            conditioning = frozenset(
                int(M[r, col])
                for r in range(row + 1, d - 1 - col)
            )
            try:
                edge_index = edge_indices[(tree, col)]
            except KeyError as exc:
                raise ValueError(
                    "R-vine sampling plan is missing active edge "
                    f"{(tree, col)}") from exc
            inverse_edges.append(edge_index)
            inverse_partner_nodes.append(node_id(partner, conditioning))
            inverse_output_nodes.append(node_id(leaf, conditioning))
            inverse_transposed.append(int(leaf > partner))
        inverse_offsets.append(len(inverse_edges))

        for tree in range(active_top + 1):
            row = d - 2 - col - tree
            partner = int(M[row, col])
            conditioning = frozenset(
                int(M[r, col])
                for r in range(row + 1, d - 1 - col)
            )
            edge_index = edge_indices[(tree, col)]
            forward_edges.append(edge_index)
            forward_leaf_nodes.append(node_id(leaf, conditioning))
            forward_partner_nodes.append(node_id(partner, conditioning))
            forward_leaf_output_nodes.append(
                node_id(leaf, conditioning | {partner}))
            forward_partner_output_nodes.append(
                node_id(partner, conditioning | {leaf}))
            forward_transposed.append(int(leaf > partner))
        forward_offsets.append(len(forward_edges))

    update_u1_nodes = []
    update_u2_nodes = []
    for tree, col in active_keys:
        try:
            orig_idx = edge_map[(tree, col)]
            conditioned, conditioning = trees[tree][orig_idx]
        except (IndexError, KeyError) as exc:
            raise ValueError(
                "R-vine sampling plan cannot resolve semantic edge "
                f"{(tree, col)}") from exc
        v1, v2 = sorted(conditioned)
        update_u1_nodes.append(node_id(v1, conditioning))
        update_u2_nodes.append(node_id(v2, conditioning))

    output_nodes = tuple(node_id(var, ()) for var in range(d))
    node_keys = [None] * len(nodes)
    for key, value in nodes.items():
        node_keys[value] = key

    return RVineTraversalPlan(
        dimension=d,
        active_keys=active_keys,
        node_keys=tuple(node_keys),
        last_uniform_column=last_uniform_column,
        last_output_node=last_output_node,
        output_nodes=output_nodes,
        column_uniforms=tuple(column_uniforms),
        inverse_offsets=tuple(inverse_offsets),
        inverse_edges=tuple(inverse_edges),
        inverse_partner_nodes=tuple(inverse_partner_nodes),
        inverse_output_nodes=tuple(inverse_output_nodes),
        inverse_transposed=tuple(inverse_transposed),
        forward_offsets=tuple(forward_offsets),
        forward_edges=tuple(forward_edges),
        forward_leaf_nodes=tuple(forward_leaf_nodes),
        forward_partner_nodes=tuple(forward_partner_nodes),
        forward_leaf_output_nodes=tuple(forward_leaf_output_nodes),
        forward_partner_output_nodes=tuple(forward_partner_output_nodes),
        forward_transposed=tuple(forward_transposed),
        update_u1_nodes=tuple(update_u1_nodes),
        update_u2_nodes=tuple(update_u2_nodes),
    )


__all__ = [
    "RVineTraversalPlan",
    "build_rvine_sampling_plan",
    "rvine_sampling_edge_keys",
]
