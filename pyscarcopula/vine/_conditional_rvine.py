"""Validation helpers for R-vine conditional prediction."""

from __future__ import annotations

import numpy as np


def validate_rvine_given(given, d):
    """Validate `given` for R-vine conditional predict."""
    if given is None:
        return {}

    if not isinstance(given, dict):
        raise TypeError("given must be a dict[int, float] or None")

    out = {}
    for key, value in given.items():
        if isinstance(key, (bool, np.bool_)) or not isinstance(
                key, (int, np.integer)):
            raise TypeError("given keys must be integers")
        idx = int(key)
        if idx < 0 or idx >= d:
            raise ValueError(f"given key must be in [0, {d - 1}], got {key!r}")
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
            raise TypeError("given values must be numeric scalars")
        val = float(value)
        if not (0.0 < val < 1.0):
            raise ValueError(
                f"given[{idx}] must be in pseudo-observation space (0, 1), got {val}"
            )
        out[idx] = val
    return out


def validate_rvine_given_vars(given_vars, d):
    """Validate fit-time given-variable indices."""
    if given_vars is None:
        return ()

    try:
        items = tuple(given_vars)
    except TypeError as exc:
        raise TypeError(
            "given_vars must be an iterable of integer variable indices "
            "or None"
        ) from exc

    out = set()
    for key in items:
        if isinstance(key, (bool, np.bool_)) or not isinstance(
                key, (int, np.integer)):
            raise TypeError("given_vars entries must be integers")
        idx = int(key)
        if idx < 0 or idx >= d:
            raise ValueError(
                f"given_vars entries must be in [0, {d - 1}], got {key!r}"
            )
        out.add(idx)
    return tuple(sorted(out))


def find_rvine_peel_order_for_given_suffix(trees, d, given_vars):
    """Find a peel order whose trailing variables equal ``given_vars``."""
    prefix_len = int(d) - len(given_vars)
    given_vars = set(int(v) for v in given_vars)

    def search(col, claimed, peeled):
        if col == d - 1:
            remaining = set(range(d)) - set(peeled)
            if len(remaining) == 1 and remaining <= given_vars:
                return peeled + [remaining.pop()]
            return None

        tree_level = d - 2 - col
        top_candidates = [
            idx for idx in range(len(trees[tree_level]))
            if idx not in claimed[tree_level]
        ]
        if len(top_candidates) != 1:
            return None

        idx_top = top_candidates[0]
        conditioned_top, conditioning_top = trees[tree_level][idx_top]
        candidates = sorted(conditioned_top)
        if col < prefix_len:
            candidates = [v for v in candidates if v not in given_vars]
        else:
            candidates = [v for v in candidates if v in given_vars]
        if not candidates:
            return None

        for leaf in candidates:
            cond_accum = set()
            walk_claims = []
            valid = True
            for t in range(tree_level):
                target_cc = frozenset(cond_accum)
                hits = [
                    idx
                    for idx, (conditioned, conditioning)
                    in enumerate(trees[t])
                    if (idx not in claimed[t]
                        and leaf in conditioned
                        and conditioning == target_cc)
                ]
                if len(hits) != 1:
                    valid = False
                    break
                idx_t = hits[0]
                conditioned_t, _ = trees[t][idx_t]
                other = next(iter(conditioned_t - {leaf}))
                cond_accum.add(other)
                walk_claims.append((t, idx_t))

            if not valid or cond_accum != set(conditioning_top):
                continue

            next_claimed = [set(level) for level in claimed]
            next_claimed[tree_level].add(idx_top)
            for t, idx_t in walk_claims:
                next_claimed[t].add(idx_t)

            result = search(
                col + 1,
                tuple(frozenset(level) for level in next_claimed),
                peeled + [leaf],
            )
            if result is not None:
                return result
        return None

    empty_claimed = tuple(frozenset() for _ in range(d - 1))
    return search(0, empty_claimed, [])
