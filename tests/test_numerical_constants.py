"""Contracts for named numerical safety boundaries."""

from pathlib import Path

import numpy as np

from pyscarcopula._constants import (
    CONDITIONAL_SAMPLE_EPS,
    H_FUNCTION_EPS,
    PDF_FLOOR,
    PSEUDO_OBS_EPS,
    ROSENBLATT_OUTPUT_EPS,
)
from pyscarcopula._utils import (
    clip_h_function_values,
    clip_pseudo_observations,
    clip_pseudo_observations_no_copy,
    clip_rosenblatt_output,
)
from pyscarcopula.numerical import _cpp_scar_ou
from pyscarcopula.vine._helpers import _clip_unit, _open_unit_uniform


def test_named_safety_boundaries_are_distinct_by_purpose():
    assert PDF_FLOOR < CONDITIONAL_SAMPLE_EPS < PSEUDO_OBS_EPS
    assert PSEUDO_OBS_EPS < H_FUNCTION_EPS
    assert H_FUNCTION_EPS == ROSENBLATT_OUTPUT_EPS


def test_pseudo_observation_clipping_has_one_python_contract():
    values = np.array([0.0, 0.25, 1.0], dtype=np.float64)
    expected = np.array([
        PSEUDO_OBS_EPS,
        0.25,
        1.0 - PSEUDO_OBS_EPS,
    ])
    np.testing.assert_array_equal(
        clip_pseudo_observations(values), expected)
    np.testing.assert_array_equal(_clip_unit(values), expected)


def test_no_copy_pseudo_observation_helper_only_copies_when_needed():
    interior = np.array([0.2, 0.8], dtype=np.float64)
    boundary = np.array([0.0, 1.0], dtype=np.float64)

    assert clip_pseudo_observations_no_copy(interior) is interior
    clipped = clip_pseudo_observations_no_copy(boundary)
    assert clipped is not boundary
    np.testing.assert_array_equal(
        clipped,
        [PSEUDO_OBS_EPS, 1.0 - PSEUDO_OBS_EPS],
    )


def test_h_and_rosenblatt_helpers_keep_separate_named_contracts():
    values = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    np.testing.assert_array_equal(
        clip_h_function_values(values),
        [H_FUNCTION_EPS, 0.5, 1.0 - H_FUNCTION_EPS],
    )
    np.testing.assert_array_equal(
        clip_rosenblatt_output(values),
        [ROSENBLATT_OUTPUT_EPS, 0.5, 1.0 - ROSENBLATT_OUTPUT_EPS],
    )


def test_vine_uniform_draws_use_shared_pseudo_observation_boundary():
    class BoundaryRng:
        def uniform(self, low, high, size):
            assert low == PSEUDO_OBS_EPS
            assert high == 1.0 - PSEUDO_OBS_EPS
            return np.full(size, 0.5)

    np.testing.assert_array_equal(
        _open_unit_uniform(BoundaryRng(), size=(2, 3)),
        np.full((2, 3), 0.5),
    )


def test_python_and_cpp_safety_constants_match():
    module = _cpp_scar_ou._cpp_extension.load()
    assert module.PSEUDO_OBS_EPS == PSEUDO_OBS_EPS
    assert module.H_FUNCTION_EPS == H_FUNCTION_EPS
    assert module.PDF_FLOOR == PDF_FLOOR


def test_vine_runtime_has_no_local_generic_eps_constants():
    vine_root = Path(__file__).resolve().parents[1] / "pyscarcopula" / "vine"
    for path in vine_root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "_EPS = 1e-10" not in text
