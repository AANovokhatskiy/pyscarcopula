"""External parity for fixed exact-suffix vine models."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
import pytest


os.environ.setdefault(
    "MPLCONFIGDIR",
    str(
        Path(tempfile.gettempdir())
        / (
            "pyscarcopula-matplotlib-cache-"
            + os.environ.get("PYTEST_XDIST_WORKER", "main")
        )
    ),
)
pv = pytest.importorskip("pyvinecopulib")

from pyscarcopula import (  # noqa: E402
    BivariateGaussianCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    IndependentCopula,
    JoeCopula,
)

from ._vine_fixtures import (  # noqa: E402
    exact_given,
    exact_structure_cases,
    homogeneous_vine,
    mixed_vine,
    truncated_gaussian_vine,
)
from ._vine_oracles import (  # noqa: E402
    pyvine_exact_suffix_sample,
    pyvine_matrix,
    pyvine_model_from_exact_state,
)


pytestmark = pytest.mark.external

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


@pytest.mark.parametrize(
    ("family", "rotation"),
    FAMILY_ROTATIONS,
    ids=lambda value: getattr(value, "__name__", f"r{value}"),
)
@pytest.mark.parametrize("path", PATHS)
def test_exact_suffix_matches_pyvine_pointwise_for_every_family_rotation(
    family, rotation, path
):
    index = FAMILY_ROTATIONS.index((family, rotation))
    structure = STRUCTURES[STRUCTURE_IDS[index % len(STRUCTURE_IDS)]]
    vine = homogeneous_vine(structure, family, rotation)
    given = exact_given(vine, path)
    seed = 20263100 + 10 * index + PATHS.index(path)

    expected, _reference = pyvine_exact_suffix_sample(
        vine, 257, given, seed, pv
    )
    observed, diagnostics = vine.predict(
        257,
        given=given,
        return_diagnostics=True,
        rng=np.random.default_rng(seed),
    )

    assert diagnostics["conditional_method"] == "suffix"
    assert diagnostics["matrix_rebuilt"] is (path == "rebuilt")
    tolerance = 1e-6 if family is JoeCopula else 3e-7
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=tolerance)


@pytest.mark.parametrize("structure_id", STRUCTURE_IDS)
@pytest.mark.parametrize("path", PATHS)
def test_mixed_family_rotation_suffix_matches_pyvine_pointwise(
    structure_id, path
):
    vine = mixed_vine(STRUCTURES[structure_id])
    given = exact_given(vine, path)
    seed = 20263200 + STRUCTURE_IDS.index(structure_id) * 2 + PATHS.index(path)

    expected, _reference = pyvine_exact_suffix_sample(
        vine, 383, given, seed, pv
    )
    observed = vine.predict(
        383, given=given, rng=np.random.default_rng(seed)
    )

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=5e-7)


@pytest.mark.parametrize("path", PATHS)
def test_truncated_suffix_matches_pyvine_pointwise(path):
    vine = truncated_gaussian_vine(STRUCTURES["r-vine"])
    given = exact_given(vine, path)
    seed = 20263300 + PATHS.index(path)

    expected, _reference = pyvine_exact_suffix_sample(
        vine, 257, given, seed, pv
    )
    observed = vine.predict(
        257, given=given, rng=np.random.default_rng(seed)
    )

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=2e-8)


@pytest.mark.parametrize("path", PATHS)
def test_pyvine_conversion_preserves_mixed_joint_density(path):
    vine = mixed_vine(STRUCTURES["r-vine"])
    given = exact_given(vine, path)
    state = vine._suffix_sampling_state(given)
    assert state is not None
    reference = pyvine_model_from_exact_state(vine, state, pv)
    np.testing.assert_array_equal(reference.matrix, pyvine_matrix(state[1]))

    rows = np.asfortranarray(np.random.default_rng(20263400).uniform(
        0.04, 0.96, size=(53, vine.d)
    ))
    parameter_paths = {
        key: np.full(len(rows), pair.param, dtype=np.float64)
        for key, pair in vine.pair_copulas.items()
    }
    observed = np.exp(vine._log_pdf_rows_with_r(rows, parameter_paths))
    expected = reference.pdf(rows)

    np.testing.assert_allclose(observed, expected, rtol=3e-8, atol=3e-9)
