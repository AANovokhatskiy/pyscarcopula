"""External density-reference validation for arbitrary vine conditioning."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np
import pytest
from scipy.stats import ks_2samp, spearmanr


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
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    JoeCopula,
)
from pyscarcopula.vine import dvine_structure  # noqa: E402

from ._vine_fixtures import (  # noqa: E402
    arbitrary_given,
    homogeneous_vine,
    mixed_vine,
)
from ._vine_mcmc_oracles import (  # noqa: E402
    pyvine_conditional_logit_mcmc,
)
from ._vine_oracles import pyvine_model_from_fitted_vine  # noqa: E402


pytestmark = pytest.mark.external


@dataclass(frozen=True)
class DensityCase:
    id: str
    dimension: int
    family: type | None
    rotation: int
    parameter: float
    levels: tuple[float, float]
    production_steps: int
    production_burnin: int
    reference_sweeps: int


DENSITY_CASES = (
    DensityCase(
        "clayton-lower-d4", 4, ClaytonCopula, 0, 1.30,
        (0.12, 0.35), 400, 160, 180,
    ),
    DensityCase(
        "gumbel-upper-r180-d4", 4, GumbelCopula, 180, 1.45,
        (0.65, 0.90), 400, 160, 180,
    ),
    DensityCase(
        "joe-opposite-tail-r270-d4", 4, JoeCopula, 270, 1.50,
        (0.20, 0.80), 400, 160, 180,
    ),
    DensityCase(
        "frank-center-d4", 4, FrankCopula, 0, 3.00,
        (0.35, 0.65), 400, 160, 180,
    ),
    DensityCase(
        "mixed-family-rotation-d6", 6, None, 0, 0.0,
        (0.20, 0.80), 600, 240, 200,
    ),
)


def _vine(case: DensityCase):
    structure = dvine_structure(case.dimension, list(range(case.dimension)))
    if case.family is None:
        return mixed_vine(structure)
    return homogeneous_vine(
        structure,
        case.family,
        rotation=case.rotation,
        parameter=case.parameter,
    )


@pytest.mark.parametrize("case", DENSITY_CASES, ids=lambda case: case.id)
def test_original_arbitrary_vine_density_matches_pyvine(case):
    vine = _vine(case)
    reference = pyvine_model_from_fitted_vine(vine, pv)
    rows = np.asfortranarray(np.random.default_rng(2026501).uniform(
        0.03, 0.97, size=(79, case.dimension)
    ))
    parameter_paths = {
        key: np.full(len(rows), pair.param, dtype=np.float64)
        for key, pair in vine.pair_copulas.items()
    }

    observed = np.exp(vine._log_pdf_rows_with_r(rows, parameter_paths))
    expected = reference.pdf(rows)
    np.testing.assert_allclose(observed, expected, rtol=4e-8, atol=4e-9)


@pytest.mark.validation
@pytest.mark.parametrize("case", DENSITY_CASES, ids=lambda case: case.id)
def test_arbitrary_non_gaussian_matches_independent_pyvine_density_mcmc(case):
    vine = _vine(case)
    given = arbitrary_given(vine, case.levels)
    reference = pyvine_model_from_fitted_vine(vine, pv)
    n = 1_200 if case.dimension == 4 else 1_000

    observed, diagnostics = vine.predict(
        n,
        given=given,
        mcmc_steps=case.production_steps,
        mcmc_burnin=case.production_burnin,
        return_diagnostics=True,
        rng=np.random.default_rng(2026510 + case.dimension),
    )
    expected, reference_diagnostics = pyvine_conditional_logit_mcmc(
        reference,
        n,
        given,
        np.random.default_rng(2026520 + case.dimension),
        sweeps=case.reference_sweeps,
    )

    assert diagnostics["conditional_method"] == "dag_mcmc"
    assert reference_diagnostics["proposal"] == "logit_random_walk"
    assert reference_diagnostics["acceptance_min"] > 0.15
    for variable, value in given.items():
        np.testing.assert_array_equal(
            observed[:, variable], np.full(n, value, dtype=np.float64)
        )
        np.testing.assert_array_equal(
            expected[:, variable], np.full(n, value, dtype=np.float64)
        )

    free = sorted(set(range(case.dimension)) - set(given))
    marginal_ks = np.array([
        ks_2samp(observed[:, variable], expected[:, variable]).statistic
        for variable in free
    ])
    mean_error = np.abs(
        np.mean(observed[:, free], axis=0)
        - np.mean(expected[:, free], axis=0)
    )
    observed_rank = np.asarray(
        spearmanr(observed[:, free], axis=0).statistic
    )
    expected_rank = np.asarray(
        spearmanr(expected[:, free], axis=0).statistic
    )
    rank_error = float(np.max(np.abs(observed_rank - expected_rank)))

    assert float(np.max(marginal_ks)) < 0.075, marginal_ks
    assert float(np.max(mean_error)) < 0.035, mean_error
    assert rank_error < 0.12, rank_error
