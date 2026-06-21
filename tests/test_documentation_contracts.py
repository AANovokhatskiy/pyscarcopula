"""Executable contracts for public documentation."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
import re

import numpy as np
import pyscarcopula


ROOT = Path(__file__).resolve().parents[1]
DOC_FILES = (
    ROOT / "README.md",
    ROOT / "ARCHITECTURE.md",
    *sorted((ROOT / "docs").rglob("*.md")),
)
MIGRATION_NOTES = ROOT / "docs/release-notes/native-core-migration.md"
WORKFLOW_FILES = sorted((ROOT / ".github/workflows").glob("*.yml"))


def _python_blocks(path):
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"^```python[^\n]*\n(.*?)^```$", re.MULTILINE | re.DOTALL)
    return pattern.findall(text)


def _resolve_documented_object(path):
    parts = path.split(".")
    for split_at in range(len(parts), 0, -1):
        module_name = ".".join(parts[:split_at])
        try:
            value = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        for attribute in parts[split_at:]:
            value = getattr(value, attribute)
        return value
    raise ImportError(path)


def test_documented_python_blocks_compile_and_import():
    for path in DOC_FILES:
        for index, source in enumerate(_python_blocks(path)):
            tree = ast.parse(source, filename=f"{path}:{index}")
            if path == MIGRATION_NOTES and "# Removed" in source:
                continue
            imports = [
                node for node in tree.body
                if isinstance(node, (ast.Import, ast.ImportFrom))
            ]
            exec(compile(
                ast.Module(body=imports, type_ignores=[]),
                filename=f"{path}:{index}",
                mode="exec",
            ), {})


def test_mkdocstrings_targets_are_importable():
    pattern = re.compile(
        r"^:::\s+([A-Za-z_][A-Za-z0-9_.]*)\s*$", re.MULTILINE)
    for path in DOC_FILES:
        for target in pattern.findall(path.read_text(encoding="utf-8")):
            assert _resolve_documented_object(target) is not None


def test_obsolete_namespace_is_confined_to_migration_notes():
    obsolete = "pyscarcopula.copula.experimental"
    for path in DOC_FILES:
        if path == MIGRATION_NOTES:
            continue
        assert obsolete not in path.read_text(encoding="utf-8")


def test_removed_native_backend_examples_do_not_return():
    forbidden = re.compile(r"\bbackend\s*=")
    for path in DOC_FILES:
        if path == MIGRATION_NOTES:
            continue
        assert forbidden.search(path.read_text(encoding="utf-8")) is None


def test_removed_public_aliases_do_not_return_to_docs_or_examples():
    forbidden = (
        "u_train=",
        "LatentResult.alpha",
        "pyscarcopula.numerical.auto_tm",
        "pyscarcopula.numerical.tm_gradient",
        "spectral_basis_order='adaptive'",
        'spectral_basis_order="adaptive"',
    )
    for path in DOC_FILES:
        if path == MIGRATION_NOTES:
            continue
        text = path.read_text(encoding="utf-8")
        for value in forbidden:
            assert value not in text, (
                f"{path.relative_to(ROOT)} contains removed API {value!r}"
            )

    for path in sorted((ROOT / "examples").glob("*.ipynb")):
        text = path.read_text(encoding="utf-8")
        assert "u_train=" not in text


def test_workflows_reference_existing_test_files():
    pattern = re.compile(r"tests/[A-Za-z0-9_./-]+\.py")
    for path in WORKFLOW_FILES:
        for test_path in pattern.findall(path.read_text(encoding="utf-8")):
            assert (ROOT / test_path).is_file(), (
                f"{path.relative_to(ROOT)} references missing {test_path}"
            )


def test_removed_experimental_namespace_is_physically_absent():
    assert not (ROOT / "pyscarcopula/copula/experimental").exists()


def test_documented_public_imports():
    from pyscarcopula import (
        BivariateCopula,
        CopulaBase,
        CopulaCapabilities,
        EquicorrGaussianCopula,
        MultivariateCopula,
        StochasticStudentCopula,
    )
    from pyscarcopula.copula.multivariate import (
        EquicorrGaussianCopula as NamespacedEquicorr,
    )
    from pyscarcopula.copula.multivariate import (
        StochasticStudentCopula as NamespacedStudent,
    )

    assert pyscarcopula.EquicorrGaussianCopula is EquicorrGaussianCopula
    assert EquicorrGaussianCopula is NamespacedEquicorr
    assert StochasticStudentCopula is NamespacedStudent
    assert issubclass(BivariateCopula, CopulaBase)
    assert issubclass(MultivariateCopula, CopulaBase)
    assert CopulaCapabilities().supports_gas is False


def test_documented_vine_signatures_match_runtime():
    from pyscarcopula import CVineCopula, RVineCopula

    assert tuple(inspect.signature(CVineCopula.sample).parameters) == (
        "self", "n", "u", "rng")
    assert tuple(inspect.signature(RVineCopula.sample).parameters) == (
        "self", "n", "u", "rng")

    for cls in (CVineCopula, RVineCopula):
        parameters = inspect.signature(cls.predict).parameters
        assert "u" in parameters
        assert "u_train" not in parameters


def test_prediction_guide_distinguishes_sampling_surfaces():
    text = (
        ROOT / "docs/guide/prediction-semantics.md"
    ).read_text(encoding="utf-8")
    assert "api.sample(copula, data, result, n)" in text
    assert "model.sample(n, u=None, rng=None)" in text
    assert "model.sample_at_parameter(n, r, rng=None)" in text


def test_gradient_matrix_matches_diagnostic_vocabulary():
    text = (
        ROOT / "docs/guide/estimation-methods.md"
    ).read_text(encoding="utf-8")
    expected_rows = (
        "| MLE | Built-in supported model | Analytical "
        "| `not_applicable` | `analytical` |",
        "| GAS | Any supported scaling | Numerical finite differences "
        "| `native` | `numerical_optimizer` |",
        "| SCAR-TM-OU | `analytical_grad=True` | Analytical native Jacobian "
        "| `not_applicable` | `analytical` |",
        "| SCAR-TM-JACOBI | `local_fixed`, analytical gradient "
        "| Model-provided | `not_applicable` | `analytical` |",
        "| SCAR-TM-JACOBI | `local`, `spectral_matrix`, or `auto`, "
        "analytical gradient | Model-provided | `not_applicable` "
        "| `semi_analytical` |",
    )
    for row in expected_rows:
        assert row in text


def test_numerical_safety_policy_documents_distinct_contracts():
    text = (
        ROOT / "docs/guide/architecture.md"
    ).read_text(encoding="utf-8")
    for name in (
        "PSEUDO_OBS_EPS",
        "H_FUNCTION_EPS",
        "ROSENBLATT_OUTPUT_EPS",
        "CONDITIONAL_SAMPLE_EPS",
        "PDF_FLOOR",
    ):
        assert f"`{name}`" in text


def test_representative_documented_workflows_execute():
    from pyscarcopula import (
        GumbelCopula,
        IndependentCopula,
        RVineCopula,
    )
    from pyscarcopula.api import fit

    copula = GumbelCopula(rotate=180, transform_type="xtanh")
    natural = np.array([1.2, 2.0])
    latent = copula.inv_transform(natural)
    assert not np.allclose(copula.transform(latent), natural)

    rng = np.random.default_rng(20260620)
    u_pair = GumbelCopula(rotate=180).sample_at_parameter(
        40, np.full(40, 1.5), rng=rng)
    result = fit(
        GumbelCopula(rotate=180),
        u_pair,
        method="mle",
        alpha0=[1.5],
    )
    assert result.diagnostics["optimizer_gradient"] == "analytical"

    u_vine = np.random.default_rng(20260621).uniform(
        0.05, 0.95, size=(30, 3))
    vine = RVineCopula(candidates=[IndependentCopula]).fit(u_vine)
    assert vine.sample(5, rng=np.random.default_rng(1)).shape == (5, 3)
    assert vine.predict(
        5, u=u_vine, rng=np.random.default_rng(2)).shape == (5, 3)
