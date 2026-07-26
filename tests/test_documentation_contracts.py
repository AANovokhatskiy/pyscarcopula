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
# OPTIONAL_DOCUMENTATION_MODULES = {"pyvinecopulib"}


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


def test_documented_python_blocks_compile_import_and_bind_public_calls():
    for path in DOC_FILES:
        for index, source in enumerate(_python_blocks(path)):
            filename = f"{path}:{index}"
            compile(source, filename=filename, mode="exec")
            tree = ast.parse(source, filename=filename)
            if path == MIGRATION_NOTES and "# Removed" in source:
                continue
            imports = [
                node for node in tree.body
                if isinstance(node, (ast.Import, ast.ImportFrom))
            ]
            namespace = {}
            try:
                exec(compile(
                    ast.Module(body=imports, type_ignores=[]),
                    filename=filename,
                    mode="exec",
                ), namespace)
            except ModuleNotFoundError as exc:
                # if exc.name in OPTIONAL_DOCUMENTATION_MODULES:
                #     continue
                raise

            for call in (
                    node for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)):
                target = namespace.get(call.func.id)
                if not callable(target):
                    continue
                if not getattr(target, "__module__", "").startswith(
                        "pyscarcopula"):
                    continue
                if any(isinstance(arg, ast.Starred) for arg in call.args):
                    continue
                if any(keyword.arg is None for keyword in call.keywords):
                    continue
                try:
                    signature = inspect.signature(target)
                except (TypeError, ValueError):
                    continue
                positional = [object()] * len(call.args)
                keywords = {keyword.arg: object() for keyword in call.keywords}
                try:
                    signature.bind_partial(*positional, **keywords)
                except TypeError as exc:
                    raise AssertionError(
                        f"invalid documented call at {filename}:{call.lineno}: "
                        f"{exc}"
                    ) from exc


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


def test_public_docs_exclude_development_plans_and_phase_reports():
    forbidden = (
        "phase-8",
        "phase 8",
        "phase-9",
        "release gate",
        "release-gate",
        "future work",
        "not implemented",
        "proposed api",
    )
    for path in DOC_FILES:
        text = path.read_text(encoding="utf-8").lower()
        for phrase in forbidden:
            assert phrase not in text, (
                f"{path.relative_to(ROOT)} contains development artifact "
                f"{phrase!r}"
            )

    assert not (ROOT / "docs/validation").exists()
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "validation/" not in nav


def test_notebooks_do_not_import_private_pyscarcopula_modules():
    for path in sorted((ROOT / "examples").glob("*.ipynb")):
        text = path.read_text(encoding="utf-8")
        assert "from pyscarcopula._" not in text
        assert "import pyscarcopula._" not in text


def test_vinecopula_is_the_discoverable_canonical_vine_api():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for public_surface in (
            "VineCopula()",
            "VineCopula.cvine(",
            "VineCopula.dvine(",
            "RVineMatrix.from_trees(",
            "natural_order_matrix"):
        assert public_surface in readme

    api = (ROOT / "docs/api/vine.md").read_text(encoding="utf-8")
    assert api.index("## VineCopula") < api.index("## RVineCopula")
    assert "RVineCopula is VineCopula" in api
    assert "legacy" in api.lower()


def test_complete_public_documentation_examples_execute():
    namespaces = {}
    for relative_path in (
            "docs/index.md",
            "docs/api/copulas.md",
            "docs/api/persistence.md"):
        path = ROOT / relative_path
        namespace = {"__name__": "__documentation_example__"}
        for source in _python_blocks(path):
            exec(compile(source, filename=str(path), mode="exec"), namespace)
        namespaces[relative_path] = namespace

    assert namespaces["docs/index.md"]["u"].shape == (400, 2)
    assert np.isfinite(
        namespaces["docs/api/copulas.md"]["fitted_log_likelihood"])
    assert (
        namespaces["docs/api/copulas.md"]["conditional_cdf"].shape
        == (200,)
    )
    assert namespaces["docs/api/persistence.md"]["samples"].shape == (20, 2)


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


def test_top_level_api_exposes_docstrings_and_complete_annotations():
    from pyscarcopula import api

    for function in (
        api.fit,
        api.log_likelihood,
        api.predictive_mean,
        api.mixture_h,
        api.sample,
        api.predict,
    ):
        assert inspect.getdoc(function), function.__name__
        signature = inspect.signature(function)
        assert signature.return_annotation is not inspect.Signature.empty
        for parameter in signature.parameters.values():
            assert parameter.annotation is not inspect.Parameter.empty, (
                f"{function.__name__}.{parameter.name} is not annotated"
            )


def test_distribution_declares_pep561_typing_marker():
    assert (ROOT / "pyscarcopula/py.typed").is_file()
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'pyscarcopula = ["py.typed"]' in pyproject


def test_documented_vine_signatures_match_runtime():
    from pyscarcopula import CVineCopula, RVineCopula, VineCopula

    assert RVineCopula is VineCopula
    for cls in (CVineCopula, VineCopula):
        sample_parameters = inspect.signature(cls.sample).parameters
        assert "n" in sample_parameters
        assert "u" in sample_parameters
        assert "rng" in sample_parameters
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
        VineCopula,
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
    vine = VineCopula(candidates=[IndependentCopula]).fit(u_vine)
    assert vine.sample(5, rng=np.random.default_rng(1)).shape == (5, 3)
    assert vine.predict(
        5, u=u_vine, rng=np.random.default_rng(2)).shape == (5, 3)
