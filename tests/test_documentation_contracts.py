"""Executable contracts for public documentation."""

from __future__ import annotations

import ast
import importlib
import inspect
import json
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
WORKFLOW_FILES = sorted((ROOT / ".github/workflows").glob("*.yml"))
OPTIONAL_DOCUMENTATION_MODULES = {"pyvinecopulib"}


def test_jacobi_docs_describe_native_gradient_sampling_and_singleton_contract():
    estimation = (ROOT / "docs/guide/estimation-methods.md").read_text(encoding="utf-8")
    backends = (ROOT / "docs/reference/scar-jacobi.md").read_text(encoding="utf-8")
    mathematics = (ROOT / "docs/guide/mathematical-contracts.md").read_text(encoding="utf-8")
    for guide in (estimation, backends):
        assert "gradient_kind='native_finite_difference'" in guide
        assert "one-row prepared evaluator" in guide or "one observation" in guide
        assert "explicitly rejects `analytical_grad=True`" not in guide
        assert "Not available with `spectral_coeff`" not in guide
    for guide in (estimation, mathematics):
        assert "Numba kernel" not in guide
        assert "separate execution paths" in guide or "separate production implementations" in guide


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
                if exc.name in OPTIONAL_DOCUMENTATION_MODULES:
                    continue
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
        assert obsolete not in path.read_text(encoding="utf-8")


def test_removed_native_backend_examples_do_not_return():
    forbidden = re.compile(r"\bbackend\s*=")
    for path in DOC_FILES:
        assert forbidden.search(path.read_text(encoding="utf-8")) is None


def test_removed_public_aliases_do_not_return_to_docs_or_examples():
    forbidden = (
        "u_train=",
        "LatentResult.alpha",
        "pyscarcopula.numerical.auto_tm",
        "pyscarcopula.numerical.tm_gradient",
        "pyscarcopula.numerical.tm_grid",
        "TMGrid",
        "CVineCopula",
        "pyscarcopula.vine.cvine",
        "CopulaCapabilities",
        "CopulaProtocol",
        "spectral_basis_order='adaptive'",
        'spectral_basis_order="adaptive"',
    )
    for path in DOC_FILES:
        text = path.read_text(encoding="utf-8")
        for value in forbidden:
            assert value not in text, (
                f"{path.relative_to(ROOT)} contains removed API {value!r}"
            )

    for path in sorted((ROOT / "examples").glob("*.ipynb")):
        text = path.read_text(encoding="utf-8")
        assert "u_train=" not in text


def test_breaking_removal_migration_notes_are_in_changelog():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    for removed_name in (
        "CopulaProtocol",
        "CommonCopulaProtocol",
        "BivariateCopulaProtocol",
        "MultivariateCopulaProtocol",
        "CopulaCapabilities",
        "CVineCopula",
        "TMGrid",
    ):
        assert removed_name in changelog
    assert "VineCopula.cvine" in changelog
    assert "no automatic migration path" in " ".join(changelog.split())
    assert "No compatibility aliases" in changelog


def test_workflows_reference_existing_test_files():
    pattern = re.compile(r"tests/[A-Za-z0-9_./-]+\.py")
    for path in WORKFLOW_FILES:
        for test_path in pattern.findall(path.read_text(encoding="utf-8")):
            assert (ROOT / test_path).is_file(), (
                f"{path.relative_to(ROOT)} references missing {test_path}"
            )


def test_removed_experimental_namespace_is_physically_absent():
    assert not (ROOT / "pyscarcopula/copula/experimental").exists()


def test_tmgrid_is_physically_absent_from_the_package():
    assert not (ROOT / "pyscarcopula/numerical/tm_grid.py").exists()
    assert not hasattr(pyscarcopula.numerical, "TMGrid")


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


def test_notebooks_only_import_the_approved_private_pobs_helper():
    approved_private_import = "from pyscarcopula._utils import pobs"
    for path in sorted((ROOT / "examples").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", ()))
            for cell in notebook.get("cells", ())
            if cell.get("cell_type") == "code"
        )
        private_imports = {
            line.strip()
            for line in code.splitlines()
            if line.strip().startswith((
                "from pyscarcopula._",
                "import pyscarcopula._",
            ))
        }
        assert private_imports <= {approved_private_import}, (
            f"{path.relative_to(ROOT)} imports an unapproved private helper: "
            f"{sorted(private_imports - {approved_private_import})}"
        )
        if re.search(r"\bpobs\(", code):
            assert approved_private_import in private_imports
            assert "def pobs(" not in code


def test_vinecopula_is_the_discoverable_canonical_vine_api():
    api = (ROOT / "docs/api/vine.md").read_text(encoding="utf-8")
    for public_surface in (
            "VineCopula()",
            "VineCopula.cvine(",
            "VineCopula.dvine(",
            "RVineMatrix.from_trees(",
            "natural_order_matrix"):
        assert public_surface in api

    assert api.index("## VineCopula") < api.index("## RVineCopula")
    assert "RVineCopula is VineCopula" in api
    assert "compatibility name" in api.lower()


def test_complete_public_documentation_examples_execute():
    namespaces = {}
    for relative_path in (
            "docs/index.md",
            "docs/api/copulas.md",
            "docs/api/persistence.md",
            "docs/api/static-models.md",
            "docs/guide/performance.md"):
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
    static = namespaces["docs/api/static-models.md"]
    assert static["conditional"].shape == (10_000, 5)
    np.testing.assert_array_equal(static["conditional"][:, 0], 0.25)
    assert static["student_result"].correlation_matrix is None
    assert np.isfinite(
        namespaces["docs/guide/performance.md"]["result"].log_likelihood)


def test_quick_start_executes_as_one_workflow():
    path = ROOT / "docs/getting-started/quickstart.md"
    namespace = {"__name__": "__documentation_example__"}
    for source in _python_blocks(path):
        exec(compile(source, filename=str(path), mode="exec"), namespace)

    assert namespace["u"].shape == (400, 2)
    assert namespace["u_pred"].shape == (500, 2)
    assert namespace["u_cond"].shape == (500, 2)
    assert namespace["v"].shape == (500, 2)


def test_bivariate_guide_executes_as_one_workflow():
    from pyscarcopula import GumbelCopula

    path = ROOT / "docs/guide/bivariate.md"
    u = GumbelCopula(rotate=180).sample_at_parameter(
        80,
        np.full(80, 1.5),
        rng=np.random.default_rng(9),
    )
    namespace = {
        "__name__": "__documentation_example__",
        "u": u,
    }
    for source in _python_blocks(path):
        exec(compile(source, filename=str(path), mode="exec"), namespace)

    assert namespace["u_pred"].shape == (500, 2)
    assert namespace["u_cond"].shape == (500, 2)
    assert namespace["u_current"].shape == (500, 2)
    assert namespace["r_t"].shape == (120,)


def test_factor_api_intro_example_executes():
    path = ROOT / "docs/api/factor.md"
    source = next(
        block for block in _python_blocks(path)
        if "FactorStudentEvaluator(operator, u).evaluate" in block
    )
    namespace = {"__name__": "__documentation_example__"}
    exec(compile(source, filename=str(path), mode="exec"), namespace)

    evaluation = namespace["evaluation"]
    assert evaluation.log_pdf.shape == (50,)
    assert np.isfinite(evaluation.log_likelihood)


def test_static_multivariate_docs_do_not_claim_predictive_mean():
    path = ROOT / "docs/guide/multivariate_models.md"
    text = path.read_text(encoding="utf-8")
    common_api = text.split("## Common API", 1)[1]
    static_example = common_api.split(
        "Dynamic scalar-parameter models", 1)[0]

    assert "cop = GaussianCopula()" in static_example
    assert "cop.predictive_mean(" not in static_example


def test_static_correlation_policy_is_documented_consistently():
    targets = (
        ROOT / "ARCHITECTURE.md",
        ROOT / "docs/guide/mathematical-contracts.md",
        ROOT / "docs/guide/multivariate_models.md",
        ROOT / "docs/guide/estimation-methods.md",
        ROOT / "docs/guide/numerical-backends.md",
        ROOT / "docs/guide/performance.md",
        ROOT / "docs/api/multivariate_models.md",
    )
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in targets)

    for term in (
            "corr_mode", "corr_estimator", "fixed", "shrinkage",
            "cholesky", "factor", "plug-in"):
        assert term in combined
    assert "MLE is the label for a static model" in combined
    assert "MLE estimates one constant copula parameter." not in combined
    assert "corr_mode=\"fixed\"" in combined
    assert "corr_mode=\"dense\"" not in combined


def test_documented_dynamic_predictive_mean_examples_execute():
    path = ROOT / "docs/guide/multivariate_models.md"
    blocks = _python_blocks(path)

    student_source = next(
        block for block in blocks
        if "# GAS is also supported" in block
    )
    student_namespace = {
        "__name__": "__documentation_example__",
        "returns": np.random.default_rng(20260728).standard_normal((60, 6)),
    }
    exec(
        compile(student_source, filename=str(path), mode="exec"),
        student_namespace,
    )
    assert student_namespace["df_t"].shape == (60,)

    setup_source = next(
        block for block in blocks
        if "np.fill_diagonal(R, 1.0)" in block
    )
    dynamic_source = next(
        block for block in blocks
        if "dynamic.predictive_mean(u)" in block
    )
    equicorr_namespace = {"__name__": "__documentation_example__"}
    exec(
        compile(setup_source, filename=str(path), mode="exec"),
        equicorr_namespace,
    )
    exec(
        compile(dynamic_source, filename=str(path), mode="exec"),
        equicorr_namespace,
    )
    assert equicorr_namespace["rho_t"].shape == (200,)


def test_documented_prediction_defaults_match_runtime():
    from pyscarcopula import VineCopula
    from pyscarcopula.api import predict

    for callable_ in (predict, VineCopula.predict):
        parameters = inspect.signature(callable_).parameters
        assert parameters["horizon"].default == "next"

    for callable_ in (VineCopula.predict,):
        parameters = inspect.signature(callable_).parameters
        assert parameters["predictive_r_mode"].default is None

    vine_parameters = inspect.signature(VineCopula.predict).parameters
    assert "conditional_method" not in vine_parameters
    assert vine_parameters["dynamic_conditioning"].default == "ignore"
    assert vine_parameters["return_diagnostics"].default is False


def test_relative_markdown_links_resolve():
    pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for path in sorted((ROOT / "docs").rglob("*.md")):
        for raw_target in pattern.findall(
                path.read_text(encoding="utf-8")):
            target = raw_target.split("#", 1)[0]
            if not target or re.match(r"^[a-z]+://", target):
                continue
            resolved = (path.parent / target).resolve()
            assert resolved.exists(), (
                f"{path.relative_to(ROOT)} links to missing {target}")


def test_conditional_method_is_documented_as_diagnostics_only():
    prediction_guide = (
        ROOT / "docs/guide/prediction-semantics.md"
    ).read_text(encoding="utf-8")
    vine_guide = (
        ROOT / "docs/guide/vine.md"
    ).read_text(encoding="utf-8")

    assert "an output diagnostics field" in prediction_guide
    assert "does not accept `conditional_method`" in vine_guide

    for path in DOC_FILES:
        for source in _python_blocks(path):
            tree = ast.parse(source, filename=str(path))
            for call in (
                    node for node in ast.walk(tree)
                    if isinstance(node, ast.Call)):
                assert all(
                    keyword.arg != "conditional_method"
                    for keyword in call.keywords
                ), (
                    f"{path.relative_to(ROOT)} passes diagnostics-only "
                    "conditional_method as an argument"
                )


def test_documented_public_imports():
    from pyscarcopula import (
        BivariateCopula,
        CopulaBase,
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
    from pyscarcopula import RVineCopula, VineCopula

    assert RVineCopula is VineCopula
    for cls in (VineCopula,):
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
        "| GAS | Any supported scaling | Native finite differences "
        "| `native` | `native_finite_difference` |",
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


def test_small_factor_walkthrough_executes_in_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = ROOT / "docs/guide/factor-models.md"
    namespace = {"__name__": "__documentation_example__"}
    for source in _python_blocks(path):
        # Explicitly separate large-output recipe is outside the small walkthrough.
        if "large_d, large_k" in source:
            continue
        exec(compile(source, filename=str(path), mode="exec"), namespace)
    assert namespace["u"].shape == (80, 20)
    assert namespace["conditional"].shape == (128, 20)
    np.testing.assert_array_equal(namespace["conditional"][:, 0], 0.25)
    np.testing.assert_array_equal(namespace["conditional"][:, 3], 0.80)
    np.testing.assert_array_equal(namespace["portable"].loadings, namespace["mapped"].loadings)


def test_result_types_in_bivariate_workflow():
    from pyscarcopula import GumbelCopula
    from pyscarcopula._types import GASResult, LatentResult, MLEResult

    u = GumbelCopula().sample_at_parameter(40, np.full(40, 1.8), rng=np.random.default_rng(9))
    model = GumbelCopula()
    for method, result_type, names in (
        ("mle", MLEResult, None),
        ("gas", GASResult, ("omega", "gamma", "beta")),
        ("scar-tm-ou", LatentResult, ("kappa", "mu", "nu")),
        ("scar-tm-jacobi", LatentResult, ("kappa", "m", "xi")),
    ):
        result = model.fit(u, method=method)
        assert isinstance(result, result_type)
        assert np.isfinite(result.log_likelihood)
        if names:
            assert tuple(result.params.names) == names


def test_documented_joint_factor_example_uses_identifiable_rank():
    from pyscarcopula import NumericalConfig, StochasticStudentCopula

    path = ROOT / "docs/guide/multivariate_models.md"
    namespace = dict(np=np, NumericalConfig=NumericalConfig,
                     StochasticStudentCopula=StochasticStudentCopula)
    blocks = _python_blocks(path)
    data = next(block for block in blocks if 'size=(200, 5)' in block)
    joint = next(block for block in blocks if 'joint = StochasticStudentCopula(' in block)
    for source in (data, joint):
        exec(compile(source, filename=str(path), mode="exec"), namespace)
    assert namespace["joint"].d >= 2 * namespace["joint"].factor_rank + 1
    assert np.isfinite(namespace["joint_result"].log_likelihood)
    assert namespace["joint_result"].correlation_matrix is None


def test_documented_archimedean_vine_transform_example_executes():
    from pyscarcopula import VineCopula

    path = ROOT / "docs/guide/transforms.md"
    source = next(block for block in _python_blocks(path) if 'bounded_vine =' in block)
    namespace = dict(VineCopula=VineCopula,
                     u=np.random.default_rng(41).uniform(0.05, 0.95, (40, 3)))
    exec(compile(source, filename=str(path), mode="exec"), namespace)
    draws = namespace["bounded_vine"].sample(10, rng=np.random.default_rng(42))
    assert draws.shape == (10, 3)
    assert np.all((draws > 0) & (draws < 1))
