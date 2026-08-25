"""Contracts for the native C++ architecture checker."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = ROOT / "tools" / "check_cpp_architecture.py"
CHECKER_MODULE_NAME = "_pyscarcopula_check_cpp_architecture"
CHECKER_SPEC = importlib.util.spec_from_file_location(
    CHECKER_MODULE_NAME,
    CHECKER_PATH,
)
if CHECKER_SPEC is None or CHECKER_SPEC.loader is None:
    raise ImportError(f"cannot load architecture checker from {CHECKER_PATH}")
CHECKER_MODULE = importlib.util.module_from_spec(CHECKER_SPEC)
sys.modules[CHECKER_MODULE_NAME] = CHECKER_MODULE
CHECKER_SPEC.loader.exec_module(CHECKER_MODULE)
check_repository = CHECKER_MODULE.check_repository


def _write(root: Path, relative: str, text: str = "") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _set_manifest(
    root: Path,
    compute: list[str] | tuple[str, ...],
    bindings: list[str] | tuple[str, ...] = ("bindings/module.cpp",),
) -> None:
    _write(
        root,
        "pyscarcopula/_cpp/build_support/sources.py",
        f"SCAR_COMPUTE_SOURCES = {tuple(compute)!r}\n"
        f"PYTHON_BINDING_SOURCES = {tuple(bindings)!r}\n",
    )


def _minimal_repository(root: Path) -> Path:
    _write(
        root,
        "setup.py",
        "SCAR_COMPUTE_SOURCES = manifest.SCAR_COMPUTE_SOURCES\n"
        "PYTHON_BINDING_SOURCES = manifest.PYTHON_BINDING_SOURCES\n",
    )
    _set_manifest(root, ())
    _write(root, "pyscarcopula/_cpp/include/scar/copula.hpp", "#pragma once\n")
    _write(root, "pyscarcopula/_cpp/include/scar/gas.hpp", "#pragma once\n")
    _write(root, "pyscarcopula/_cpp/include/scar/ou.hpp", "#pragma once\n")
    _write(
        root,
        "pyscarcopula/_cpp/src/bindings/module.cpp",
        '#include "module.hpp"\n\n'
        "PYBIND11_MODULE(_scar_cpp, module) {\n"
        "    pyscarcopula::bindings::bind_common(module);\n"
        "}\n",
    )
    _write(
        root,
        "pyscarcopula/_cpp/src/bindings/module.hpp",
        "#include <pybind11/pybind11.h>\n",
    )
    return root


def _rules(root: Path) -> set[str]:
    return {violation.rule for violation in check_repository(root)}


def test_current_repository_satisfies_cpp_architecture_contract():
    assert check_repository(ROOT) == []


def test_stage2_foundation_helpers_have_single_canonical_owners():
    include = ROOT / "pyscarcopula" / "_cpp" / "include" / "scar"
    source = ROOT / "pyscarcopula" / "_cpp" / "src"
    expected_headers = {
        "core/span.hpp",
        "core/matrix_view.hpp",
        "core/checked_arithmetic.hpp",
        "core/threading.hpp",
        "core/status.hpp",
        "core/result.hpp",
        "math/normal.hpp",
        "copula/model_descriptor.hpp",
        "copula/transforms.hpp",
        "copula/rotation.hpp",
    }
    assert all((include / path).is_file() for path in expected_headers)

    cpp_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(source.rglob("*.cpp"))
    )
    header_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(include.rglob("*.hpp"))
    )

    for removed_duplicate in (
        "validate_threads",
        "effective_threads",
        "checked_product",
    ):
        assert removed_duplicate not in cpp_text

    assert len(re.findall(r"\bdouble\s+normal_cdf\s*\(", cpp_text)) == 1
    assert len(re.findall(r"\bdouble\s+normal_quantile\s*\(", cpp_text)) == 1
    assert len(re.findall(
        r"\bResult<std::size_t>\s+rosenblatt_output_size\s*\(",
        cpp_text,
    )) == 1
    assert "CopulaSpec::expected_dimension" not in cpp_text
    assert cpp_text.count(
        "TypedModelDescriptor CopulaSpec::model_descriptor()"
    ) == 1
    descriptor_text = (
        include / "copula" / "model_descriptor.hpp"
    ).read_text(encoding="utf-8")
    assert "class TypedModelDescriptor" in descriptor_text
    assert "int expected_dimension() const noexcept" in descriptor_text
    spec_text = (include / "copula" / "spec.hpp").read_text(encoding="utf-8")
    assert "TypedModelDescriptor model_descriptor() const noexcept;" in spec_text
    assert "int expected_dimension() const noexcept;" not in spec_text
    assert "struct DoubleView" not in header_text
    assert "using DoubleView = Span<const double>;" in (
        include / "core" / "span.hpp"
    ).read_text(encoding="utf-8")
    assert "n_threads > 256" not in cpp_text

    foundation_files = [
        *sorted((include / "core").glob("*.hpp")),
        *sorted((include / "math").glob("*.hpp")),
        include / "copula" / "transforms.hpp",
        include / "copula" / "rotation.hpp",
        source / "math" / "normal.cpp",
        source / "copula" / "transforms.cpp",
        source / "copula" / "rotation.cpp",
    ]
    forbidden_foundation_include = re.compile(
        r'#\s*include\s+"scar/(?:detail/|factor\.hpp|gas(?:_rvine)?\.hpp|'
        r'ou\.hpp|rvine(?:_plan)?\.hpp|copula\.hpp)'
    )
    for path in foundation_files:
        assert forbidden_foundation_include.search(
            path.read_text(encoding="utf-8")
        ) is None, path

    validation_text = (
        source / "scar_ou" / "validation.cpp"
    ).read_text(encoding="utf-8")
    assert "Result<std::size_t> rosenblatt_output_size(" in validation_text
    assert "result.failure.coordinate" in validation_text
    assert "return success(output_size);" in validation_text
    for name in ("gaussian_rosenblatt.cpp", "student_rosenblatt.cpp"):
        consumer = (source / "scar_ou" / name).read_text(encoding="utf-8")
        assert "const Result<std::size_t> output_shape" in consumer
        assert "output_shape.is_ok()" in consumer


def test_stage3_pair_copulas_are_vertical_and_prepared_once():
    include = ROOT / "pyscarcopula" / "_cpp" / "include" / "scar"
    source = ROOT / "pyscarcopula" / "_cpp" / "src" / "copula"
    manifest = include / "copula" / "pair" / "families.def"
    entries = re.findall(
        r"^SCAR_PAIR_FAMILY\(\s*([A-Za-z][A-Za-z0-9_]*)\s*,\s*"
        r"([a-z][a-z0-9_]*)\s*,",
        manifest.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert entries
    families = tuple(package for _, package in entries)

    assert not (include / "detail" / "copula.hpp").exists()
    assert (include / "copula" / "prepared_pair_kernel.hpp").is_file()
    for family in families:
        assert (include / "copula" / "pair" / f"{family}.hpp").is_file()
        assert (source / "pair" / f"{family}.cpp").is_file()

    registry = (source / "pair" / "runtime_registry.cpp").read_text(
        encoding="utf-8")
    assert registry.count("switch (family)") == 1
    assert registry.count(
        '#include "scar/copula/pair/families.def"') == 2

    kernel = (include / "copula" / "pair" / "kernel.hpp").read_text(
        encoding="utf-8")
    assert "PairSupportKernel" not in kernel
    assert "Rotation" not in kernel
    assert "Transform" not in kernel
    for family in families:
        implementation = (source / "pair" / f"{family}.cpp").read_text(
            encoding="utf-8")
        assert "_h_rotated" not in implementation
        assert "_h_inverse_rotated" not in implementation
        assert "scar/copula/rotation.hpp" not in implementation

    dispatch = (source / "dispatch.cpp").read_text(encoding="utf-8")
    for family in ("clayton", "gumbel", "frank", "joe"):
        assert re.search(rf"\b{family}_[A-Za-z0-9_]+", dispatch) is None
    assert "const scar::PreparedPairKernel kernel(spec);" in dispatch
    assert "is_pair_copula_family" not in dispatch

    core = (source / "core.cpp").read_text(encoding="utf-8")
    assert "const PreparedPairKernel kernel(spec);" in core
    assert core.count("scar_internal::copula_log_pdf_unrotated(") == 0
    assert "is_pair_copula_family" not in core

    binding = (
        ROOT / "pyscarcopula" / "_cpp" / "src" / "bindings" / "copula.cpp"
    ).read_text(encoding="utf-8")
    assert binding.count(
        '#include "scar/copula/pair/families.def"') == 1
    adapter = (
        ROOT / "pyscarcopula" / "numerical" / "_cpp_copula.py"
    ).read_text(encoding="utf-8")
    rvine_adapter = (
        ROOT / "pyscarcopula" / "numerical" / "_cpp_rvine.py"
    ).read_text(encoding="utf-8")
    for enum_name, _ in entries:
        assert f'.value("{enum_name}",' not in binding
        assert f"CopulaFamily.{enum_name}" not in adapter
    assert "_builtin_copula_types" not in rvine_adapter
    assert "from pyscarcopula.copula." not in rvine_adapter

    sources = (
        ROOT / "pyscarcopula" / "_cpp" / "build_support" / "sources.py"
    ).read_text(encoding="utf-8")
    assert "PAIR_FAMILY_SOURCES = _pair_family_sources()" in sources
    assert "*PAIR_FAMILY_SOURCES" in sources


def test_stage4_multivariate_models_are_vertical_and_typed():
    cpp = ROOT / "pyscarcopula" / "_cpp"
    include = cpp / "include" / "scar"
    source = cpp / "src" / "copula"
    multivariate_include = include / "copula" / "multivariate"
    multivariate_source = source / "multivariate"

    assert not (source / "multivariate.cpp").exists()
    assert not (source / "families" / "student.cpp").exists()
    assert not (include / "detail" / "copula" / "student.hpp").exists()

    for relative in (
        "correlation/dense.hpp",
        "correlation/factor.hpp",
        "gaussian/model.hpp",
        "gaussian/density.hpp",
        "gaussian/conditional.hpp",
        "equicorrelation/model.hpp",
        "equicorrelation/kernel.hpp",
        "student/model.hpp",
        "student/distribution.hpp",
        "student/quantile.hpp",
        "student/ppf_cache.hpp",
        "student/density.hpp",
        "student/conditional.hpp",
        "student/rosenblatt.hpp",
    ):
        assert (multivariate_include / relative).is_file()

    for relative in (
        "correlation/conditional.cpp",
        "correlation/dense.cpp",
        "correlation/factor.cpp",
        "equicorrelation/evaluator.cpp",
        "gaussian/density.cpp",
        "gaussian/conditional.cpp",
        "equicorrelation/model.cpp",
        "equicorrelation/kernel.cpp",
        "student/distribution.cpp",
        "student/density.cpp",
        "student/evaluator.cpp",
        "student/conditional.cpp",
        "student/factor_density.cpp",
        "student/factor_grid.cpp",
        "student/ppf_cache.cpp",
        "student/quantile.cpp",
        "student/rosenblatt.cpp",
    ):
        assert (multivariate_source / relative).is_file()

    dispatch = (multivariate_source / "dispatch.cpp").read_text(
        encoding="utf-8")
    assert len(dispatch.splitlines()) <= 100
    for model_implementation in (
        "StudentWorkspace",
        "EquicorrStats",
        "conditional_df",
        "parallel_for_blocks",
        "student_fill_",
        "equicorr_log_pdf_from_stats(",
        "normal_quantile",
        "cholesky",
    ):
        assert model_implementation not in dispatch

    conditional_engine = (
        multivariate_source / "correlation" / "conditional.cpp"
    ).read_text(encoding="utf-8")
    for model_specific in (
        "Student", "student_", "conditional_df", "chi_square",
    ):
        assert model_specific not in conditional_engine
    student_conditional = (
        multivariate_source / "student" / "conditional.cpp"
    ).read_text(encoding="utf-8")
    assert "student_conditional_scale" in student_conditional
    assert "conditional_df" in student_conditional

    spec = (include / "copula" / "spec.hpp").read_text(encoding="utf-8")
    for old_field in (
        "l_inv;", "log_det;", "ppf_n_obs;", "ppf_nodes;", "ppf_table;",
        "gaussian_z1_cache;", "equicorr_sum_cache;",
    ):
        assert old_field not in spec
    storage = (include / "copula" / "model_storage.hpp").read_text(
        encoding="utf-8")
    assert "std::variant<" in storage
    assert "student::DenseModelStorage" in storage
    assert "gaussian::FactorModelStorage" in storage

    factor = (
        multivariate_include / "correlation" / "factor.hpp"
    ).read_text(encoding="utf-8")
    assert "FactorCorrelationOperator" in factor
    assert "FactorStudent" not in factor

    for package in (
        multivariate_include / "gaussian",
        multivariate_source / "gaussian",
    ):
        for path in package.rglob("*"):
            if path.is_file():
                text = path.read_text(encoding="utf-8")
                assert "/student/" not in text
                assert "Student" not in text


def test_stage5_application_modules_use_prepared_copula_interfaces():
    cpp = ROOT / "pyscarcopula" / "_cpp"
    include = cpp / "include" / "scar"
    source = cpp / "src"

    emission_header = include / "copula" / "prepared_dynamic_emission.hpp"
    emission_source = source / "copula" / "prepared_dynamic_emission.cpp"
    assert (include / "copula" / "grid_values.hpp").is_file()
    assert emission_header.is_file()
    assert emission_source.is_file()
    emission_contract = emission_header.read_text(encoding="utf-8")
    assert "class PreparedDynamicEmission" in emission_contract
    assert "class PreparedDynamicEmissionWorkspace" in emission_contract
    assert "DynamicEmissionRowResult evaluate_parameter(" in emission_contract
    assert "bool observation_cache_compatible(" in emission_contract
    assert "copula/pair/" not in emission_contract
    assert "copula/multivariate/" not in emission_contract
    assert "detail/copula/" not in emission_contract

    gas = (source / "gas" / "evaluator.cpp").read_text(encoding="utf-8")
    assert "PreparedDynamicEmission" in gas
    assert '#include "scar/copula.hpp"' not in gas
    assert "detail/copula/dispatch.hpp" not in gas
    assert "copula/multivariate/" not in gas
    assert "copula/pair/" not in gas
    gas_header = (include / "gas.hpp").read_text(encoding="utf-8")
    assert "GasStateResult initial_state_prepared(" in gas_header
    assert "GasUpdateResult update_one_prepared(" in gas_header
    assert "GasUpdateResult update_observation_prepared(" in gas_header

    static_header = (include / "copula.hpp").read_text(encoding="utf-8")
    static_source = (source / "likelihood" / "static.cpp").read_text(
        encoding="utf-8")
    assert "PreparedDynamicEmission" in static_header
    assert "emission_" in static_header
    assert "emission_->evaluate_parameter(" in static_source

    ou_header = (include / "ou.hpp").read_text(encoding="utf-8")
    assert '#include "scar/copula.hpp"' not in ou_header
    assert "const PreparedDynamicEmission* prepared_emission_" in ou_header
    assert "PreparedDynamicEmission emission_;" in ou_header
    for name in (
        "evaluator.cpp",
        "likelihood.cpp",
        "monte_carlo.cpp",
        "prediction.cpp",
        "state_distribution.cpp",
        "transition.cpp",
        "validation.cpp",
    ):
        text = (source / "scar_ou" / name).read_text(encoding="utf-8")
        assert "PreparedDynamicEmission" in text, name
        assert "detail/copula/dispatch.hpp" not in text, name
        assert "copula/multivariate/" not in text, name
        assert "copula/pair/" not in text, name
        assert "factor_operator(" not in text, name
        assert "student_ppf_nodes(" not in text, name

    gradient = (source / "scar_ou" / "gradient.cpp").read_text(
        encoding="utf-8")
    assert "&prepared->compatibility_spec() == &copula" not in gradient
    assert "if (prepared != nullptr)" in gradient
    assert "const CopulaSpec& copula = emission.compatibility_spec();" in gradient

    vine_header = (include / "rvine.hpp").read_text(encoding="utf-8")
    assert "PreparedPairKernel kernel;" in vine_header
    assert "PreparedPairKernel transposed_kernel;" in vine_header
    for path in (source / "vine").glob("*.cpp"):
        text = path.read_text(encoding="utf-8")
        assert "detail/copula/dispatch.hpp" not in text, path
        assert "copula/multivariate/" not in text, path
        assert "copula/pair/" not in text, path

    composition = (source / "gas" / "rvine_sampler.cpp").read_text(
        encoding="utf-8")
    assert '#include "scar/gas_rvine.hpp"' in composition
    assert '#include "scar/rvine.hpp"' in composition
    assert '#include "scar/copula/prepared_dynamic_emission.hpp"' in composition
    assert "gas_emissions" in composition
    assert "gas_workspaces" in composition
    assert "update_one_prepared(" in composition
    assert "evaluator.update_one(" not in composition

    composition_header = (include / "gas_rvine.hpp").read_text(
        encoding="utf-8")
    assert '#include "scar/copula/spec.hpp"' in composition_header
    assert '#include "scar/copula.hpp"' not in composition_header


def test_stage6_public_cpp_api_is_domain_scoped_and_caller_neutral():
    cpp = ROOT / "pyscarcopula" / "_cpp"
    include = cpp / "include" / "scar"
    source = cpp / "src"

    contracts = {
        include / "static" / "result.hpp": "StaticObjectiveResult",
        include / "gas" / "result.hpp": "GasLogLikResult",
        include / "scar_ou" / "result.hpp": "ScarOuVectorResult",
        include / "copula" / "result.hpp": "MultivariateRowsResult",
        include / "vine" / "result.hpp": "SampleResult",
        include / "gas_rvine" / "result.hpp": "GasRvineSampleResult",
    }
    for path, marker in contracts.items():
        text = path.read_text(encoding="utf-8")
        assert marker in text
        assert "Status status" in text
        assert "FailureContext failure" in text
        assert re.search(r"\bint\s+status\b|\bfailure_index\b", text) is None

    umbrella_expectations = {
        include / "copula.hpp": (
            ('#include "scar/copula/result.hpp"',
             '#include "scar/static/result.hpp"'),
            ("struct MultivariateRowsResult", "struct StaticObjectiveResult")),
        include / "gas.hpp": (
            '#include "scar/gas/result.hpp"',
            "struct GasLogLikResult"),
        include / "ou.hpp": (
            '#include "scar/scar_ou/result.hpp"',
            "struct LogLikResult"),
        include / "rvine.hpp": (
            '#include "scar/vine/result.hpp"',
            "struct SampleResult"),
        include / "gas_rvine.hpp": (
            '#include "scar/gas_rvine/result.hpp"',
            "struct GasRvineSampleResult"),
    }
    for path, (required, forbidden) in umbrella_expectations.items():
        text = path.read_text(encoding="utf-8")
        required = required if isinstance(required, tuple) else (required,)
        forbidden = forbidden if isinstance(forbidden, tuple) else (forbidden,)
        assert all(marker in text for marker in required)
        assert all(marker not in text for marker in forbidden)

    copula_header = (include / "copula.hpp").read_text(encoding="utf-8")
    assert "copula/multivariate/" not in copula_header
    assert "copula/pair/" not in copula_header
    assert "copula/prepared_dynamic_emission.hpp" not in copula_header
    status_header = (include / "core" / "status.hpp").read_text(
        encoding="utf-8")
    assert "operator==(Status" not in status_header
    assert "operator!=(Status" not in status_header

    ou_header = (include / "ou.hpp").read_text(encoding="utf-8")
    assert "struct Workspace;" in ou_header
    assert "std::unique_ptr<Workspace>" in ou_header
    assert "ScarOuGridGradientWorkspace" not in ou_header
    assert "ScarOuSpectralGradientWorkspace" not in ou_header
    assert "ScarOuVectorResult predictive_mean_local_gh(" in ou_header
    assert "ScarOuVectorResult mixture_h_pair_auto(" in ou_header
    assert "ScarOuVectorResult predictive_mean(const OuParams& params)" in ou_header
    assert "ScarOuVectorResult mixture_h_pair(const OuParams& params)" in ou_header
    workspace = source / "scar_ou" / "gradient_workspace.hpp"
    assert workspace.is_file()
    assert "ScarOuSpectralGradientWorkspace" in workspace.read_text(
        encoding="utf-8")

    caller_specific = re.compile(
        r"\b(?:Python|pybind11|NumPy|PyObject)\b", re.IGNORECASE)
    compute_files = [
        *include.rglob("*.hpp"),
        *(path for path in source.rglob("*.cpp")
          if "bindings" not in path.relative_to(source).parts),
        *(path for path in source.rglob("*.hpp")
          if "bindings" not in path.relative_to(source).parts),
    ]
    for path in compute_files:
        assert caller_specific.search(path.read_text(encoding="utf-8")) is None


def test_stage6_checker_rejects_workspace_and_caller_leaks(tmp_path):
    cpp = "pyscarcopula/_cpp"
    include = f"{cpp}/include/scar"
    source = f"{cpp}/src"

    result_contracts = {
        "static/result.hpp": ("StaticObjectiveResult",),
        "gas/result.hpp": (
            "GasLogLikResult", "GasFilterResult", "GasUpdateResult",
            "GasStateResult", "GasPredictResult", "GasPathResult"),
        "scar_ou/result.hpp": (
            "ScarOuVectorResult", "LogLikResult", "GradLogLikResult",
            "StateDistribution", "SmoothedStateDistribution",
            "OuGridFilterResult", "TrajectoryLogPdfResult"),
        "copula/result.hpp": (
            "MultivariateRowsResult", "MultivariateGridResult",
            "EquicorrPreparationResult"),
        "vine/result.hpp": (
            "SampleResult", "ConditionalSampleResult", "DensityResult",
            "RosenblattResult", "MCMCResult"),
        "gas_rvine/result.hpp": ("GasRvineSampleResult",),
    }
    for relative, result_names in result_contracts.items():
        _write(
            tmp_path,
            f"{include}/{relative}",
            "".join(
                f"struct {name} {{\n"
                "  Status status;\n"
                "  FailureContext failure;\n"
                "};\n"
                for name in result_names
            ),
        )

    _write(
        tmp_path,
        f"{include}/copula.hpp",
        '#include "scar/copula/result.hpp"\n'
        '#include "scar/static/result.hpp"\n',
    )
    _write(
        tmp_path,
        f"{include}/gas.hpp",
        '#include "scar/gas/result.hpp"\n',
    )
    _write(
        tmp_path,
        f"{include}/rvine.hpp",
        '#include "scar/vine/result.hpp"\n',
    )
    _write(
        tmp_path,
        f"{include}/gas_rvine.hpp",
        '#include "scar/gas_rvine/result.hpp"\n',
    )
    ou_header = _write(
        tmp_path,
        f"{include}/ou.hpp",
        '#include "scar/scar_ou/result.hpp"\n'
        "struct Workspace;\n"
        "std::unique_ptr<Workspace> workspace_;\n"
        "ScarOuVectorResult predictive_mean_local_gh();\n"
        "ScarOuVectorResult mixture_h_pair_auto();\n"
        "ScarOuVectorResult predictive_mean(const OuParams& params);\n"
        "ScarOuVectorResult mixture_h_pair(const OuParams& params);\n",
    )
    _write(
        tmp_path,
        f"{source}/scar_ou/gradient_workspace.hpp",
        "struct ScarOuSpectralGradientWorkspace {};\n",
    )

    assert CHECKER_MODULE.check_public_cpp_api(tmp_path) == []

    status_header = _write(
        tmp_path,
        f"{include}/core/status.hpp",
        "bool operator==(Status lhs, int rhs);\n",
    )
    violations = CHECKER_MODULE.check_public_cpp_api(tmp_path)
    assert any(
        "must not compare implicitly" in violation.message
        for violation in violations
    )
    status_header.write_text("#pragma once\n", encoding="utf-8")

    copula_header = tmp_path / f"{include}/copula.hpp"
    copula_contract = copula_header.read_text(encoding="utf-8")
    copula_header.write_text(
        copula_contract
        + '#include "scar/copula/multivariate/student/conditional.hpp"\n',
        encoding="utf-8",
    )
    violations = CHECKER_MODULE.check_public_cpp_api(tmp_path)
    assert any(
        "imports concrete implementation" in violation.message
        for violation in violations
    )
    copula_header.write_text(copula_contract, encoding="utf-8")

    ou_contract = ou_header.read_text(encoding="utf-8")
    ou_header.write_text(
        ou_contract
        + "class ScarOuEvaluator { public:\n"
        + "  void legacy(int& status);\nprivate:\n};\n",
        encoding="utf-8",
    )
    violations = CHECKER_MODULE.check_public_cpp_api(tmp_path)
    assert any(
        "legacy status out-parameter" in violation.message
        for violation in violations
    )
    ou_header.write_text(ou_contract, encoding="utf-8")

    vine_header = tmp_path / f"{include}/rvine.hpp"
    vine_contract = vine_header.read_text(encoding="utf-8")
    vine_header.write_text(
        vine_contract + "void legacy(std::int64_t& failure_row);\n",
        encoding="utf-8",
    )
    violations = CHECKER_MODULE.check_public_cpp_api(tmp_path)
    assert any(
        "failure out-parameters" in violation.message
        for violation in violations
    )
    vine_header.write_text(vine_contract, encoding="utf-8")

    ou_header.write_text(
        ou_header.read_text(encoding="utf-8")
        + "struct ScarOuGridGradientWorkspace {};\n",
        encoding="utf-8",
    )
    violations = CHECKER_MODULE.check_public_cpp_api(tmp_path)
    assert any(
        "exposes private workspace" in violation.message
        for violation in violations
    )

    _write(
        tmp_path,
        f"{source}/scar_ou/caller_leak.cpp",
        "// Python-specific computational contract.\n",
    )
    violations = CHECKER_MODULE.check_public_cpp_api(tmp_path)
    assert any(
        "caller-neutral" in violation.message for violation in violations
    )


def test_stage7_bindings_are_thin_and_domain_scoped():
    bindings = ROOT / "pyscarcopula/_cpp/src/bindings"
    assert not (bindings / "common.hpp").exists()
    for name in ("module.hpp", "array.hpp", "array.cpp"):
        assert (bindings / name).is_file()

    module = (bindings / "module.hpp").read_text(encoding="utf-8")
    assert 'scar/' not in module
    array = (bindings / "array.hpp").read_text(encoding="utf-8")
    for marker in (
            "CopulaSpec", "GridValues", "Gas", "OuBackend", "RVine",
            "Student", "Result"):
        assert marker not in array

    owners = {
        "copula.cpp": ("StaticObjectiveResult",),
        "multivariate.cpp": (
            "MultivariateRowsResult", "ConditionalSampleResult"),
        "gas.cpp": ("GasLogLikResult", "GasFilterResult"),
        "scar_ou.cpp": (
            "LogLikResult", "GradLogLikResult",
            "SmoothedStateDistribution", "OuGridFilterResult",
            "TrajectoryLogPdfResult"),
        "factor.cpp": (
            "FactorStudentRowsResult", "FactorStudentJointResult",
            "FactorStudentGridResult"),
        "rvine.cpp": (
            "SampleResult", "ConditionalSampleResult", "DensityResult",
            "RosenblattResult", "MCMCResult"),
    }
    for owner, results in owners.items():
        source = (bindings / owner).read_text(encoding="utf-8")
        assert '#include "module.hpp"' in source
        for result in results:
            assert result in source

    for name in (
            "common.cpp", "parallel.cpp", "gas.cpp", "rvine.cpp",
            "scar_ou_types.cpp", "scar_ou.cpp"):
        source = (bindings / name).read_text(encoding="utf-8")
        assert 'copula/multivariate/student/' not in source

    factor = (bindings / "factor.cpp").read_text(encoding="utf-8")
    assert factor.count(
        'output["status"] = static_cast<int>(result.status);') >= 3
    assert "matrix_copy(" not in factor
    assert "vector_copy(" not in factor
    assert re.search(
        r"py::gil_scoped_release\s+\w+\s*;"
        r"(?:(?!^\s*\}).)*?\.(?:request|mutable_data|writeable)\s*\(",
        factor,
        re.MULTILINE | re.DOTALL,
    ) is None

    scar_ou = (bindings / "scar_ou.cpp").read_text(encoding="utf-8")
    for model_call in (
            "build_ou_grid(", "build_grid_transition_operator(",
            "forward_filter_emissions(", "backward_filter_emissions(",
            "smooth_state_emissions("):
        assert model_call not in scar_ou
    copula = (bindings / "copula.cpp").read_text(encoding="utf-8")
    assert '#include "scar/factor.hpp"' not in copula
    assert (
        '#include "scar/copula/multivariate/correlation/factor.hpp"'
        in copula)

    ou_header = (
        ROOT / "pyscarcopula/_cpp/include/scar/ou.hpp"
    ).read_text(encoding="utf-8")
    ou_result = (
        ROOT / "pyscarcopula/_cpp/include/scar/scar_ou/result.hpp"
    ).read_text(encoding="utf-8")
    assert "OuGridFilterResult filter_ou_grid_emissions(" in ou_header
    assert "struct OuGridFilterResult" in ou_result


def test_stage7_checker_rejects_umbrella_and_cross_domain_binding_leaks(
    tmp_path,
):
    root = _minimal_repository(tmp_path)
    bindings = root / "pyscarcopula/_cpp/src/bindings"
    _write(root, "pyscarcopula/_cpp/src/bindings/common.hpp", "#pragma once\n")
    _write(
        root,
        "pyscarcopula/_cpp/src/bindings/common.cpp",
        '#include "module.hpp"\n#include "scar/ou.hpp"\n',
    )
    _write(
        root,
        "pyscarcopula/_cpp/src/bindings/array.hpp",
        "struct CopulaSpec;\n",
    )
    _write(root, "pyscarcopula/_cpp/src/bindings/array.cpp", "")
    _write(
        root,
        "pyscarcopula/_cpp/src/bindings/gas.cpp",
        '#include "module.hpp"\n#include "scar/ou.hpp"\n',
    )

    violations = CHECKER_MODULE.check_thin_bindings(root)
    messages = {violation.message for violation in violations}
    assert "the umbrella bindings/common.hpp must not be restored" in messages
    assert any("shared binding helper imports" in message for message in messages)
    assert any("unrelated domain API" in message for message in messages)
    assert any("array/view conversion depends" in message for message in messages)
    assert bindings.is_dir()


def test_stage7_checker_rejects_policy_gil_and_model_logic_in_bindings(
    tmp_path,
):
    root = _minimal_repository(tmp_path)
    _write(root, "pyscarcopula/_cpp/src/bindings/array.hpp", "")
    _write(root, "pyscarcopula/_cpp/src/bindings/array.cpp", "")
    _write(
        root,
        "pyscarcopula/_cpp/src/bindings/factor.cpp",
        '#include "module.hpp"\n'
        "void bind() {\n"
        "    py::array_t<double> output(2);\n"
        "    {\n"
        "        py::gil_scoped_release release;\n"
        "        output.mutable_data();\n"
        "    }\n"
        "    FactorStudentRowsResult result;\n"
        "    if (result.failure.index >= 0) { throw 1; }\n"
        "    matrix_copy(values, 1, 1);\n"
        "}\n",
    )
    _write(
        root,
        "pyscarcopula/_cpp/src/bindings/scar_ou.cpp",
        '#include "module.hpp"\nvoid bind() { build_ou_grid(); }\n',
    )
    _write(
        root,
        "pyscarcopula/_cpp/src/bindings/copula.cpp",
        '#include "module.hpp"\n#include "scar/factor.hpp"\n',
    )

    messages = {
        violation.message
        for violation in CHECKER_MODULE.check_thin_bindings(root)
    }
    assert any("result-dependent policy" in message for message in messages)
    assert any("GIL is released" in message for message in messages)
    assert any("model orchestration" in message for message in messages)
    assert any("duplicates shared array" in message for message in messages)
    assert any("focused factor correlation" in message for message in messages)


def test_setup_build_path_does_not_mutate_path():
    source = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert 'os.environ["PATH"]' not in source
    assert "shutil.which" not in source


def test_gate4_workflow_covers_required_compilers_and_build_boundaries():
    source = (
        ROOT / ".github/workflows/parallel-release-gates.yml"
    ).read_text(encoding="utf-8")
    for configuration in (
        "linux-gcc-py310",
        "linux-gcc-py314",
        "linux-clang-py312",
        "windows-msvc-py312",
        "windows-mingw64-py312",
        "macos-arm64-clang-py312",
    ):
        assert source.count(f"name: {configuration}") == 1

    assert "msystem: MINGW64" in source
    assert "install: mingw-w64-x86_64-gcc" in source
    assert "python tools/build_cpp_tests.py --force" in source
    assert "python -m pyscarcopula._native.smoke" in source
    assert "Run full non-benchmark suite against wheel" in source
    assert source.index("Verify Python-free C++ build boundary") < source.index(
        "Build strict wheel")


def test_stateless_scar_bindings_release_gil_after_array_validation():
    source = (
        ROOT / "pyscarcopula/_cpp/src/bindings/scar_ou.cpp"
    ).read_text(encoding="utf-8")

    assert "with_observation_view_without_gil" in source
    assert source.count("observation_view_from_array(") == 3
    helper = source.index("with_observation_view_without_gil")
    view = source.index("observation_view_from_array(", helper)
    release = source.index("py::gil_scoped_release release", view)
    assert helper < view < release


def test_rvine_sample_binding_keeps_arrays_alive_and_releases_gil():
    source = (
        ROOT / "pyscarcopula/_cpp/src/bindings/rvine.cpp"
    ).read_text(encoding="utf-8")

    request = source.index('"rvine_sample"')
    release = source.index("py::gil_scoped_release release", request)
    native_call = source.index("scar::rvine::sample(", release)
    result_dict = source.index("py::dict diagnostics", native_call)
    assert request < release < native_call < result_dict
    for name in ("scalar_parameters", "row_parameters", "uniforms"):
        assert name in source[request:release]


def test_rvine_conditional_binding_keeps_arrays_alive_and_releases_gil():
    source = (
        ROOT / "pyscarcopula/_cpp/src/bindings/rvine.cpp"
    ).read_text(encoding="utf-8")

    request = source.index('"rvine_conditional_sample"')
    release = source.index("py::gil_scoped_release release", request)
    native_call = source.index("scar::rvine::conditional_sample(", release)
    result_dict = source.index("py::dict diagnostics", native_call)
    assert request < release < native_call < result_dict
    for name in (
            "scalar_parameters",
            "row_parameters",
            "given_values",
            "uniforms"):
        assert name in source[request:release]


def test_rvine_density_binding_keeps_arrays_alive_and_releases_gil():
    source = (
        ROOT / "pyscarcopula/_cpp/src/bindings/rvine.cpp"
    ).read_text(encoding="utf-8")

    request = source.index('"rvine_log_pdf_rows"')
    release = source.index("py::gil_scoped_release release", request)
    native_call = source.index("scar::rvine::log_pdf_rows(", release)
    result_dict = source.index("py::dict diagnostics", native_call)
    assert request < release < native_call < result_dict
    for name in (
            "scalar_parameters", "row_parameters", "observations"):
        assert name in source[request:release]


def test_rvine_rosenblatt_binding_keeps_arrays_alive_and_releases_gil():
    source = (
        ROOT / "pyscarcopula/_cpp/src/bindings/rvine.cpp"
    ).read_text(encoding="utf-8")

    request = source.index('"rvine_rosenblatt_transform"')
    release = source.index("py::gil_scoped_release release", request)
    native_call = source.index(
        "scar::rvine::rosenblatt_transform(", release)
    result_dict = source.index("py::dict diagnostics", native_call)
    assert request < release < native_call < result_dict
    for name in (
            "scalar_parameters", "row_parameters", "observations"):
        assert name in source[request:release]


def test_dense_student_binding_keeps_arrays_alive_and_releases_gil():
    source = (
        ROOT / "pyscarcopula/_cpp/src/bindings/multivariate.cpp"
    ).read_text(encoding="utf-8")

    request = source.index('"dense_student_rosenblatt_transform"')
    release = source.index("py::gil_scoped_release release", request)
    native_call = source.index("scar::student_rosenblatt_dense(", release)
    result_dict = source.index(
        "return multivariate_rosenblatt_result_to_dict(result);",
        native_call,
    )
    helper = source.index("py::dict multivariate_rosenblatt_result_to_dict(")
    assert helper < request < release < native_call < result_dict
    for name in ("correlation_view", "df_view", "observations"):
        assert name in source[request:release]


def test_rvine_mcmc_binding_keeps_arrays_alive_and_releases_gil():
    source = (
        ROOT / "pyscarcopula/_cpp/src/bindings/rvine.cpp"
    ).read_text(encoding="utf-8")

    request = source.index("const auto mcmc_binding")
    release = source.index("py::gil_scoped_release release", request)
    native_call = source.index("scar::rvine::mcmc_chunk(", release)
    result_dict = source.index("py::dict diagnostics", native_call)
    assert request < release < native_call < result_dict
    for name in (
            "scalar_parameters",
            "row_parameters",
            "given_values",
            "current_state",
            "current_log_pdf",
            "proposal_uniforms",
            "acceptance_uniforms"):
        assert name in source[request:release]


@pytest.mark.parametrize(
    ("relative", "content", "expected_rule"),
    [
        (
            "pyscarcopula/_cpp/src/gas/bad.cpp",
            '#include "scar/ou.hpp"\n',
            "gas-independent-of-ou",
        ),
        (
            "pyscarcopula/_cpp/src/math/bad.cpp",
            '#include "scar/detail/safety.hpp"\n',
            "foundation-independent-of-models",
        ),
        (
            "pyscarcopula/_cpp/src/copula/bad.cpp",
            '#include "scar/gas.hpp"\n',
            "copula-independent-of-gas",
        ),
        (
            "pyscarcopula/_cpp/src/copula/families/bad.cpp",
            '#include "scar/detail/scar_ou/grid.hpp"\n',
            "families-independent-of-ou",
        ),
        (
            "pyscarcopula/_cpp/src/vine/bad.cpp",
            '#include "scar/gas_rvine.hpp"\n',
            "rvine-independent-of-dynamic-models",
        ),
        (
            "pyscarcopula/_cpp/src/gas/evaluator.cpp",
            '#include "scar/copula/multivariate/student/density.hpp"\n',
            "prepared-application-modules",
        ),
        (
            "pyscarcopula/_cpp/src/scar_ou/likelihood.cpp",
            '#include "scar/detail/copula/dispatch.hpp"\n',
            "prepared-application-modules",
        ),
        (
            "pyscarcopula/_cpp/src/copula/bad.cpp",
            '#include "scar/rvine.hpp"\n',
            "prepared-application-modules",
        ),
        (
            "pyscarcopula/_cpp/src/scar_ou/gradient.cpp",
            "&prepared->compatibility_spec() == &copula;\n",
            "prepared-application-modules",
        ),
        (
            "pyscarcopula/_cpp/src/scar_ou/monte_carlo.cpp",
            "spec.student_ppf_nodes();\n",
            "prepared-application-modules",
        ),
        (
            "pyscarcopula/_cpp/include/scar/gas_rvine.hpp",
            '#include "scar/copula.hpp"\n',
            "prepared-application-modules",
        ),
        (
            "pyscarcopula/_cpp/src/gas/rvine_sampler.cpp",
            "evaluator.update_one();\n",
            "prepared-application-modules",
        ),
        (
            "pyscarcopula/_cpp/include/scar/detail/internal.hpp",
            "#pragma once\n",
            "removed-internal-header",
        ),
    ],
)
def test_forbidden_dependencies_produce_clear_rule(
    tmp_path,
    relative,
    content,
    expected_rule,
):
    root = _minimal_repository(tmp_path)
    _write(root, relative, content)
    if relative.endswith(".cpp"):
        source = Path(relative).relative_to(
            "pyscarcopula/_cpp/src").as_posix()
        if source.startswith("bindings/"):
            _set_manifest(root, (), ("bindings/module.cpp", source))
        else:
            _set_manifest(root, (source,))
    assert expected_rule in _rules(root)


def test_source_manifest_detects_unlisted_cpp(tmp_path):
    root = _minimal_repository(tmp_path)
    _write(root, "pyscarcopula/_cpp/src/gas/unlisted.cpp")
    assert "source-manifest" in _rules(root)


def test_module_entrypoint_rejects_binding_implementation(tmp_path):
    root = _minimal_repository(tmp_path)
    module = root / "pyscarcopula/_cpp/src/bindings/module.cpp"
    module.write_text(
        module.read_text(encoding="utf-8")
        + 'module.def("unexpected", [] { return 1; });\n',
        encoding="utf-8",
    )
    assert "minimal-module-entrypoint" in _rules(root)


def test_public_header_cycle_is_rejected(tmp_path):
    root = _minimal_repository(tmp_path)
    _write(
        root,
        "pyscarcopula/_cpp/include/scar/copula.hpp",
        '#include "scar/gas.hpp"\n',
    )
    _write(
        root,
        "pyscarcopula/_cpp/include/scar/gas.hpp",
        '#include "scar/copula.hpp"\n',
    )
    assert "public-header-cycle" in _rules(root)


@pytest.mark.parametrize(
    "content",
    [
        '#include <pybind11/pybind11.h>\n',
        '#include <Python.h>\n',
        '#include <numpy/arrayobject.h>\n',
        "PyObject* callback = nullptr;\n",
        "auto value = py::none();\n",
    ],
)
def test_python_dependencies_are_rejected_from_compute_boundary(
    tmp_path, content,
):
    root = _minimal_repository(tmp_path)
    _write(root, "pyscarcopula/_cpp/src/copula/bad.cpp", content)
    _set_manifest(root, ("copula/bad.cpp",))
    assert "python-free-compute-boundary" in _rules(root)


def test_binding_source_cannot_be_declared_as_compute(tmp_path):
    root = _minimal_repository(tmp_path)
    _set_manifest(root, ("bindings/module.cpp",), ())
    assert "source-manifest" in _rules(root)


def test_setup_must_consume_both_canonical_manifests(tmp_path):
    root = _minimal_repository(tmp_path)
    (root / "setup.py").write_text(
        "SCAR_COMPUTE_SOURCES = manifest.SCAR_COMPUTE_SOURCES\n",
        encoding="utf-8",
    )
    assert "source-manifest" in _rules(root)
