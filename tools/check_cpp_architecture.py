"""Check architectural boundaries of the native C++ source tree."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import re
import sys
from typing import Iterable


_SCAR_INCLUDE = re.compile(
    r'^\s*#\s*include\s+"scar/([^"]+)"', re.MULTILINE)
_MODULE_LINE = re.compile(
    r"pyscarcopula::bindings::bind_[A-Za-z0-9_]+\(module\);")
_FORBIDDEN_COMPUTE_DEPENDENCIES = (
    (
        "pybind11",
        re.compile(
            r"#\s*include\s*[<\"]pybind11/|\bpybind11::|\bpy::",
            re.MULTILINE,
        ),
    ),
    (
        "Python C API",
        re.compile(
            r"#\s*include\s*[<\"]Python\.h[>\"]|\bPyObject\b|\bPy_[A-Z]",
            re.MULTILINE,
        ),
    ),
    (
        "NumPy C API",
        re.compile(
            r"#\s*include\s*[<\"]numpy/|\bPyArray_|\bNPY_[A-Z0-9_]",
            re.MULTILINE,
        ),
    ),
)
_CALLER_SPECIFIC_CONTRACT_TERMS = re.compile(
    r"\b(?:Python|pybind11|NumPy|PyObject)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Violation:
    rule: str
    path: Path
    message: str
    line: int | None = None

    def format(self, root: Path) -> str:
        try:
            relative = self.path.relative_to(root)
        except ValueError:
            relative = self.path
        location = f"{relative}:{self.line}" if self.line else str(relative)
        return f"[{self.rule}] {location}: {self.message}"


def _source_files(directory: Path) -> Iterable[Path]:
    if not directory.is_dir():
        return ()
    return sorted(
        path for path in directory.rglob("*")
        if path.suffix in {".cpp", ".hpp"}
    )


def _include_lines(path: Path) -> list[tuple[str, int]]:
    text = path.read_text(encoding="utf-8")
    return [
        (match.group(1), text.count("\n", 0, match.start()) + 1)
        for match in _SCAR_INCLUDE.finditer(text)
    ]


def _forbid_includes(
    root: Path,
    files: Iterable[Path],
    rule: str,
    forbidden,
    description: str,
) -> list[Violation]:
    violations = []
    for path in files:
        for include, line in _include_lines(path):
            if forbidden(include):
                violations.append(Violation(
                    rule,
                    path,
                    f'{description}; found #include "scar/{include}"',
                    line,
                ))
    return violations


def check_include_boundaries(root: Path) -> list[Violation]:
    cpp_root = root / "pyscarcopula" / "_cpp"
    src = cpp_root / "src"
    include = cpp_root / "include" / "scar"
    violations = []
    foundation_files = [
        *list(_source_files(include / "core")),
        *list(_source_files(include / "math")),
        *list(_source_files(src / "math")),
        include / "copula" / "rotation.hpp",
        include / "copula" / "transforms.hpp",
        src / "copula" / "rotation.cpp",
        src / "copula" / "transforms.cpp",
    ]
    violations.extend(_forbid_includes(
        root,
        (path for path in foundation_files if path.is_file()),
        "foundation-independent-of-models",
        lambda value: value.startswith("detail/")
        or value in {
            "copula.hpp",
            "copula/model_descriptor.hpp",
            "factor.hpp",
            "gas.hpp",
            "gas_rvine.hpp",
            "ou.hpp",
            "rvine.hpp",
            "rvine_plan.hpp",
        },
        "foundation helpers must not depend on model or workflow headers",
    ))
    gas_files = [
        *list(_source_files(src / "gas")),
        include / "gas.hpp",
        include / "gas_rvine.hpp",
    ]
    violations.extend(_forbid_includes(
        root,
        (path for path in gas_files if path.is_file()),
        "gas-independent-of-ou",
        lambda value: value == "ou.hpp"
        or value.startswith("detail/scar_ou/"),
        "GAS must not depend on SCAR-OU",
    ))
    violations.extend(_forbid_includes(
        root,
        _source_files(src / "copula"),
        "copula-independent-of-gas",
        lambda value: value == "gas.hpp",
        "copula implementations must not depend on GAS",
    ))
    violations.extend(_forbid_includes(
        root,
        (
            *list(_source_files(src / "copula" / "families")),
            *list(_source_files(src / "copula" / "pair")),
        ),
        "families-independent-of-ou",
        lambda value: value == "ou.hpp"
        or value.startswith("detail/scar_ou/"),
        "copula family implementations must not depend on SCAR-OU",
    ))
    rvine_files = [
        *list(_source_files(src / "vine")),
        include / "rvine.hpp",
        include / "rvine_plan.hpp",
    ]
    violations.extend(_forbid_includes(
        root,
        (path for path in rvine_files if path.is_file()),
        "rvine-independent-of-dynamic-models",
        lambda value: value in {"gas.hpp", "gas_rvine.hpp", "ou.hpp"}
        or value.startswith("detail/scar_ou/"),
        "the common R-vine runtime must not depend on GAS or SCAR-OU",
    ))
    return violations


def check_module_entrypoint(root: Path) -> list[Violation]:
    path = (
        root / "pyscarcopula" / "_cpp" / "src" / "bindings" / "module.cpp")
    if not path.is_file():
        return [Violation(
            "minimal-module-entrypoint",
            path,
            "bindings/module.cpp is missing",
        )]
    significant = [
        (line_number, line.strip())
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1)
        if line.strip() and not line.lstrip().startswith("//")
    ]
    module_count = sum(
        line.startswith("PYBIND11_MODULE(") for _, line in significant)
    violations = []
    if module_count != 1:
        violations.append(Violation(
            "minimal-module-entrypoint",
            path,
            f"expected exactly one PYBIND11_MODULE, found {module_count}",
        ))
    for line_number, line in significant:
        allowed = (
            line == '#include "module.hpp"'
            or line.startswith("PYBIND11_MODULE(")
            or line == "}"
            or _MODULE_LINE.fullmatch(line) is not None
        )
        if not allowed:
            violations.append(Violation(
                "minimal-module-entrypoint",
                path,
                "only module.hpp, PYBIND11_MODULE, and bind_* calls are allowed",
                line_number,
            ))
    return violations


def _source_manifest(root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    path = (
        root / "pyscarcopula" / "_cpp" / "build_support" / "sources.py")
    spec = importlib.util.spec_from_file_location(
        "_pyscarcopula_cpp_source_manifest", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load canonical source manifest from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    values = []
    for name in ("SCAR_COMPUTE_SOURCES", "PYTHON_BINDING_SOURCES"):
        value = getattr(module, name, None)
        if not isinstance(value, (list, tuple)) or not all(
                isinstance(item, str) for item in value):
            raise ValueError(f"{name} must be a list or tuple of strings")
        normalized = tuple(Path(item).as_posix() for item in value)
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"{name} contains duplicate paths")
        if any(
                Path(item).is_absolute() or ".." in Path(item).parts
                for item in normalized):
            raise ValueError(f"{name} must contain relative paths below src")
        values.append(normalized)
    return values[0], values[1]


def check_source_manifest(root: Path) -> list[Violation]:
    manifest_path = (
        root / "pyscarcopula" / "_cpp" / "build_support" / "sources.py")
    src = root / "pyscarcopula" / "_cpp" / "src"
    try:
        compute, bindings = _source_manifest(root)
    except (ImportError, OSError, SyntaxError, ValueError) as error:
        return [Violation("source-manifest", manifest_path, str(error))]

    declared_compute = set(compute)
    declared_bindings = set(bindings)
    actual = {
        path.relative_to(src).as_posix()
        for path in src.rglob("*.cpp")
    }
    violations = []
    overlap = declared_compute & declared_bindings
    for path in sorted(overlap):
        violations.append(Violation(
            "source-manifest",
            manifest_path,
            f"source is listed in both canonical manifests: {path}",
        ))

    actual_bindings = {
        path for path in actual if path.startswith("bindings/")
    }
    actual_compute = actual - actual_bindings
    for label, declared, discovered in (
        ("SCAR_COMPUTE_SOURCES", declared_compute, actual_compute),
        ("PYTHON_BINDING_SOURCES", declared_bindings, actual_bindings),
    ):
        for path in sorted(discovered - declared):
            violations.append(Violation(
                "source-manifest",
                manifest_path,
                f"C++ source is not listed in {label}: {path}",
            ))
        for path in sorted(declared - discovered):
            violations.append(Violation(
                "source-manifest",
                manifest_path,
                f"{label} references a missing or mispartitioned file: {path}",
            ))

    setup_path = root / "setup.py"
    try:
        setup_text = setup_path.read_text(encoding="utf-8")
    except OSError as error:
        violations.append(Violation(
            "source-manifest",
            setup_path, str(error),
        ))
        return violations
    for name in ("SCAR_COMPUTE_SOURCES", "PYTHON_BINDING_SOURCES"):
        if name not in setup_text:
            violations.append(Violation(
                "source-manifest",
                setup_path,
                f"setup.py must consume canonical {name}",
            ))
    if "SCAR_CORE_SOURCES" in setup_text:
        violations.append(Violation(
            "source-manifest",
            setup_path,
            "legacy combined SCAR_CORE_SOURCES manifest must not be restored",
        ))
    return violations


def check_python_free_compute_boundary(root: Path) -> list[Violation]:
    cpp_root = root / "pyscarcopula" / "_cpp"
    src = cpp_root / "src"
    try:
        compute, _ = _source_manifest(root)
    except (ImportError, OSError, SyntaxError, ValueError):
        return []  # check_source_manifest reports the canonical root cause.

    files = [src / relative for relative in compute]
    files.extend(_source_files(cpp_root / "include"))
    files.extend(
        path for path in _source_files(src)
        if path.suffix == ".hpp"
        and "bindings" not in path.relative_to(src).parts
    )
    violations = []
    for path in sorted(set(files)):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for dependency, pattern in _FORBIDDEN_COMPUTE_DEPENDENCIES:
            for match in pattern.finditer(text):
                violations.append(Violation(
                    "python-free-compute-boundary",
                    path,
                    f"computational C++ must not depend on {dependency}",
                    text.count("\n", 0, match.start()) + 1,
                ))
    return violations


def check_removed_monolith(root: Path) -> list[Violation]:
    detail = (
        root / "pyscarcopula" / "_cpp" / "include" / "scar" / "detail")
    violations = []
    for name in ("internal.hpp", "copula.hpp"):
        path = detail / name
        if path.exists():
            violations.append(Violation(
                "removed-internal-header",
                path,
                f"the monolithic detail/{name} must not be reintroduced",
            ))
    return violations


def check_pair_verticalization(root: Path) -> list[Violation]:
    cpp = root / "pyscarcopula" / "_cpp"
    include = cpp / "include" / "scar" / "copula"
    source = cpp / "src" / "copula"
    violations = []
    manifest = include / "pair" / "families.def"
    entry_pattern = re.compile(
        r"^SCAR_PAIR_FAMILY\(\s*"
        r"([A-Za-z][A-Za-z0-9_]*)\s*,\s*"
        r"([a-z][a-z0-9_]*)\s*,\s*"
        r"([0-9]+)\s*,\s*"
        r"(Any|Archimedean|GaussianTanh)\s*,\s*"
        r"(Any|R0Only)\s*,\s*"
        r"(Softplus|XTanh|GaussianTanh|Exponential|Logistic)\s*,\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*\)$"
    )
    entries: list[tuple[str, str, int]] = []
    if manifest.is_file():
        for line_number, raw_line in enumerate(
                manifest.read_text(encoding="utf-8").splitlines(), 1):
            line = raw_line.strip()
            if not line or line.startswith("//"):
                continue
            match = entry_pattern.fullmatch(line)
            if match is None:
                violations.append(Violation(
                    "pair-copula-verticalization",
                    manifest,
                    "invalid pair-family registry entry",
                    line_number,
                ))
                continue
            entries.append((match.group(1), match.group(2), int(match.group(3))))
    family_names = tuple(package for _, package, _ in entries)

    if not entries:
        violations.append(Violation(
            "pair-copula-verticalization",
            manifest,
            "pair-family registry must contain at least one entry",
        ))
    elif (
            len({enum_name for enum_name, _, _ in entries}) != len(entries)
            or len(set(family_names)) != len(entries)
            or len({value for _, _, value in entries}) != len(entries)):
        violations.append(Violation(
            "pair-copula-verticalization",
            manifest,
            "pair-family enum names, package names, and values must be unique",
        ))

    required = [
        manifest,
        include / "pair" / "kernel.hpp",
        include / "prepared_pair_kernel.hpp",
        source / "pair" / "runtime_registry.cpp",
    ]
    required.extend(include / "pair" / f"{name}.hpp" for name in family_names)
    required.extend(source / "pair" / f"{name}.cpp" for name in family_names)
    for path in required:
        if not path.is_file():
            violations.append(Violation(
                "pair-copula-verticalization",
                path,
                "required pair-copula package file is missing",
            ))

    registry = source / "pair" / "runtime_registry.cpp"
    if registry.is_file():
        text = registry.read_text(encoding="utf-8")
        if text.count("switch (family)") != 1:
            violations.append(Violation(
                "pair-copula-verticalization",
                registry,
                "pair families must have exactly one runtime registration switch",
            ))
        if text.count('#include "scar/copula/pair/families.def"') != 2:
            violations.append(Violation(
                "pair-copula-verticalization",
                registry,
                "runtime declarations and registration must use families.def",
            ))

    kernel_contract = include / "pair" / "kernel.hpp"
    if kernel_contract.is_file():
        text = kernel_contract.read_text(encoding="utf-8")
        for forbidden in ("Rotation", "Transform", "supports"):
            if re.search(rf"\b{forbidden}\b", text):
                violations.append(Violation(
                    "pair-copula-verticalization",
                    kernel_contract,
                    f"{forbidden} must remain outside the family contract",
                ))

    dispatch = source / "dispatch.cpp"
    if dispatch.is_file():
        text = dispatch.read_text(encoding="utf-8")
        for family in ("clayton", "gumbel", "frank", "joe"):
            if re.search(rf"\b{family}_[A-Za-z0-9_]+", text):
                violations.append(Violation(
                    "pair-copula-verticalization",
                    dispatch,
                    f"{family} implementation leaked into generic dispatch",
                ))

    for generic_source in (source / "core.cpp", dispatch):
        if (
                generic_source.is_file()
                and "is_pair_copula_family" in generic_source.read_text(
                    encoding="utf-8")):
            violations.append(Violation(
                "pair-copula-verticalization",
                generic_source,
                "construct PreparedPairKernel directly to avoid a second lookup",
            ))

    pair_source = source / "pair"
    for name in family_names:
        path = pair_source / f"{name}.cpp"
        if not path.is_file():
            continue
        includes = {value for value, _ in _include_lines(path)}
        expected = f"copula/pair/{name}.hpp"
        other_families = {
            f"copula/pair/{other}.hpp"
            for other in family_names
            if other != name
        }
        if expected not in includes or includes & other_families:
            violations.append(Violation(
                "pair-copula-verticalization",
                path,
                "a pair implementation must include its own package header only",
            ))
        text = path.read_text(encoding="utf-8")
        for forbidden in (
                "scar/copula/rotation.hpp",
                "scar::Rotation",
                "scar::Transform",
                "_h_rotated",
                "_h_inverse_rotated"):
            if forbidden in text:
                violations.append(Violation(
                    "pair-copula-verticalization",
                    path,
                    f"common rotation/transform concern leaked into family: {forbidden}",
                ))

    binding = cpp / "src" / "bindings" / "copula.cpp"
    if binding.is_file():
        text = binding.read_text(encoding="utf-8")
        if text.count('#include "scar/copula/pair/families.def"') != 1:
            violations.append(Violation(
                "pair-copula-verticalization",
                binding,
                "generic CopulaFamily binding must consume families.def",
            ))
        for enum_name, _, _ in entries:
            if f'.value("{enum_name}",' in text:
                violations.append(Violation(
                    "pair-copula-verticalization",
                    binding,
                    f"pair family {enum_name} is hard-coded in generic binding",
                ))

    adapter = root / "pyscarcopula" / "numerical" / "_cpp_copula.py"
    if adapter.is_file():
        text = adapter.read_text(encoding="utf-8")
        for enum_name, _, _ in entries:
            if f"CopulaFamily.{enum_name}" in text:
                violations.append(Violation(
                    "pair-copula-verticalization",
                    adapter,
                    f"pair family {enum_name} is hard-coded in generic adapter",
                ))

    rvine_adapter = root / "pyscarcopula" / "numerical" / "_cpp_rvine.py"
    if rvine_adapter.is_file():
        text = rvine_adapter.read_text(encoding="utf-8")
        if (
                "_builtin_copula_types" in text
                or "from pyscarcopula.copula." in text):
            violations.append(Violation(
                "pair-copula-verticalization",
                rvine_adapter,
                "generic R-vine adapter must discover pair families by marker",
            ))
    return violations


def check_multivariate_verticalization(root: Path) -> list[Violation]:
    cpp = root / "pyscarcopula" / "_cpp"
    include = cpp / "include" / "scar"
    source = cpp / "src" / "copula"
    multivariate_include = include / "copula" / "multivariate"
    multivariate_source = source / "multivariate"
    violations = []

    removed = (
        source / "multivariate.cpp",
        source / "families" / "student.cpp",
        source / "student_rosenblatt.cpp",
        cpp / "src" / "factor" / "operator.cpp",
        cpp / "src" / "factor" / "student.cpp",
        cpp / "src" / "factor" / "grid.cpp",
        include / "detail" / "copula" / "student.hpp",
    )
    for path in removed:
        if path.exists():
            violations.append(Violation(
                "multivariate-model-verticalization",
                path,
                "legacy horizontal multivariate source/header must be removed",
            ))

    required = (
        include / "copula" / "spec.hpp",
        include / "copula" / "model_storage.hpp",
        include / "detail" / "copula" / "multivariate" / "batch.hpp",
        multivariate_include / "correlation" / "dense.hpp",
        multivariate_include / "correlation" / "factor.hpp",
        multivariate_include / "gaussian" / "model.hpp",
        multivariate_include / "gaussian" / "density.hpp",
        multivariate_include / "gaussian" / "conditional.hpp",
        multivariate_include / "equicorrelation" / "model.hpp",
        multivariate_include / "equicorrelation" / "kernel.hpp",
        multivariate_include / "student" / "model.hpp",
        multivariate_include / "student" / "distribution.hpp",
        multivariate_include / "student" / "quantile.hpp",
        multivariate_include / "student" / "ppf_cache.hpp",
        multivariate_include / "student" / "density.hpp",
        multivariate_include / "student" / "conditional.hpp",
        multivariate_include / "student" / "rosenblatt.hpp",
        multivariate_source / "correlation" / "dense.cpp",
        multivariate_source / "correlation" / "factor.cpp",
        multivariate_source / "correlation" / "conditional.cpp",
        multivariate_source / "gaussian" / "density.cpp",
        multivariate_source / "gaussian" / "conditional.cpp",
        multivariate_source / "equicorrelation" / "evaluator.cpp",
        multivariate_source / "equicorrelation" / "model.cpp",
        multivariate_source / "equicorrelation" / "kernel.cpp",
        multivariate_source / "student" / "distribution.cpp",
        multivariate_source / "student" / "density.cpp",
        multivariate_source / "student" / "evaluator.cpp",
        multivariate_source / "student" / "conditional.cpp",
        multivariate_source / "student" / "factor_density.cpp",
        multivariate_source / "student" / "factor_grid.cpp",
        multivariate_source / "student" / "ppf_cache.cpp",
        multivariate_source / "student" / "quantile.cpp",
        multivariate_source / "student" / "rosenblatt.cpp",
    )
    for path in required:
        if not path.is_file():
            violations.append(Violation(
                "multivariate-model-verticalization",
                path,
                "required vertical multivariate package file is missing",
            ))

    dispatch = multivariate_source / "dispatch.cpp"
    if dispatch.is_file():
        text = dispatch.read_text(encoding="utf-8")
        forbidden_dispatch_markers = (
            "StudentWorkspace",
            "EquicorrStats",
            "conditional_df",
            "parallel_for_blocks",
            "student_fill_",
            "equicorr_log_pdf_from_stats(",
            "normal_quantile",
            "cholesky",
        )
        if len(text.splitlines()) > 100:
            violations.append(Violation(
                "multivariate-model-verticalization",
                dispatch,
                "multivariate dispatch must remain a thin translation unit",
            ))
        for marker in forbidden_dispatch_markers:
            if marker in text:
                violations.append(Violation(
                    "multivariate-model-verticalization",
                    dispatch,
                    f"model implementation leaked into dispatch: {marker}",
                ))

    conditional_engine = (
        multivariate_source / "correlation" / "conditional.cpp"
    )
    if conditional_engine.is_file():
        text = conditional_engine.read_text(encoding="utf-8")
        for marker in ("Student", "student_", "conditional_df", "chi_square"):
            if marker in text:
                violations.append(Violation(
                    "multivariate-model-verticalization",
                    conditional_engine,
                    f"model-specific conditional policy leaked into correlation algebra: {marker}",
                ))

    student_conditional = multivariate_source / "student" / "conditional.cpp"
    if student_conditional.is_file():
        text = student_conditional.read_text(encoding="utf-8")
        if (
                "student_conditional_scale" not in text
                or "conditional_df" not in text):
            violations.append(Violation(
                "multivariate-model-verticalization",
                student_conditional,
                "Student conditional scaling must be owned by the Student package",
            ))

    spec = include / "copula" / "spec.hpp"
    if spec.is_file():
        text = spec.read_text(encoding="utf-8")
        for field in (
                "l_inv", "log_det", "ppf_n_obs", "ppf_nodes", "ppf_table",
                "gaussian_z1_cache", "gaussian_z2_cache",
                "equicorr_sum_cache", "equicorr_sum_squares_cache"):
            if re.search(rf"\b{field}\s*(?:=|;)", text):
                violations.append(Violation(
                    "multivariate-model-verticalization",
                    spec,
                    f"model-specific field remains in CopulaSpec: {field}",
                ))

    storage = include / "copula" / "model_storage.hpp"
    if storage.is_file():
        text = storage.read_text(encoding="utf-8")
        required_alternatives = (
            "gaussian::DenseModelStorage",
            "gaussian::FactorModelStorage",
            "equicorrelation::ModelStorage",
            "student::DenseModelStorage",
            "student::FactorModelStorage",
        )
        if "std::variant<" not in text or any(
                alternative not in text for alternative in required_alternatives):
            violations.append(Violation(
                "multivariate-model-verticalization",
                storage,
                "typed model storage must enumerate every multivariate model",
            ))

    factor_contract = multivariate_include / "correlation" / "factor.hpp"
    if factor_contract.is_file():
        text = factor_contract.read_text(encoding="utf-8")
        if "FactorCorrelationOperator" not in text or "FactorStudent" in text:
            violations.append(Violation(
                "multivariate-model-verticalization",
                factor_contract,
                "factor correlation contract must be model-independent",
            ))

    for package, forbidden in (
        (multivariate_include / "gaussian", ("/student/", "Student")),
        (multivariate_source / "gaussian", ("/student/", "Student")),
        (include / "copula" / "pair", ("/multivariate/", "scar/factor.hpp", "scar/ou.hpp")),
        (source / "pair", ("/multivariate/", "scar/factor.hpp", "scar/ou.hpp")),
    ):
        if not package.is_dir():
            continue
        for path in _source_files(package):
            text = path.read_text(encoding="utf-8")
            for marker in forbidden:
                if marker in text:
                    violations.append(Violation(
                        "multivariate-model-verticalization",
                        path,
                        f"forbidden cross-model dependency: {marker}",
                    ))

    return violations


def check_prepared_application_modules(root: Path) -> list[Violation]:
    cpp = root / "pyscarcopula" / "_cpp"
    include = cpp / "include" / "scar"
    source = cpp / "src"
    violations = []
    rule = "prepared-application-modules"

    required = (
        include / "copula" / "grid_values.hpp",
        include / "copula" / "prepared_pair_kernel.hpp",
        include / "copula" / "prepared_dynamic_emission.hpp",
        source / "copula" / "prepared_dynamic_emission.cpp",
    )
    for path in required:
        if not path.is_file():
            violations.append(Violation(
                rule,
                path,
                "required prepared copula interface is missing",
            ))

    prepared_emission_header = (
        include / "copula" / "prepared_dynamic_emission.hpp")
    if prepared_emission_header.is_file():
        text = prepared_emission_header.read_text(encoding="utf-8")
        for marker in (
                "class PreparedDynamicEmission",
                "class PreparedDynamicEmissionWorkspace",
                "DynamicEmissionRowResult evaluate_parameter(",
                "bool observation_cache_compatible("):
            if marker not in text:
                violations.append(Violation(
                    rule,
                    prepared_emission_header,
                    f"dynamic emission contract is missing: {marker}",
                ))
        for include_name, line in _include_lines(prepared_emission_header):
            if (
                    include_name.startswith("copula/pair/")
                    or include_name.startswith("copula/multivariate/")
                    or include_name.startswith("detail/copula/")):
                violations.append(Violation(
                    rule,
                    prepared_emission_header,
                    "public dynamic-emission contract exposed a concrete model",
                    line,
                ))

    gas_evaluator = source / "gas" / "evaluator.cpp"
    if gas_evaluator.is_file():
        text = gas_evaluator.read_text(encoding="utf-8")
        if "PreparedDynamicEmission" not in text:
            violations.append(Violation(
                rule,
                gas_evaluator,
                "GAS must evaluate through the scalar dynamic-emission interface",
            ))
        violations.extend(_forbid_includes(
            root,
            (gas_evaluator,),
            rule,
            lambda value: value in {
                "copula.hpp", "detail/copula/dispatch.hpp",
            }
            or value.startswith("copula/pair/")
            or value.startswith("copula/multivariate/"),
            "GAS must not include concrete copula implementations",
        ))

    gas_interface = include / "gas.hpp"
    if gas_interface.is_file():
        text = gas_interface.read_text(encoding="utf-8")
        for marker in (
                "GasStateResult initial_state_prepared(",
                "GasUpdateResult update_one_prepared(",
                "GasUpdateResult update_observation_prepared("):
            if marker not in text:
                violations.append(Violation(
                    rule,
                    gas_interface,
                    f"prepared GAS update contract is missing: {marker}",
                ))

    static_header = include / "copula.hpp"
    static_source = source / "likelihood" / "static.cpp"
    for path, markers in (
        (static_header, ("PreparedDynamicEmission", "emission_")),
        (static_source, ("emission_->evaluate_parameter(",)),
    ):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                violations.append(Violation(
                    rule,
                    path,
                    f"static evaluator is not prepared through: {marker}",
                ))

    scar_ou_interface = include / "ou.hpp"
    if scar_ou_interface.is_file():
        text = scar_ou_interface.read_text(encoding="utf-8")
        for marker in (
                "const PreparedDynamicEmission* prepared_emission_",
                "PreparedDynamicEmission emission_;"):
            if marker not in text:
                violations.append(Violation(
                    rule,
                    scar_ou_interface,
                    f"SCAR-OU prepared dependency is missing: {marker}",
                ))

    generic_scar_ou_files = tuple(
        source / "scar_ou" / name
        for name in (
            "evaluator.cpp",
            "evaluator_internal.hpp",
            "likelihood.cpp",
            "prediction.cpp",
            "state_distribution.cpp",
            "transition.cpp",
            "validation.cpp",
        )
    )
    violations.extend(_forbid_includes(
        root,
        (path for path in generic_scar_ou_files if path.is_file()),
        rule,
        lambda value: value in {
            "detail/copula/dispatch.hpp", "factor.hpp",
        }
        or value.startswith("copula/pair/")
        or value.startswith("copula/multivariate/"),
        "generic SCAR-OU execution must use its prepared emission dependency",
    ))
    forbidden_scar_ou_model_storage = (
        "factor_operator(",
        "dense_inverse_cholesky(",
        "student_ppf_nodes(",
        "student_ppf_table(",
        "student_ppf_observation_count(",
    )
    for path in generic_scar_ou_files:
        if path.is_file() and path.name not in {"evaluator_internal.hpp"}:
            text = path.read_text(encoding="utf-8")
            if "PreparedDynamicEmission" not in text:
                violations.append(Violation(
                    rule,
                    path,
                    "generic SCAR-OU execution lost its prepared emission dependency",
                ))
            for marker in forbidden_scar_ou_model_storage:
                if marker in text:
                    violations.append(Violation(
                        rule,
                        path,
                        "generic SCAR-OU execution accessed concrete model "
                        f"storage: {marker}",
                    ))

    scar_ou_gradient = source / "scar_ou" / "gradient.cpp"
    if scar_ou_gradient.is_file():
        text = scar_ou_gradient.read_text(encoding="utf-8")
        if "&prepared->compatibility_spec() == &copula" in text:
            violations.append(Violation(
                rule,
                scar_ou_gradient,
                "prepared SCAR-OU gradient emission must not be selected by "
                "CopulaSpec address identity",
            ))
        for marker in (
                "if (prepared != nullptr)",
                "const CopulaSpec& copula = emission.compatibility_spec();"):
            if marker not in text:
                violations.append(Violation(
                    rule,
                    scar_ou_gradient,
                    f"prepared SCAR-OU gradient lifecycle is missing: {marker}",
                ))

    vine_header = include / "rvine.hpp"
    vine_files = (
        *tuple(_source_files(source / "vine")),
        vine_header,
        include / "rvine_plan.hpp",
    )
    violations.extend(_forbid_includes(
        root,
        (path for path in vine_files if path.is_file()),
        rule,
        lambda value: value in {
            "copula.hpp", "detail/copula/dispatch.hpp",
        }
        or value.startswith("copula/pair/")
        or value.startswith("copula/multivariate/"),
        "R-vine runtime must use PreparedPairKernel",
    ))
    if vine_header.is_file():
        text = vine_header.read_text(encoding="utf-8")
        if (
                "PreparedPairKernel kernel;" not in text
                or "PreparedPairKernel transposed_kernel;" not in text):
            violations.append(Violation(
                rule,
                vine_header,
                "each prepared vine edge must own forward and transposed pair kernels",
            ))

    scar_ou_files = (
        *tuple(_source_files(source / "scar_ou")),
        include / "ou.hpp",
    )
    violations.extend(_forbid_includes(
        root,
        (path for path in scar_ou_files if path.is_file()),
        rule,
        lambda value: value in {
            "copula.hpp", "gas.hpp", "gas_rvine.hpp",
        },
        "SCAR-OU must depend on prepared copulas and remain independent of GAS",
    ))

    model_files = (
        *tuple(_source_files(include / "copula")),
        *tuple(_source_files(source / "copula")),
    )
    violations.extend(_forbid_includes(
        root,
        model_files,
        rule,
        lambda value: value in {
            "gas.hpp", "gas_rvine.hpp", "ou.hpp", "rvine.hpp",
            "rvine_plan.hpp",
        } or value.startswith("detail/scar_ou/"),
        "copula models must not depend on application modules",
    ))

    composition = source / "gas" / "rvine_sampler.cpp"
    composition_header = include / "gas_rvine.hpp"
    if composition_header.is_file():
        includes = {value for value, _ in _include_lines(composition_header)}
        if "copula.hpp" in includes or "copula/spec.hpp" not in includes:
            violations.append(Violation(
                rule,
                composition_header,
                "GAS-vine composition must depend on CopulaSpec rather than "
                "the static/copula umbrella",
            ))
    if composition.is_file():
        includes = {value for value, _ in _include_lines(composition)}
        for expected in (
                "gas_rvine.hpp",
                "rvine.hpp",
                "copula/prepared_dynamic_emission.hpp"):
            if expected not in includes:
                violations.append(Violation(
                    rule,
                    composition,
                    f"GAS-vine composition must explicitly own {expected}",
                ))
        text = composition.read_text(encoding="utf-8")
        for marker in (
                "gas_emissions",
                "gas_workspaces",
                "update_one_prepared("):
            if marker not in text:
                violations.append(Violation(
                    rule,
                    composition,
                    f"GAS-vine hot path lost prepared state: {marker}",
                ))
        if "evaluator.update_one(" in text:
            violations.append(Violation(
                rule,
                composition,
                "GAS-vine must not prepare a scalar emission inside its row loop",
            ))

    return violations


def check_public_cpp_api(root: Path) -> list[Violation]:
    """Enforce the Stage 6 domain contracts and opaque workspace boundary."""

    cpp = root / "pyscarcopula" / "_cpp"
    include = cpp / "include" / "scar"
    source = cpp / "src"
    rule = "public-cpp-api"
    violations = []

    result_contracts = {
        include / "static" / "result.hpp": ("StaticObjectiveResult",),
        include / "gas" / "result.hpp": (
            "GasLogLikResult", "GasFilterResult", "GasUpdateResult",
            "GasStateResult", "GasPredictResult", "GasPathResult"),
        include / "scar_ou" / "result.hpp": (
            "ScarOuVectorResult", "LogLikResult", "GradLogLikResult",
            "StateDistribution", "SmoothedStateDistribution",
            "OuGridFilterResult",),
        include / "copula" / "result.hpp": (
            "MultivariateRowsResult", "MultivariateGridResult",
            "EquicorrPreparationResult"),
        include / "vine" / "result.hpp": (
            "SampleResult", "ConditionalSampleResult", "DensityResult",
            "RosenblattResult", "MCMCResult"),
        include / "gas_rvine" / "result.hpp": ("GasRvineSampleResult",),
    }
    for path, result_names in result_contracts.items():
        if not path.is_file():
            violations.append(Violation(
                rule,
                path,
                "domain result contract is missing",
            ))
            continue
        text = path.read_text(encoding="utf-8")
        for result_name in result_names:
            result_match = re.search(
                rf"(?ms)^struct\s+{re.escape(result_name)}\s*\{{(.*?)^\}};",
                text,
            )
            if result_match is None:
                violations.append(Violation(
                    rule,
                    path,
                    f"domain result contract is missing: {result_name}",
                ))
                continue
            body = result_match.group(1)
            for required in ("Status status", "FailureContext failure"):
                if required not in body:
                    violations.append(Violation(
                        rule,
                        path,
                        f"{result_name} is missing {required}",
                    ))
            if re.search(
                    r"\bint\s+status\b|\bfailure_(?:index|row|edge|operation|coordinate)\b",
                    body):
                violations.append(Violation(
                    rule,
                    path,
                    f"{result_name} must use Status and FailureContext",
                ))

    for path in _source_files(include):
        if "detail" in path.relative_to(include).parts:
            continue
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(
                r"(?ms)^struct\s+(\w*Result)\s*\{(.*?)^\};", text):
            name, body = match.groups()
            for required in ("Status status", "FailureContext failure"):
                if required not in body:
                    violations.append(Violation(
                        rule,
                        path,
                        f"public result {name} is missing {required}",
                        text.count("\n", 0, match.start()) + 1,
                    ))
            if re.search(
                    r"\bint\s+status\b|\bfailure_(?:index|row|edge|operation|coordinate)\b",
                    body):
                violations.append(Violation(
                    rule,
                    path,
                    f"public result {name} exposes legacy integer errors",
                    text.count("\n", 0, match.start()) + 1,
                ))

    umbrella_contracts = {
        include / "copula.hpp": (
            ("copula/result.hpp", "static/result.hpp"),
            ("struct MultivariateRowsResult", "struct StaticObjectiveResult")),
        include / "gas.hpp": ("gas/result.hpp", "struct GasLogLikResult"),
        include / "ou.hpp": ("scar_ou/result.hpp", "struct LogLikResult"),
        include / "rvine.hpp": ("vine/result.hpp", "struct SampleResult"),
        include / "gas_rvine.hpp": (
            "gas_rvine/result.hpp", "struct GasRvineSampleResult"),
    }
    for path, (required_includes, forbidden_definitions) in (
            umbrella_contracts.items()):
        if not path.is_file():
            continue
        includes = {value for value, _ in _include_lines(path)}
        text = path.read_text(encoding="utf-8")
        if isinstance(required_includes, str):
            required_includes = (required_includes,)
        if isinstance(forbidden_definitions, str):
            forbidden_definitions = (forbidden_definitions,)
        for required_include in required_includes:
            if required_include not in includes:
                violations.append(Violation(
                    rule,
                    path,
                    f"public API must import scar/{required_include}",
                ))
        for forbidden_definition in forbidden_definitions:
            if forbidden_definition in text:
                violations.append(Violation(
                    rule,
                    path,
                    "domain result is defined in an umbrella header",
                ))

    copula_header = include / "copula.hpp"
    if copula_header.is_file():
        concrete_imports = {
            value for value, _ in _include_lines(copula_header)
            if value.startswith("copula/multivariate/")
            or value.startswith("copula/pair/")
            or value == "copula/prepared_dynamic_emission.hpp"
        }
        for concrete_import in sorted(concrete_imports):
            violations.append(Violation(
                rule,
                copula_header,
                "static copula umbrella imports concrete implementation: "
                f"{concrete_import}",
            ))

    status_header = include / "core" / "status.hpp"
    if status_header.is_file():
        text = status_header.read_text(encoding="utf-8")
        if re.search(r"operator\s*[!=]=\s*\(\s*(?:Status\s+\w+\s*,\s*int|int\s+\w+\s*,\s*Status)", text):
            violations.append(Violation(
                rule,
                status_header,
                "Status must not compare implicitly with legacy integer codes",
            ))

    ou_header = include / "ou.hpp"
    if ou_header.is_file():
        text = ou_header.read_text(encoding="utf-8")
        for marker in (
                "ScarOuGridGradientOperators",
                "ScarOuGridGradientWorkspace",
                "ScarOuSpectralGradientWorkspace"):
            if marker in text:
                violations.append(Violation(
                    rule,
                    ou_header,
                    f"public SCAR-OU API exposes private workspace: {marker}",
                ))
        for marker in ("struct Workspace;", "std::unique_ptr<Workspace>"):
            if marker not in text:
                violations.append(Violation(
                    rule,
                    ou_header,
                    f"SCAR-OU evaluator PImpl boundary is missing: {marker}",
                ))
        for class_name in ("ScarOuEvaluator", "PreparedScarOuEvaluator"):
            public = re.search(
                rf"(?ms)class\s+{class_name}\s*\{{\s*public:(.*?)^private:",
                text,
            )
            if public is not None and re.search(
                    r"\bint\s*&\s*status\b|\bint&\s*status\b",
                    public.group(1)):
                violations.append(Violation(
                    rule,
                    ou_header,
                    f"{class_name} exposes a legacy status out-parameter",
                ))
        for marker in (
                "ScarOuVectorResult predictive_mean_local_gh(",
                "ScarOuVectorResult mixture_h_pair_auto(",
                "ScarOuVectorResult predictive_mean(const OuParams& params)",
                "ScarOuVectorResult mixture_h_pair(const OuParams& params)"):
            if marker not in text:
                violations.append(Violation(
                    rule,
                    ou_header,
                    f"typed SCAR-OU vector contract is missing: {marker}",
                ))

    rvine_header = include / "rvine.hpp"
    if rvine_header.is_file():
        text = rvine_header.read_text(encoding="utf-8")
        if re.search(r"\bfailure_(?:row|edge|operation)\s*[,);]", text):
            violations.append(Violation(
                rule,
                rvine_header,
                "public R-vine API exposes internal failure out-parameters",
            ))

    private_workspace = source / "scar_ou" / "gradient_workspace.hpp"
    if not private_workspace.is_file():
        violations.append(Violation(
            rule,
            private_workspace,
            "private SCAR-OU gradient workspace is missing",
        ))
    elif "ScarOuSpectralGradientWorkspace" not in (
            private_workspace.read_text(encoding="utf-8")):
        violations.append(Violation(
            rule,
            private_workspace,
            "private SCAR-OU gradient workspace is incomplete",
        ))

    compute_contract_files = (
        *tuple(_source_files(include)),
        *tuple(
            path for path in _source_files(source)
            if "bindings" not in path.relative_to(source).parts
        ),
    )
    for path in compute_contract_files:
        text = path.read_text(encoding="utf-8")
        match = _CALLER_SPECIFIC_CONTRACT_TERMS.search(text)
        if match is not None:
            violations.append(Violation(
                rule,
                path,
                "computational contracts must be caller-neutral; found "
                f"{match.group(0)!r}",
                text.count("\n", 0, match.start()) + 1,
            ))

    return violations


def check_thin_bindings(root: Path) -> list[Violation]:
    """Enforce the Stage 7 pybind include and conversion boundaries."""

    bindings = (
        root / "pyscarcopula" / "_cpp" / "src" / "bindings")
    rule = "thin-python-bindings"
    violations = []

    legacy_umbrella = bindings / "common.hpp"
    if legacy_umbrella.exists():
        violations.append(Violation(
            rule,
            legacy_umbrella,
            "the umbrella bindings/common.hpp must not be restored",
        ))

    required = (
        bindings / "module.hpp",
        bindings / "array.hpp",
        bindings / "array.cpp",
    )
    for path in required:
        if not path.is_file():
            violations.append(Violation(
                rule,
                path,
                "required focused binding helper is missing",
            ))

    module_header = bindings / "module.hpp"
    if module_header.is_file():
        text = module_header.read_text(encoding="utf-8")
        if _SCAR_INCLUDE.search(text):
            violations.append(Violation(
                rule,
                module_header,
                "module declarations must not import computational APIs",
            ))
        if "pybind11/numpy.h" in text:
            violations.append(Violation(
                rule,
                module_header,
                "module declarations must not import array conversion",
            ))

    array_header = bindings / "array.hpp"
    if array_header.is_file():
        text = array_header.read_text(encoding="utf-8")
        for marker in (
                "CopulaSpec", "GridValues", "Gas", "OuBackend",
                "RVine", "Student", "Result"):
            if marker in text:
                violations.append(Violation(
                    rule,
                    array_header,
                    f"array/view conversion depends on a domain type: {marker}",
                ))
        for signature in (
                "const Float64Array& values",
                "const Float64Array& values,"):
            if signature not in text:
                violations.append(Violation(
                    rule,
                    array_header,
                    "view-producing conversion must retain its array owner "
                    "through a const reference",
                ))
                break

    shared_files = (
        bindings / "common.cpp",
        bindings / "array.cpp",
        bindings / "array.hpp",
        bindings / "module.hpp",
    )
    domain_include = re.compile(
        r"^(?:copula(?:\.hpp|/)|factor\.hpp|gas(?:\.hpp|_rvine\.hpp|/)|"
        r"ou\.hpp|rvine(?:\.hpp|_plan\.hpp)|scar_ou/|vine/|gas_rvine/)"
    )
    for path in shared_files:
        if not path.is_file():
            continue
        for include, line in _include_lines(path):
            if domain_include.match(include):
                violations.append(Violation(
                    rule,
                    path,
                    "shared binding helper imports a domain API: "
                    f"scar/{include}",
                    line,
                ))

    binder_sources = {
        "common.cpp", "parallel.cpp", "copula.cpp", "factor.cpp",
        "multivariate.cpp", "scar_ou_types.cpp", "rvine.cpp", "gas.cpp",
        "scar_ou.cpp",
    }
    for name in sorted(binder_sources):
        path = bindings / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if '#include "module.hpp"' not in text:
            violations.append(Violation(
                rule,
                path,
                "domain binder must import only the focused module contract",
            ))

    forbidden_by_binder = {
        "parallel.cpp": domain_include,
        "gas.cpp": re.compile(
            r"^(?:factor\.hpp|ou\.hpp|rvine\.hpp|scar_ou/|vine/|"
            r"copula/multivariate/student/)"),
        "rvine.cpp": re.compile(
            r"^(?:factor\.hpp|gas(?:\.hpp|_rvine\.hpp|/)|ou\.hpp|"
            r"scar_ou/|copula/multivariate/)"),
        "factor.cpp": re.compile(
            r"^(?:gas(?:\.hpp|_rvine\.hpp|/)|ou\.hpp|rvine\.hpp|"
            r"scar_ou/|vine/)"),
        "copula.cpp": re.compile(
            r"^(?:gas(?:\.hpp|_rvine\.hpp|/)|ou\.hpp|rvine\.hpp|"
            r"scar_ou/|vine/)"),
        "multivariate.cpp": re.compile(
            r"^(?:gas(?:\.hpp|_rvine\.hpp|/)|ou\.hpp|rvine\.hpp|"
            r"scar_ou/|vine/)"),
        "scar_ou_types.cpp": re.compile(
            r"^(?:copula(?:\.hpp|/)|factor\.hpp|gas(?:\.hpp|_rvine\.hpp|/)|"
            r"ou\.hpp|rvine\.hpp|vine/)"),
        "scar_ou.cpp": re.compile(
            r"^(?:factor\.hpp|gas(?:\.hpp|_rvine\.hpp|/)|rvine\.hpp|vine/)"),
    }
    for name, forbidden in forbidden_by_binder.items():
        path = bindings / name
        if not path.is_file():
            continue
        for include, line in _include_lines(path):
            if forbidden.match(include):
                violations.append(Violation(
                    rule,
                    path,
                    "binder imports an unrelated domain API: "
                    f"scar/{include}",
                    line,
                ))

    result_owners = {
        "copula.cpp": ("StaticObjectiveResult",),
        "multivariate.cpp": (
            "MultivariateRowsResult", "MultivariateGridResult",
            "EquicorrPreparationResult", "ConditionalSampleResult"),
        "gas.cpp": (
            "GasLogLikResult", "GasFilterResult", "GasUpdateResult",
            "GasStateResult", "GasPredictResult", "GasPathResult"),
        "scar_ou.cpp": (
            "ScarOuVectorResult", "LogLikResult", "GradLogLikResult",
            "StateDistribution", "SmoothedStateDistribution",
            "OuGridFilterResult",),
        "factor.cpp": (
            "FactorStudentRowsResult", "FactorStudentJointResult",
            "FactorStudentGridResult"),
        "rvine.cpp": (
            "SampleResult", "ConditionalSampleResult", "DensityResult",
            "RosenblattResult", "MCMCResult"),
    }
    all_result_names = {
        result_name
        for names in result_owners.values()
        for result_name in names
    }
    for owner, names in result_owners.items():
        path = bindings / owner
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for result_name in names:
            if result_name not in text:
                violations.append(Violation(
                    rule,
                    path,
                    f"domain result serialization is missing: {result_name}",
                ))
    for path in _source_files(bindings):
        if path.suffix == ".hpp":
            text = path.read_text(encoding="utf-8")
            if "_result_to_dict" in text or any(
                    name in text for name in all_result_names):
                violations.append(Violation(
                    rule,
                    path,
                    "model-specific result conversion must stay beside its binder",
                ))

    student_allowed = {"copula.cpp", "factor.cpp", "multivariate.cpp"}
    for path in bindings.glob("*.cpp"):
        if path.name in student_allowed:
            continue
        for include, line in _include_lines(path):
            if "/student/" in include:
                violations.append(Violation(
                    rule,
                    path,
                    "Student API leaked into a Student-independent binder",
                    line,
                ))

    factor_binding = bindings / "factor.cpp"
    if factor_binding.is_file():
        text = factor_binding.read_text(encoding="utf-8")
        for marker in ("matrix_copy(", "vector_copy("):
            if marker in text:
                violations.append(Violation(
                    rule,
                    factor_binding,
                    "factor binder duplicates shared array conversion: "
                    f"{marker[:-1]}",
                ))
        for serializer in (
                "factor_student_rows_result_to_dict",
                "factor_student_joint_result_to_dict",
                "factor_student_grid_result_to_dict"):
            if serializer not in text:
                violations.append(Violation(
                    rule,
                    factor_binding,
                    "factor result serialization is missing: " + serializer,
                ))
        status_serialization = (
            'output["status"] = static_cast<int>(result.status);')
        if text.count(status_serialization) < 3:
            violations.append(Violation(
                rule,
                factor_binding,
                "every factor Student result must serialize Status",
            ))

    status_policy = re.compile(
        r"\bif\s*\([^;{}]*\bresult\.(?:status|failure|is_ok\s*\()",
        re.DOTALL,
    )
    gil_python_access = re.compile(
        r"py::gil_scoped_release\s+\w+\s*;"
        r"(?:(?!^\s*\}).)*?\b\w+\."
        r"(?:request|mutable_data|writeable)\s*\(",
        re.MULTILINE | re.DOTALL,
    )
    orchestration_calls = (
        "build_ou_grid(",
        "build_grid_transition_operator(",
        "forward_filter_emissions(",
        "backward_filter_emissions(",
        "smooth_state_emissions(",
    )
    for path in bindings.glob("*.cpp"):
        text = path.read_text(encoding="utf-8")
        match = status_policy.search(text)
        if match is not None:
            violations.append(Violation(
                rule,
                path,
                "binding must serialize Status without result-dependent policy",
                text.count("\n", 0, match.start()) + 1,
            ))
        match = gil_python_access.search(text)
        if match is not None:
            violations.append(Violation(
                rule,
                path,
                "Python/NumPy array API used while the GIL is released",
                text.count("\n", 0, match.start()) + 1,
            ))
        for marker in orchestration_calls:
            index = text.find(marker)
            if index >= 0:
                violations.append(Violation(
                    rule,
                    path,
                    "model orchestration belongs in the computational API: "
                    f"{marker[:-1]}",
                    text.count("\n", 0, index) + 1,
                ))

    copula_binding = bindings / "copula.cpp"
    if copula_binding.is_file():
        for include, line in _include_lines(copula_binding):
            if include == "factor.hpp":
                violations.append(Violation(
                    rule,
                    copula_binding,
                    "generic copula binder must import the focused factor "
                    "correlation contract instead of scar/factor.hpp",
                    line,
                ))

    return violations


def _find_cycle(graph: dict[str, set[str]]) -> list[str] | None:
    visited: set[str] = set()
    active: set[str] = set()
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        visited.add(node)
        active.add(node)
        stack.append(node)
        for neighbour in sorted(graph[node]):
            if neighbour not in visited:
                cycle = visit(neighbour)
                if cycle:
                    return cycle
            elif neighbour in active:
                start = stack.index(neighbour)
                return [*stack[start:], neighbour]
        stack.pop()
        active.remove(node)
        return None

    for node in sorted(graph):
        if node not in visited:
            cycle = visit(node)
            if cycle:
                return cycle
    return None


def check_public_header_cycles(root: Path) -> list[Violation]:
    include_root = root / "pyscarcopula" / "_cpp" / "include"
    scar_root = include_root / "scar"
    headers = sorted(scar_root.rglob("*.hpp"))
    graph = {
        path.relative_to(include_root).as_posix(): set()
        for path in headers
    }
    for path in headers:
        name = path.relative_to(include_root).as_posix()
        for include, _ in _include_lines(path):
            target = f"scar/{include}"
            if target in graph:
                graph[name].add(target)
    cycle = _find_cycle(graph)
    if not cycle:
        return []
    return [Violation(
        "public-header-cycle",
        include_root / cycle[0],
        f"cyclic public-header dependency: {' -> '.join(cycle)}",
    )]


def check_repository(root: Path) -> list[Violation]:
    root = root.resolve()
    checks = (
        check_include_boundaries,
        check_module_entrypoint,
        check_source_manifest,
        check_python_free_compute_boundary,
        check_removed_monolith,
        check_pair_verticalization,
        check_multivariate_verticalization,
        check_prepared_application_modules,
        check_public_cpp_api,
        check_thin_bindings,
        check_public_header_cycles,
    )
    return [
        violation
        for check in checks
        for violation in check(root)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (defaults to the parent of tools/)",
    )
    args = parser.parse_args(argv)
    violations = check_repository(args.root)
    if violations:
        print("C++ architecture check failed:", file=sys.stderr)
        for violation in violations:
            print(f"  {violation.format(args.root)}", file=sys.stderr)
        return 1
    print("C++ architecture check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
