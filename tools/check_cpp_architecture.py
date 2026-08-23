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
            line == '#include "common.hpp"'
            or line.startswith("PYBIND11_MODULE(")
            or line == "}"
            or _MODULE_LINE.fullmatch(line) is not None
        )
        if not allowed:
            violations.append(Violation(
                "minimal-module-entrypoint",
                path,
                "only common.hpp, PYBIND11_MODULE, and bind_* calls are allowed",
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

    binding = cpp / "src" / "bindings" / "common.cpp"
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
