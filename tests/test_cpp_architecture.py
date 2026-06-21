"""Contracts for the native C++ architecture checker."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.check_cpp_architecture import check_repository


ROOT = Path(__file__).resolve().parents[1]


def _write(root: Path, relative: str, text: str = "") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _minimal_repository(root: Path) -> Path:
    _write(
        root,
        "setup.py",
        'SCAR_CORE_SOURCES = ["bindings/module.cpp"]\n',
    )
    _write(root, "pyscarcopula/_cpp/include/scar/copula.hpp", "#pragma once\n")
    _write(root, "pyscarcopula/_cpp/include/scar/gas.hpp", "#pragma once\n")
    _write(root, "pyscarcopula/_cpp/include/scar/ou.hpp", "#pragma once\n")
    _write(
        root,
        "pyscarcopula/_cpp/src/bindings/module.cpp",
        '#include "common.hpp"\n\n'
        "PYBIND11_MODULE(_scar_cpp, module) {\n"
        "    pyscarcopula::bindings::bind_common(module);\n"
        "}\n",
    )
    return root


def _rules(root: Path) -> set[str]:
    return {violation.rule for violation in check_repository(root)}


def test_current_repository_satisfies_cpp_architecture_contract():
    assert check_repository(ROOT) == []


@pytest.mark.parametrize(
    ("relative", "content", "expected_rule"),
    [
        (
            "pyscarcopula/_cpp/src/gas/bad.cpp",
            '#include "scar/ou.hpp"\n',
            "gas-independent-of-ou",
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
        setup = root / "setup.py"
        setup.write_text(
            "SCAR_CORE_SOURCES = "
            f"{['bindings/module.cpp', source]!r}\n",
            encoding="utf-8",
        )
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


def test_module_entrypoint_allows_compiler_diagnostic_guard(tmp_path):
    root = _minimal_repository(tmp_path)
    module = root / "pyscarcopula/_cpp/src/bindings/module.cpp"
    text = module.read_text(encoding="utf-8")
    text = text.replace(
        "PYBIND11_MODULE",
        "#if defined(__GNUC__) || defined(__clang__)\n"
        "#pragma GCC diagnostic push\n"
        '#pragma GCC diagnostic ignored "-Wpedantic"\n'
        "#endif\n\n"
        "PYBIND11_MODULE",
        1,
    )
    text += (
        "\n#if defined(__GNUC__) || defined(__clang__)\n"
        "#pragma GCC diagnostic pop\n"
        "#endif\n"
    )
    module.write_text(text, encoding="utf-8")
    assert "minimal-module-entrypoint" not in _rules(root)


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
