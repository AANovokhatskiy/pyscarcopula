"""Contracts for the native C++ architecture checker."""

from __future__ import annotations

import importlib.util
from pathlib import Path
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
    assert "python -m pyscarcopula._native_smoke" in source
    assert "Run full non-benchmark suite against wheel" in source
    assert source.index("Verify Python-free C++ build boundary") < source.index(
        "Build strict wheel")


def test_stateless_scar_bindings_release_gil_after_array_validation():
    source = (
        ROOT / "pyscarcopula/_cpp/src/bindings/scar_ou.cpp"
    ).read_text(encoding="utf-8")

    assert "with_observation_view_without_gil" in source
    assert source.count("observation_view_from_array(copula, u)") == 2


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
    result_dict = source.index("py::dict diagnostics", native_call)
    assert request < release < native_call < result_dict
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
