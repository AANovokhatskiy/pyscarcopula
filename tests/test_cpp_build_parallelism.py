"""Contracts for opt-in source-level C++ build parallelism."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import threading

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILD_SUPPORT_PATH = (
    ROOT / "pyscarcopula" / "_cpp" / "build_support" / "build_parallel.py"
)
TOOLCHAIN_PATH = (
    ROOT / "pyscarcopula" / "_cpp" / "build_support" / "toolchain.py"
)
WHEEL_LAYOUT_PATH = (
    ROOT / "pyscarcopula" / "_cpp" / "build_support" / "wheel_layout.py"
)
BUILD_TOOL_PATH = ROOT / "tools" / "build_cpp_tests.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load test module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD_PARALLEL = _load_module(
    "_pyscarcopula_test_build_parallel", BUILD_SUPPORT_PATH)
TOOLCHAIN = _load_module(
    "_pyscarcopula_test_toolchain", TOOLCHAIN_PATH)
WHEEL_LAYOUT = _load_module(
    "_pyscarcopula_test_wheel_layout", WHEEL_LAYOUT_PATH)
BUILD_CPP_TESTS = _load_module(
    "_pyscarcopula_test_build_cpp_tests", BUILD_TOOL_PATH)


def test_build_jobs_default_is_strictly_sequential(monkeypatch):
    monkeypatch.delenv(BUILD_PARALLEL.BUILD_JOBS_ENV, raising=False)
    assert BUILD_PARALLEL.DEFAULT_BUILD_JOBS == 1
    assert BUILD_PARALLEL.resolve_build_jobs() == 1

    monkeypatch.setenv(BUILD_PARALLEL.BUILD_JOBS_ENV, "  ")
    assert BUILD_PARALLEL.resolve_build_jobs() == 1


def test_build_jobs_cli_value_precedes_environment(monkeypatch):
    monkeypatch.setenv(BUILD_PARALLEL.BUILD_JOBS_ENV, "3")
    assert BUILD_PARALLEL.resolve_build_jobs() == 3
    assert BUILD_PARALLEL.resolve_build_jobs(2) == 2
    assert BUILD_PARALLEL.resolve_build_jobs(" 4 ") == 4


@pytest.mark.parametrize(
    "value",
    [0, -1, True, "0", "-2", "invalid", "1.5", 2.5],
)
def test_build_jobs_reject_invalid_explicit_values(value):
    with pytest.raises(ValueError, match="positive integer"):
        BUILD_PARALLEL.resolve_build_jobs(value)


@pytest.mark.parametrize("value", ["0", "-1", "invalid", "1.5"])
def test_build_jobs_reject_invalid_environment_values(monkeypatch, value):
    monkeypatch.setenv(BUILD_PARALLEL.BUILD_JOBS_ENV, value)
    with pytest.raises(
            ValueError, match=BUILD_PARALLEL.BUILD_JOBS_ENV):
        BUILD_PARALLEL.resolve_build_jobs()


class _CompilerProbe:
    """Minimal compiler implementing the protocol used by ParallelCompile."""

    def __init__(self) -> None:
        self.serial_calls = 0
        self.worker_ids: list[int] = []
        self._worker_lock = threading.Lock()
        self._worker_barrier = threading.Barrier(2)

    def compile(self, sources, **_kwargs):
        self.serial_calls += 1
        return [f"serial:{source}" for source in sources]

    def _setup_compile(
        self, _output_dir, macros, _include_dirs, sources, _depends, extra,
    ):
        objects = [f"{source}.o" for source in sources]
        build = {
            obj: (source, Path(source).suffix)
            for obj, source in zip(objects, sources)
        }
        return macros or [], objects, extra or [], [], build

    @staticmethod
    def _get_cc_args(_pp_opts, _debug, _extra_preargs):
        return []

    def _compile(
        self, _obj, _src, _ext, _cc_args, _extra_postargs, _pp_opts,
    ):
        self._worker_barrier.wait(timeout=5)
        with self._worker_lock:
            self.worker_ids.append(threading.get_ident())


def test_sequential_context_leaves_compiler_method_untouched():
    compiler = _CompilerProbe()
    with BUILD_PARALLEL.parallel_compilation(compiler, 1):
        result = compiler.compile(["a.cpp"])

    assert result == ["serial:a.cpp"]
    assert compiler.serial_calls == 1
    assert compiler.worker_ids == []
    assert compiler.compile.__func__ is _CompilerProbe.compile


def test_parallel_context_uses_pybind_pool_and_restores_compiler_method():
    compiler = _CompilerProbe()
    with BUILD_PARALLEL.parallel_compilation(compiler, 2):
        result = compiler.compile(["a.cpp", "b.cpp"])

    assert result == ["a.cpp.o", "b.cpp.o"]
    assert len(compiler.worker_ids) == 2
    assert len(set(compiler.worker_ids)) == 2
    assert compiler.compile.__func__ is _CompilerProbe.compile


class _MsvcCompilerProbe(_CompilerProbe):
    compiler_type = "msvc"

    def __init__(self) -> None:
        super().__init__()
        self.initialized = False

    def initialize(self) -> None:
        self.initialized = True

    def compile(self, sources, **_kwargs):
        assert self.initialized
        self._worker_barrier.wait(timeout=5)
        with self._worker_lock:
            self.worker_ids.append(threading.get_ident())
        return [f"msvc:{source}" for source in sources]


def test_parallel_context_adapts_and_initializes_msvc_compile_override():
    compiler = _MsvcCompilerProbe()
    with BUILD_PARALLEL.parallel_compilation(compiler, 2):
        result = compiler.compile(["a.cpp", "b.cpp"])

    assert result == ["a.cpp.o", "b.cpp.o"]
    assert compiler.initialized is True
    assert len(compiler.worker_ids) == 2
    assert len(set(compiler.worker_ids)) == 2
    assert compiler.compile.__func__ is _MsvcCompilerProbe.compile


def test_msvc_environment_is_normalized_initialized_and_exported(monkeypatch):
    class Compiler:
        compiler_type = "msvc"
        initialized = False

        def initialize(self):
            assert TOOLCHAIN.os.environ["PATH"] == "inherited-path"
            self._paths = TOOLCHAIN.os.pathsep.join([
                "msvc-bin", "windows-sdk-bin", "inherited-path",
            ])
            self.initialized = True

    compiler = Compiler()
    monkeypatch.setenv("PATH", "inherited-path")

    TOOLCHAIN.prepare_compiler_environment(compiler)

    assert compiler.initialized is True
    assert TOOLCHAIN.os.environ["PATH"] == compiler._paths


def test_non_msvc_environment_is_left_untouched(monkeypatch):
    class Compiler:
        compiler_type = "mingw32"

    monkeypatch.setenv("PATH", "inherited-path")
    TOOLCHAIN.prepare_compiler_environment(Compiler())
    assert TOOLCHAIN.os.environ["PATH"] == "inherited-path"


def test_strict_msvc_replaces_default_warning_level(monkeypatch):
    class Compiler:
        compiler_type = "msvc"
        initialized = True
        compile_options = ["/nologo", "/O2", "/W3", "/MD"]
        compile_options_debug = ["/nologo", "/Od", "/W3", "/MDd"]

    monkeypatch.setenv("PYSCA_CPP_STRICT", "1")
    compiler = Compiler()
    TOOLCHAIN.prepare_compiler_environment(compiler)

    assert compiler.compile_options == ["/nologo", "/O2", "/MD"]
    assert compiler.compile_options_debug == ["/nologo", "/Od", "/MDd"]
    assert TOOLCHAIN.standalone_compile_args("msvc")[-2:] == ["/W4", "/WX"]


def test_build_cpp_tests_cli_forwards_explicit_jobs(monkeypatch, tmp_path):
    captured = {}

    def fake_build_cpp_tests(**kwargs):
        captured.update(kwargs)
        return tmp_path / "scar_compute_smoke"

    monkeypatch.setattr(
        BUILD_CPP_TESTS, "build_cpp_tests", fake_build_cpp_tests)
    assert BUILD_CPP_TESTS.main([
        "--build-dir", str(tmp_path),
        "--build-jobs", "3",
        "--skip-header-checks",
        "--skip-run",
    ]) == 0
    assert captured["build_jobs"] == 3
    assert captured["check_headers"] is False
    assert captured["run"] is False


@pytest.mark.parametrize(
    ("mode", "enabled", "disabled"),
    [
        (
            "address-undefined",
            "PYSCA_CPP_SANITIZE",
            "PYSCA_CPP_THREAD_SANITIZE",
        ),
        (
            "thread",
            "PYSCA_CPP_THREAD_SANITIZE",
            "PYSCA_CPP_SANITIZE",
        ),
    ],
)
def test_cpp_test_sanitizer_is_standalone_and_scoped(
    monkeypatch, tmp_path, mode, enabled, disabled,
):
    observed = {}
    monkeypatch.setenv(enabled, "previous")
    monkeypatch.setenv(disabled, "previous-other")

    def fake_build_cpp_tests(**_kwargs):
        observed[enabled] = BUILD_CPP_TESTS.os.environ.get(enabled)
        observed[disabled] = BUILD_CPP_TESTS.os.environ.get(disabled)
        return tmp_path / "scar_compute_smoke"

    monkeypatch.setattr(
        BUILD_CPP_TESTS, "build_cpp_tests", fake_build_cpp_tests)
    assert BUILD_CPP_TESTS.main([
        "--build-dir", str(tmp_path),
        "--sanitize", mode,
        "--skip-run",
    ]) == 0
    assert observed == {enabled: "1", disabled: None}
    assert BUILD_CPP_TESTS.os.environ[enabled] == "previous"
    assert BUILD_CPP_TESTS.os.environ[disabled] == "previous-other"


def test_python_free_executable_has_required_model_suites():
    names = {path.name for path in BUILD_CPP_TESTS.REQUIRED_MODEL_TEST_SOURCES}
    assert names == {
        "pair_models.cpp",
        "multivariate_gaussian.cpp",
        "multivariate_student.cpp",
        "equicorr_stochastic_student.cpp",
        "gas_models.cpp",
        "ou_models.cpp",
        "vine_models.cpp",
        "multivariate_models.cpp",
        "application_models.cpp",
        "jacobi_domain.cpp",
        "jacobi_transition.cpp",
        "jacobi_evaluator.cpp",
        "jacobi_sampling.cpp",
    }
    assert all(path.is_file() for path in BUILD_CPP_TESTS.REQUIRED_MODEL_TEST_SOURCES)
    entrypoint = (ROOT / "tests/cpp/compute_smoke.cpp").read_text(
        encoding="utf-8")
    for function in (
        "run_pair_model_tests",
        "run_multivariate_gaussian_tests",
        "run_multivariate_student_tests",
        "run_equicorr_stochastic_student_tests",
        "run_gas_model_tests",
        "run_ou_model_tests",
        "run_vine_model_tests",
        "run_multivariate_model_tests",
        "run_application_model_tests",
        "run_jacobi_domain_tests",
        "run_jacobi_transition_tests",
        "run_jacobi_evaluator_tests",
        "run_jacobi_sampling_tests",
    ):
        assert entrypoint.count(f"{function}()") == 2


def test_multivariate_gaussian_cpp_suite_is_distribution_isolated():
    source = (ROOT / "tests/cpp/multivariate_gaussian.cpp").read_text(
        encoding="utf-8")
    assert "student" not in source.lower()
    assert '"scar/copula/multivariate/gaussian/density.hpp"' in source
    assert '"scar/copula/multivariate/correlation/factor.hpp"' in source


@pytest.mark.parametrize("fail_compile", [False, True])
def test_cpp_build_passes_relative_sources_and_restores_working_directory(
        tmp_path, monkeypatch, fail_compile):
    from contextlib import nullcontext
    from types import SimpleNamespace

    calls = []
    class Compiler:
        compiler_type = "unix"
        def compile(self, sources, **options):
            cwd = Path.cwd()
            assert all(not Path(source).is_absolute() for source in sources)
            assert all((cwd / source).is_file() for source in sources)
            assert Path(options["output_dir"]).is_absolute()
            calls.append((cwd, list(sources)))
            if fail_compile:
                raise RuntimeError("compile failed")
            return [str(Path(options["output_dir"]) / (Path(source).stem + ".o"))
                    for source in sources]
        def link_executable(self, objects, name, **options):
            assert objects
            (Path(options["output_dir"]) / name).touch()
        def executable_filename(self, name):
            return name

    compiler = Compiler()
    real_loader = BUILD_CPP_TESTS._load_build_support
    support = {
        "sources": real_loader("sources"),
        "toolchain": SimpleNamespace(
            prepare_compiler_environment=lambda value: None,
            standalone_compile_args=lambda value: [],
            standalone_link_args=lambda value: []),
        "build_parallel": SimpleNamespace(
            resolve_build_jobs=lambda value: value,
            parallel_compilation=lambda value, jobs: nullcontext()),
    }
    monkeypatch.setattr(BUILD_CPP_TESTS, "_load_build_support", support.__getitem__)
    monkeypatch.setattr(BUILD_CPP_TESTS, "new_compiler", lambda **kwargs: compiler)
    monkeypatch.setattr(BUILD_CPP_TESTS, "customize_compiler", lambda value: None)
    original = Path.cwd()
    options = dict(build_dir=tmp_path, compiler_name="unix", build_jobs=1,
                   debug=False, force=False, check_headers=True, run=False)
    if fail_compile:
        with pytest.raises(RuntimeError, match="compile failed"):
            BUILD_CPP_TESTS.build_cpp_tests(**options)
    else:
        assert BUILD_CPP_TESTS.build_cpp_tests(**options).is_file()
        assert len(calls) == 3
        assert calls[0][0] == BUILD_CPP_TESTS.CPP_SOURCE_ROOT
        assert calls[1][0] == ROOT / "tests" / "cpp"
        assert calls[2][0] == tmp_path / "unix" / "header-units"
    assert Path.cwd() == original


def test_extension_and_cpp_boundary_use_shared_build_support():
    setup_source = (ROOT / "setup.py").read_text(encoding="utf-8")
    tool_source = BUILD_TOOL_PATH.read_text(encoding="utf-8")

    for source in (setup_source, tool_source):
        assert '_load_build_support("build_parallel")' in source
        assert ".resolve_build_jobs(" in source
        assert ".parallel_compilation(" in source
        assert ".prepare_compiler_environment(" in source
    assert "self.parallel" in setup_source
    assert '"-j", "--build-jobs"' in tool_source


def test_sdist_contains_cpp_registry_inputs_needed_during_metadata_build():
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-include pyscarcopula/_cpp *.cpp *.hpp *.def" in manifest


def test_build_only_cpp_support_is_excluded_from_wheel_packages():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"pyscarcopula._cpp*"' in pyproject

    setup_source = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert '_load_build_support("wheel_layout")' in setup_source
    assert '"build_py": build_py' in setup_source


def test_wheel_staging_prunes_deleted_modules_and_build_only_support(tmp_path):
    source = tmp_path / "source" / "pyscarcopula"
    build = tmp_path / "build"
    staged = build / "pyscarcopula"
    (source / "_native").mkdir(parents=True)
    (staged / "_native").mkdir(parents=True)
    (staged / "numerical").mkdir(parents=True)
    (staged / "_cpp" / "build_support").mkdir(parents=True)

    (source / "__init__.py").write_text("", encoding="utf-8")
    (source / "_native" / "pair.py").write_text("", encoding="utf-8")
    (staged / "__init__.py").write_text("", encoding="utf-8")
    (staged / "_native" / "pair.py").write_text("", encoding="utf-8")
    expected_extension = staged / "_native" / "_scar_cpp.test.pyd"
    expected_extension.write_bytes(b"native")
    (staged / "numerical" / "tm_grid.py").write_text(
        "stale = True\n", encoding="utf-8")
    (staged / "_scar_cpp.test.pyd").write_bytes(b"removed")
    (staged / "_cpp" / "build_support" / "sources.py").write_text(
        "", encoding="utf-8")

    removed = WHEEL_LAYOUT.prune_stale_package_files(build, source)

    assert (staged / "__init__.py").is_file()
    assert (staged / "_native" / "pair.py").is_file()
    assert expected_extension.is_file()
    assert not (staged / "numerical" / "tm_grid.py").exists()
    assert not (staged / "_scar_cpp.test.pyd").exists()
    assert not (staged / "_cpp").exists()
    assert set(removed) == {
        "pyscarcopula/_cpp/build_support/sources.py",
        "pyscarcopula/_scar_cpp.test.pyd",
        "pyscarcopula/numerical/tm_grid.py",
    }
