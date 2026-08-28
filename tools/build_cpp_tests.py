"""Build and run Python-free C++ boundary tests with setuptools compilers."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import subprocess

from setuptools._distutils.ccompiler import new_compiler
from setuptools._distutils.sysconfig import customize_compiler


ROOT = Path(__file__).resolve().parents[1]
CPP_ROOT = ROOT / "pyscarcopula" / "_cpp"
CPP_SOURCE_ROOT = CPP_ROOT / "src"
CPP_INCLUDE_ROOT = CPP_ROOT / "include"
DEFAULT_BUILD_DIR = ROOT / "build" / "cpp-tests"
SMOKE_SOURCE = ROOT / "tests" / "cpp" / "compute_smoke.cpp"
CPP_TEST_SOURCES = tuple(sorted((ROOT / "tests" / "cpp").glob("*.cpp")))
REQUIRED_MODEL_TEST_SOURCES = tuple(
    ROOT / "tests" / "cpp" / name
    for name in (
        "pair_models.cpp",
        "multivariate_models.cpp",
        "application_models.cpp",
        "jacobi_domain.cpp",
        "jacobi_transition.cpp",
        "jacobi_evaluator.cpp",
        "jacobi_sampling.cpp",
    )
)


@contextmanager
def _sanitizer_environment(mode: str | None):
    variables = ("PYSCA_CPP_SANITIZE", "PYSCA_CPP_THREAD_SANITIZE")
    previous = {name: os.environ.get(name) for name in variables}
    try:
        if mode is not None:
            for name in variables:
                os.environ.pop(name, None)
            selected = (
                "PYSCA_CPP_SANITIZE"
                if mode == "address-undefined"
                else "PYSCA_CPP_THREAD_SANITIZE"
            )
            os.environ[selected] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _load_build_support(name: str):
    path = CPP_ROOT / "build_support" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(
        f"_pyscarcopula_cpp_tests_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load C++ build support module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_header_units(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    units = []
    for header in sorted((CPP_INCLUDE_ROOT / "scar").rglob("*.hpp")):
        include = header.relative_to(CPP_INCLUDE_ROOT).as_posix()
        unit = directory / (include.replace("/", "__") + ".cpp")
        content = f'#include "{include}"\n'
        if not unit.is_file() or unit.read_text(encoding="utf-8") != content:
            unit.write_text(content, encoding="utf-8", newline="\n")
        units.append(unit)
    return units


def _compiler_executable(compiler, output_dir: Path) -> Path:
    filename = compiler.executable_filename("scar_compute_smoke")
    return output_dir / filename


def build_cpp_tests(
    *,
    build_dir: Path,
    compiler_name: str | None,
    build_jobs: int | None = None,
    debug: bool,
    force: bool,
    check_headers: bool,
    run: bool,
) -> Path:
    sources = _load_build_support("sources")
    toolchain = _load_build_support("toolchain")
    build_parallel = _load_build_support("build_parallel")
    compiler_name = compiler_name or toolchain.requested_compiler()
    build_jobs = build_parallel.resolve_build_jobs(build_jobs)

    compiler = new_compiler(compiler=compiler_name, force=force)
    customize_compiler(compiler)
    toolchain.prepare_compiler_environment(compiler)
    compiler_type = compiler.compiler_type
    compile_args = toolchain.standalone_compile_args(compiler_type)
    link_args = toolchain.standalone_link_args(compiler_type)

    build_dir = build_dir.resolve()
    object_dir = build_dir / compiler_type / "objects"
    executable_dir = build_dir / compiler_type / "bin"
    header_dir = build_dir / compiler_type / "header-units"
    object_dir.mkdir(parents=True, exist_ok=True)
    executable_dir.mkdir(parents=True, exist_ok=True)

    compute_sources = [
        CPP_SOURCE_ROOT / relative
        for relative in sources.SCAR_COMPUTE_SOURCES
    ]
    missing = [path for path in [
        *compute_sources,
        *CPP_TEST_SOURCES,
        *REQUIRED_MODEL_TEST_SOURCES,
    ]
               if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing C++ boundary sources: "
            + ", ".join(str(path) for path in missing)
        )

    print(
        f"Building {len(compute_sources)} computational sources with "
        f"setuptools compiler '{compiler_type}' (C++17, "
        f"build jobs: {build_jobs})"
    )
    with build_parallel.parallel_compilation(compiler, build_jobs):
        compute_objects = compiler.compile(
            [str(path) for path in compute_sources],
            output_dir=str(object_dir),
            include_dirs=[str(CPP_INCLUDE_ROOT)],
            debug=debug,
            extra_postargs=compile_args,
        )
        smoke_objects = compiler.compile(
            [str(path) for path in CPP_TEST_SOURCES],
            output_dir=str(object_dir),
            include_dirs=[str(CPP_INCLUDE_ROOT)],
            debug=debug,
            extra_postargs=compile_args,
        )

        if check_headers:
            header_units = _write_header_units(header_dir)
            print(f"Compiling {len(header_units)} self-contained header units")
            # Generated names are already flattened and unique.  Keep their
            # objects beside them instead of expanding each absolute source
            # path below output_dir, which can exceed Windows path limits.
            compiler.compile(
                [str(path) for path in header_units],
                output_dir=None,
                include_dirs=[str(CPP_INCLUDE_ROOT)],
                debug=debug,
                extra_postargs=compile_args,
            )

    compiler.link_executable(
        [*compute_objects, *smoke_objects],
        "scar_compute_smoke",
        output_dir=str(executable_dir),
        debug=debug,
        extra_postargs=link_args,
        target_lang="c++",
    )
    executable = _compiler_executable(compiler, executable_dir)
    if not executable.is_file():
        raise FileNotFoundError(
            f"setuptools did not produce the expected executable: {executable}"
        )

    if run:
        print(f"Running Python-free smoke executable: {executable}")
        subprocess.run([str(executable)], cwd=executable_dir, check=True)
    print("Python-free C++ compile/link boundary passed")
    return executable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir", type=Path, default=DEFAULT_BUILD_DIR,
        help="build output root (default: build/cpp-tests)",
    )
    parser.add_argument(
        "--compiler",
        help=(
            "setuptools compiler name, for example mingw32; defaults to "
            "PYSCA_CPP_COMPILER or the platform compiler"
        ),
    )
    parser.add_argument(
        "-j", "--build-jobs", type=int,
        help=(
            "number of parallel C++ compilation jobs; defaults to "
            "PYSCA_CPP_BUILD_JOBS or 1"
        ),
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--sanitize",
        choices=("address-undefined", "thread"),
        help=(
            "instrument the standalone Python-free executable with "
            "ASan/UBSan or TSan; does not build the Python extension"
        ),
    )
    parser.add_argument(
        "--skip-header-checks", action="store_true",
        help="do not compile each scar/*.hpp header in isolation",
    )
    parser.add_argument(
        "--skip-run", action="store_true",
        help="compile and link the executable without running it",
    )
    args = parser.parse_args(argv)
    with _sanitizer_environment(args.sanitize):
        build_cpp_tests(
            build_dir=args.build_dir,
            compiler_name=args.compiler,
            build_jobs=args.build_jobs,
            debug=args.debug,
            force=args.force,
            check_headers=not args.skip_header_checks,
            run=not args.skip_run,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
