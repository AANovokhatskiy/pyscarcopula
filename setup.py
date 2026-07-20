import os
import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension
from pybind11.setup_helpers import build_ext as _build_ext
from setuptools import setup


class build_ext(_build_ext):
    """Support building the extension with MinGW GCC on Windows.

    MSVC remains the default Windows toolchain. To use GCC instead:

        python setup.py build_ext --compiler=mingw32

    or, for pip/PEP 517 builds, set the environment variable:

        PYSCA_CPP_COMPILER=mingw32
    """

    def finalize_options(self):
        compiler = os.environ.get("PYSCA_CPP_COMPILER", "").strip()
        if compiler:
            self.compiler = compiler
        super().finalize_options()

    def build_extension(self, ext):
        if self.compiler.compiler_type == "mingw32":
            # pybind11 assumes MSVC on Windows and injects cl-style flags;
            # translate them for GCC.
            translated = []
            for arg in ext.extra_compile_args:
                if arg.startswith("/std:"):
                    translated.append("-std=" + arg[len("/std:"):])
                elif arg == "/W4":
                    translated.extend(["-Wall", "-Wextra"])
                elif arg == "/WX":
                    translated.append("-Werror")
                elif arg in ("/EHsc", "/bigobj"):
                    continue  # no GCC equivalent needed
                else:
                    translated.append(arg)
            if "-fvisibility=hidden" not in translated:
                translated.append("-fvisibility=hidden")
            # Match the optimization level of the MSVC release build
            # (/O2 /DNDEBUG). Prepended so that user-supplied CFLAGS
            # (appended by pybind11 at the end) can still override it.
            ext.extra_compile_args = ["-O2", "-DNDEBUG", *translated]
            # Link the GCC runtime statically so the built .pyd does not
            # depend on MSYS2 DLLs (libstdc++-6.dll, libgcc_s_seh-1.dll,
            # libwinpthread-1.dll) being importable at runtime.
            ext.extra_link_args = [
                *ext.extra_link_args,
                "-static-libstdc++",
                "-static-libgcc",
                "-Wl,-Bstatic",
                "-lwinpthread",
                "-Wl,-Bdynamic",
            ]
        super().build_extension(ext)


ROOT = Path(__file__).resolve().parent
CPP_ROOT = ROOT / "pyscarcopula" / "_cpp"
CPP_SRC = Path("pyscarcopula") / "_cpp" / "src"

SCAR_CORE_SOURCES = [
    "copula/core.cpp",
    "copula/common.cpp",
    "copula/dispatch.cpp",
    "copula/families/clayton.cpp",
    "copula/families/gumbel.cpp",
    "copula/families/frank.cpp",
    "copula/families/joe.cpp",
    "copula/families/gaussian.cpp",
    "copula/kendall.cpp",
    "copula/families/student.cpp",
    "copula/multivariate.cpp",
    "likelihood/static.cpp",
    "gas/evaluator.cpp",
    "gas/rvine_sampler.cpp",
    "scar_ou/monte_carlo.cpp",
    "scar_ou/validation.cpp",
    "scar_ou/likelihood.cpp",
    "scar_ou/gradient.cpp",
    "scar_ou/prediction.cpp",
    "scar_ou/state_distribution.cpp",
    "scar_ou/evaluator.cpp",
    "scar_ou/prepared.cpp",
    "scar_ou/grid.cpp",
    "scar_ou/quadrature.cpp",
    "scar_ou/transition.cpp",
    "bindings/common.cpp",
    "bindings/copula.cpp",
    "bindings/multivariate.cpp",
    "bindings/scar_ou_types.cpp",
    "bindings/gas.cpp",
    "bindings/scar_ou.cpp",
    "bindings/module.cpp",
]


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


extra_compile_args = []
extra_link_args = []

if _env_flag("PYSCA_CPP_STRICT"):
    if sys.platform == "win32":
        extra_compile_args.extend(["/W4", "/WX"])
    else:
        extra_compile_args.extend(["-Wall", "-Wextra", "-Wpedantic", "-Werror"])

if _env_flag("PYSCA_CPP_SANITIZE"):
    if sys.platform == "win32":
        raise RuntimeError(
            "PYSCA_CPP_SANITIZE requires a GCC- or Clang-compatible platform"
        )
    sanitizer_flags = [
        "-fsanitize=address,undefined",
        "-fno-omit-frame-pointer",
        "-fno-sanitize-recover=all",
    ]
    extra_compile_args.extend([*sanitizer_flags, "-O1", "-g"])
    extra_link_args.extend(sanitizer_flags)


ext_modules = [
    Pybind11Extension(
        "pyscarcopula._scar_cpp",
        [str(CPP_SRC / name) for name in SCAR_CORE_SOURCES],
        include_dirs=[str(CPP_ROOT / "include")],
        cxx_std=17,
        optional=False,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]


setup(
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    zip_safe=False,
)
