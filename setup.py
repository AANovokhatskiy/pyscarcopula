import os
import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


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
    "scar_ou/monte_carlo.cpp",
    "scar_ou/validation.cpp",
    "scar_ou/likelihood.cpp",
    "scar_ou/gradient.cpp",
    "scar_ou/prediction.cpp",
    "scar_ou/state_distribution.cpp",
    "scar_ou/evaluator.cpp",
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
