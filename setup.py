import os
import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


ROOT = Path(__file__).resolve().parent
CPP_ROOT = ROOT / "pyscarcopula" / "_cpp"
CPP_SRC = Path("pyscarcopula") / "_cpp" / "src"

SCAR_CORE_SOURCES = [
    "copula.cpp",
    "copula_common.cpp",
    "copula_dispatch.cpp",
    "clayton_copula.cpp",
    "gumbel_copula.cpp",
    "frank_copula.cpp",
    "joe_copula.cpp",
    "gaussian_copula.cpp",
    "kendall.cpp",
    "student_copula.cpp",
    "multivariate_copula.cpp",
    "static_likelihood.cpp",
    "gas_evaluator.cpp",
    "scar_mc.cpp",
    "scar_ou_evaluator.cpp",
    "scar_ou_grid.cpp",
    "scar_quadrature.cpp",
    "scar_ou_transition.cpp",
    "bindings_pybind.cpp",
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
