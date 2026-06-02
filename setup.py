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
    "scar_ou_evaluator.cpp",
    "scar_ou_grid.cpp",
    "scar_quadrature.cpp",
    "scar_ou_transition.cpp",
    "bindings_pybind.cpp",
]

ext_modules = [
    Pybind11Extension(
        "pyscarcopula._scar_cpp",
        [str(CPP_SRC / name) for name in SCAR_CORE_SOURCES],
        include_dirs=[str(CPP_ROOT / "include")],
        cxx_std=17,
    )
]


setup(cmdclass={"build_ext": build_ext}, ext_modules=ext_modules)
