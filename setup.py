import importlib.util
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension
from pybind11.setup_helpers import build_ext as _build_ext
from setuptools import setup


ROOT = Path(__file__).resolve().parent
CPP_ROOT = ROOT / "pyscarcopula" / "_cpp"
CPP_SRC = Path("pyscarcopula") / "_cpp" / "src"


def _load_build_support(name: str):
    path = CPP_ROOT / "build_support" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(
        f"_pyscarcopula_build_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load C++ build support module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_sources = _load_build_support("sources")
_toolchain = _load_build_support("toolchain")
_build_parallel = _load_build_support("build_parallel")
SCAR_COMPUTE_SOURCES = _sources.SCAR_COMPUTE_SOURCES
PYTHON_BINDING_SOURCES = _sources.PYTHON_BINDING_SOURCES


class build_ext(_build_ext):
    """Support opt-in source parallelism and MinGW GCC on Windows.

    MSVC remains the default Windows toolchain. To use GCC instead:

        python setup.py build_ext --compiler=mingw32

    or, for pip/PEP 517 builds, set the environment variable:

        PYSCA_CPP_COMPILER=mingw32

    C++ source compilation remains sequential unless ``--parallel N`` or
    ``PYSCA_CPP_BUILD_JOBS=N`` is supplied explicitly.
    """

    def finalize_options(self):
        compiler = _toolchain.requested_compiler()
        if compiler:
            self.compiler = compiler
        super().finalize_options()

    def build_extension(self, ext):
        if self.compiler.compiler_type == "mingw32":
            _toolchain.prepare_mingw_extension(ext)
        super().build_extension(ext)

    def build_extensions(self):
        build_jobs = _build_parallel.resolve_build_jobs(self.parallel)
        self.announce(
            f"C++ source compilation jobs: {build_jobs}", level=2)
        extension_parallel = self.parallel
        self.parallel = None
        try:
            with _build_parallel.parallel_compilation(
                    self.compiler, build_jobs):
                super().build_extensions()
        finally:
            self.parallel = extension_parallel


ext_modules = [
    Pybind11Extension(
        "pyscarcopula._scar_cpp",
        [
            str(CPP_SRC / name)
            for name in (*SCAR_COMPUTE_SOURCES, *PYTHON_BINDING_SOURCES)
        ],
        include_dirs=[str(CPP_ROOT / "include")],
        cxx_std=_toolchain.CXX_STANDARD,
        optional=False,
        extra_compile_args=_toolchain.extension_compile_args(),
        extra_link_args=_toolchain.extension_link_args(),
    )
]


setup(
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    zip_safe=False,
)
