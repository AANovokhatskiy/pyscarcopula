"""Shared opt-in parallel compilation policy for C++ build entry points."""

from __future__ import annotations

from contextlib import contextmanager
import os
from types import MethodType
from typing import Any, Callable, Iterator

from pybind11.setup_helpers import ParallelCompile


BUILD_JOBS_ENV = "PYSCA_CPP_BUILD_JOBS"
DEFAULT_BUILD_JOBS = 1


def resolve_build_jobs(explicit: int | str | None = None) -> int:
    """Resolve a positive build-job count from CLI/config or the environment."""

    source = "build jobs"
    value: int | str | None = explicit
    if value is None:
        source = BUILD_JOBS_ENV
        value = os.environ.get(BUILD_JOBS_ENV)
        if value is None or not value.strip():
            return DEFAULT_BUILD_JOBS

    if isinstance(value, bool):
        raise ValueError(
            f"{source} must be a positive integer, got {value!r}"
        )
    if isinstance(value, int):
        jobs = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(
                f"{source} must be a positive integer, got {value!r}"
            )
        try:
            jobs = int(text)
        except ValueError as error:
            raise ValueError(
                f"{source} must be a positive integer, got {value!r}"
            ) from error
    else:
        raise ValueError(
            f"{source} must be a positive integer, got {value!r}"
        )

    if jobs < 1:
        raise ValueError(
            f"{source} must be a positive integer, got {value!r}"
        )
    return jobs


@contextmanager
def parallel_compilation(
    compiler: Any,
    build_jobs: int,
) -> Iterator[None]:
    """Apply pybind11's source-level compiler pool to one compiler instance.

    Binding the helper to the selected compiler instance covers compiler
    subclasses that implement their own ``compile`` method. Modern setuptools
    MSVC compilers also lack the per-source ``_compile`` implementation used by
    pybind11, so a thin adapter delegates each pooled source back to MSVC's
    original compiler method. The pool and job scheduling remain pybind11's.
    The sequential default deliberately leaves the compiler untouched.
    """

    jobs = resolve_build_jobs(build_jobs)
    if jobs == 1:
        yield
        return

    initialize = getattr(compiler, "initialize", None)
    if (
        callable(initialize)
        and not getattr(compiler, "initialized", True)
    ):
        initialize()

    previous_compile = compiler.compile
    pybind_compile = ParallelCompile(default=jobs).function()
    compile_function = _compiler_compile_function(
        compiler,
        previous_compile,
        pybind_compile,
    )
    compiler.compile = MethodType(compile_function, compiler)
    try:
        yield
    finally:
        compiler.compile = previous_compile


def _compiler_compile_function(
    compiler: Any,
    original_compile: Callable[..., Any],
    pybind_compile: Callable[..., Any],
) -> Callable[..., Any]:
    if getattr(compiler, "compiler_type", None) != "msvc":
        return pybind_compile

    def compile_with_msvc_adapter(
        selected_compiler: Any,
        sources: list[str],
        output_dir: str | None = None,
        macros: list[Any] | None = None,
        include_dirs: list[str] | tuple[str, ...] | None = None,
        debug: bool = False,
        extra_preargs: list[str] | None = None,
        extra_postargs: list[str] | None = None,
        depends: list[str] | tuple[str, ...] | None = None,
    ) -> Any:
        previous_source_compile = selected_compiler._compile

        def compile_one_source(
            _obj: str,
            src: str,
            _ext: str,
            _cc_args: list[str],
            _source_extra_postargs: list[str],
            _pp_opts: list[str],
        ) -> None:
            original_compile(
                [src],
                output_dir=output_dir,
                macros=macros,
                include_dirs=include_dirs,
                debug=debug,
                extra_preargs=(
                    list(extra_preargs) if extra_preargs is not None else None
                ),
                extra_postargs=(
                    list(extra_postargs)
                    if extra_postargs is not None else None
                ),
                depends=depends,
            )

        selected_compiler._compile = compile_one_source
        try:
            return pybind_compile(
                selected_compiler,
                sources,
                output_dir=output_dir,
                macros=macros,
                include_dirs=include_dirs,
                debug=debug,
                extra_preargs=extra_preargs,
                extra_postargs=extra_postargs,
                depends=depends,
            )
        finally:
            selected_compiler._compile = previous_source_compile

    return compile_with_msvc_adapter
