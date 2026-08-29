"""Keep wheel staging independent from stale files in ``build/lib.*``."""

from __future__ import annotations

from pathlib import Path
import shutil


_PYTHON_OUTPUT_SUFFIXES = frozenset({".py", ".pyc", ".pyo", ".pyi"})
_NATIVE_OUTPUT_SUFFIXES = frozenset({".dll", ".dylib", ".pyd", ".so"})


def _inside(root: Path, target: Path) -> bool:
    return target == root or root in target.parents


def prune_stale_package_files(build_lib, source_package) -> tuple[str, ...]:
    """Remove deleted Python/build-only files from a wheel staging package."""
    build_root = Path(build_lib).resolve()
    source_root = Path(source_package).resolve()
    built_package = (build_root / source_root.name).resolve()
    if not _inside(build_root, built_package):
        raise RuntimeError(
            f"wheel staging package escapes build root: {built_package}")
    if not built_package.is_dir():
        return ()

    removed = []
    build_only = (built_package / "_cpp").resolve()
    if build_only.is_dir():
        if not _inside(built_package, build_only):
            raise RuntimeError(
                f"build-only package escapes staging root: {build_only}")
        removed.extend(
            path.relative_to(build_root).as_posix()
            for path in build_only.rglob("*")
            if path.is_file()
        )
        shutil.rmtree(build_only)

    for path in tuple(built_package.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(built_package)
        source = source_root / relative
        suffix = path.suffix.lower()
        stale_python = (
            suffix in _PYTHON_OUTPUT_SUFFIXES
            and (suffix in {".pyc", ".pyo"} or not source.is_file())
        )
        stale_native = (
            suffix in _NATIVE_OUTPUT_SUFFIXES
            and not (
                relative.parent.as_posix() == "_native"
                and relative.name.startswith("_scar_cpp.")
            )
        )
        if stale_python or stale_native:
            removed.append(path.relative_to(build_root).as_posix())
            path.unlink()

    directories = sorted(
        (path for path in built_package.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            pass
    return tuple(sorted(removed))
