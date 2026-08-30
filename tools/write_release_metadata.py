"""Write reproducible toolchain, source, wheel, and binary provenance."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import zipfile


def _inside(root: Path, target: Path) -> bool:
    return target == root or root in target.parents


def _version(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return platform.python_compiler()
    lines = (completed.stdout or completed.stderr).splitlines()
    return lines[0] if lines else platform.python_compiler()


def _git(source_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load build support module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest_files(source_root: Path, paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        resolved = path.resolve()
        if not _inside(source_root, resolved) or not resolved.is_file():
            raise RuntimeError(f"native source is missing or external: {path}")
        relative = resolved.relative_to(source_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(resolved.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def source_provenance(source_root: Path) -> dict:
    source_root = source_root.resolve()
    support = source_root / "pyscarcopula" / "_cpp" / "build_support"
    sources_module = _load_module(
        support / "sources.py", "_pyscarcopula_release_sources"
    )
    toolchain_module = _load_module(
        support / "toolchain.py", "_pyscarcopula_release_toolchain"
    )
    cpp_root = source_root / "pyscarcopula" / "_cpp"
    source_dir = cpp_root / "src"
    compute = tuple(
        source_dir / name for name in sources_module.SCAR_COMPUTE_SOURCES
    )
    bindings = tuple(
        source_dir / name for name in sources_module.PYTHON_BINDING_SOURCES
    )
    headers = tuple(
        path
        for path in sorted((cpp_root / "include").rglob("*"))
        if path.is_file()
    )
    build_policy = (
        support / "sources.py",
        support / "toolchain.py",
        cpp_root / "include" / "scar" / "copula" / "pair" / "families.def",
    )
    native_files = tuple(
        dict.fromkeys((*compute, *bindings, *headers, *build_policy))
    )
    return {
        "cxx_standard": int(toolchain_module.CXX_STANDARD),
        "compute_source_count": len(compute),
        "binding_source_count": len(bindings),
        "public_header_count": len(headers),
        "compute_source_sha256": _digest_files(source_root, compute),
        "binding_source_sha256": _digest_files(source_root, bindings),
        "native_source_sha256": _digest_files(source_root, native_files),
    }


def wheel_provenance(wheel_dir: Path | None) -> list[dict]:
    if wheel_dir is None:
        return []
    records = []
    for path in sorted(wheel_dir.glob("*.whl")):
        with zipfile.ZipFile(path) as archive:
            members = tuple(sorted(archive.namelist()))
            extension_members = tuple(
                name
                for name in members
                if name.startswith("pyscarcopula/_native/_scar_cpp.")
                and name.lower().endswith((".pyd", ".so", ".dylib"))
            )
            if len(extension_members) != 1:
                raise RuntimeError(
                    f"{path.name} must contain exactly one native extension; "
                    f"found {extension_members!r}"
                )
            binary = archive.read(extension_members[0])
        records.append({
            "file": path.name,
            "sha256": _sha256(path),
            "files": len(members),
            "extension": extension_members[0],
            "extension_sha256": hashlib.sha256(binary).hexdigest(),
        })
    return records


def build_report(
    *,
    source_root: Path,
    configuration: str,
    status: str,
    wheel_dir: Path | None,
    validation_file: Path | None,
    test_workers: int,
    build_jobs: int,
) -> dict:
    source_root = source_root.resolve()
    source = source_provenance(source_root)
    head = _git(source_root, "rev-parse", "HEAD")
    dirty_lines = _git(source_root, "status", "--porcelain").splitlines()
    validation = None
    if validation_file is not None:
        validation_path = validation_file.resolve()
        validation_payload = json.loads(
            validation_path.read_text(encoding="utf-8")
        )
        if validation_payload.get("status") != "passed":
            raise RuntimeError(
                f"wheel validation is not passed: {validation_path}"
            )
        if validation_payload.get("configuration") != configuration:
            raise RuntimeError(
                "wheel validation configuration does not match release "
                f"metadata: {validation_payload.get('configuration')!r} != "
                f"{configuration!r}"
            )
        validation = {
            "file": validation_path.name,
            "sha256": _sha256(validation_path),
            "status": validation_payload["status"],
        }

    compiler = os.environ.get("CXX") or (
        "cl" if os.name == "nt" else "c++"
    )
    compiler_command = [compiler] if compiler.lower().endswith("cl") else [
        compiler,
        "--version",
    ]
    return {
        "schema_version": 2,
        "record_type": "release_provenance",
        "status": status,
        "configuration": configuration,
        "head": head,
        "dirty": bool(dirty_lines),
        "dirty_paths": dirty_lines,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "cc": os.environ.get("CC") or "toolchain-default",
        "cxx": os.environ.get("CXX") or "toolchain-default",
        "cxx_version": _version(compiler_command),
        "test_workers": int(test_workers),
        "build_jobs": int(build_jobs),
        "source": source,
        "wheels": wheel_provenance(
            None if wheel_dir is None else wheel_dir.resolve()
        ),
        "wheel_validation": validation,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--wheel-dir", type=Path)
    parser.add_argument("--validation-file", type=Path)
    parser.add_argument("--test-workers", type=int, default=4)
    parser.add_argument("--build-jobs", type=int, default=4)
    parser.add_argument(
        "--status",
        default="passed",
        help="status recorded in the artifact (default: passed)",
    )
    arguments = parser.parse_args(argv)

    source_root = arguments.source_root.resolve()
    output = arguments.output.resolve()
    if _inside(source_root, output):
        parser.error("--output must be outside the product repository")
    if output.exists():
        parser.error("refusing to overwrite existing release evidence")
    if arguments.test_workers < 1 or arguments.build_jobs < 1:
        parser.error("worker and build-job counts must be positive")

    report = build_report(
        source_root=source_root,
        configuration=arguments.configuration,
        status=arguments.status,
        wheel_dir=arguments.wheel_dir,
        validation_file=arguments.validation_file,
        test_workers=arguments.test_workers,
        build_jobs=arguments.build_jobs,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
