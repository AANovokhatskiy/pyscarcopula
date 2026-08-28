"""Audit the compiled extension's direct binary dependencies.

The script intentionally uses platform toolchain utilities already present on
wheel builders. It fails when OpenMP or external BLAS/LAPACK runtimes appear
and emits a machine-readable report for release validation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys


_BANNED_DEPENDENCY_MARKERS = (
    "libgomp",
    "libomp",
    "vcomp",
    "openblas",
    "mkl",
    "libblas",
    "liblapack",
    "accelerate.framework",
)
_BANNED_SOURCE_MARKERS = (
    "#include <eigen",
    "#include \"eigen",
    "#include <cblas",
    "#include <openblas",
    "#include <omp.h",
)


def _extension_path() -> Path:
    spec = importlib.util.find_spec("pyscarcopula._native._scar_cpp")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "pyscarcopula._native._scar_cpp is not importable")
    return Path(spec.origin).resolve()


def _dependency_command(extension: Path) -> list[str]:
    if sys.platform == "darwin":
        executable = shutil.which("otool")
        if executable is None:
            raise RuntimeError("otool is required for the macOS audit")
        return [executable, "-L", str(extension)]
    executable = shutil.which("ldd")
    if executable is None:
        raise RuntimeError("ldd is required for the Linux audit")
    return [executable, str(extension)]


def _pe_dependencies(extension: Path) -> list[str]:
    data = extension.read_bytes()
    if data[:2] != b"MZ":
        raise RuntimeError("Windows extension does not have an MZ header")
    pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe_offset:pe_offset + 4] != b"PE\0\0":
        raise RuntimeError("Windows extension does not have a PE header")

    coff_offset = pe_offset + 4
    section_count = struct.unpack_from("<H", data, coff_offset + 2)[0]
    optional_size = struct.unpack_from("<H", data, coff_offset + 16)[0]
    optional_offset = coff_offset + 20
    magic = struct.unpack_from("<H", data, optional_offset)[0]
    if magic == 0x20B:
        directories_offset = optional_offset + 112
    elif magic == 0x10B:
        directories_offset = optional_offset + 96
    else:
        raise RuntimeError(f"unsupported PE optional-header magic: {magic:#x}")

    import_rva, _ = struct.unpack_from(
        "<II", data, directories_offset + 8)
    section_offset = optional_offset + optional_size
    sections = []
    for index in range(section_count):
        offset = section_offset + index * 40
        virtual_size, virtual_address, raw_size, raw_offset = (
            struct.unpack_from("<IIII", data, offset + 8)
        )
        sections.append((
            virtual_address,
            max(virtual_size, raw_size),
            raw_offset,
        ))

    def rva_to_offset(rva: int) -> int:
        for virtual_address, size, raw_offset in sections:
            if virtual_address <= rva < virtual_address + size:
                return raw_offset + rva - virtual_address
        raise RuntimeError(f"PE RVA {rva:#x} is outside all sections")

    dependencies = []
    descriptor_offset = rva_to_offset(import_rva)
    while True:
        descriptor = struct.unpack_from("<IIIII", data, descriptor_offset)
        if descriptor == (0, 0, 0, 0, 0):
            break
        name_offset = rva_to_offset(descriptor[3])
        name_end = data.index(b"\0", name_offset)
        dependencies.append(
            data[name_offset:name_end].decode("ascii", errors="replace"))
        descriptor_offset += 20
    return dependencies


def _dependency_output(extension: Path) -> tuple[list[str], str]:
    if sys.platform == "win32":
        dependencies = _pe_dependencies(extension)
        return ["internal-pe-parser", str(extension)], "\n".join(dependencies)
    command = _dependency_command(extension)
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return command, completed.stdout + completed.stderr


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_findings() -> list[dict[str, str]]:
    root = Path(__file__).resolve().parents[1] / "pyscarcopula" / "_cpp"
    findings = []
    for path in sorted(root.rglob("*")):
        if path.suffix not in {".cpp", ".hpp"}:
            continue
        lowered = path.read_text(encoding="utf-8").lower()
        for marker in _BANNED_SOURCE_MARKERS:
            if marker in lowered:
                findings.append({
                    "file": str(path.relative_to(root)),
                    "marker": marker,
                })
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    extension = _extension_path()
    command, dependency_output = _dependency_output(extension)
    lowered = dependency_output.lower()
    findings = [
        marker
        for marker in _BANNED_DEPENDENCY_MARKERS
        if marker in lowered
    ]
    source_findings = _source_findings()
    report = {
        "status": (
            "passed" if not findings and not source_findings else "failed"
        ),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "extension": str(extension),
        "extension_sha256": _sha256(extension),
        "command": command,
        "banned_markers": list(_BANNED_DEPENDENCY_MARKERS),
        "findings": findings,
        "banned_source_markers": list(_BANNED_SOURCE_MARKERS),
        "source_findings": source_findings,
        "dependencies": dependency_output.splitlines(),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    return 1 if findings or source_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
