"""Finalize or verify an external append-only release artifact directory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


INDEX_NAME = "artifact_index.json"
CHECKSUM_NAME = "checksums.sha256"


def _inside(root: Path, target: Path) -> bool:
    return target == root or root in target.parents


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".whl":
        return "wheel"
    if suffix == ".json":
        return "json"
    if suffix in {".md", ".txt", ".log"}:
        return "text"
    if suffix in {".pyd", ".so", ".dylib", ".dll"}:
        return "binary"
    return "file"


def _payload_files(root: Path) -> tuple[Path, ...]:
    files = tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {INDEX_NAME, CHECKSUM_NAME}
    )
    for path in files:
        if path.is_symlink() or not _inside(root, path.resolve()):
            raise RuntimeError(f"artifact path escapes root: {path}")
    return files


def finalize(root: Path, producer: str) -> dict:
    root = root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"artifact root does not exist: {root}")
    index_path = root / INDEX_NAME
    checksum_path = root / CHECKSUM_NAME
    if index_path.exists() or checksum_path.exists():
        raise RuntimeError(
            "artifact root is already finalized; refusing to overwrite"
        )
    payload_files = _payload_files(root)
    if not payload_files:
        raise RuntimeError("cannot finalize an empty artifact root")

    entries = []
    for path in payload_files:
        entries.append({
            "path": path.relative_to(root).as_posix(),
            "type": _artifact_type(path),
            "producer": producer,
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        })
    index = {
        "schema_version": 1,
        "finalized_at": datetime.now(timezone.utc).isoformat(),
        "producer": producer,
        "files": entries,
    }
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksum_files = (*payload_files, index_path)
    checksum_path.write_text(
        "".join(
            f"{_sha256(path)} *{path.relative_to(root).as_posix()}\n"
            for path in checksum_files
        ),
        encoding="utf-8",
    )
    verify(root)
    return index


def _read_checksums(path: Path) -> dict[str, str]:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition(" *")
        if not separator or len(digest) != 64 or not relative:
            raise RuntimeError(f"invalid checksum line: {line!r}")
        if relative in records:
            raise RuntimeError(f"duplicate checksum path: {relative}")
        records[relative] = digest
    return records


def verify(root: Path) -> dict:
    root = root.resolve()
    index_path = root / INDEX_NAME
    checksum_path = root / CHECKSUM_NAME
    if not index_path.is_file() or not checksum_path.is_file():
        raise RuntimeError("artifact index or checksums are missing")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    checksums = _read_checksums(checksum_path)

    expected_paths = {
        path.relative_to(root).as_posix() for path in _payload_files(root)
    }
    indexed = {entry["path"]: entry for entry in index.get("files", [])}
    if set(indexed) != expected_paths:
        raise RuntimeError(
            "artifact index paths differ from payload files: "
            f"indexed={sorted(indexed)!r}, actual={sorted(expected_paths)!r}"
        )
    expected_checksums = {*expected_paths, INDEX_NAME}
    if set(checksums) != expected_checksums:
        raise RuntimeError(
            "checksum paths differ from finalized files: "
            f"checksums={sorted(checksums)!r}, "
            f"expected={sorted(expected_checksums)!r}"
        )
    for relative, digest in checksums.items():
        path = (root / relative).resolve()
        if not _inside(root, path) or not path.is_file():
            raise RuntimeError(f"checksummed artifact is missing: {relative}")
        if _sha256(path) != digest:
            raise RuntimeError(f"checksum mismatch: {relative}")
    for relative, entry in indexed.items():
        path = root / relative
        if entry.get("sha256") != checksums[relative]:
            raise RuntimeError(f"index checksum mismatch: {relative}")
        if entry.get("size") != path.stat().st_size:
            raise RuntimeError(f"index size mismatch: {relative}")
        if not entry.get("producer") or not entry.get("type"):
            raise RuntimeError(f"incomplete artifact index entry: {relative}")
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--producer", default="release validation workflow")
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args(argv)

    source_root = arguments.source_root.resolve()
    artifact_root = arguments.artifact_root.resolve()
    if _inside(source_root, artifact_root) and not _inside(source_root / "build", artifact_root):
        parser.error("--artifact-root must be inside build/ or outside the product repository")
    if arguments.verify:
        verify(artifact_root)
    else:
        finalize(artifact_root, arguments.producer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
