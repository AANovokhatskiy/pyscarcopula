"""Write platform/compiler/wheel metadata for release-gate artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--wheel-dir", type=Path)
    parser.add_argument(
        "--status",
        default="passed",
        help="status recorded in the artifact (default: passed)",
    )
    arguments = parser.parse_args()

    wheels = []
    if arguments.wheel_dir is not None:
        wheels = [
            {"file": path.name, "sha256": _sha256(path)}
            for path in sorted(arguments.wheel_dir.glob("*.whl"))
        ]
    report = {
        "status": arguments.status,
        "configuration": arguments.configuration,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "cc": os.environ.get("CC") or "toolchain-default",
        "cxx": os.environ.get("CXX") or "toolchain-default",
        "cxx_version": _version([
            os.environ.get("CXX") or (
                "cl" if os.name == "nt" else "c++"
            ),
            "--version",
        ]),
        "wheels": wheels,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
