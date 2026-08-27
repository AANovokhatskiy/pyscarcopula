"""Convert a pytest JUnit report into a reproducible R-vine test inventory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Sequence
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyscarcopula._native import _extension as _cpp_extension


DEFAULT_OUTPUT = (
    ROOT / "benchmark_artifacts" / "rvine_runtime_test_inventory.json"
)


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    value = completed.stdout.strip()
    return value or None


def _version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def _case_id(case: ET.Element) -> str:
    classname = case.attrib.get("classname", "")
    name = case.attrib.get("name", "")
    return f"{classname}::{name}" if classname else name


def record_suite(junit: Path, command: str) -> dict[str, object]:
    root = ET.parse(junit).getroot()
    cases = list(root.iter("testcase"))
    skipped = []
    xfailed = []
    failed = []
    errors = []
    for case in cases:
        node_id = _case_id(case)
        skipped_element = case.find("skipped")
        if skipped_element is not None:
            record = {
                "node_id": node_id,
                "type": skipped_element.attrib.get("type"),
                "message": skipped_element.attrib.get("message", ""),
            }
            if "xfail" in str(record["type"]).lower():
                xfailed.append(record)
            else:
                skipped.append(record)
        failure = case.find("failure")
        if failure is not None:
            failed.append({
                "node_id": node_id,
                "type": failure.attrib.get("type"),
                "message": failure.attrib.get("message", ""),
            })
        error = case.find("error")
        if error is not None:
            errors.append({
                "node_id": node_id,
                "type": error.attrib.get("type"),
                "message": error.attrib.get("message", ""),
            })

    module = _cpp_extension.load()
    duration = sum(float(case.attrib.get("time", 0.0)) for case in cases)
    return {
        "schema_version": 1,
        "artifact_kind": "rvine_runtime_test_inventory",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "command": command,
        "backend": "native",
        "counts": {
            "collected": len(cases),
            "passed": (
                len(cases) - len(skipped) - len(xfailed)
                - len(failed) - len(errors)
            ),
            "skipped": len(skipped),
            "xfailed": len(xfailed),
            "failed": len(failed),
            "errors": len(errors),
            "summed_case_seconds": duration,
        },
        "skipped": skipped,
        "xfailed": xfailed,
        "failed": failed,
        "errors": errors,
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_compiler_runtime": platform.python_compiler(),
            "numpy": np.__version__,
            "scipy": _version("scipy"),
            "pytest": _version("pytest"),
            "pyscarcopula": _version("pyscarcopula"),
            "native_extension": str(getattr(module, "__file__", "unknown")),
            "native_rvine_symbols": sorted(
                name for name in dir(module) if "rvine" in name.lower()),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    junit = args.junit if args.junit.is_absolute() else ROOT / args.junit
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = record_suite(junit, args.command)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"recorded {report['counts']['collected']} tests with "
        f"{report['counts']['skipped']} skips in {output}"
    )
    return 1 if report["failed"] or report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
