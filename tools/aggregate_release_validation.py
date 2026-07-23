"""Aggregate successful release-gate artifacts into one validation report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    arguments = parser.parse_args()

    records = []
    for path in sorted(arguments.artifacts.rglob("*.json")):
        if path.resolve() == arguments.json_output.resolve():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        records.append({"artifact": str(path), "payload": payload})

    failed = [
        record for record in records
        if record["payload"].get("status") not in (None, "passed")
    ]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "passed" if records and not failed else "failed",
        "artifact_count": len(records),
        "failed_artifacts": failed,
        "records": records,
        "subinterpreter_contract": (
            "multiple_interpreters::not_supported; immediate rejection"
        ),
    }
    arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Native parallel release validation",
        "",
        f"- Verdict: **{report['verdict']}**",
        f"- Generated: `{report['generated_at']}`",
        f"- Machine-readable artifacts: `{len(records)}`",
        "- Subinterpreters: explicitly unsupported; import must fail "
        "immediately.",
        "",
        "## Configurations",
        "",
    ]
    for record in records:
        payload = record["payload"]
        label = payload.get(
            "configuration",
            payload.get("platform", record["artifact"]),
        )
        lines.append(
            f"- `{label}`: `{payload.get('status', 'recorded')}`")
    arguments.markdown_output.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return 1 if report["verdict"] != "passed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
