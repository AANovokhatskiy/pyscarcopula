"""Aggregate a complete release-gate artifact matrix and reject drift."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re

try:
    from tools.finalize_release_artifacts import verify as verify_artifacts
except ModuleNotFoundError:
    from finalize_release_artifacts import verify as verify_artifacts


def _inside(root: Path, target: Path) -> bool:
    return target == root or root in target.parents


def aggregate(artifacts: Path, required: tuple[str, ...]) -> dict:
    records = []
    for path in sorted(artifacts.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        records.append({"artifact": str(path), "payload": payload})

    failed = [
        record
        for record in records
        if record["payload"].get("status") not in (None, "passed")
    ]
    configurations = Counter(
        record["payload"].get("configuration")
        for record in records
        if record["payload"].get("configuration")
    )
    missing = sorted(set(required) - set(configurations))

    provenance = [
        record["payload"]
        for record in records
        if record["payload"].get("record_type") == "release_provenance"
    ]
    validation = [
        record["payload"]
        for record in records
        if record["payload"].get("record_type")
        == "installed_wheel_validation"
    ]
    provenance_configurations = {
        record.get("configuration") for record in provenance
    }
    validation_configurations = {
        record.get("configuration") for record in validation
    }
    missing_provenance = sorted(
        set(required) - provenance_configurations
    )
    required_wheels = {
        configuration
        for configuration in required
        if configuration.startswith("wheel-")
        or re.search(r"-py\d{3}$", configuration)
    }
    missing_wheel_validation = sorted(
        required_wheels - validation_configurations
    )

    heads = {record.get("head") for record in provenance if record.get("head")}
    source_digests = {
        record.get("source", {}).get("native_source_sha256")
        for record in provenance
        if record.get("source", {}).get("native_source_sha256")
    }
    cxx_standards = {
        record.get("source", {}).get("cxx_standard")
        for record in provenance
    }
    dirty = [
        record.get("configuration")
        for record in provenance
        if record.get("dirty")
    ]
    invalid_concurrency = [
        record.get("configuration")
        for record in provenance
        if record.get("test_workers") != 4
        or record.get("build_jobs") != 4
    ]
    wheel_provenance_failures = [
        record.get("configuration")
        for record in provenance
        if record.get("configuration") in required_wheels
        and (
            not record.get("wheels")
            or (record.get("wheel_validation") or {}).get("status")
            != "passed"
        )
    ]
    integrity_failures = []
    checked_roots = set()
    for record in records:
        payload = record["payload"]
        configuration = payload.get("configuration")
        if (
            payload.get("record_type") != "release_provenance"
            or configuration not in required
        ):
            continue
        root = str(Path(record["artifact"]).resolve().parent)
        if root in checked_roots:
            continue
        checked_roots.add(root)
        try:
            verify_artifacts(Path(root))
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
            integrity_failures.append({
                "configuration": configuration,
                "artifact_root": root,
                "error": str(error),
            })
    consistency_errors = []
    if provenance and len(heads) != 1:
        consistency_errors.append(f"multiple HEAD values: {sorted(heads)!r}")
    if provenance and len(source_digests) != 1:
        consistency_errors.append(
            "multiple native source digests: "
            f"{sorted(source_digests)!r}"
        )
    if provenance and cxx_standards != {17}:
        consistency_errors.append(
            f"release matrix is not uniformly C++17: {cxx_standards!r}"
        )

    passed = bool(records) and not any((
        failed,
        missing,
        missing_provenance,
        missing_wheel_validation,
        dirty,
        invalid_concurrency,
        wheel_provenance_failures,
        integrity_failures,
        consistency_errors,
    ))
    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "passed" if passed else "failed",
        "artifact_count": len(records),
        "required_configurations": list(required),
        "configuration_artifact_counts": dict(sorted(configurations.items())),
        "missing_configurations": missing,
        "missing_provenance": missing_provenance,
        "missing_wheel_validation": missing_wheel_validation,
        "dirty_configurations": dirty,
        "invalid_concurrency": invalid_concurrency,
        "wheel_provenance_failures": wheel_provenance_failures,
        "artifact_integrity_failures": integrity_failures,
        "consistency_errors": consistency_errors,
        "heads": sorted(heads),
        "native_source_digests": sorted(source_digests),
        "failed_artifacts": failed,
        "records": records,
        "subinterpreter_contract": (
            "multiple_interpreters::not_supported; immediate rejection"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--required-configuration",
        action="append",
        default=[],
    )
    arguments = parser.parse_args(argv)

    source_root = arguments.source_root.resolve()
    json_output = arguments.json_output.resolve()
    markdown_output = arguments.markdown_output.resolve()
    for output in (json_output, markdown_output):
        if _inside(source_root, output):
            parser.error("report outputs must be outside the product repository")
        if output.exists():
            parser.error("refusing to overwrite release validation evidence")

    required = tuple(arguments.required_configuration)
    if len(required) != len(set(required)):
        parser.error("required configurations must be unique")
    report = aggregate(arguments.artifacts.resolve(), required)

    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Native release validation",
        "",
        f"- Verdict: **{report['verdict']}**",
        f"- Generated: `{report['generated_at']}`",
        f"- Machine-readable artifacts: `{report['artifact_count']}`",
        f"- Required configurations: `{len(required)}`",
        "- Subinterpreters: explicitly unsupported; import must fail "
        "immediately.",
        "",
        "## Configurations",
        "",
    ]
    for configuration in required:
        count = report["configuration_artifact_counts"].get(configuration, 0)
        lines.append(f"- `{configuration}`: `{count}` artifact(s)")
    if report["missing_configurations"]:
        lines.extend((
            "",
            "## Missing",
            "",
            *(
                f"- `{configuration}`"
                for configuration in report["missing_configurations"]
            ),
        ))
    markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if report["verdict"] != "passed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
