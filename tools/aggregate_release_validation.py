"""Aggregate a complete release-validation artifact matrix and reject drift."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

try:
    from tools.finalize_release_artifacts import verify as verify_artifacts
except ModuleNotFoundError:
    from finalize_release_artifacts import verify as verify_artifacts


def _inside(root: Path, target: Path) -> bool:
    return target == root or root in target.parents


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wheel_identity_error(payload: dict) -> str | None:
    wheel = payload.get("wheel")
    if not isinstance(wheel, dict):
        return "missing wheel identity"
    required = ("file", "sha256", "files", "extension", "extension_sha256")
    if any(field not in wheel for field in required):
        return "incomplete wheel identity"
    if (
            not isinstance(wheel["file"], str)
            or not wheel["file"]
            or Path(wheel["file"]).name != wheel["file"]
            or not isinstance(wheel["files"], int)
            or isinstance(wheel["files"], bool)
            or wheel["files"] < 1
            or not isinstance(wheel["extension"], str)
            or not wheel["extension"]):
        return "invalid wheel identity"
    for field in ("sha256", "extension_sha256"):
        digest = wheel[field]
        if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            return f"invalid wheel {field}"
    return None


def _runtime_contract_error(payload: dict) -> str | None:
    if payload.get("schema_version") != 2 or payload.get("status") != "passed":
        return "installed-wheel validation schema or status is invalid"
    wheel_error = _wheel_identity_error(payload)
    if wheel_error is not None:
        return wheel_error
    contract = payload.get("parallel_runtime_contract")
    if not isinstance(contract, dict):
        return "missing parallel_runtime_contract"
    default = contract.get("default_call") or {}
    parallel = contract.get("parallel_call") or {}
    if (
            contract.get("default_n_threads") != 1
            or default.get("runtime_initialized") is not False
            or default.get("batches_submitted") != 0
            or default.get("tasks_submitted") != 0):
        return "default call did not prove the one-thread no-pool contract"
    if (
            parallel.get("requested_n_threads") != 2
            or parallel.get("runtime_initialized") is not True
            or parallel.get("owner_pid_matches") is not True
            or parallel.get("worker_count") != 2
            or parallel.get("batches_submitted") != 1
            or parallel.get("tasks_submitted") != 2):
        return "explicit call did not prove one two-runner batch"
    if contract.get("shutdown_initialized") is not False:
        return "runtime shutdown was not recorded"
    for field in ("extension_sha256", "result_sha256"):
        digest = contract.get(field)
        if not isinstance(digest, str) or len(digest) != 64:
            return f"invalid {field}"
    extension_digest = contract["extension_sha256"]
    if extension_digest != (payload.get("wheel") or {}).get(
            "extension_sha256"):
        return "runtime contract does not match the validated wheel"
    if extension_digest != (payload.get("import_boundary") or {}).get(
            "extension_sha256"):
        return "runtime contract does not match the installed extension"
    return None


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

    provenance_entries = [
        record
        for record in records
        if record["payload"].get("record_type") == "release_provenance"
    ]
    validation_entries = [
        record
        for record in records
        if record["payload"].get("record_type")
        == "installed_wheel_validation"
    ]
    provenance = [record["payload"] for record in provenance_entries]
    validation = [record["payload"] for record in validation_entries]
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
    runtime_contract_failures = []
    runtime_contract_provenance_failures = []
    for configuration in sorted(required_wheels):
        matching_validation = [
            record for record in validation_entries
            if record["payload"].get("configuration") == configuration
        ]
        if len(matching_validation) != 1:
            runtime_contract_failures.append({
                "configuration": configuration,
                "error": (
                    "expected exactly one installed-wheel validation, got "
                    f"{len(matching_validation)}"
                ),
            })
            continue
        validation_entry = matching_validation[0]
        validation_record = validation_entry["payload"]
        error = _runtime_contract_error(validation_record)
        if error is not None:
            runtime_contract_failures.append({
                "configuration": configuration,
                "error": error,
            })
            continue

        matching_provenance = [
            record for record in provenance
            if record.get("configuration") == configuration
        ]
        contract = validation_record["parallel_runtime_contract"]
        if len(matching_provenance) != 1:
            runtime_contract_provenance_failures.append({
                "configuration": configuration,
                "error": (
                    "expected exactly one provenance record, got "
                    f"{len(matching_provenance)}"
                ),
            })
            continue
        provenance_record = matching_provenance[0]
        wheels = provenance_record.get("wheels") or []
        recorded_validation = provenance_record.get("wheel_validation") or {}
        recorded_contract = recorded_validation.get("parallel_runtime_contract")
        validation_path = Path(validation_entry["artifact"])
        validation_wheel = validation_record["wheel"]
        if (
                provenance_record.get("schema_version") != 3
                or provenance_record.get("status") != "passed"
                or len(wheels) != 1
                or wheels[0] != validation_wheel
                or recorded_validation.get("wheel") != validation_wheel
                or recorded_validation.get("file") != validation_path.name
                or recorded_validation.get("sha256") != _sha256(validation_path)
                or recorded_contract != contract):
            runtime_contract_provenance_failures.append({
                "configuration": configuration,
                "error": (
                    "runtime contract does not match wheel provenance"
                ),
            })

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
        runtime_contract_failures,
        runtime_contract_provenance_failures,
        dirty,
        invalid_concurrency,
        wheel_provenance_failures,
        integrity_failures,
        consistency_errors,
    ))
    return {
        "schema_version": 3,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "passed" if passed else "failed",
        "artifact_count": len(records),
        "required_configurations": list(required),
        "configuration_artifact_counts": dict(sorted(configurations.items())),
        "missing_configurations": missing,
        "missing_provenance": missing_provenance,
        "missing_wheel_validation": missing_wheel_validation,
        "parallel_runtime_contract_failures": runtime_contract_failures,
        "parallel_runtime_provenance_failures": (
            runtime_contract_provenance_failures
        ),
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


def _failure_lines(report: dict) -> list[str]:
    lines = []
    if not report["artifact_count"]:
        lines.append("- No machine-readable artifacts found.")
    for field in (
        "missing_configurations",
        "missing_provenance",
        "missing_wheel_validation",
        "parallel_runtime_contract_failures",
        "parallel_runtime_provenance_failures",
        "dirty_configurations",
        "invalid_concurrency",
        "wheel_provenance_failures",
        "artifact_integrity_failures",
        "consistency_errors",
    ):
        for failure in report[field]:
            detail = (
                json.dumps(failure, sort_keys=True)
                if isinstance(failure, dict) else failure
            )
            lines.append(f"- {field}: {detail}")
    for record in report["failed_artifacts"]:
        lines.append(
            f"- failed_artifacts: {record['artifact']} "
            f"(status={record['payload'].get('status')!r})"
        )
    for record in report["records"]:
        payload = record["payload"]
        if payload.get("record_type") == "release_provenance" and payload.get("dirty"):
            for path in payload.get("dirty_paths", []):
                lines.append(f"- {payload.get('configuration')} dirty path: {path}")
    return lines


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
        if _inside(source_root, output) and not _inside(source_root / "build", output):
            parser.error("report outputs must be inside build/ or outside the product repository")
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
    if report["verdict"] != "passed":
        lines.extend(("", "## Failures", "", *_failure_lines(report)))
    markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nJSON report: {json_output}")
    print(f"Markdown report: {markdown_output}")
    return 1 if report["verdict"] != "passed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
