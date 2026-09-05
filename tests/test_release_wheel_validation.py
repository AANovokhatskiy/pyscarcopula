"""Permanent release wheel, provenance, and matrix contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

import pytest

from tools.aggregate_release_validation import aggregate, main as aggregate_main
from tools.finalize_release_artifacts import finalize, verify
from tools.validate_installed_wheel import (
    EXPECTED_NATIVE_IDS,
    REMOVED_METHODS,
    REMOVED_MODULES,
    REMOVED_PUBLIC_NAMES,
)
from tools.write_release_metadata import (
    build_report,
    source_provenance,
    wheel_provenance,
)


ROOT = Path(__file__).resolve().parents[1]
EXTENSION_DIGEST = "b" * 64
WHEEL_DIGEST = "a" * 64


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wheel_record(
        extension_digest: str = EXTENSION_DIGEST,
        wheel_digest: str = WHEEL_DIGEST) -> dict:
    return {
        "file": "candidate.whl",
        "sha256": wheel_digest,
        "files": 2,
        "extension": "pyscarcopula/_native/_scar_cpp.fake.pyd",
        "extension_sha256": extension_digest,
    }


def _runtime_contract(extension_digest: str = EXTENSION_DIGEST) -> dict:
    return {
        "extension_sha256": extension_digest,
        "default_n_threads": 1,
        "default_call": {
            "runtime_initialized": False,
            "batches_submitted": 0,
            "tasks_submitted": 0,
        },
        "parallel_call": {
            "requested_n_threads": 2,
            "runtime_initialized": True,
            "owner_pid_matches": True,
            "worker_count": 2,
            "batches_submitted": 1,
            "tasks_submitted": 2,
        },
        "result_sha256": "c" * 64,
        "shutdown_initialized": False,
    }


def _validation(
        configuration: str,
        extension_digest: str = EXTENSION_DIGEST,
        wheel: dict | None = None) -> dict:
    wheel = (
        _wheel_record(extension_digest)
        if wheel is None
        else dict(wheel)
    )
    extension_digest = wheel["extension_sha256"]
    return {
        "schema_version": 2,
        "record_type": "installed_wheel_validation",
        "status": "passed",
        "configuration": configuration,
        "wheel": wheel,
        "import_boundary": {"extension_sha256": extension_digest},
        "parallel_runtime_contract": _runtime_contract(extension_digest),
    }


def _provenance(
        configuration: str,
        digest: str = "source-digest",
        extension_digest: str = EXTENSION_DIGEST,
        wheel: dict | None = None,
        validation_sha256: str = "d" * 64) -> dict:
    is_wheel = configuration.startswith("wheel-") or bool(
        re.search(r"-py\d{3}$", configuration)
    )
    wheel = (
        _wheel_record(extension_digest)
        if wheel is None
        else dict(wheel)
    )
    extension_digest = wheel["extension_sha256"]
    return {
        "schema_version": 3,
        "record_type": "release_provenance",
        "status": "passed",
        "configuration": configuration,
        "head": "a" * 40,
        "dirty": False,
        "test_workers": 4,
        "build_jobs": 4,
        "source": {
            "cxx_standard": 17,
            "native_source_sha256": digest,
        },
        "wheels": (
            [wheel]
            if is_wheel
            else []
        ),
        "wheel_validation": (
            {
                "file": "validation.json",
                "sha256": validation_sha256,
                "status": "passed",
                "wheel": wheel,
                "parallel_runtime_contract": _runtime_contract(
                    extension_digest),
            }
            if is_wheel
            else None
        ),
    }


def _write_release_pair(
        root: Path,
        configuration: str,
        *,
        validation: dict | None = None,
        provenance: dict | None = None) -> tuple[Path, Path]:
    validation = (
        _validation(configuration)
        if validation is None
        else validation
    )
    validation_path = root / configuration / "validation.json"
    _write_json(validation_path, validation)
    if provenance is None:
        provenance = _provenance(
            configuration,
            wheel=validation["wheel"],
            validation_sha256=_sha256(validation_path),
        )
    provenance_path = root / configuration / "provenance.json"
    _write_json(provenance_path, provenance)
    return validation_path, provenance_path


def test_native_source_provenance_is_canonical_and_cxx17():
    first = source_provenance(ROOT)
    second = source_provenance(ROOT)

    assert first == second
    assert first["cxx_standard"] == 17
    assert first["compute_source_count"] > 50
    assert first["binding_source_count"] > 10
    assert first["public_header_count"] > 50
    for key in (
        "compute_source_sha256",
        "binding_source_sha256",
        "native_source_sha256",
    ):
        assert len(first[key]) == 64
        int(first[key], 16)


def test_wheel_provenance_hashes_the_namespaced_extension(tmp_path):
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    wheel = wheel_dir / "pyscarcopula-0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("pyscarcopula/__init__.py", "")
        archive.writestr(
            "pyscarcopula/_native/_scar_cpp.fake.pyd", b"native-binary"
        )

    records = wheel_provenance(wheel_dir)

    assert len(records) == 1
    assert records[0]["file"] == wheel.name
    assert records[0]["extension"] == (
        "pyscarcopula/_native/_scar_cpp.fake.pyd"
    )
    assert len(records[0]["sha256"]) == 64
    assert len(records[0]["extension_sha256"]) == 64


def test_release_aggregator_rejects_missing_or_inconsistent_matrix(tmp_path):
    artifacts = tmp_path / "artifacts"
    required = (
        "wheel-linux-cp312",
        "linux-clang-py312",
        "linux-clang-asan-ubsan",
    )
    for configuration in required[:2]:
        _write_release_pair(artifacts, configuration)
    _write_json(
        artifacts / required[-1] / "provenance.json",
        _provenance(required[-1]),
    )
    for configuration in required:
        finalize(artifacts / configuration, "release unit test")

    assert aggregate(artifacts, required)["verdict"] == "passed"
    missing = aggregate(artifacts, (*required, "windows-msvc-py312"))
    assert missing["verdict"] == "failed"
    assert missing["missing_configurations"] == ["windows-msvc-py312"]

    payload_path = artifacts / required[-1] / "provenance.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["source"]["native_source_sha256"] = "different"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    inconsistent = aggregate(artifacts, required)
    assert inconsistent["verdict"] == "failed"
    assert inconsistent["consistency_errors"]
    assert inconsistent["artifact_integrity_failures"]


def test_in_place_native_extensions_are_ignored_but_sources_are_not():
    binaries = [
        "pyscarcopula/_native/_scar_cpp.cpython-310-x86_64-linux-gnu.so",
        "pyscarcopula/_native/_scar_cpp.cpython-314-x86_64-linux-gnu.so",
        "pyscarcopula/_native/_scar_cpp.cpython-312-darwin.so",
        "pyscarcopula/_native/_scar_cpp.cp312-win_amd64.pyd",
    ]
    sources = [
        "pyscarcopula/_native/new_module.py",
        "pyscarcopula/_cpp/src/new_source.cpp",
        "pyscarcopula/_native/unexpected.so",
    ]
    completed = subprocess.run(
        ["git", "-C", str(ROOT), "-c", "core.excludesFile=",
         "check-ignore", "--no-index", "--", *binaries, *sources],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.splitlines() == binaries


@pytest.mark.parametrize("failure", [None, "dirty", "missing", "integrity"])
def test_release_aggregator_cli_reports_verdict_and_reasons(
        tmp_path, capsys, failure):
    configuration = "linux-clang-py312"
    artifacts = tmp_path / "artifacts"
    _, provenance_path = _write_release_pair(artifacts, configuration)
    dirty_path = " M pyscarcopula/_cpp/src/native.cpp"
    if failure == "dirty":
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
        payload.update(dirty=True, dirty_paths=[dirty_path])
        _write_json(provenance_path, payload)
    finalize(provenance_path.parent, "release CLI test")
    if failure == "integrity":
        provenance_path.write_bytes(provenance_path.read_bytes() + b"\n")
    required = (
        "windows-msvc-py312" if failure == "missing" else configuration
    )
    json_output = tmp_path / "report.json"
    markdown_output = tmp_path / "report.md"
    result = aggregate_main([
        "--source-root", str(ROOT),
        "--artifacts", str(artifacts),
        "--json-output", str(json_output),
        "--markdown-output", str(markdown_output),
        "--required-configuration", required,
    ])
    assert result == (0 if failure is None else 1)
    report = json.loads(json_output.read_text(encoding="utf-8"))
    markdown = markdown_output.read_text(encoding="utf-8")
    stdout = capsys.readouterr().out
    assert markdown.strip() in stdout
    assert f"Verdict: **{report['verdict']}**" in stdout
    assert str(json_output) in stdout
    if failure == "dirty":
        assert f"dirty_configurations: {configuration}" in stdout
        assert dirty_path in stdout
    elif failure == "missing":
        assert f"missing_configurations: {required}" in stdout
    elif failure == "integrity":
        assert "artifact_integrity_failures" in stdout
        assert "checksum mismatch: provenance.json" in stdout
    else:
        assert "## Failures" not in stdout


def test_release_aggregator_rejects_missing_or_foreign_runtime_contract(
        tmp_path):
    configuration = "wheel-windows-msvc-amd64-cp312"

    missing_root = tmp_path / "missing"
    missing_validation = _validation(configuration)
    missing_validation.pop("parallel_runtime_contract")
    _write_release_pair(
        missing_root,
        configuration,
        validation=missing_validation,
    )
    finalize(missing_root / configuration, "runtime contract fixture")
    missing = aggregate(missing_root, (configuration,))
    assert missing["verdict"] == "failed"
    assert missing["parallel_runtime_contract_failures"] == [{
        "configuration": configuration,
        "error": "missing parallel_runtime_contract",
    }]

    foreign_root = tmp_path / "foreign"
    foreign_validation = _validation(configuration, "d" * 64)
    foreign_validation_path = (
        foreign_root / configuration / "validation.json"
    )
    _write_json(
        foreign_validation_path,
        foreign_validation,
    )
    _write_json(
        foreign_root / configuration / "provenance.json",
        _provenance(
            configuration,
            validation_sha256=_sha256(foreign_validation_path),
        ),
    )
    finalize(foreign_root / configuration, "runtime contract fixture")
    foreign = aggregate(foreign_root, (configuration,))
    assert foreign["verdict"] == "failed"
    assert foreign["parallel_runtime_contract_failures"] == []
    assert foreign["parallel_runtime_provenance_failures"] == [{
        "configuration": configuration,
        "error": "runtime contract does not match wheel provenance",
    }]

    incomplete_root = tmp_path / "incomplete"
    incomplete_validation = _validation(configuration)
    incomplete_validation_path = (
        incomplete_root / configuration / "validation.json"
    )
    _write_json(incomplete_validation_path, incomplete_validation)
    incomplete_provenance = _provenance(
        configuration,
        wheel=incomplete_validation["wheel"],
        validation_sha256=_sha256(incomplete_validation_path),
    )
    incomplete_provenance.pop("status")
    _write_json(
        incomplete_root / configuration / "provenance.json",
        incomplete_provenance,
    )
    finalize(incomplete_root / configuration, "runtime contract fixture")
    incomplete = aggregate(incomplete_root, (configuration,))
    assert incomplete["verdict"] == "failed"
    assert incomplete["parallel_runtime_provenance_failures"] == [{
        "configuration": configuration,
        "error": "runtime contract does not match wheel provenance",
    }]


def test_release_aggregator_requires_one_validation_and_provenance(
        tmp_path):
    configuration = "wheel-windows-msvc-amd64-cp312"

    missing_validation_root = tmp_path / "missing-validation"
    _write_json(
        missing_validation_root / configuration / "provenance.json",
        _provenance(configuration),
    )
    finalize(missing_validation_root / configuration, "record fixture")
    missing_validation = aggregate(
        missing_validation_root, (configuration,))
    assert missing_validation["verdict"] == "failed"
    assert missing_validation["missing_wheel_validation"] == [configuration]
    assert missing_validation["parallel_runtime_contract_failures"] == [{
        "configuration": configuration,
        "error": "expected exactly one installed-wheel validation, got 0",
    }]

    missing_provenance_root = tmp_path / "missing-provenance"
    _write_json(
        missing_provenance_root / configuration / "validation.json",
        _validation(configuration),
    )
    finalize(missing_provenance_root / configuration, "record fixture")
    missing_provenance = aggregate(
        missing_provenance_root, (configuration,))
    assert missing_provenance["verdict"] == "failed"
    assert missing_provenance["missing_provenance"] == [configuration]
    assert missing_provenance[
        "parallel_runtime_provenance_failures"
    ] == [{
        "configuration": configuration,
        "error": "expected exactly one provenance record, got 0",
    }]

    duplicate_validation_root = tmp_path / "duplicate-validation"
    _write_json(
        duplicate_validation_root / configuration / "provenance.json",
        _provenance(configuration),
    )
    for index in range(2):
        _write_json(
            duplicate_validation_root / configuration
            / f"validation-{index}.json",
            _validation(configuration),
        )
    finalize(duplicate_validation_root / configuration, "record fixture")
    duplicate_validation = aggregate(
        duplicate_validation_root, (configuration,))
    assert duplicate_validation["verdict"] == "failed"
    assert duplicate_validation["parallel_runtime_contract_failures"] == [{
        "configuration": configuration,
        "error": "expected exactly one installed-wheel validation, got 2",
    }]

    duplicate_provenance_root = tmp_path / "duplicate-provenance"
    validation_path = (
        duplicate_provenance_root / configuration / "validation.json"
    )
    _write_json(
        validation_path,
        _validation(configuration),
    )
    for index in range(2):
        _write_json(
            duplicate_provenance_root / configuration
            / f"provenance-{index}.json",
            _provenance(
                configuration,
                validation_sha256=_sha256(validation_path),
            ),
        )
    finalize(duplicate_provenance_root / configuration, "record fixture")
    duplicate_provenance = aggregate(
        duplicate_provenance_root, (configuration,))
    assert duplicate_provenance["verdict"] == "failed"
    assert duplicate_provenance[
        "parallel_runtime_provenance_failures"
    ] == [{
        "configuration": configuration,
        "error": "expected exactly one provenance record, got 2",
    }]


def test_release_metadata_rejects_substituted_wheel_with_same_extension(
        tmp_path):
    configuration = "wheel-windows-msvc-amd64-cp312"
    original_dir = tmp_path / "original"
    substitute_dir = tmp_path / "substitute"
    original_dir.mkdir()
    substitute_dir.mkdir()
    wheel_name = "candidate.whl"
    extension = "pyscarcopula/_native/_scar_cpp.fake.pyd"
    for wheel_dir, marker in (
            (original_dir, "original"),
            (substitute_dir, "substitute")):
        with zipfile.ZipFile(wheel_dir / wheel_name, "w") as archive:
            archive.writestr(extension, b"same-native-binary")
            archive.writestr(
                "pyscarcopula/__init__.py", f'MARKER = "{marker}"\n')

    validated_wheel = wheel_provenance(original_dir)[0]
    substituted_wheel = wheel_provenance(substitute_dir)[0]
    assert substituted_wheel["sha256"] != validated_wheel["sha256"]
    for field in ("file", "files", "extension", "extension_sha256"):
        assert substituted_wheel[field] == validated_wheel[field]
    validation_path = tmp_path / "validation.json"
    _write_json(
        validation_path,
        _validation(configuration, wheel=validated_wheel),
    )

    accepted = build_report(
        source_root=ROOT,
        configuration=configuration,
        status="passed",
        wheel_dir=original_dir,
        validation_file=validation_path,
        test_workers=4,
        build_jobs=4,
    )
    assert accepted["wheels"] == [validated_wheel]
    assert accepted["wheel_validation"]["wheel"] == validated_wheel
    assert accepted["wheel_validation"]["sha256"] == _sha256(
        validation_path)

    with pytest.raises(
            RuntimeError,
            match="does not match the release wheel archive"):
        build_report(
            source_root=ROOT,
            configuration=configuration,
            status="passed",
            wheel_dir=substitute_dir,
            validation_file=validation_path,
            test_workers=4,
            build_jobs=4,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda provenance: provenance["wheels"][0].update(
            sha256="e" * 64),
        lambda provenance: provenance["wheel_validation"].update(
            sha256="f" * 64),
    ],
    ids=("substituted-wheel", "validation-file-digest"),
)
def test_release_aggregator_rejects_broken_validation_provenance_link(
        tmp_path, mutation):
    configuration = "wheel-windows-msvc-amd64-cp312"
    artifacts = tmp_path / "artifacts"
    validation = _validation(configuration)
    validation_path = artifacts / configuration / "validation.json"
    _write_json(validation_path, validation)
    provenance = _provenance(
        configuration,
        wheel=validation["wheel"],
        validation_sha256=_sha256(validation_path),
    )
    mutation(provenance)
    _write_json(
        artifacts / configuration / "provenance.json",
        provenance,
    )
    finalize(artifacts / configuration, "validation link fixture")

    report = aggregate(artifacts, (configuration,))

    assert report["verdict"] == "failed"
    assert report["parallel_runtime_contract_failures"] == []
    assert report["parallel_runtime_provenance_failures"] == [{
        "configuration": configuration,
        "error": "runtime contract does not match wheel provenance",
    }]


@pytest.mark.parametrize("mutation, message", [
    (
        lambda record: record.update(record_type="foreign_record"),
        "installed wheel validation schema version 2",
    ),
    (
        lambda record: record.pop("parallel_runtime_contract"),
        "no parallel runtime contract",
    ),
    (
        lambda record: record["parallel_runtime_contract"].update(
            extension_sha256="d" * 64),
        "different native binary",
    ),
    (
        lambda record: record.update(configuration="foreign-configuration"),
        "configuration does not match",
    ),
])
def test_release_metadata_rejects_foreign_or_incomplete_validation(
        tmp_path, mutation, message):
    configuration = "wheel-windows-msvc-amd64-cp312"
    validation = _validation(configuration)
    mutation(validation)
    validation_path = tmp_path / "validation.json"
    _write_json(validation_path, validation)

    with pytest.raises(RuntimeError, match=message):
        build_report(
            source_root=ROOT,
            configuration=configuration,
            status="passed",
            wheel_dir=None,
            validation_file=validation_path,
            test_workers=4,
            build_jobs=4,
        )


def test_artifact_finalizer_detects_append_only_mutation(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "evidence.json").write_text('{"status":"passed"}\n')

    index = finalize(root, "release test producer")

    assert index["files"][0]["path"] == "evidence.json"
    assert verify(root) == index
    assert (root / "artifact_index.json").is_file()
    assert (root / "checksums.sha256").is_file()

    (root / "evidence.json").write_text('{"status":"changed"}\n')
    try:
        verify(root)
    except RuntimeError as error:
        assert "mismatch" in str(error)
    else:
        raise AssertionError("mutated finalized artifact was accepted")


def test_release_wheel_workflow_covers_every_supported_python_and_platform():
    workflow = (ROOT / ".github/workflows/wheels.yml").read_text(
        encoding="utf-8"
    )
    for version, tag in (
        ("3.10", "cp310"),
        ("3.11", "cp311"),
        ("3.12", "cp312"),
        ("3.13", "cp313"),
        ("3.14", "cp314"),
    ):
        assert workflow.count(f'version: "{version}"') >= 1
        assert workflow.count(f"tag: {tag}") == 1
    for platform_name in (
        "linux-gcc-x86_64",
        "windows-msvc-amd64",
        "macos-appleclang-x86_64",
        "macos-appleclang-arm64",
    ):
        assert workflow.count(f"name: {platform_name}") == 1

    assert "os: macos-15" in workflow
    assert "arch: arm64" in workflow
    assert "CIBW_TEST_COMMAND" in workflow
    assert "tools/validate_installed_wheel.py" in workflow
    assert "Validate wheel again in a fresh venv" in workflow
    assert "tools/write_release_metadata.py" in workflow
    assert "tools/finalize_release_artifacts.py" in workflow
    assert "--validation-file" in workflow
    assert "${{ github.workspace }}/build/ci/pyscarcopula-release" in workflow
    assert "--required-configuration" in workflow
    assert workflow.count("--required-configuration wheel-") == 20
    assert "--parallel 8" not in workflow
    assert "PYSCA_CPP_BUILD_JOBS=4" in workflow


def test_toolchain_and_sanitizer_workflow_has_complete_release_provenance():
    workflow = (
        ROOT / ".github/workflows/parallel-release-validation.yml"
    ).read_text(encoding="utf-8")
    for configuration in (
        "linux-gcc-py310",
        "linux-gcc-py314",
        "linux-clang-py312",
        "windows-msvc-py312",
        "windows-mingw64-py312",
        "macos-arm64-clang-py312",
        "linux-clang-asan-ubsan",
        "linux-gcc-tsan",
    ):
        assert f"--required-configuration {configuration}" in workflow
    assert "os: macos-15" in workflow
    assert "tools/validate_installed_wheel.py" in workflow
    assert "tools/finalize_release_artifacts.py" in workflow
    assert "--sanitize address-undefined" in workflow
    assert "--sanitize thread" in workflow
    assert "pip install -e" not in workflow
    assert workflow.count("Build and install external") == 4
    assert workflow.count("--import-mode=importlib") >= 5
    assert workflow.count("--validation-file") >= 5
    assert "libstdcxx_path" in workflow
    assert "${{ github.workspace }}/build/ci/pyscarcopula-release" in workflow
    assert "build/release" not in workflow


def test_installed_wheel_validator_covers_all_removal_surfaces():
    assert EXPECTED_NATIVE_IDS == (
        "Independent",
        "Clayton",
        "Frank",
        "Gumbel",
        "Joe",
        "BivariateGaussian",
        "Gaussian",
        "Student",
        "EquicorrGaussian",
        "StochasticStudent",
        "Vine",
    )
    assert {
        "pyscarcopula._scar_cpp",
        "pyscarcopula.copula._protocol",
        "pyscarcopula.numerical.tm_grid",
        "pyscarcopula.strategy.scar_mc",
        "pyscarcopula.vine.cvine",
    } <= set(REMOVED_MODULES)
    assert {"CVineCopula", "TMGrid", "CopulaProtocol"} <= set(
        REMOVED_PUBLIC_NAMES
    )
    assert {"SCAR-P-OU", "SCAR-M-OU"} <= set(REMOVED_METHODS)


def test_sdist_keeps_executable_wheel_and_ownership_gates():
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    for path in (
        "tools/validate_installed_wheel.py",
        "tools/check_python_ownership.py",
        "tools/python_ownership_policy.py",
        "tools/write_release_metadata.py",
        "tools/aggregate_release_validation.py",
        "tools/finalize_release_artifacts.py",
    ):
        assert f"include {path}" in manifest
