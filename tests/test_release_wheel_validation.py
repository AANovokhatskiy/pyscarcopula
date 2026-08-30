"""Permanent FV-7 wheel, provenance, and matrix contracts."""

from __future__ import annotations

import json
from pathlib import Path
import re
import zipfile

from tools.aggregate_release_validation import aggregate
from tools.finalize_release_artifacts import finalize, verify
from tools.validate_installed_wheel import (
    EXPECTED_NATIVE_IDS,
    REMOVED_METHODS,
    REMOVED_MODULES,
    REMOVED_PUBLIC_NAMES,
)
from tools.write_release_metadata import (
    source_provenance,
    wheel_provenance,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _provenance(configuration: str, digest: str = "source-digest") -> dict:
    is_wheel = configuration.startswith("wheel-") or bool(
        re.search(r"-py\d{3}$", configuration)
    )
    return {
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
            [{"file": "candidate.whl", "extension_sha256": "binary"}]
            if is_wheel
            else []
        ),
        "wheel_validation": (
            {"status": "passed"}
            if is_wheel
            else None
        ),
    }


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
    for configuration in required:
        _write_json(
            artifacts / configuration / "provenance.json",
            _provenance(configuration),
        )
    for configuration in required[:2]:
        _write_json(
            artifacts / configuration / "validation.json",
            {
                "record_type": "installed_wheel_validation",
                "status": "passed",
                "configuration": configuration,
            },
        )
    for configuration in required:
        finalize(artifacts / configuration, "FV-7 unit test")

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


def test_artifact_finalizer_detects_append_only_mutation(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "evidence.json").write_text('{"status":"passed"}\n')

    index = finalize(root, "FV-7 test producer")

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
    assert "${{ runner.temp }}/pyscarcopula-fv7" in workflow
    assert "--required-configuration" in workflow
    assert workflow.count("--required-configuration wheel-") == 20
    assert "--parallel 8" not in workflow
    assert "PYSCA_CPP_BUILD_JOBS=4" in workflow


def test_toolchain_and_sanitizer_workflow_has_complete_gate4_provenance():
    workflow = (
        ROOT / ".github/workflows/parallel-release-gates.yml"
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
    assert "${{ runner.temp }}/pyscarcopula-fv7" in workflow
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
