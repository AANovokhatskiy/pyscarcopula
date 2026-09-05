"""Workspace-local temporary files, caches, and installed-wheel isolation."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools import run_in_workspace as workspace
from tools import validate_installed_wheel as wheel_validator


def test_child_process_uses_workspace_temp_and_cache_even_with_global_defaults(
        tmp_path, monkeypatch):
    outside = tmp_path / "profile"
    outside.mkdir()
    root = tmp_path / "checkout"
    root.mkdir()
    for key in ("TMP", "TEMP", "TMPDIR", *workspace.CACHE_VARIABLES):
        monkeypatch.setenv(key, str(outside))
    environment, run = workspace.command_environment(root, "first")
    code = """
import json, os, pathlib, sys, tempfile
with tempfile.NamedTemporaryFile(delete=False) as f:
    f.write(b'workspace')
    temporary = f.name
import importlib.util
cache = importlib.util.cache_from_source(os.__file__)
print(json.dumps({'temporary': temporary, 'cache': cache,
                  'prefix': sys.pycache_prefix}))
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=root,
                            env=environment, text=True, capture_output=True,
                            check=True, timeout=30)
    record = json.loads(result.stdout)
    assert Path(record["temporary"]).is_relative_to(run)
    assert Path(record["temporary"]).read_bytes() == b"workspace"
    for key in ("cache", "prefix"):
        assert Path(record[key]).is_relative_to(root / "build" / "cache")
    for key in workspace.CACHE_VARIABLES:
        assert Path(environment[key]).is_relative_to(root / "build" / "cache")
    assert list(outside.iterdir()) == []
    _, other = workspace.command_environment(root, "second")
    assert other != run


def test_launcher_uses_selected_local_cwd_and_records_failed_command(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "ROOT", tmp_path)
    assert workspace.main([
        "--cwd", "build/isolated", "--", "-c",
        "from pathlib import Path; Path('output.txt').write_text('local'); raise SystemExit(3)",
    ]) == 3
    assert (tmp_path / "build/isolated/output.txt").read_text() == "local"
    record, = (tmp_path / "build/workspace-runs").glob("*/command.json")
    assert json.loads(record.read_text())["returncode"] == 3


def test_launcher_rejects_working_directory_outside_workspace(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        workspace.local_directory(tmp_path, "../outside")


def test_plain_pytest_creates_temp_parent_on_a_fresh_checkout(tmp_path, monkeypatch):
    import conftest
    monkeypatch.setattr(conftest, "_PROJECT_ROOT", tmp_path)
    config = SimpleNamespace(option=SimpleNamespace(basetemp=None))
    conftest.pytest_configure(config)
    path = Path(config.option.basetemp)
    assert path.parent == tmp_path / "build/pytest"
    # pytest creates the last path component without parents=True.
    path.mkdir()
    assert path.is_dir()


def test_container_preparation_does_not_replace_existing_files(tmp_path):
    from tools.prepare_wheel_container import prepare
    container = tmp_path / "container"
    path = container / "tmp/cibuildwheel"
    path.parent.mkdir(parents=True)
    path.write_text("existing")
    with pytest.raises(RuntimeError, match="refusing to replace"):
        prepare(tmp_path / "project", container)
    assert path.read_text() == "existing"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux wheel container links")
def test_container_outputs_resolve_inside_project_and_preparation_is_idempotent(tmp_path):
    from tools.prepare_wheel_container import prepare
    project = tmp_path / "project"
    container = tmp_path / "container"
    prepare(project, container)
    prepare(project, container)
    for relative in ("tmp/cibuildwheel", "output", "constraints.txt"):
        assert (container / relative).resolve().is_relative_to(project / "build")


def test_github_environment_exports_workspace_paths_without_changing_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "ROOT", tmp_path)
    destination = tmp_path / "github-env"
    monkeypatch.setenv("GITHUB_ENV", str(destination))
    monkeypatch.setenv("TEMP", "parent-temp")
    assert workspace.main(["--github-env"]) == 0
    exported = dict(line.split("=", 1) for line in destination.read_text().splitlines())
    for key in ("TMP", "TEMP", "TMPDIR", *workspace.CACHE_VARIABLES,
                "PYSCA_WORKSPACE_TEMP"):
        assert Path(exported[key]).is_relative_to(tmp_path / "build")
        assert Path(exported[key]).is_dir()
    import os
    assert os.environ["TEMP"] == "parent-temp"


@pytest.mark.parametrize("origin", ["installed", "source", "different-distribution"])
def test_workspace_venv_imports_are_accepted_only_from_the_installed_distribution(
        tmp_path, monkeypatch, origin):
    import pyscarcopula
    from pyscarcopula._native import _extension

    root = tmp_path / "checkout"
    installed = root / "build/venv/site-packages/pyscarcopula"
    origins = {"installed": installed, "source": root / "pyscarcopula",
               "different-distribution": tmp_path / "other/pyscarcopula"}
    package = origins[origin]
    distribution = SimpleNamespace(
        version="test", read_text=lambda name: "Wheel-Version: 1.0",
        locate_file=lambda name: installed,
    )
    monkeypatch.setattr(wheel_validator.metadata, "distribution", lambda name: distribution)
    monkeypatch.setattr(pyscarcopula, "__file__", str(package / "__init__.py"))
    monkeypatch.setattr(_extension, "load", lambda: SimpleNamespace(
        __file__=str(package / "_native/_scar_cpp.pyd")))
    if origin == "installed":
        assert Path(wheel_validator._assert_wheel_import(root)["package"]).is_relative_to(installed)
    else:
        with pytest.raises(RuntimeError, match="source-tree leakage"):
            wheel_validator._assert_wheel_import(root)


def test_python_ci_jobs_initialize_workspace_before_installing_dependencies():
    root = Path(__file__).resolve().parents[1]
    for path in (root / ".github/workflows").glob("*.yml"):
        text = path.read_text(encoding="utf-8")
        if "actions/setup-python@" not in text:
            continue
        assert "runner.temp" not in text and "RUNNER_TEMP" not in text, path
        steps = text.split("uses: actions/checkout@")[1:]
        for job in steps:
            if "actions/setup-python@" in job:
                assert job.index("--github-env") < job.index("actions/setup-python@"), path
                assert job.index("actions/setup-python@") < job.index("--github-venv"), path
                if "pip install" in job:
                    assert job.index("--github-venv") < job.index("pip install"), path
