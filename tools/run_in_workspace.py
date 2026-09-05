"""Run a Python development command with temporary files and caches in build/."""

from __future__ import annotations

import sys

# The launcher itself must not create bytecode beside an external interpreter.
if __name__ == "__main__":
    sys.dont_write_bytecode = True

import argparse
import json
import os
from pathlib import Path
import subprocess
import uuid


ROOT = Path(__file__).resolve().parents[1]
CACHE_VARIABLES = {
    "PIP_CACHE_DIR": "pip",
    "UV_CACHE_DIR": "uv",
    "XDG_CACHE_HOME": "xdg",
    "PYTHONPYCACHEPREFIX": "python",
    "NUMBA_CACHE_DIR": "numba",
    "JOBLIB_TEMP_FOLDER": "joblib",
    "MPLCONFIGDIR": "matplotlib",
    "MYPY_CACHE_DIR": "mypy",
    "HYPOTHESIS_STORAGE_DIRECTORY": "hypothesis",
    "CCACHE_DIR": "ccache",
    "SCCACHE_DIR": "sccache",
    "CIBW_CACHE_PATH": "cibuildwheel",
    "PLAYWRIGHT_BROWSERS_PATH": "playwright",
    "VIRTUALENV_OVERRIDE_APP_DATA": "virtualenv",
}


def local_directory(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"workspace path escapes through a link: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def command_environment(root: Path, run_id: str):
    """Override inherited global caches before Python/pip/compiler startup."""
    environment = os.environ.copy()
    run = local_directory(root, f"build/workspace-runs/{run_id}")
    # multiprocessing appends /pymp-XXXXXXXX/listener-XXXXXXXX. Keep TMPDIR
    # short enough for macOS's 104-byte Unix socket address on CI checkouts.
    # Command records can retain their descriptive path independently.
    temporary = local_directory(root, f"build/t/{run_id}")
    for name in ("TMP", "TEMP", "TMPDIR"):
        environment[name] = str(temporary)
    for name, directory in CACHE_VARIABLES.items():
        environment[name] = str(local_directory(root, f"build/cache/{directory}"))
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    return environment, run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", default=".", help="working directory relative to the repository")
    parser.add_argument("--timeout", type=float, default=900, help="command timeout in seconds")
    parser.add_argument("--github-env", action="store_true",
                        help="export local temp/cache paths to subsequent GitHub Actions steps")
    parser.add_argument("--github-venv", action="store_true",
                        help="create a workspace venv and use it in subsequent GitHub Actions steps")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="Python arguments after --, e.g. -- -m pytest -q")
    args = parser.parse_args(argv)
    if args.github_venv:
        if not os.environ.get("GITHUB_PATH") or not os.environ.get("GITHUB_ENV"):
            parser.error("--github-venv requires GitHub Actions environment files")
        environment, _ = command_environment(ROOT, "ci")
        venv = local_directory(ROOT, "build/venv")
        subprocess.run([sys.executable, "-m", "venv", str(venv)],
                       env=environment, check=True, timeout=args.timeout)
        scripts = venv / ("Scripts" if os.name == "nt" else "bin")
        with open(os.environ["GITHUB_PATH"], "a", encoding="utf-8") as output:
            output.write(f"{scripts}\n")
        with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as output:
            output.write(f"VIRTUAL_ENV={venv}\n")
        return 0
    if args.github_env:
        if not os.environ.get("GITHUB_ENV"):
            parser.error("--github-env requires GitHub Actions GITHUB_ENV")
        environment, _ = command_environment(ROOT, "ci")
        environment["PYSCA_WORKSPACE_TEMP"] = str(local_directory(ROOT, "build/ci"))
        keys = ("TMP", "TEMP", "TMPDIR", *CACHE_VARIABLES,
                "PIP_DISABLE_PIP_VERSION_CHECK", "PYTHONNOUSERSITE", "PYSCA_WORKSPACE_TEMP")
        with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as output:
            for key in keys:
                output.write(f"{key}={environment[key]}\n")
        return 0
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("provide Python arguments after --")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    cwd = local_directory(ROOT, args.cwd)
    environment, run = command_environment(ROOT, uuid.uuid4().hex[:12])
    if command[:2] == ["-m", "pytest"]:
        if not any(arg == "--basetemp" or arg.startswith("--basetemp=") for arg in command):
            command += [f"--basetemp={run / 'pytest'}"]
        command += ["-o", f"cache_dir={ROOT / 'build' / 'cache' / 'pytest'}"]
    invocation = [sys.executable, *command]
    record = {
        "cwd": str(cwd), "command": invocation,
        "environment": {key: environment[key] for key in (
            "TMP", "TEMP", "TMPDIR", *CACHE_VARIABLES,
            "PIP_DISABLE_PIP_VERSION_CHECK", "PYTHONNOUSERSITE")},
    }
    print(f"Workspace run: {run}", flush=True)
    try:
        result = subprocess.run(invocation, cwd=cwd, env=environment, timeout=args.timeout)
        record["returncode"] = result.returncode
        return result.returncode
    except subprocess.TimeoutExpired:
        record["timeout_seconds"] = args.timeout
        raise
    finally:
        (run / "command.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
