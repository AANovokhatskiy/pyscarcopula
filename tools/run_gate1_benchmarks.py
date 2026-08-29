"""Capture and compare the permanent Gate 1 native benchmark matrix.

The manifest owns workload identity.  This driver owns deterministic fixture
construction, calibration, paired timing samples, memory probes, checksums,
and host/toolchain metadata.  It deliberately times the
existing Python/native boundary because the refactor must preserve that
boundary until its compatibility inventory says otherwise.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import importlib.util
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "benchmarks" / "gate1_manifest_v2.json"
TESTS = ROOT / "tests"


def _artifact_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Resolve append-only outputs and reject product-checkout artifacts."""

    artifact_root = args.artifact_root.resolve()
    if artifact_root == ROOT or ROOT in artifact_root.parents:
        raise SystemExit(
            "--artifact-root must be outside the product repository")
    artifact_root.mkdir(parents=True, exist_ok=True)

    def resolve_output(value: Path | None, default_name: str) -> Path:
        path = Path(default_name) if value is None else value
        if not path.is_absolute():
            path = artifact_root / path
        path = path.resolve()
        if artifact_root != path.parent and artifact_root not in path.parents:
            raise SystemExit("benchmark outputs must stay below --artifact-root")
        if path.exists():
            raise SystemExit(f"refusing to overwrite append-only artifact: {path}")
        return path

    output = resolve_output(args.output, "gate1_candidate.json")
    summary = resolve_output(args.summary, "performance_summary.md")
    return artifact_root, output, summary


@dataclass(frozen=True)
class PreparedCase:
    call: Callable[[], Any]
    diagnostics: dict[str, Any]


def _run_text(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _git_value(*args: str) -> str | None:
    return _run_text(["git", *args])


def _windows_physical_cpu_ids() -> list[int]:
    """Return one group-0 logical CPU id for every Windows physical core."""

    if sys.platform != "win32":
        return []

    class GroupAffinity(ctypes.Structure):
        _fields_ = [
            ("mask", ctypes.c_size_t),
            ("group", ctypes.c_ushort),
            ("reserved", ctypes.c_ushort * 3),
        ]

    class ProcessorRelationship(ctypes.Structure):
        _fields_ = [
            ("flags", ctypes.c_ubyte),
            ("efficiency_class", ctypes.c_ubyte),
            ("reserved", ctypes.c_ubyte * 20),
            ("group_count", ctypes.c_ushort),
            ("group_mask", GroupAffinity * 1),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_information = kernel32.GetLogicalProcessorInformationEx
    get_information.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    get_information.restype = ctypes.c_int
    relation_processor_core = 0
    required = ctypes.c_ulong(0)
    get_information(relation_processor_core, None, ctypes.byref(required))
    if required.value == 0:
        return []
    buffer = ctypes.create_string_buffer(required.value)
    if not get_information(
            relation_processor_core, buffer, ctypes.byref(required)):
        return []

    result = []
    offset = 0
    while offset + 8 <= required.value:
        relationship = ctypes.c_uint32.from_buffer(buffer, offset).value
        size = ctypes.c_uint32.from_buffer(buffer, offset + 4).value
        if size < 8 or offset + size > required.value:
            return []
        if relationship == relation_processor_core:
            processor = ProcessorRelationship.from_buffer_copy(
                buffer.raw[offset + 8:offset + size])
            affinity = processor.group_mask[0]
            mask = int(affinity.mask)
            if processor.group_count == 1 and affinity.group == 0 and mask:
                result.append((mask & -mask).bit_length() - 1)
        offset += size
    return sorted(set(result))


def _physical_core_count() -> int:
    if sys.platform == "win32":
        physical_cpu_ids = _windows_physical_cpu_ids()
        if physical_cpu_ids:
            return len(physical_cpu_ids)
        value = _run_text([
            "powershell.exe",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_Processor | "
            "Measure-Object NumberOfCores -Sum).Sum",
        ])
        if value:
            try:
                return max(1, int(value.splitlines()[-1]))
            except ValueError:
                pass
    elif sys.platform == "darwin":
        value = _run_text(["sysctl", "-n", "hw.physicalcpu"])
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                pass
    elif sys.platform.startswith("linux"):
        cpuinfo = Path("/proc/cpuinfo")
        try:
            records = cpuinfo.read_text(encoding="utf-8").split("\n\n")
            cores = set()
            for record in records:
                fields = {}
                for line in record.splitlines():
                    if ":" in line:
                        key, value = line.split(":", 1)
                        fields[key.strip()] = value.strip()
                if "physical id" in fields and "core id" in fields:
                    cores.add((fields["physical id"], fields["core id"]))
            if cores:
                return len(cores)
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def _resolved_threads(value: int | str, physical: int) -> int:
    if value == "physical":
        return max(1, min(256, physical))
    return int(value)


def _required_thread_capacity(cases: list[dict[str, Any]]) -> int:
    """Return the minimum CPU pool needed to exercise declared scaling."""

    return max(
        (int(case["n_threads"]) for case in cases
         if isinstance(case["n_threads"], int)),
        default=1,
    )


def _parse_cpu_set(value: str, logical: int) -> list[int]:
    if value == "physical" or value.startswith("physical:"):
        physical_cpu_ids = _windows_physical_cpu_ids()
        if not physical_cpu_ids:
            raise SystemExit(
                "--cpu-set physical requires Windows physical-core topology")
        if value == "physical":
            requested = len(physical_cpu_ids)
        else:
            try:
                requested = int(value.split(":", 1)[1])
            except ValueError as exc:
                raise SystemExit(
                    "--cpu-set physical:N requires a positive integer N"
                ) from exc
        if requested < 1 or requested > len(physical_cpu_ids):
            raise SystemExit(
                "--cpu-set physical:N must be between 1 and the detected "
                f"physical-core count ({len(physical_cpu_ids)})"
            )
        return physical_cpu_ids[:requested]

    try:
        cpu_pool = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise SystemExit(
            "--cpu-set must contain comma-separated integers or physical:N"
        ) from exc
    if (
            not cpu_pool
            or len(cpu_pool) != len(set(cpu_pool))
            or any(cpu < 0 or cpu >= logical for cpu in cpu_pool)):
        raise SystemExit(
            f"--cpu-set must contain unique values in [0, {logical - 1}]")
    return cpu_pool


def _peak_rss_bytes() -> int | None:
    if sys.platform == "win32":
        class MemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(MemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        if psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(),
                ctypes.byref(counters),
                counters.cb):
            return int(counters.PeakWorkingSetSize)
        return None

    try:
        import resource
    except ImportError:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _compute_source_paths() -> list[Path]:
    """Return the canonical Python-free computational source boundary."""

    cpp_root = ROOT / "pyscarcopula" / "_cpp"
    manifest_path = cpp_root / "build_support" / "sources.py"
    spec = importlib.util.spec_from_file_location(
        "_pyscarcopula_benchmark_source_manifest", manifest_path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"cannot load canonical source manifest from {manifest_path}")
    manifest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(manifest)

    source_root = cpp_root / "src"
    compute_sources = [
        source_root / relative
        for relative in manifest.SCAR_COMPUTE_SOURCES
    ]
    compute_headers = [
        *cpp_root.joinpath("include").rglob("*.hpp"),
        *(
            path for path in source_root.rglob("*.hpp")
            if "bindings" not in path.relative_to(source_root).parts
        ),
    ]
    return sorted(set([*compute_sources, *compute_headers]))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_digest() -> str:
    digest = hashlib.sha256()
    for path in _compute_source_paths():
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _extension_compiler_identity(extension: Any) -> str:
    identity = getattr(extension, "__cpp_compiler__", None)
    if identity is not None:
        return str(identity)
    return platform.python_compiler()


def _optimization_flags(compiler_identity: str) -> str:
    if compiler_identity.startswith("MSVC "):
        return "/O2 /DNDEBUG (MSVC setup.py release defaults)"
    if sys.platform == "win32" and compiler_identity.startswith("GCC "):
        return "-O2 -DNDEBUG (MinGW setup.py release defaults)"
    return "setuptools/pybind11 release defaults"


def _stabilize_process_priority() -> dict[str, Any]:
    """Reduce scheduler interference without changing clocks or affinity."""
    if sys.platform != "win32":
        return {
            "requested": "unchanged",
            "applied": True,
            "detail": "process priority and affinity unchanged",
        }
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel32.SetPriorityClass.restype = ctypes.c_int
    # ABOVE_NORMAL_PRIORITY_CLASS avoids starving interactive/system work while
    # reducing the long scheduler stalls that invalidate timing samples.
    above_normal = 0x00008000
    applied = bool(kernel32.SetPriorityClass(
        kernel32.GetCurrentProcess(), above_normal))
    return {
        "requested": "ABOVE_NORMAL_PRIORITY_CLASS",
        "applied": applied,
        "detail": (
            "process priority only; CPU affinity, clocks, and power policy "
            "unchanged"
        ),
    }


def _set_process_affinity(
        n_threads: int, cpu_pool: list[int] | None) -> dict[str, Any]:
    logical = max(1, os.cpu_count() or 1)
    available = list(range(logical)) if cpu_pool is None else list(cpu_pool)
    cpu_count = max(1, min(int(n_threads), len(available)))
    cpus = available[:cpu_count]
    if sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = ctypes.c_int
        mask = sum(1 << cpu for cpu in cpus)
        applied = bool(kernel32.SetProcessAffinityMask(
            kernel32.GetCurrentProcess(), mask))
        return {"requested_cpus": cpus, "applied": applied}
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, set(cpus))
            return {"requested_cpus": cpus, "applied": True}
        except OSError:
            pass
    return {
        "requested_cpus": cpus,
        "applied": False,
        "reason": "process affinity API unavailable",
    }


def _metadata(
        physical_cores: int,
        command: str,
        process_stabilization: dict[str, Any],
        benchmark_cpu_pool: list[int] | None,
        reference_label: str | None) -> dict[str, Any]:
    packages = {}
    for package in ("numpy", "scipy", "pybind11", "setuptools", "pyscarcopula"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    from pyscarcopula._native import _extension

    extension = _extension.load()

    status = _git_value("status", "--short")
    compute_paths = _compute_source_paths()
    compute_status = _run_text([
        "git", "status", "--short", "--untracked-files=no", "--",
        *(path.relative_to(ROOT).as_posix() for path in compute_paths),
    ])
    extension_path = Path(extension.__file__).resolve()
    newest_source_mtime_ns = max(
        path.stat().st_mtime_ns for path in compute_paths)
    extension_mtime_ns = extension_path.stat().st_mtime_ns
    compiler_identity = _extension_compiler_identity(extension)
    return {
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_commit_time": _git_value("show", "-s", "--format=%cI", "HEAD"),
        "worktree_dirty": bool(status),
        "compute_worktree_dirty": bool(compute_status),
        "compute_source_sha256": _source_digest(),
        "python": sys.version,
        "python_executable": sys.executable,
        "python_compiler": platform.python_compiler(),
        "extension_path": str(extension_path),
        "extension_sha256": _file_sha256(extension_path),
        "build_freshness": {
            "newest_compute_source_mtime_ns": newest_source_mtime_ns,
            "extension_mtime_ns": extension_mtime_ns,
            "extension_not_older_than_compute_sources": (
                extension_mtime_ns >= newest_source_mtime_ns),
        },
        "compiler_identity": compiler_identity,
        "target_architecture": platform.machine(),
        "platform": platform.platform(),
        "cpu_model": (
            platform.processor()
            or os.environ.get("PROCESSOR_IDENTIFIER")
            or "unknown"
        ),
        "logical_cores": os.cpu_count(),
        "physical_cores": physical_cores,
        "benchmark_cpu_pool": benchmark_cpu_pool,
        "reference_label": reference_label,
        "benchmark_accessible_cores": (
            physical_cores
            if benchmark_cpu_pool is None else len(benchmark_cpu_pool)
        ),
        "build_type": "release extension already present in source tree",
        "cxx_standard": "C++17",
        "optimization_flags": _optimization_flags(compiler_identity),
        "fp_flags": "default; no fast-math requested by setup.py",
        "dependencies": packages,
        "benchmark_command": command,
        "process_stabilization": process_stabilization,
        "native_allocation_probe": {
            "available": False,
            "reason": (
                "The baseline records Python-boundary allocations with tracemalloc; "
                "the existing native allocation_probe.c is an LD_PRELOAD CI "
                "probe and is unavailable on this Windows reference host."
            ),
        },
    }


def _hash_value(digest: Any, value: Any) -> None:
    if is_dataclass(value) and not isinstance(value, type):
        digest.update(b"dataclass\0")
        for field in fields(value):
            _hash_value(digest, field.name)
            _hash_value(digest, getattr(value, field.name))
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"array\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
        return
    if isinstance(value, np.generic):
        _hash_value(digest, value.item())
        return
    if isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=str):
            _hash_value(digest, str(key))
            _hash_value(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        digest.update(b"sequence\0")
        for item in value:
            _hash_value(digest, item)
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError(f"benchmark returned non-finite scalar {value}")
        digest.update(value.hex().encode("ascii"))
        return
    digest.update(repr(value).encode("utf-8"))


def _checksum(value: Any) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, value)
    return digest.hexdigest()


def _equicorrelation(dimension: int, rho: float = 0.15) -> np.ndarray:
    correlation = np.full((dimension, dimension), rho, dtype=np.float64)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def _loadings(dimension: int, rank: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, 0.12, size=(dimension, rank))
    row_norm = np.linalg.norm(values, axis=1)
    scale = np.maximum(1.0, row_norm / 0.65)
    return np.ascontiguousarray(values / scale[:, None])


def _pair(case: dict[str, Any]):
    from pyscarcopula import (
        BivariateGaussianCopula,
        ClaytonCopula,
        FrankCopula,
        GumbelCopula,
        IndependentCopula,
        JoeCopula,
    )

    factories = {
        "independent": IndependentCopula,
        "clayton": ClaytonCopula,
        "frank": FrankCopula,
        "gumbel": GumbelCopula,
        "joe": JoeCopula,
        "gaussian": BivariateGaussianCopula,
    }
    family = case["family"]
    kwargs: dict[str, Any] = {"rotate": int(case["rotation"])}
    if family in {"clayton", "frank", "gumbel", "joe"}:
        kwargs["transform_type"] = case["transform"]
    return factories[family](**kwargs)


def _pair_parameter(family: str) -> float:
    return {
        "independent": 0.0,
        "clayton": 0.8,
        "frank": 2.0,
        "gumbel": 1.5,
        "joe": 1.6,
        "gaussian": -0.35,
    }[family]


def _prepare_pair_grid(case: dict[str, Any], _: int) -> PreparedCase:
    copula = _pair(case)
    shape = case["shape"]
    rng = np.random.default_rng(case["seed"])
    observations = rng.uniform(0.01, 0.99, size=(shape["n_obs"], 2))
    grid = np.linspace(-2.0, 2.0, shape["n_grid"], dtype=np.float64)
    quantiles = np.linspace(0.01, 0.99, shape["n_obs"], dtype=np.float64)
    parameter = _pair_parameter(case["family"])

    def call():
        pdf, gradient = copula.pdf_and_grad_on_grid_batch(observations, grid)
        first_h, second_h = copula.h_pair(
            observations[:, 0], observations[:, 1], parameter)
        inverse = copula.h_inverse(
            quantiles, observations[:, 1], parameter)
        return pdf, gradient, first_h, second_h, inverse

    return PreparedCase(call, {"parameter": parameter})


def _prepare_transform(case: dict[str, Any], _: int) -> PreparedCase:
    copula = _pair(case)
    values = np.linspace(
        -3.0, 3.0, case["shape"]["n_values"], dtype=np.float64)

    def call():
        transformed = copula.transform(values)
        return transformed, copula.dtransform(values), copula.inv_transform(
            transformed)

    return PreparedCase(call, {})


def _prepare_static_dense_gaussian(
        case: dict[str, Any], threads: int) -> PreparedCase:
    from pyscarcopula._native import static as static_likelihood

    dimension = int(case["dimension"])
    observations = np.random.default_rng(case["seed"]).uniform(
        0.01, 0.99, size=(case["shape"]["n_obs"], dimension))
    evaluator = static_likelihood.prepare_gaussian(
        _equicorrelation(dimension), observations, n_threads=threads)
    return PreparedCase(
        lambda: evaluator.log_likelihood(0.0),
        {"prepared_evaluator": True},
    )


def _prepare_static_factor_gaussian(
        case: dict[str, Any], threads: int) -> PreparedCase:
    from pyscarcopula import GaussianCopula
    from pyscarcopula._native import static as static_likelihood

    dimension = int(case["dimension"])
    rank = int(case["shape"]["rank"])
    observations = np.random.default_rng(case["seed"]).uniform(
        0.01, 0.99, size=(case["shape"]["n_obs"], dimension))
    copula = GaussianCopula(
        d=dimension,
        corr_mode="factor",
        factor_rank=rank,
        factor_loadings=_loadings(dimension, rank, case["seed"] + 1),
    )
    evaluator = static_likelihood.prepare(
        copula, observations, n_threads=threads)
    return PreparedCase(
        lambda: evaluator.log_likelihood(0.0),
        {
            "prepared_evaluator": True,
            "factor_workspace_bytes": (
                dimension * rank + dimension + rank * rank) * 8,
        },
    )


def _prepare_equicorr_grid(
        case: dict[str, Any], threads: int) -> PreparedCase:
    from pyscarcopula import EquicorrGaussianCopula
    from pyscarcopula._native import multivariate as multivariate_native

    dimension = int(case["dimension"])
    observations = np.random.default_rng(case["seed"]).uniform(
        0.01, 0.99, size=(case["shape"]["n_obs"], dimension))
    grid = np.linspace(-2.0, 2.0, case["shape"]["n_grid"])
    copula = EquicorrGaussianCopula(dimension)

    def call():
        return multivariate_native.pdf_and_grad_grid_info(
            copula, observations, grid, n_threads=threads)

    first = call()
    return PreparedCase(call, first[2])


def _prepare_static_estimated_gaussian(
        case: dict[str, Any], threads: int) -> PreparedCase:
    from pyscarcopula._native import multivariate as multivariate_native
    from pyscarcopula._native import static as static_likelihood

    dimension = int(case["dimension"])
    observations = np.random.default_rng(case["seed"]).uniform(
        0.01, 0.99, size=(case["shape"]["n_obs"], dimension))
    evaluator = static_likelihood.prepare_gaussian(
        np.eye(dimension), observations, n_threads=threads)
    mode = case["correlation_mode"]
    target = _equicorrelation(dimension, 0.15)
    if mode == "shrinkage":
        base = _equicorrelation(dimension, 0.3)
        parameters = np.array([0.4], dtype=np.float64)

        def call():
            correlation = multivariate_native.make_shrinkage_correlation(
                parameters[0], base)
            value, gradient = evaluator.gaussian_objective_and_gradient(
                correlation)
            raw_gradient = multivariate_native.correlation_gradient_to_raw(
                mode, parameters, correlation, gradient, base=base)
            return value, raw_gradient, correlation
    elif mode == "cholesky":
        parameters = multivariate_native.pack_cholesky_correlation(target)

        def call():
            correlation = multivariate_native.unpack_cholesky_correlation(
                parameters, dimension)
            value, gradient = evaluator.gaussian_objective_and_gradient(
                correlation)
            raw_gradient = multivariate_native.correlation_gradient_to_raw(
                mode, parameters, correlation, gradient)
            return value, raw_gradient, correlation
    else:
        raise ValueError(f"unknown estimated Gaussian mode: {mode}")
    return PreparedCase(call, {
        "correlation_mode": mode,
        "parameter_count": len(parameters),
        "prepared_observations": True,
    })


def _prepare_student_grid(
        case: dict[str, Any], threads: int, *, factor: bool) -> PreparedCase:
    from pyscarcopula import (
        FactorCorrelation,
        FactorStudentEvaluator,
        StochasticStudentCopula,
    )
    from pyscarcopula._native import multivariate as multivariate_native

    dimension = int(case["dimension"])
    shape = case["shape"]
    observations = np.random.default_rng(case["seed"]).uniform(
        0.01, 0.99, size=(shape["n_obs"], dimension))
    if factor:
        rank = int(shape["rank"])
        evaluator = FactorStudentEvaluator(
            FactorCorrelation(
                _loadings(dimension, rank, case["seed"] + 1)),
            observations,
        )
        grid = np.linspace(3.0, 15.0, shape["n_grid"])

        def call():
            result = evaluator.evaluate_grid(
                grid,
                dimension_tile=64,
                n_threads=threads,
            )
            return result.log_pdf, result.dlog_ddf, dict(result.diagnostics)
    else:
        grid = np.linspace(-1.5, 2.5, shape["n_grid"])
        copula = StochasticStudentCopula(
            d=dimension, R=_equicorrelation(dimension))
        cache = copula.prepare_emission_cache(observations)

        def call():
            return multivariate_native.pdf_and_grad_grid_info(
                copula,
                observations,
                grid,
                cache=cache,
                n_threads=threads,
            )

    first = call()
    return PreparedCase(call, first[2])


def _prepare_student_dense_grid(
        case: dict[str, Any], threads: int) -> PreparedCase:
    return _prepare_student_grid(case, threads, factor=False)


def _prepare_student_factor_grid(
        case: dict[str, Any], threads: int) -> PreparedCase:
    return _prepare_student_grid(case, threads, factor=True)


def _student_ppf_fixture(case: dict[str, Any]):
    from pyscarcopula._native import multivariate as multivariate_native

    observations = np.random.default_rng(case["seed"]).uniform(
        1e-7, 1.0 - 1e-7,
        size=(case["shape"]["n_obs"], int(case["dimension"])),
    )
    options = {
        "df_hi": 50.0,
        "n_boundary": 2,
        "n_lo": 8,
        "n_hi": 8,
        "max_table_bytes": 512 * 1024 * 1024,
    }
    return multivariate_native, observations, options


def _prepare_student_ppf_cold(
        case: dict[str, Any], _: int) -> PreparedCase:
    multivariate_native, observations, options = _student_ppf_fixture(case)

    def call():
        return multivariate_native.prepare_student_ppf_table(
            observations, **options)

    first = call()
    return PreparedCase(call, {
        "node_count": len(first[1]),
        "has_table": first[2] is not None,
        "preparation_in_timed_call": True,
    })


def _prepare_student_ppf_prepared(
        case: dict[str, Any], _: int) -> PreparedCase:
    multivariate_native, observations, options = _student_ppf_fixture(case)
    clipped, nodes, table = multivariate_native.prepare_student_ppf_table(
        observations, **options)

    def call():
        return multivariate_native.evaluate_student_ppf_table(
            clipped, nodes, table, 7.25)

    return PreparedCase(call, {
        "node_count": len(nodes),
        "has_table": table is not None,
        "prepared_table_reused": True,
    })


def _prepare_correlation_preprocess(
        case: dict[str, Any], _: int) -> PreparedCase:
    from pyscarcopula._native import multivariate as multivariate_native

    dimension = int(case["dimension"])
    matrix = _equicorrelation(dimension, 0.2)
    matrix[0, 1] = matrix[1, 0] = 0.95

    def call():
        return multivariate_native.preprocess_correlation(matrix)

    first = call()
    info = first[1]
    return PreparedCase(call, {
        "min_eigenvalue_before": info["min_eigenvalue_before"],
        "min_eigenvalue_after": info["min_eigenvalue_after"],
        "projection_applied": info["projection_applied"],
        "nonfinite_kendall_pairs": len(info["nonfinite_kendall_pairs"]),
    })


def _prepare_correlation_dense(
        case: dict[str, Any], _: int) -> PreparedCase:
    from pyscarcopula._native import multivariate as multivariate_native

    matrix = _equicorrelation(int(case["dimension"]), 0.15)
    return PreparedCase(
        lambda: multivariate_native.prepare_dense_correlation(matrix),
        {"prepared_representation": "inverse_cholesky_logdet"},
    )


def _prepare_gas_pair(case: dict[str, Any], _: int) -> PreparedCase:
    from pyscarcopula import GumbelCopula
    from pyscarcopula._native import gas as _cpp_gas

    observations = np.random.default_rng(case["seed"]).uniform(
        0.01, 0.99, size=(case["shape"]["n_obs"], 2))
    copula = GumbelCopula(rotate=90)
    return PreparedCase(
        lambda: _cpp_gas.filter_result(
            0.07, 0.35, 0.62, observations, copula, scaling="unit"),
        {"scaling": "unit"},
    )


def _prepare_gas_student(case: dict[str, Any], _: int) -> PreparedCase:
    from pyscarcopula import StochasticStudentCopula
    from pyscarcopula._native import gas as _cpp_gas

    dimension = int(case["dimension"])
    observations = np.random.default_rng(case["seed"]).uniform(
        0.01, 0.99, size=(case["shape"]["n_obs"], dimension))
    copula = StochasticStudentCopula(
        d=dimension, R=_equicorrelation(dimension))
    return PreparedCase(
        lambda: _cpp_gas.filter_result(
            0.03, 0.15, 0.75, observations, copula, scaling="unit"),
        {"scaling": "unit", "student_cache": True},
    )


def _prepare_gas_multivariate(
        case: dict[str, Any], _: int) -> PreparedCase:
    from pyscarcopula import EquicorrGaussianCopula, StochasticStudentCopula
    from pyscarcopula._native import gas as _cpp_gas

    dimension = int(case["dimension"])
    observations = np.random.default_rng(case["seed"]).uniform(
        0.01, 0.99, size=(case["shape"]["n_obs"], dimension))
    if case["model"] == "gas_equicorr":
        copula = EquicorrGaussianCopula(dimension)
    elif case["model"] == "gas_stochastic_student_factor":
        rank = int(case["shape"]["rank"])
        copula = StochasticStudentCopula(
            d=dimension,
            corr_mode="factor",
            factor_rank=rank,
            factor_loadings=_loadings(dimension, rank, case["seed"]),
        )
    else:
        raise ValueError(f"unknown multivariate GAS model: {case['model']}")
    return PreparedCase(
        lambda: _cpp_gas.filter_result(
            0.03, 0.15, 0.75, observations, copula, scaling="unit"),
        {"scaling": "unit", "model": case["model"]},
    )


def _scar_fixture(case: dict[str, Any], threads: int):
    from pyscarcopula import (
        ClaytonCopula,
        EquicorrGaussianCopula,
        StochasticStudentCopula,
    )
    from pyscarcopula.numerical._scar_ou_config import AutoTMConfig

    shape = case["shape"]
    model = case["model"]
    dimension = int(case["dimension"])
    observations = np.random.default_rng(case["seed"]).uniform(
        0.02, 0.98, size=(shape["n_obs"], dimension))
    if model == "scar_ou_pair":
        copula = ClaytonCopula(rotate=180)
    elif model == "scar_ou_equicorr":
        copula = EquicorrGaussianCopula(dimension)
    elif model == "scar_ou_stochastic_student_dense":
        copula = StochasticStudentCopula(
            d=dimension, R=_equicorrelation(dimension))
    elif model == "scar_ou_stochastic_student_factor":
        rank = int(shape["rank"])
        copula = StochasticStudentCopula(
            d=dimension,
            corr_mode="factor",
            factor_rank=rank,
            factor_loadings=_loadings(dimension, rank, case["seed"]),
        )
    else:
        raise ValueError(f"unknown SCAR-OU benchmark model: {model}")
    kwargs = {
        "transition_method": case["backend"],
        "K": int(shape.get("grid", 48)),
        "max_K": int(shape.get("grid", 48)),
        "adaptive": False,
        "basis_order": int(shape.get("basis", 32)),
        "gh_order": 7,
        "n_threads": threads,
    }
    return observations, copula, AutoTMConfig(**kwargs)


def _scar_parameters(case: dict[str, Any]) -> tuple[float, float, float]:
    values = case.get("parameters", (1.2, 0.1, 0.7))
    return tuple(float(value) for value in values)


def _prepare_scar_ou(case: dict[str, Any], threads: int) -> PreparedCase:
    from pyscarcopula._native import scar_ou as _cpp_scar_ou
    from pyscarcopula._native import _extension

    observations, copula, config = _scar_fixture(case, threads)
    parameters = _scar_parameters(case)
    native = _extension.load()

    def call():
        native._clear_hermite_rule_cache()
        return _cpp_scar_ou.neg_loglik_with_grad_info(
            *parameters, observations, copula, config)

    first = call()
    diagnostics = dict(first[2])
    diagnostics.update({
        "cache_reset_each_call": True,
        "preparation_in_timed_call": True,
    })
    return PreparedCase(call, diagnostics)


def _prepare_scar_ou_prepared(
        case: dict[str, Any], threads: int) -> PreparedCase:
    from pyscarcopula._native import scar_ou as _cpp_scar_ou

    observations, copula, config = _scar_fixture(case, threads)
    evaluator = _cpp_scar_ou.prepare_objective(observations, copula, config)
    parameters = _scar_parameters(case)

    def call():
        return evaluator.neg_loglik_with_grad_info(*parameters)

    first = call()
    diagnostics = dict(first[2])
    diagnostics.update({
        "prepared_evaluator": True,
        "preparation_count": 1,
    })
    return PreparedCase(call, diagnostics)


def _jacobi_fixture(case: dict[str, Any]):
    from pyscarcopula import GumbelCopula
    from pyscarcopula._native import jacobi as jacobi_native

    observations = np.random.default_rng(case["seed"]).uniform(
        0.02, 0.98, size=(case["shape"]["n_obs"], 2))
    options = {
        "basis_order": int(case["shape"].get("basis", 4)),
        "quad_order": int(case["shape"].get("quad", 16)),
        "gh_order": 3,
        "transition_method": case["backend"],
        "storage": case.get("storage", "dense"),
    }
    return jacobi_native, observations, GumbelCopula(), options


def _prepare_scar_jacobi(
        case: dict[str, Any], _: int) -> PreparedCase:
    jacobi_native, observations, copula, options = _jacobi_fixture(case)

    def call():
        evaluator = jacobi_native.PreparedScarJacobiEvaluator(
            observations, copula, **options)
        return evaluator.neg_loglik_with_grad(1.2, 0.4, 0.25)

    return PreparedCase(call, {
        "preparation_in_timed_call": True,
        "transition_backend": case["backend"],
        "storage": case.get("storage", "dense"),
    })


def _prepare_scar_jacobi_prepared(
        case: dict[str, Any], _: int) -> PreparedCase:
    jacobi_native, observations, copula, options = _jacobi_fixture(case)
    evaluator = jacobi_native.PreparedScarJacobiEvaluator(
        observations, copula, **options)

    def call():
        return evaluator.neg_loglik_with_grad(1.2, 0.4, 0.25)

    first = call()
    return PreparedCase(call, {
        "prepared_evaluator": True,
        "preparation_count": evaluator.preparation_count,
        "transition_backend": case["backend"],
        "storage": case.get("storage", "dense"),
        "first_checksum": _checksum(first),
    })


def _rvine_fixture(case: dict[str, Any], threads: int):
    if str(TESTS) not in sys.path:
        sys.path.insert(0, str(TESTS))
    from rvine_runtime_cases import (
        configured_mixed_family_vine,
        configured_static_dvine,
        fitted_pair,
        scalar_parameters,
    )
    from pyscarcopula import (
        BivariateGaussianCopula,
        IndependentCopula,
        RVineCopula,
    )
    from pyscarcopula._native import _extension as _cpp_extension, vine as _cpp_rvine
    from pyscarcopula.vine._rvine_matrix_builder import (
        build_rvine_matrix_with_edge_map,
    )
    from pyscarcopula.vine._structure import cvine_structure

    topology = case.get("topology", "dvine")
    if topology == "dvine":
        vine = configured_static_dvine(int(case["dimension"]))
    elif topology == "rvine":
        vine = configured_mixed_family_vine()
    elif topology == "cvine":
        dimension = int(case["dimension"])
        structure = cvine_structure(dimension)
        trees = structure.to_trees()
        matrix, edge_map = build_rvine_matrix_with_edge_map(
            dimension, trees)
        vine = RVineCopula(structure=structure, vine_type="cvine")
        vine.d = dimension
        vine.matrix = matrix
        vine._structure = structure
        vine._trees = tuple(tuple(level) for level in trees)
        vine._edge_map = dict(edge_map)
        vine._orig_edge_key = {
            (tree, original): key
            for key, original in edge_map.items()
            for tree in (key[0],)
        }
        vine.pair_copulas = {
            key: (
                fitted_pair(BivariateGaussianCopula(), 0.2 / (key[0] + 1))
                if key[0] < 2
                else fitted_pair(IndependentCopula(), 0.0)
            )
            for key in edge_map
        }
        vine._last_u = None
        vine._target_given_vars = ()
        vine._conditional_fit_supported = True
        vine._T = 0
        vine._log_likelihood = 0.0
        vine.method = "MLE"
    else:
        raise ValueError(f"unknown vine topology: {topology}")
    parameters = scalar_parameters(vine)
    module = _cpp_extension.load()
    return vine, parameters, module, _cpp_rvine, threads


def _prepare_vine_dynamic_rosenblatt(
        case: dict[str, Any], _: int) -> PreparedCase:
    if str(TESTS) not in sys.path:
        sys.path.insert(0, str(TESTS))
    from rvine_runtime_cases import (
        configured_mixed_gas_vine,
        configured_mixed_jacobi_vine,
        configured_mixed_scar_vine,
    )
    from pyscarcopula.stattests import rvine_rosenblatt_transform

    factories = {
        "gas": configured_mixed_gas_vine,
        "scar_ou": configured_mixed_scar_vine,
        "scar_jacobi": configured_mixed_jacobi_vine,
    }
    dynamic = case["dynamic"]
    vine = factories[dynamic]()
    observations = np.random.default_rng(case["seed"]).uniform(
        0.02, 0.98, size=(case["shape"]["n_obs"], vine.d))

    def call():
        return rvine_rosenblatt_transform(
            vine,
            observations,
            K=int(case["shape"].get("grid", 24)),
            grid_range=3.0,
        )

    return PreparedCase(call, {
        "dynamic": dynamic,
        "topology": "rvine",
        "prepared_native_traversal": True,
    })


def _rvine_density_context(case: dict[str, Any], threads: int):
    vine, paths, module, adapter, threads = _rvine_fixture(case, threads)
    n_rows = int(case["shape"]["n_obs"])
    active = adapter.density_active_keys(vine._trees, vine._edge_map)
    normalized, sources = adapter.density_parameter_layout(
        vine.pair_copulas, active, paths, n_rows)
    edges, pack = adapter.compile_edge_specs(
        module,
        vine.pair_copulas,
        active,
        normalized,
        n_rows,
        parameter_sources=sources,
    )
    plan = adapter.compile_density_plan(
        module, vine.d, vine._trees, vine._edge_map, active)
    return vine, paths, module, adapter, active, normalized, sources, edges, pack, plan


def _prepare_vine_density(
        case: dict[str, Any], threads: int) -> PreparedCase:
    context = _rvine_density_context(case, threads)
    vine, paths, module, adapter, active, normalized, sources, edges, pack, plan = context
    observations = np.random.default_rng(case["seed"]).uniform(
        0.01, 0.99, size=(case["shape"]["n_obs"], vine.d))

    def call():
        return adapter.log_pdf_rows(
            module,
            vine.pair_copulas,
            vine.d,
            vine._trees,
            vine._edge_map,
            paths,
            observations,
            active_keys=active,
            normalized_parameter_paths=normalized,
            parameter_sources=sources,
            native_plan=plan,
            native_edges=edges,
            parameter_pack=pack,
            n_threads=threads,
        )

    return PreparedCase(call, {"active_edges": len(active)})


def _prepare_vine_rosenblatt(
        case: dict[str, Any], threads: int) -> PreparedCase:
    vine, paths, module, adapter, threads = _rvine_fixture(case, threads)
    n_rows = int(case["shape"]["n_obs"])
    observations = np.random.default_rng(case["seed"]).uniform(
        0.01, 0.99, size=(n_rows, vine.d))
    active = adapter.density_active_keys(vine._trees, vine._edge_map)
    layout = adapter.static_rosenblatt_parameter_layout(
        vine.pair_copulas, active)
    if layout is None:
        raise RuntimeError("static R-vine fixture produced no layout")
    parameter_paths, sources = layout
    edges, pack = adapter.compile_edge_specs(
        module,
        vine.pair_copulas,
        active,
        parameter_paths,
        n_rows,
        parameter_sources=sources,
    )
    residuals = adapter.rosenblatt_residual_node_keys(vine.matrix)
    plan = adapter.compile_density_plan(
        module,
        vine.d,
        vine._trees,
        vine._edge_map,
        active,
        residual_node_keys=residuals,
    )

    def call():
        return adapter.rosenblatt(
            module,
            vine.pair_copulas,
            vine.d,
            vine._trees,
            vine._edge_map,
            vine.matrix,
            observations,
            active_keys=active,
            parameter_paths=parameter_paths,
            parameter_sources=sources,
            residual_node_keys=residuals,
            native_plan=plan,
            native_edges=edges,
            parameter_pack=pack,
            n_threads=threads,
        )

    return PreparedCase(call, {"active_edges": len(active)})


def _prepare_vine_sampling(
        case: dict[str, Any], threads: int) -> PreparedCase:
    from pyscarcopula.vine._rvine_sampling_plan import build_rvine_sampling_plan

    vine, paths, module, adapter, threads = _rvine_fixture(case, threads)
    n_rows = int(case["shape"]["n_obs"])
    max_tree = vine._max_non_independent_tree_level()
    active = vine._sample_active_edge_keys(max_tree)
    python_plan = build_rvine_sampling_plan(
        vine.d,
        vine.matrix,
        vine.trees,
        vine._edge_map,
        active,
        max_tree,
    )
    sources = {key: "scalar" for key in active}
    edges, pack = adapter.compile_edge_specs(
        module,
        vine.pair_copulas,
        active,
        paths,
        n_rows,
        parameter_sources=sources,
    )
    native_plan = adapter.compile_traversal_plan(module, python_plan)
    uniforms = np.random.default_rng(case["seed"]).uniform(
        0.01, 0.99, size=(n_rows, vine.d))
    unused_rng = np.random.default_rng(0)

    def call():
        return adapter.sample(
            module,
            vine,
            n_rows,
            unused_rng,
            active,
            python_plan,
            paths,
            uniforms=uniforms,
            parameter_sources=sources,
            native_plan=native_plan,
            native_edges=edges,
            parameter_pack=pack,
            n_threads=threads,
        )

    return PreparedCase(call, {"active_edges": len(active)})


def _prepare_vine_mcmc(
        case: dict[str, Any], threads: int) -> PreparedCase:
    vine, paths, module, adapter, threads = _rvine_fixture(case, threads)
    n_rows = int(case["shape"]["n_obs"])
    active = adapter.density_active_keys(vine._trees, vine._edge_map)
    normalized, sources = adapter.density_parameter_layout(
        vine.pair_copulas, active, paths, n_rows)
    edges, pack = adapter.compile_edge_specs(
        module,
        vine.pair_copulas,
        active,
        normalized,
        n_rows,
        parameter_sources=sources,
    )
    plan = adapter.compile_density_plan(
        module, vine.d, vine._trees, vine._edge_map, active)
    rng = np.random.default_rng(case["seed"])
    initial_uniforms = rng.uniform(0.01, 0.99, size=(n_rows, vine.d))
    total_steps = int(case["shape"]["steps"] + case["shape"]["burnin"])
    draws = rng.uniform(0.01, 0.99, size=(total_steps, n_rows, 2))
    given = {0: 0.4}
    unused_rng = np.random.default_rng(0)

    def call():
        return adapter.mcmc(
            module,
            vine.pair_copulas,
            vine.d,
            vine._trees,
            vine._edge_map,
            paths,
            n_rows,
            unused_rng,
            given,
            n_steps=case["shape"]["steps"],
            burnin_steps=case["shape"]["burnin"],
            initial_uniforms=initial_uniforms,
            random_draws=draws,
            active_keys=active,
            normalized_parameter_paths=normalized,
            parameter_sources=sources,
            native_plan=plan,
            native_edges=edges,
            parameter_pack=pack,
            n_threads=threads,
            density_algorithm="incremental",
        )

    return PreparedCase(
        call,
        {"active_edges": len(active), "random_draws_precomputed": True},
    )


def _prepare_gaussian_conditional(
        case: dict[str, Any], threads: int) -> PreparedCase:
    from pyscarcopula._native import multivariate as multivariate_native

    dimension = int(case["dimension"])
    n_given = int(case["shape"]["n_given"])
    n_rows = int(case["shape"]["n_obs"])
    rng = np.random.default_rng(case["seed"])
    given_indices = np.arange(n_given, dtype=np.int32)
    given_latent = rng.normal(size=(n_rows, n_given))
    normal_draws = rng.normal(size=(n_rows, dimension - n_given))
    correlation = _equicorrelation(dimension)

    def call():
        return multivariate_native.gaussian_conditional_latent_info(
            correlation,
            given_indices,
            given_latent,
            normal_draws,
            n_threads=threads,
        )

    first = call()
    return PreparedCase(call, first[1])


def _prepare_student_conditional(
        case: dict[str, Any], threads: int) -> PreparedCase:
    from pyscarcopula._native import multivariate as multivariate_native

    dimension = int(case["dimension"])
    n_given = int(case["shape"]["n_given"])
    n_rows = int(case["shape"]["n_obs"])
    rng = np.random.default_rng(case["seed"])
    given_indices = np.arange(n_given, dtype=np.int32)
    given_latent = rng.normal(size=(n_rows, n_given))
    normal_draws = rng.normal(size=(n_rows, dimension - n_given))
    chi_square_draws = rng.chisquare(6.0 + n_given, size=n_rows)
    degrees = np.full(n_rows, 6.0, dtype=np.float64)
    correlation = _equicorrelation(dimension)

    def call():
        return multivariate_native.student_conditional_latent_info(
            correlation,
            given_indices,
            given_latent,
            degrees,
            normal_draws,
            chi_square_draws,
            n_threads=threads,
        )

    first = call()
    return PreparedCase(call, first[1])


RUNNERS = {
    "pair_grid": _prepare_pair_grid,
    "transform": _prepare_transform,
    "static_dense_gaussian": _prepare_static_dense_gaussian,
    "static_factor_gaussian": _prepare_static_factor_gaussian,
    "static_estimated_gaussian": _prepare_static_estimated_gaussian,
    "equicorr_grid": _prepare_equicorr_grid,
    "student_dense_grid": _prepare_student_dense_grid,
    "student_factor_grid": _prepare_student_factor_grid,
    "student_ppf_cold": _prepare_student_ppf_cold,
    "student_ppf_prepared": _prepare_student_ppf_prepared,
    "correlation_preprocess": _prepare_correlation_preprocess,
    "correlation_dense": _prepare_correlation_dense,
    "gas_pair": _prepare_gas_pair,
    "gas_student": _prepare_gas_student,
    "gas_multivariate": _prepare_gas_multivariate,
    "scar_ou": _prepare_scar_ou,
    "scar_ou_prepared": _prepare_scar_ou_prepared,
    "scar_jacobi": _prepare_scar_jacobi,
    "scar_jacobi_prepared": _prepare_scar_jacobi_prepared,
    "vine_density": _prepare_vine_density,
    "vine_rosenblatt": _prepare_vine_rosenblatt,
    "vine_sampling": _prepare_vine_sampling,
    "vine_mcmc": _prepare_vine_mcmc,
    "vine_dynamic_rosenblatt": _prepare_vine_dynamic_rosenblatt,
    "gaussian_conditional": _prepare_gaussian_conditional,
    "student_conditional": _prepare_student_conditional,
}


def prepare_case(case: dict[str, Any], physical_cores: int) -> PreparedCase:
    threads = _resolved_threads(case["n_threads"], physical_cores)
    try:
        factory = RUNNERS[case["runner"]]
    except KeyError as exc:
        raise KeyError(f"unknown benchmark runner {case['runner']!r}") from exc
    return factory(case, threads)


def _measure(call: Callable[[], Any], repetitions: int) -> tuple[float, str]:
    start = time.perf_counter_ns()
    result = None
    for _ in range(repetitions):
        result = call()
    elapsed = (time.perf_counter_ns() - start) / 1e9 / repetitions
    return elapsed, _checksum(result)


def _calibrate(call: Callable[[], Any], minimum_seconds: float) -> int:
    """Choose a batch size that reliably spans the timing minimum.

    A single-call estimate is especially fragile for sub-millisecond parallel
    cases: one scheduler stall can make the call appear several times slower
    and leave every measured batch shorter than ``minimum_seconds``.  Measure
    complete batches instead and require the selected size to clear the
    minimum twice consecutively.  The small growth margin avoids repeatedly
    landing just below the boundary because of ordinary timing noise.
    """
    repetitions = 1
    confirmations = 0
    # Calls may get faster while calibration warms clocks and caches.  Keep
    # enough headroom that the subsequent measured batches still span the
    # declared minimum after that speed-up.
    target_seconds = minimum_seconds * 1.25
    for _ in range(64):
        start = time.perf_counter_ns()
        for _ in range(repetitions):
            call()
        elapsed = max((time.perf_counter_ns() - start) / 1e9, 1e-9)
        if elapsed >= target_seconds:
            confirmations += 1
            if confirmations == 2:
                return repetitions
            continue

        confirmations = 0
        estimated = int(math.ceil(
            repetitions * target_seconds / elapsed))
        repetitions = max(repetitions + 1, estimated)

    raise RuntimeError(
        "benchmark calibration did not converge after 64 batches")


def _memory_probe(call: Callable[[], Any]) -> dict[str, int | None]:
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    tracemalloc.reset_peak()
    result = call()
    checksum = _checksum(result)
    del result
    _current, peak = tracemalloc.get_traced_memory()
    after = tracemalloc.take_snapshot()
    differences = after.compare_to(before, "lineno")
    count = sum(max(0, entry.count_diff) for entry in differences)
    allocated = sum(max(0, entry.size_diff) for entry in differences)
    tracemalloc.stop()
    return {
        "python_allocation_count": int(count),
        "python_allocated_bytes": int(allocated),
        "python_peak_bytes": int(peak),
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "memory_probe_checksum": checksum,
    }


def _relative_mad(values: list[float]) -> float:
    median = statistics.median(values)
    if median == 0.0:
        return 0.0
    return statistics.median(abs(value - median) for value in values) / median


def _q95(values: list[float]) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), 0.95))


def run_case(
        case: dict[str, Any],
        protocol: dict[str, Any],
        physical_cores: int,
        *,
        samples_override: int | None,
        duration_override: float | None,
        cpu_pool: list[int] | None) -> dict[str, Any]:
    resolved_threads = _resolved_threads(case["n_threads"], physical_cores)
    affinity = _set_process_affinity(resolved_threads, cpu_pool)
    prepared = prepare_case(case, physical_cores)
    warmups = int(protocol["warmup_calls"])
    samples = int(samples_override or protocol["paired_samples"])
    minimum = float(
        protocol["minimum_sample_seconds"]
        if duration_override is None else duration_override)
    for _ in range(warmups):
        prepared.call()
    repetitions = _calibrate(prepared.call, minimum)
    raw = []
    reference_checksum = None
    for index in range(samples):
        order = "AB" if index % 2 == 0 else "BA"
        measured = {}
        for label in order:
            seconds, checksum = _measure(prepared.call, repetitions)
            if reference_checksum is None:
                reference_checksum = checksum
            elif checksum != reference_checksum:
                raise RuntimeError(
                    f"{case['id']} returned non-deterministic checksum: "
                    f"{checksum} != {reference_checksum}")
            measured[label] = seconds
        raw.append({
            "index": index,
            "order": order,
            "a_seconds": measured["A"],
            "b_seconds": measured["B"],
            "b_over_a": measured["B"] / measured["A"],
        })

    a_values = [item["a_seconds"] for item in raw]
    b_values = [item["b_seconds"] for item in raw]
    ratios = [abs(item["b_over_a"] - 1.0) for item in raw]
    noise = _q95(ratios)
    relative_mad = max(_relative_mad(a_values), _relative_mad(b_values))
    return {
        "id": case["id"],
        "case": case,
        "resolved_n_threads": resolved_threads,
        "process_affinity": affinity,
        "warmup_calls": warmups,
        "paired_samples": samples,
        "calibrated_repetitions": repetitions,
        "minimum_sample_seconds": minimum,
        "checksum": reference_checksum,
        "raw_samples": raw,
        "a_median_seconds": statistics.median(a_values),
        "b_median_seconds": statistics.median(b_values),
        "session_median_seconds": statistics.median(a_values + b_values),
        "relative_mad": relative_mad,
        "noise_envelope": noise,
        "domain_diagnostics": prepared.diagnostics,
        "memory": _memory_probe(prepared.call),
    }


_ENVIRONMENT_COMPATIBILITY_FIELDS = (
    "python",
    "python_compiler",
    "compiler_identity",
    "target_architecture",
    "platform",
    "cpu_model",
    "logical_cores",
    "physical_cores",
    "benchmark_cpu_pool",
    "benchmark_accessible_cores",
    "reference_label",
    "build_type",
    "cxx_standard",
    "optimization_flags",
    "fp_flags",
    "dependencies",
    "process_stabilization",
)


def _coarse_metric_comparison(
        reference: int | float | None,
        measured: int | float | None,
        maximum_ratio: float,
        minimum_increase: int | float) -> tuple[float | None, float | None, bool]:
    if reference is None or measured is None:
        return None, None, reference != measured
    increase = measured - reference
    if reference > 0:
        ratio = measured / reference
        regression = ratio > maximum_ratio and increase > minimum_increase
    else:
        ratio = None
        regression = increase > minimum_increase
    return ratio, increase, regression


def _parallel_group_key(record: dict[str, Any]) -> str:
    case = {
        key: value
        for key, value in record["case"].items()
        if key not in {"id", "n_threads"}
    }
    return json.dumps(case, sort_keys=True, separators=(",", ":"))


def compare_benchmark_artifacts(
        baseline: dict[str, Any],
        candidate: dict[str, Any]) -> dict[str, Any]:
    """Apply the declared coarse runtime, memory, and scaling policy."""
    policy = baseline["protocol"]["regression_policy"]
    failures: list[str] = []
    if baseline.get("manifest_id") != candidate.get("manifest_id"):
        failures.append("baseline and candidate use different manifests")
    if baseline.get("manifest_sha256") != candidate.get("manifest_sha256"):
        failures.append("baseline and candidate use different manifest content")
    if not baseline.get("valid_for_regression_check", False):
        failures.append("baseline is not an eligible full-protocol capture")
    if not candidate.get("valid_for_regression_check", False):
        failures.append("candidate is not an eligible full-protocol capture")

    baseline_environment = baseline.get("environment", {})
    candidate_environment = candidate.get("environment", {})
    environment_mismatches = []
    for field in _ENVIRONMENT_COMPATIBILITY_FIELDS:
        reference_value = baseline_environment.get(field)
        measured_value = candidate_environment.get(field)
        if reference_value != measured_value:
            environment_mismatches.append({
                "field": field,
                "baseline": reference_value,
                "candidate": measured_value,
            })
            failures.append(f"environment mismatch: {field}")

    baseline_cases = {case["id"]: case for case in baseline.get("cases", [])}
    candidate_cases = {case["id"]: case for case in candidate.get("cases", [])}
    missing = sorted(baseline_cases.keys() - candidate_cases.keys())
    extra = sorted(candidate_cases.keys() - baseline_cases.keys())
    if missing:
        failures.append(f"candidate is missing cases: {', '.join(missing)}")
    if extra:
        failures.append(f"candidate has unexpected cases: {', '.join(extra)}")

    maximum_runtime_ratio = float(policy["maximum_runtime_ratio"])
    require_checksum = bool(policy["require_checksum_match"])
    require_diagnostics = bool(policy["require_domain_diagnostics_match"])
    metric_policies = {
        "python_allocation_count": (
            float(policy["maximum_python_allocation_count_ratio"]),
            int(policy["minimum_python_allocation_count_increase"]),
        ),
        "python_allocated_bytes": (
            float(policy["maximum_python_allocated_bytes_ratio"]),
            int(policy["minimum_python_allocated_bytes_increase"]),
        ),
        "python_peak_bytes": (
            float(policy["maximum_python_peak_memory_ratio"]),
            int(policy["minimum_python_peak_memory_increase_bytes"]),
        ),
        "process_peak_rss_bytes": (
            float(policy["maximum_process_peak_rss_ratio"]),
            int(policy["minimum_process_peak_rss_increase_bytes"]),
        ),
    }
    comparisons = []
    for case_id in baseline_cases.keys() & candidate_cases.keys():
        reference = baseline_cases[case_id]
        measured = candidate_cases[case_id]
        reference_seconds = float(reference["session_median_seconds"])
        measured_seconds = float(measured["session_median_seconds"])
        runtime_ratio = measured_seconds / reference_seconds
        checksum_match = reference["checksum"] == measured["checksum"]
        diagnostics_match = (
            reference["domain_diagnostics"] == measured["domain_diagnostics"])
        affinity_match = (
            reference["resolved_n_threads"] == measured["resolved_n_threads"]
            and reference["process_affinity"] == measured["process_affinity"]
        )

        violations = []
        if require_checksum and not checksum_match:
            violations.append("checksum changed")
        if require_diagnostics and not diagnostics_match:
            violations.append("domain diagnostics changed")
        if not affinity_match:
            violations.append("thread count or process affinity changed")
        if runtime_ratio > maximum_runtime_ratio:
            violations.append(
                f"runtime ratio {runtime_ratio:.3g} exceeds "
                f"{maximum_runtime_ratio:.3g}")

        memory_comparisons = {}
        for metric, (maximum_ratio, minimum_increase) in metric_policies.items():
            ratio, increase, regression = _coarse_metric_comparison(
                reference["memory"].get(metric),
                measured["memory"].get(metric),
                maximum_ratio,
                minimum_increase,
            )
            memory_comparisons[metric] = {
                "ratio": ratio,
                "increase": increase,
                "regression": regression,
            }
            if regression:
                violations.append(
                    f"{metric} exceeds the coarse ratio/absolute limit")
        if violations:
            failures.append(f"{case_id}: {'; '.join(violations)}")
        comparisons.append({
            "id": case_id,
            "runtime_ratio": runtime_ratio,
            "checksum_match": checksum_match,
            "domain_diagnostics_match": diagnostics_match,
            "affinity_match": affinity_match,
            "memory": memory_comparisons,
            "violations": violations,
        })

    baseline_groups: dict[str, dict[str, dict[str, Any]]] = {}
    candidate_groups: dict[str, dict[str, dict[str, Any]]] = {}
    for record in baseline_cases.values():
        baseline_groups.setdefault(_parallel_group_key(record), {})[
            str(record["case"]["n_threads"])] = record
    for record in candidate_cases.values():
        candidate_groups.setdefault(_parallel_group_key(record), {})[
            str(record["case"]["n_threads"])] = record
    maximum_scaling_loss = float(
        policy["maximum_parallel_scaling_loss_ratio"])
    scaling_comparisons = []
    for group_key, reference_group in baseline_groups.items():
        measured_group = candidate_groups.get(group_key, {})
        if "1" not in reference_group or len(reference_group) < 2:
            continue
        for thread_key, reference in reference_group.items():
            if thread_key == "1" or thread_key not in measured_group:
                continue
            baseline_speedup = (
                reference_group["1"]["session_median_seconds"]
                / reference["session_median_seconds"])
            candidate_speedup = (
                measured_group["1"]["session_median_seconds"]
                / measured_group[thread_key]["session_median_seconds"])
            scaling_loss_ratio = baseline_speedup / candidate_speedup
            regression = scaling_loss_ratio > maximum_scaling_loss
            case_id = reference["id"]
            if regression:
                failures.append(
                    f"{case_id}: parallel scaling loss ratio "
                    f"{scaling_loss_ratio:.3g} exceeds "
                    f"{maximum_scaling_loss:.3g}")
            scaling_comparisons.append({
                "id": case_id,
                "declared_n_threads": reference["case"]["n_threads"],
                "baseline_speedup": baseline_speedup,
                "candidate_speedup": candidate_speedup,
                "scaling_loss_ratio": scaling_loss_ratio,
                "regression": regression,
            })

    comparisons.sort(key=lambda item: item["id"])
    scaling_comparisons.sort(key=lambda item: item["id"])
    return {
        "passed": not failures,
        "policy": policy,
        "failures": failures,
        "environment_mismatches": environment_mismatches,
        "cases": comparisons,
        "parallel_scaling": scaling_comparisons,
    }


def _write_summary(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# C++ refactor benchmark capture",
        "",
        f"- Manifest: `{payload['manifest_id']}`",
        f"- Commit: `{payload['environment']['git_commit']}`",
        f"- Compute source digest: `{payload['environment']['compute_source_sha256']}`",
        f"- Cases: {len(payload['cases'])}",
        f"- Valid for regression check: {payload['valid_for_regression_check']}",
        f"- Validity: {payload['validity_reason']}",
    ]
    comparison = payload.get("comparison")
    if comparison is not None:
        lines.extend([
            f"- Comparison passed: {comparison['passed']}",
            f"- Comparison failures: {len(comparison['failures'])}",
        ])
    lines.extend([
        "",
        "Percentage noise metrics below are diagnostic and never block a change.",
        "",
        "| Case | Median, s | relMAD | Pair noise |",
        "|---|---:|---:|---:|",
    ])
    for case in payload["cases"]:
        lines.append(
            f"| `{case['id']}` | {case['session_median_seconds']:.9g} | "
            f"{case['relative_mad']:.3%} | {case['noise_envelope']:.3%} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--build-jobs", type=int, default=1)
    parser.add_argument("--test-workers", type=int, default=1)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run only an exact id or id prefix; repeatable.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Override paired sample count (smoke/debug only).",
    )
    parser.add_argument(
        "--minimum-sample-seconds",
        type=float,
        help="Override calibration duration (smoke/debug only).",
    )
    parser.add_argument(
        "--cpu-set",
        help=(
            "Comma-separated homogeneous logical-CPU pool for a pinned "
            "reference runner, or physical:N to select one logical CPU from "
            "each of N distinct Windows physical cores. A case declaring "
            "'physical' resolves to the pool size."
        ),
    )
    parser.add_argument(
        "--compare-to",
        type=Path,
        help=(
            "Compare the captured candidate with this baseline and return "
            "a non-zero status on a radical regression."
        ),
    )
    parser.add_argument(
        "--reference-label",
        help=(
            "Stable identifier for the dedicated hardware/toolchain "
            "configuration. Required for an eligible full capture."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    artifact_root, output, summary = _artifact_paths(args)
    if not 1 <= args.build_jobs <= 4:
        raise SystemExit("--build-jobs must be between 1 and 4")
    if not 1 <= args.test_workers <= 4:
        raise SystemExit("--test-workers must be between 1 and 4")
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    if args.case:
        cases = [
            case for case in cases
            if any(
                case["id"] == requested or case["id"].startswith(requested)
                for requested in args.case
            )
        ]
        if not cases:
            raise SystemExit("no benchmark cases matched --case filters")
    if args.samples is not None and args.samples < 2:
        raise SystemExit("--samples must be at least 2")
    if (
            args.minimum_sample_seconds is not None
            and args.minimum_sample_seconds <= 0.0):
        raise SystemExit("--minimum-sample-seconds must be positive")

    host_physical = _physical_core_count()
    cpu_pool = None
    if args.cpu_set:
        logical = max(1, os.cpu_count() or 1)
        cpu_pool = _parse_cpu_set(args.cpu_set, logical)
    benchmark_physical = (
        host_physical if cpu_pool is None else len(cpu_pool))
    required_thread_capacity = _required_thread_capacity(manifest["cases"])
    process_stabilization = _stabilize_process_priority()
    command = " ".join([str(Path(sys.executable).resolve()), *sys.argv])
    output_cases = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case['id']}", flush=True)
        output_cases.append(run_case(
            case,
            manifest["protocol"],
            benchmark_physical,
            samples_override=args.samples,
            duration_override=args.minimum_sample_seconds,
            cpu_pool=cpu_pool,
        ))

    smoke_override = (
        args.samples is not None
        or args.minimum_sample_seconds is not None
    )
    full_manifest = not args.case and len(output_cases) == len(manifest["cases"])
    environment = _metadata(
        host_physical,
        command,
        process_stabilization,
        cpu_pool,
        args.reference_label,
    )
    eligibility_failures = []
    if not full_manifest:
        eligibility_failures.append("capture does not cover the full manifest")
    if smoke_override:
        eligibility_failures.append("smoke timing overrides were used")
    if benchmark_physical < required_thread_capacity:
        eligibility_failures.append(
            "benchmark CPU pool cannot exercise the manifest's maximum "
            f"thread count ({benchmark_physical} < {required_thread_capacity})"
        )
    if not args.reference_label:
        eligibility_failures.append("--reference-label was not declared")
    if environment["compute_worktree_dirty"]:
        eligibility_failures.append("computational sources have tracked changes")
    if not environment["build_freshness"][
            "extension_not_older_than_compute_sources"]:
        eligibility_failures.append(
            "loaded extension is older than a computational source")
    if not process_stabilization["applied"]:
        eligibility_failures.append("process stabilization was not applied")
    if not all(
            case["process_affinity"].get("applied", False)
            for case in output_cases):
        eligibility_failures.append("process affinity was not applied to all cases")
    valid_for_regression_check = not eligibility_failures
    payload = {
        "schema_version": 4,
        "artifact_type": "pyscarcopula-gate1-benchmark-capture",
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": manifest["protocol"],
        "environment": environment,
        "smoke_override": smoke_override,
        "cases": output_cases,
        "summary": {
            "total": len(output_cases),
        },
        "resource_profile": {
            "build_jobs": args.build_jobs,
            "test_workers": args.test_workers,
            "benchmark_driver_workers": 1,
            "required_thread_capacity": required_thread_capacity,
        },
        "artifact_root": str(artifact_root),
        "valid_for_regression_check": valid_for_regression_check,
        "eligibility_failures": eligibility_failures,
        "validity_reason": (
            "eligible capture under the declared regression protocol"
            if valid_for_regression_check
            else "; ".join(eligibility_failures)
        ),
    }
    exit_code = 0
    if args.compare_to is not None:
        baseline = json.loads(
            args.compare_to.resolve().read_text(encoding="utf-8"))
        payload["comparison"] = compare_benchmark_artifacts(baseline, payload)
        if not payload["comparison"]["passed"]:
            exit_code = 1
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_summary(payload, summary)
    print(f"wrote {output}", flush=True)
    print(f"wrote {summary}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
