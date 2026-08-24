"""Write the machine-readable C++ architecture/config inventory."""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, fields, is_dataclass
import hashlib
import inspect
import json
from pathlib import Path
import re
import runpy
import subprocess
from typing import Any

try:
    from tools.benchmark_cpp_refactor import ROOT, _source_digest
except ImportError:  # Direct script execution.
    from benchmark_cpp_refactor import ROOT, _source_digest


DEFAULT_OUTPUT = (
    ROOT / "benchmarks" / "cpp_refactor_inventory.json"
)


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    value_type = type(value)
    type_name = f"{value_type.__module__}.{value_type.__qualname__}"
    enum_name = getattr(value, "name", None)
    if isinstance(enum_name, str):
        return {"type": type_name, "name": enum_name}
    return {"type": type_name}


def _dataclass_defaults(cls) -> dict[str, Any]:
    instance = cls()
    return {
        field.name: _json_value(getattr(instance, field.name))
        for field in fields(cls)
    }


def _pybind_defaults(cls) -> dict[str, Any]:
    try:
        instance = cls()
    except TypeError:
        return {}
    values = {}
    for name in dir(instance):
        if name.startswith("_"):
            continue
        try:
            value = getattr(instance, name)
        except Exception:
            continue
        if callable(value):
            continue
        values[name] = _json_value(value)
    return values


def _extension_api() -> dict[str, Any]:
    import pyscarcopula._scar_cpp as module

    symbols = sorted(name for name in dir(module) if not name.startswith("__"))
    types = {}
    for name in symbols:
        value = getattr(module, name)
        if inspect.isclass(value):
            types[name] = {
                "attributes": sorted(
                    item for item in dir(value) if not item.startswith("__")),
                "defaults": _pybind_defaults(value),
            }
    return {
        "policy": (
            "These raw extension names and DTO attribute names remain "
            "compatible until their Python callers have moved behind the "
            "_native adapter. Additive diagnostics are allowed; "
            "removal or semantic changes require updating this inventory "
            "and a transition adapter."
        ),
        "symbols": symbols,
        "types": types,
    }


_CONSTANT_WORDS = (
    "EPS", "FLOOR", "TOL", "BOUND", "MAX", "MIN", "OFFSET", "CLIP",
    "THRESHOLD", "LIMIT", "BUDGET", "TILE", "ORDER", "DEFAULT",
)


def _looks_like_constant(name: str) -> bool:
    return name.isupper() or any(word in name.upper() for word in _CONSTANT_WORDS)


def _python_constant_inventory() -> list[dict[str, Any]]:
    records = []

    def walk_body(path: Path, body, owner: str) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                walk_body(path, node.body, f"{owner}.{node.name}")
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            targets = []
            value_node = None
            if isinstance(node, ast.Assign):
                targets = [target.id for target in node.targets
                           if isinstance(target, ast.Name)]
                value_node = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(
                    node.target, ast.Name):
                targets = [node.target.id]
                value_node = node.value
            if value_node is None:
                continue
            try:
                value = ast.literal_eval(value_node)
            except (ValueError, TypeError):
                continue
            for name in targets:
                if _looks_like_constant(name):
                    records.append({
                        "owner": owner,
                        "symbol": name,
                        "value": _json_value(value),
                        "source": path.relative_to(ROOT).as_posix(),
                        "line": node.lineno,
                    })

    for path in sorted((ROOT / "pyscarcopula").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module = path.relative_to(ROOT).with_suffix("").as_posix().replace("/", ".")
        walk_body(path, tree.body, module)

    # Source manifests derived from families.def deliberately are not Python
    # literals.  Evaluate this build-only module so the generated source lists
    # remain covered by the compatibility inventory.
    sources_path = (
        ROOT / "pyscarcopula" / "_cpp" / "build_support" / "sources.py")
    sources_module = runpy.run_path(str(sources_path))
    source_lines = {
        node.targets[0].id: node.lineno
        for node in ast.parse(
            sources_path.read_text(encoding="utf-8"),
            filename=str(sources_path),
        ).body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    owner = "pyscarcopula._cpp.build_support.sources"
    recorded = {(item["owner"], item["symbol"]) for item in records}
    for name in ("PAIR_FAMILY_SOURCES", "SCAR_COMPUTE_SOURCES"):
        if (owner, name) in recorded:
            continue
        records.append({
            "owner": owner,
            "symbol": name,
            "value": _json_value(sources_module[name]),
            "source": sources_path.relative_to(ROOT).as_posix(),
            "line": source_lines[name],
        })
    return records


_CPP_CONSTANT = re.compile(
    r"(?:inline\s+|static\s+)*constexpr\s+[\w:<>]+\s+"
    r"(?P<name>\w+)\s*=\s*(?P<value>[^;]+);"
)


def _cpp_constant_inventory() -> list[dict[str, Any]]:
    records = []
    cpp_root = ROOT / "pyscarcopula" / "_cpp"
    for path in sorted([*cpp_root.rglob("*.hpp"), *cpp_root.rglob("*.cpp")]):
        text = path.read_text(encoding="utf-8")
        for match in _CPP_CONSTANT.finditer(text):
            name = match.group("name")
            if not _looks_like_constant(name):
                continue
            records.append({
                "owner": "current C++ translation unit/namespace",
                "symbol": name,
                "value_expression": " ".join(
                    match.group("value").split()),
                "source": path.relative_to(ROOT).as_posix(),
                "line": text.count("\n", 0, match.start()) + 1,
            })
    return records


def _target_owner_for_source(source: str) -> str:
    normalized = source.replace("\\", "/")
    if "/bindings/" in normalized:
        return "python_bindings"
    if "/scar_ou/" in normalized or "_scar_ou" in normalized:
        return "scar_ou"
    if "/gas/" in normalized or "gas" in Path(normalized).stem:
        return "gas"
    if "/vine/" in normalized or "rvine" in normalized:
        return "vine"
    if "/copula/families/" in normalized:
        family = Path(normalized).stem
        if family == "student":
            return "copula::multivariate::student"
        return f"copula::pair::{family}"
    if "/copula/factor/" in normalized or "factor" in normalized:
        return "copula::correlation::factor"
    if "/copula/" in normalized or "multivariate" in normalized:
        return "copula_models"
    if normalized.startswith("pyscarcopula/_cpp/"):
        return "foundation"
    if normalized.endswith("pyscarcopula/_types.py"):
        return "Python public config/result facade"
    if "/strategy/" in normalized:
        return "Python fit orchestration"
    if "/numerical/" in normalized:
        return "Python native adapter/orchestration"
    return "Python user layer"


def _complete_constant_mappings(
        python_constants: list[dict[str, Any]],
        cpp_constants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    curated = {entry["old"]: entry for entry in _named_constant_mappings()}
    records = []
    for kind, constants in (
            ("python", python_constants), ("cpp", cpp_constants)):
        for item in constants:
            old = (
                f"{item['owner']}.{item['symbol']}"
                if kind == "python"
                else f"{item['source']}:{item['symbol']}"
            )
            curated_entry = curated.get(old)
            records.append({
                "kind": kind,
                "old": old,
                "source": item["source"],
                "line": item["line"],
                "value": item.get("value", item.get("value_expression")),
                "target_owner": (
                    curated_entry["target_owner"]
                    if curated_entry is not None
                    else _target_owner_for_source(item["source"])
                ),
                "target": (
                    curated_entry["target"]
                    if curated_entry is not None else item["symbol"]
                ),
                "semantic": (
                    curated_entry["semantic"]
                    if curated_entry is not None
                    else "preserve the current value, bounds, and usage semantics"
                ),
            })
    return records


def _include_graph() -> dict[str, list[str]]:
    include_pattern = re.compile(r'^\s*#\s*include\s+"(scar/[^"]+)"', re.MULTILINE)
    root = ROOT / "pyscarcopula" / "_cpp"
    graph = {}
    for path in sorted([*root.rglob("*.hpp"), *root.rglob("*.cpp")]):
        includes = sorted(set(include_pattern.findall(
            path.read_text(encoding="utf-8"))))
        if includes:
            graph[path.relative_to(ROOT).as_posix()] = includes
    return graph


def _config_mappings() -> list[dict[str, Any]]:
    from pyscarcopula._types import NumericalConfig

    default = NumericalConfig()
    target = {
        "fail_value": ("Python native adapter error policy", "failure_value"),
        "n_threads": ("Per-domain configs/call arguments", "n_threads"),
        "default_K": ("scar_ou::OuNumericalConfig", "K"),
        "default_grid_range": ("scar_ou::OuNumericalConfig", "grid_range"),
        "default_pts_per_sigma": ("scar_ou::OuNumericalConfig", "pts_per_sigma"),
        "default_grid_method": ("scar_ou::OuNumericalConfig", "grid_method"),
        "default_adaptive": ("scar_ou::OuNumericalConfig", "adaptive"),
        "mle_optimizer": ("Python fit orchestration", "mle_optimizer"),
        "gas_optimizer": ("Python GAS fit orchestration", "gas_optimizer"),
        "scar_optimizer": ("Python SCAR-OU fit orchestration", "scar_optimizer"),
        "bivariate_scar_optimizer": ("Python pair SCAR-OU fit orchestration", "bivariate_scar_optimizer"),
        "bivariate_log_scar_optimizer": ("Python log-stationary SCAR fit orchestration", "bivariate_log_scar_optimizer"),
        "equicorr_optimizer": ("Python equicorrelation fit orchestration", "equicorr_optimizer"),
        "stochastic_student_optimizer": ("Python Student fit orchestration", "stochastic_student_optimizer"),
        "static_student_optimizer": ("Python static Student fit orchestration", "static_student_optimizer"),
        "stochastic_student_gas_optimizer": ("Python Student GAS fit orchestration", "stochastic_student_gas_optimizer"),
        "stochastic_student_scar_optimizer": ("Python Student SCAR-OU fit orchestration", "stochastic_student_scar_optimizer"),
        "bisection_tol": ("copula::pair inverse-h policy", "tolerance"),
        "bisection_maxiter": ("copula::pair inverse-h policy", "max_iterations"),
        "gas_score_eps": ("gas::GasConfig", "score_eps"),
        "gas_gamma_bound": ("Python GAS parameter bounds", "gamma_bound"),
        "gas_beta_bound": ("Python GAS parameter bounds", "beta_bound"),
        "default_n_tr": ("Python SCAR-MC orchestration", "n_trajectories"),
        "default_M_iterations": ("Python SCAR-MC orchestration", "m_iterations"),
    }
    records = []
    for field in fields(NumericalConfig):
        owner, target_name = target[field.name]
        records.append({
            "old_owner": "pyscarcopula._types.NumericalConfig",
            "old_field": field.name,
            "default": _json_value(getattr(default, field.name)),
            "target_owner": owner,
            "target_field_or_constant": target_name,
            "public_facade_preserved": True,
        })
    return records


def _named_constant_mappings() -> list[dict[str, Any]]:
    return [
        {"old":"pyscarcopula._constants.PDF_FLOOR","value":1e-300,"target_owner":"copula::pair common","target":"kPdfFloor","semantic":"positive density/log floor"},
        {"old":"pyscarcopula._constants.PSEUDO_OBS_EPS","value":1e-10,"target_owner":"math::probability transforms","target":"kPseudoObservationEps","semantic":"guard before inverse Gaussian/Student CDF"},
        {"old":"pyscarcopula._constants.H_FUNCTION_EPS","value":1e-10,"target_owner":"copula::pair conditional kernels","target":"kHFunctionEps","semantic":"h and inverse-h boundary"},
        {"old":"pyscarcopula._constants.ROSENBLATT_OUTPUT_EPS","value":1e-6,"target_owner":"vine::rosenblatt output policy","target":"kRosenblattOutputEps","semantic":"final GoF-safe Rosenblatt clipping"},
        {"old":"pyscarcopula._constants.CONDITIONAL_SAMPLE_EPS","value":1e-12,"target_owner":"Python sampling orchestration","target":"conditional_sample_eps","semantic":"newly sampled free coordinates only"},
        {"old":"pyscarcopula.numerical._scar_ou_config.CPP_MAX_GRID_SIZE","value":100000,"target_owner":"scar_ou::validation","target":"kMaxGridSize","semantic":"non-dense grid upper bound"},
        {"old":"pyscarcopula.numerical._scar_ou_config.CPP_MAX_DENSE_GRID_SIZE","value":10000,"target_owner":"scar_ou::validation","target":"kMaxDenseGridSize","semantic":"dense matrix grid upper bound"},
        {"old":"pyscarcopula.numerical._scar_ou_config.CPP_MAX_SPECTRAL_ORDER","value":1024,"target_owner":"scar_ou::spectral validation","target":"kMaxSpectralOrder","semantic":"basis/quadrature/GH order upper bound"},
        {"old":"pyscarcopula.numerical.multivariate_native._DENSE_STUDENT_NATIVE_MIN_DF","value":0.1,"target_owner":"copula::multivariate::student validation","target":"kMinDegreesOfFreedom","semantic":"dense native Student lower df bound"},
        {"old":"pyscarcopula.numerical.multivariate_native._DENSE_STUDENT_NATIVE_MAX_CONDITION","value":10000.0,"target_owner":"copula::multivariate::student correlation","target":"kMaxCondition","semantic":"accepted dense Student correlation condition"},
        {"old":"pyscarcopula.numerical.multivariate_native._DENSE_STUDENT_CORRELATION_TOLERANCE","value":1e-12,"target_owner":"copula::multivariate::student correlation","target":"kCorrelationTolerance","semantic":"correlation validation tolerance"},
        {"old":"pyscarcopula.copula.multivariate.stochastic_student._DF_OFFSET","value":2.000001,"target_owner":"copula::multivariate::student model","target":"kDynamicDfOffset","semantic":"dynamic df transform offset"},
        {"old":"scar::GasConfig.score_eps","value":0.0001,"target_owner":"gas::GasConfig","target":"score_eps","semantic":"score finite-difference safeguard"},
        {"old":"scar::GasConfig.g_clip","value":50.0,"target_owner":"gas::GasConfig","target":"g_clip","semantic":"latent GAS state clipping"},
        {"old":"scar::GasConfig.score_clip","value":100.0,"target_owner":"gas::GasConfig","target":"score_clip","semantic":"GAS score clipping"},
        {"old":"scar::GasConfig.fisher_floor","value":1e-6,"target_owner":"gas::GasConfig","target":"fisher_floor","semantic":"Fisher scaling floor"},
        {"old":"scar::GasConfig.stationary_beta_tol","value":1e-8,"target_owner":"gas::GasConfig","target":"stationary_beta_tol","semantic":"stationary initialization tolerance"},
        {"old":"scar::OuNumericalConfig.auto_small_kdt","value":0.01,"target_owner":"scar_ou::OuNumericalConfig","target":"auto_small_kdt","semantic":"auto backend threshold"},
        {"old":"pyscarcopula.numerical._cpp_rvine._DEFAULT_MCMC_DRAW_MEMORY_BUDGET_BYTES","value":67108864,"target_owner":"vine::McmcConfig/Python RNG adapter","target":"draw_memory_budget_bytes","semantic":"bounded proposal draw buffer"}
    ]


def _model_operation_matrix() -> list[dict[str, Any]]:
    return [
        {"model":"pair built-ins","families":["independent","clayton","frank","gumbel","joe","gaussian"],"dynamics":["static","GAS","SCAR-OU","SCAR-MC"],"corr_modes":["n/a"],"operations":["transform","density","gradient","h","inverse-h","row/grid","static likelihood","sampling"]},
        {"model":"GaussianCopula","families":["multivariate gaussian"],"dynamics":["static"],"corr_modes":["fixed","shrinkage","cholesky","factor"],"operations":["fit","density","correlation gradient","sampling","conditional sampling"]},
        {"model":"EquicorrGaussianCopula","families":["equicorrelation gaussian"],"dynamics":["static","GAS","SCAR-OU","SCAR-MC"],"corr_modes":["equicorrelation"],"operations":["row/grid density and gradient","prepared sufficient statistics","likelihood","Rosenblatt","sampling"]},
        {"model":"StudentCopula","families":["static student"],"dynamics":["static"],"corr_modes":["fixed","shrinkage","cholesky","factor"],"operations":["fit","density","correlation gradient","sampling","conditional sampling"]},
        {"model":"StochasticStudentCopula","families":["dynamic student"],"dynamics":["static","GAS","SCAR-OU","SCAR-MC"],"corr_modes":["fixed","shrinkage","cholesky","factor"],"operations":["cached/exact row-grid density and gradient","factor tiled density","likelihood","Rosenblatt","sampling","conditional sampling"]},
        {"model":"R-vine","families":["mixed built-in pair edges"],"dynamics":["static","GAS edge composition","SCAR edge composition"],"corr_modes":["n/a"],"operations":["density","Rosenblatt","unconditional sampling","conditional DAG sampling","conditional MCMC"]}
    ]


def _build_matrix() -> dict[str, Any]:
    sources = [
        ".github/workflows/parallel-release-gates.yml",
        ".github/workflows/wheels.yml",
        "setup.py",
        "pyproject.toml",
        "pyscarcopula/_cpp/build_support/sources.py",
        "pyscarcopula/_cpp/build_support/toolchain.py",
        "tools/build_cpp_tests.py",
        "tools/check_cpp_architecture.py",
    ]
    return {
        "current_no_regression_matrix": [
            {"id":"linux-gcc-py310","os":"ubuntu-latest","compiler":"GCC","python":"3.10"},
            {"id":"linux-gcc-py314","os":"ubuntu-latest","compiler":"GCC","python":"3.14"},
            {"id":"linux-clang-py312","os":"ubuntu-latest","compiler":"Clang","python":"3.12"},
            {"id":"windows-msvc-py312","os":"windows-latest","compiler":"MSVC","python":"3.12"},
            {"id":"windows-mingw64-py312","os":"windows-latest","compiler":"MinGW-w64 GCC","python":"3.12"},
            {"id":"macos-arm64-clang-py312","os":"macos-14","compiler":"AppleClang","python":"3.12"}
        ],
        "wheel_matrix": {"python":["cp310","cp311","cp312","cp313","cp314"],"linux":["x86_64"],"windows":["AMD64"],"macos":["universal2","arm64 tested"]},
        "sanitizers":["Linux Clang ASan+UBSan","Linux GCC TSan"],
        "cxx_standard":"C++17",
        "required_addition":{"id":"windows-mingw64-py312","compiler":"MinGW-w64 GCC","status":"implemented in parallel-release-gates.yml; full Gate 4 is active"},
        "sources": sources,
        "source_sha256": {
            source: hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
            for source in sources
        },
    }


def _allowed_dependencies() -> dict[str, Any]:
    return {
        "rule":"Dependencies flow only from the application/binding layers toward model/foundation layers; bindings never appear below the binding layer.",
        "layers":[
            {"name":"foundation","may_depend_on":[]},
            {"name":"copula_models","may_depend_on":["foundation"]},
            {"name":"static","may_depend_on":["copula_models","foundation"]},
            {"name":"gas","may_depend_on":["copula_models","foundation"]},
            {"name":"scar_ou","may_depend_on":["copula_models","foundation"]},
            {"name":"vine","may_depend_on":["copula_models","foundation"]},
            {"name":"gas_rvine_composition","may_depend_on":["gas","vine","copula_models","foundation"]},
            {"name":"python_bindings","may_depend_on":["static","gas","scar_ou","vine","gas_rvine_composition","copula_models","foundation"]}
        ],
        "forbidden":["foundation -> model/application/bindings","copula_models -> application/bindings","GAS <-> SCAR-OU","vine -> dynamic applications","any compute source -> pybind11/Python.h/NumPy C API/PyObject"]
    }


def _breaking_changes() -> dict[str, Any]:
    return {
        "approval":"Approved by the architecture refactoring plan for a breaking release; no compatibility adapter is required for arbitrary Python copulas or replaced Python-core backends.",
        "target_release":"next explicitly documented breaking release after native adapters are complete",
        "public_imports_and_protocols":[
            {"surface":"Subclassing CopulaBase/BivariateCopula as a user-defined numerical copula","paths":["pyscarcopula/copula/base.py","pyscarcopula/copula/_protocol.py"],"replacement":"Implement and register a native C++ model"},
            {"surface":"CopulaCapabilities flags as a custom-backend protocol","paths":["pyscarcopula/copula/base.py","pyscarcopula/strategy/_base.py","pyscarcopula/vine/_selection.py"],"replacement":"Native descriptor capability query; flags may remain only for user-level strategy selection"},
            {"surface":"__pyscarcopula_native_rvine__ subclass opt-in","paths":["pyscarcopula/numerical/_cpp_rvine.py"],"replacement":"Exact registered native model descriptor"}
        ],
        "backend_selectors":[
            {"name":"PYSCA_RVINE_BACKEND","values":["auto","python_executor","native_strict"],"path":"pyscarcopula/numerical/_rvine_backend.py","removal":"Remove python_executor and auto fallback; native behavior is mandatory"}
        ],
        "fallback_paths":[
            "pyscarcopula.vine.vine.VineCopula._sample_stepwise_stateful",
            "pyscarcopula.vine.vine.VineCopula._sample_with_r_python",
            "pyscarcopula.vine.vine.VineCopula._sample_suffix_given_with_r_python",
            "pyscarcopula.vine.vine.VineCopula._sample_dag_given_with_r_python",
            "pyscarcopula.vine.vine.VineCopula._log_pdf_rows_with_r_python",
            "pyscarcopula.vine.vine.VineCopula._sample_arbitrary_given_mcmc_python",
            "pyscarcopula.numerical._rvine_backend.dispatch_rvine_backend",
            "pyscarcopula.stattests._rvine_rosenblatt_transform_python",
            "pyscarcopula.stattests._student_rosenblatt_transform_python",
            "custom subclass capability fallbacks in numerical/_cpp_copula.py and numerical/_cpp_rvine.py",
            "production pair transform/rotation formulas retained only as test oracles",
            "replaced OU/TM core paths in numerical/ou_kernels.py, hermite_tm.py, predictive_tm.py, tm_functions.py, and tm_grid.py"
        ],
        "docs_to_update":[
            "README.md",
            "docs/api/contrib.md",
            "docs/api/copulas.md",
            "docs/api/vine.md",
            "docs/guide/architecture.md",
            "docs/guide/numerical-backends.md",
            "CHANGELOG.md"
        ],
        "tests_to_replace_with_removal_contracts":[
            "tests/test_copula_strategy_support.py custom classes",
            "tests/test_vine_strategy_generic.py custom protocols",
            "tests/test_cpp_rvine_runtime.py::test_custom_builtin_subclass_is_fallback_only",
            "tests/test_cpp_rvine_density_mcmc.py custom-subclass fallback cases",
            "tests/test_cpp_rvine_rosenblatt.py custom-subclass fallback cases",
            "tests/test_cpp_rvine_conditional.py custom-subclass and Python fallback cases",
            "tests/test_removed_compatibility.py additions for rejected legacy imports/selectors"
        ],
        "migration_note_entries":[
            "Custom Python copula numerical protocols are removed; new families require native C++ implementation and registration.",
            "R-vine python_executor/auto numerical fallback selection is removed.",
            "Replaced Python numerical kernels are no longer importable production backends; fit, RNG, approved sampling, and orchestration stay in Python.",
            "Built-in public models, serialized results/configs, defaults, and numerical behavior remain covered by the numerical and configuration contracts."
        ]
    }


def _regression_oracles() -> dict[str, Any]:
    return {
        "architecture_oracles":["tests/fixtures/cpp_refactor_goldens_v1.json","tests/test_cpp_refactor_contracts.py","benchmarks/cpp_refactor_manifest.json","benchmark_artifacts/cpp_refactor_baseline.json"],
        "existing_pair_and_transform":["tests/test_native_bivariate_operations.py","tests/test_kendall_mapping.py","tests/test_h_stability.py","tests/test_math_properties.py"],
        "existing_multivariate":["tests/test_native_multivariate_operations.py","tests/test_multivariate_math.py","tests/test_factor_student.py","tests/test_factor_correlation.py","tests/test_cpp_dense_student_rosenblatt.py"],
        "existing_application_modules":["tests/test_native_static_likelihood.py","tests/test_cpp_gas_wrapper.py","tests/test_native_scar_ou.py","tests/test_cpp_rvine_runtime.py","tests/test_cpp_rvine_density_mcmc.py","tests/test_cpp_rvine_rosenblatt.py"],
        "reduction_order_sensitive":["tests/test_multivariate_parallelization_baselines.py","tests/test_static_parallelization.py","tests/test_sampling_parallelization.py","tests/test_cpp_refactor_contracts.py"],
        "config_and_serialization":["tests/test_numerical_constants.py","tests/test_persistence.py","tests/test_multivariate_model_contracts.py","tests/test_correlation_policy.py"]
    }


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def build_payload() -> dict[str, Any]:
    from pyscarcopula._types import LBFGSBConfig, NumericalConfig, PredictConfig
    from pyscarcopula.numerical._scar_ou_config import AutoTMConfig
    import pyscarcopula._scar_cpp as native

    python_constants = _python_constant_inventory()
    cpp_constants = _cpp_constant_inventory()

    return {
        "schema_version": 1,
        "inventory_id": "cpp-architecture-refactor-v1",
        "source_commit": _git_commit(),
        "compute_source_sha256": _source_digest(),
        "configuration_contracts": {
            "public_config_defaults": {
                "LBFGSBConfig": _dataclass_defaults(LBFGSBConfig),
                "NumericalConfig": _dataclass_defaults(NumericalConfig),
                "PredictConfig": _dataclass_defaults(PredictConfig),
                "AutoTMConfig": _dataclass_defaults(AutoTMConfig),
            },
            "native_config_defaults": {
                "GasConfig": _pybind_defaults(native.GasConfig),
                "OuNumericalConfig": _pybind_defaults(native.OuNumericalConfig),
            },
            "numerical_config_mappings": _config_mappings(),
            "named_constant_mappings": _named_constant_mappings(),
            "discovered_python_constants": python_constants,
            "discovered_cpp_constants": cpp_constants,
            "complete_constant_mappings": _complete_constant_mappings(
                python_constants, cpp_constants),
            "model_operation_matrix": _model_operation_matrix(),
        },
        "build_matrix": _build_matrix(),
        "dependencies": {
            "allowed_target_graph": _allowed_dependencies(),
            "include_graph": _include_graph(),
        },
        "extension_compatibility": _extension_api(),
        "breaking_changes": _breaking_changes(),
        "regression_oracles": _regression_oracles(),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def _contract_view(payload: dict[str, Any]) -> dict[str, Any]:
    """Return contracts that must not drift without inventory edits."""
    return {
        "public_config_defaults": payload["configuration_contracts"]["public_config_defaults"],
        "native_config_defaults": payload["configuration_contracts"]["native_config_defaults"],
        "numerical_config_mappings": payload["configuration_contracts"]["numerical_config_mappings"],
        "named_constant_mappings": payload["configuration_contracts"]["named_constant_mappings"],
        "model_operation_matrix": payload["configuration_contracts"]["model_operation_matrix"],
        "discovered_python_constants": payload["configuration_contracts"]["discovered_python_constants"],
        "discovered_cpp_constants": payload["configuration_contracts"]["discovered_cpp_constants"],
        "complete_constant_mappings": payload["configuration_contracts"]["complete_constant_mappings"],
        "build_matrix": payload["build_matrix"],
        "allowed_target_graph": payload["dependencies"]["allowed_target_graph"],
        "include_graph": payload["dependencies"]["include_graph"],
        "extension_compatibility": payload["extension_compatibility"],
        "breaking_changes": payload["breaking_changes"],
        "regression_oracles": payload["regression_oracles"],
    }


def _gate3_contract_view(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the preserved config/constants part of the Stage 0 artifact.

    Stage 8 deliberately changes build graphs, extension symbols, owners and
    source locations.  The repository inventory is the historical Stage 0
    artifact; the external Stage 8 inventory owns those approved boundary
    changes.  Its blocking responsibility here is therefore Gate 3: values,
    mappings and model-operation policy must remain unchanged.
    """
    contracts = payload["configuration_contracts"]
    return {
        "public_config_defaults": contracts["public_config_defaults"],
        "native_config_defaults": contracts["native_config_defaults"],
        "numerical_config_mappings": contracts[
            "numerical_config_mappings"],
        "named_constant_mappings": contracts["named_constant_mappings"],
        "model_operation_matrix": contracts["model_operation_matrix"],
        "discovered_cpp_constants": contracts["discovered_cpp_constants"],
    }


_GATE3_PYTHON_CONSTANT_RELOCATIONS = {
    "pyscarcopula.numerical._cpp_extension._CPP_STATUS_NAMES":
        "pyscarcopula._native.errors._STATUS_NAMES",
    "pyscarcopula.numerical._cpp_extension._MODULE":
        "pyscarcopula._native._extension._MODULE",
    "pyscarcopula.numerical._cpp_extension._MODULE_ERROR":
        "pyscarcopula._native._extension._MODULE_ERROR",
}

_GATE3_APPROVED_ARCHITECTURE_MANIFESTS = {
    "pyscarcopula._cpp.build_support.sources.SCAR_COMPUTE_SOURCES",
    "pyscarcopula._cpp.build_support.sources.PYTHON_BINDING_SOURCES",
}


def _python_constant_values(payload: dict[str, Any]) -> dict[str, Any]:
    mappings = payload["configuration_contracts"][
        "complete_constant_mappings"]
    return {
        entry["old"]: entry["value"]
        for entry in mappings
        if entry["kind"] == "python"
    }


def _gate3_python_constant_drift(
        expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """Compare frozen Python values while permitting approved relocations.

    Source paths and line numbers are deliberately ignored.  Additions made by
    later architectural stages are allowed, but every Stage 0 Python constant
    remains required with its frozen value unless the manifest itself is an
    approved build-boundary list.
    """
    frozen = _python_constant_values(expected)
    current = _python_constant_values(actual)
    drift = []
    for old, expected_value in frozen.items():
        if old in _GATE3_APPROVED_ARCHITECTURE_MANIFESTS:
            continue
        current_name = _GATE3_PYTHON_CONSTANT_RELOCATIONS.get(old, old)
        if current_name not in current:
            drift.append(f"missing Python constant {old} ({current_name})")
        elif current[current_name] != expected_value:
            drift.append(
                f"Python constant {old} changed from {expected_value!r} "
                f"to {current[current_name]!r}"
            )
    return drift


def _gate3_drift(
        expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    drift = []
    if _gate3_contract_view(expected) != _gate3_contract_view(actual):
        drift.append("config, mapping, model-operation, or C++ constant drift")
    drift.extend(_gate3_python_constant_drift(expected, actual))
    return drift


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output = args.output.resolve()
    payload = build_payload()
    if args.check:
        expected = json.loads(output.read_text(encoding="utf-8"))
        drift = _gate3_drift(expected, payload)
        if drift:
            raise SystemExit(
                "C++ refactor Gate 3 contract drifted; update the mapping or "
                "add a compatibility adapter before accepting the change: "
                + "; ".join(drift)
            )
        print(f"C++ refactor Gate 3 contracts are unchanged: {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
