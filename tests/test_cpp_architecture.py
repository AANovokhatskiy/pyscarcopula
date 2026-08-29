"""Contracts for the native C++ architecture checker."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = ROOT / "tools" / "check_cpp_architecture.py"
CHECKER_MODULE_NAME = "_pyscarcopula_check_cpp_architecture"
CHECKER_SPEC = importlib.util.spec_from_file_location(
    CHECKER_MODULE_NAME,
    CHECKER_PATH,
)
if CHECKER_SPEC is None or CHECKER_SPEC.loader is None:
    raise ImportError(f"cannot load architecture checker from {CHECKER_PATH}")
CHECKER_MODULE = importlib.util.module_from_spec(CHECKER_SPEC)
sys.modules[CHECKER_MODULE_NAME] = CHECKER_MODULE
CHECKER_SPEC.loader.exec_module(CHECKER_MODULE)
check_repository = CHECKER_MODULE.check_repository
check_raw_extension_imports = CHECKER_MODULE.check_raw_extension_imports
check_registry_completeness = CHECKER_MODULE.check_registry_completeness


def _write(root: Path, relative: str, text: str = "") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _set_manifest(
    root: Path,
    compute: list[str] | tuple[str, ...],
    bindings: list[str] | tuple[str, ...] = ("bindings/module.cpp",),
) -> None:
    _write(
        root,
        "pyscarcopula/_cpp/build_support/sources.py",
        f"SCAR_COMPUTE_SOURCES = {tuple(compute)!r}\n"
        f"PYTHON_BINDING_SOURCES = {tuple(bindings)!r}\n",
    )


def _minimal_repository(root: Path) -> Path:
    _write(
        root,
        "setup.py",
        "SCAR_COMPUTE_SOURCES = manifest.SCAR_COMPUTE_SOURCES\n"
        "PYTHON_BINDING_SOURCES = manifest.PYTHON_BINDING_SOURCES\n",
    )
    _set_manifest(root, ())
    _write(root, "pyscarcopula/_cpp/include/scar/copula.hpp", "#pragma once\n")
    _write(root, "pyscarcopula/_cpp/include/scar/gas.hpp", "#pragma once\n")
    _write(root, "pyscarcopula/_cpp/include/scar/ou.hpp", "#pragma once\n")
    _write(
        root,
        "pyscarcopula/_cpp/src/bindings/module.cpp",
        '#include "module.hpp"\n\n'
        "PYBIND11_MODULE(_scar_cpp, module) {\n"
        "    pyscarcopula::bindings::bind_common(module);\n"
        "}\n",
    )
    _write(
        root,
        "pyscarcopula/_cpp/src/bindings/module.hpp",
        "#include <pybind11/pybind11.h>\n",
    )
    return root


def _rules(root: Path) -> set[str]:
    return {violation.rule for violation in check_repository(root)}


def test_current_repository_satisfies_cpp_architecture_contract():
    assert check_repository(ROOT) == []


@pytest.mark.parametrize("relative", [
    "pyscarcopula/_native/extra.py",
    "pyscarcopula/numerical/extra.py",
    "pyscarcopula/strategy/extra.py",
    "pyscarcopula/copula/multivariate/extra.py",
    "pyscarcopula/vine/extra.py",
    "pyscarcopula/stattests.py",
    "pyscarcopula/new_package/extra.py",
])
def test_python_ownership_scans_every_production_directory(tmp_path, relative):
    from tools.check_python_ownership import audit_package

    _write(tmp_path, relative, "def state(x, rho):\n    return x * rho + 1\n")
    result = audit_package(tmp_path, exceptions={})
    assert result["verdict"] == "FAIL"
    assert any(v["rule"] == "python-ownership-arithmetic" for v in result["violations"])
    assert result["entries"][-1]["category"] == "model_math"
    # The general gate, not only a standalone audit command, must enforce it.
    assert any(v.rule.startswith("python-ownership-") for v in
               CHECKER_MODULE.check_python_numerical_ownership(tmp_path))


@pytest.mark.parametrize("source,rule", [
    ("from scipy.special import stdtrit as quantile\n"
     "def value(u):\n    return quantile(4, u)\n", "numerical-call"),
    ("import numpy as n\nsolve = n.linalg.inv\n"
     "def value(x):\n    return solve(x)\n", "numerical-call"),
    ("import numba as nb\n@nb.njit\ndef value(x):\n    return x\n", "numba-kernel"),
    ("def outer():\n    return lambda x: x ** 2\n", "arithmetic"),
    ("class Model:\n    scale = 1.0 / 3\n", "arithmetic"),
    ("class Model:\n    _bounds = [(0.01, 5.0)]\n", "model-policy"),
    ("def fit(problem):\n    return problem(initial_parameters=[1.0, 0.5])\n", "model-policy"),
    ("def value(native):\n    try:\n        return native()\n"
     "    except RuntimeError:\n        return 1e10\n", "numeric-fallback"),
    ("def value(native, fail_value):\n    try:\n        return native()\n"
     "    except Exception:\n        return float(fail_value)\n", "numeric-fallback"),
    ("import numpy as np\ndef value(result, fail_value):\n"
     "    if result['status'] != 0:\n"
     "        return float(fail_value), np.array([0.0])\n"
     "    return result['value'], result['gradient']\n", "numeric-fallback"),
    ("import numpy as np\ndef value(valid):\n"
     "    fail = 1e10, np.zeros(3)\n"
     "    if not valid:\n        return fail\n"
     "    return 0.0, np.ones(3)\n", "numeric-fallback"),
    ("def value(shape_is_supported):\n"
     "    if not shape_is_supported():\n        return 1e10\n"
     "    return native()\n", "numeric-fallback"),
    ("import numpy as np\ndef value(native, x):\n"
     "    try:\n        return native(x)\n"
     "    except RuntimeError:\n        return np.zeros_like(x)\n", "numeric-fallback"),
    ("def value(rng, df):\n    return rng.chisquare(df)\n", "model-rng"),
    ("def value(rng, weights):\n    return rng.choice(3, p=weights)\n", "model-rng"),
    ("def value(rng, mu):\n    return rng.normal(mu, 1)\n", "model-rng"),
    ("import importlib as il\nil.import_module('pyscarcopula._native._scar_cpp')\n", "raw-import"),
    ("from ._native import _scar_cpp\n", "raw-import"),
    ("from pyscarcopula._scar_cpp import objective\n", "raw-import"),
    ("def load_raw():\n    from importlib import import_module as load\n"
     "    return load('pyscarcopula._native._scar_cpp')\n", "raw-import"),
    ("from importlib import import_module as load\nload('pyscarcopula.' + name)\n", "dynamic-import"),
    ("from scipy.special import *\n", "opaque-import"),
    ("def initial_point():\n    return [1.0, 0.5, 0.2]\n", "model-policy"),
    ("def gradient(x, objective):\n    g = []\n    for i in range(len(x)):\n"
     "        trial = x.copy()\n        trial[i] += 1e-5\n"
     "        g.append((objective(trial)-objective(x))/1e-5)\n"
     "    return g\n", "arithmetic"),
])
def test_python_ownership_mutations_are_rejected(tmp_path, source, rule):
    from tools.check_python_ownership import audit_package

    _write(tmp_path, "pyscarcopula/api.py", source)
    result = audit_package(tmp_path, exceptions={})
    assert any(v["rule"] == "python-ownership-" + rule for v in result["violations"])


def test_python_ownership_aliases_respect_function_scopes(tmp_path):
    from tools.check_python_ownership import audit_package

    _write(tmp_path, "pyscarcopula/api.py",
           "def unsafe(x):\n    from scipy.special import stdtrit as f\n"
           "    return f(4, x)\n"
           "def safe(x):\n    from numpy import asarray as f\n    return f(x)\n")
    result = audit_package(tmp_path, exceptions={})
    assert any(v["symbol"] == "unsafe" and v["rule"].endswith("numerical-call")
               for v in result["violations"])
    assert not any(v["symbol"] == "safe" for v in result["violations"])


def test_python_ownership_allows_plain_adapters_and_raw_draws(tmp_path):
    from tools.check_python_ownership import audit_package

    _write(tmp_path, "pyscarcopula/api.py",
           "import numpy as np\n"
           "def adapter(x, native):\n    x = np.asarray(x)\n"
           "    if x.ndim != 2:\n        raise ValueError('shape')\n"
           "    return native(x)\n"
           "def draws(rng, n):\n    return rng.standard_normal(n), rng.uniform(0, 1, n)\n")
    assert audit_package(tmp_path, exceptions={})["violations"] == []


def test_python_ownership_allowlist_cannot_hide_symbol_mutations(tmp_path):
    import ast
    from tools.check_python_ownership import audit_package, fingerprint

    source = "def size(n):\n    return n * 8\n"
    path = _write(tmp_path, "pyscarcopula/api.py", source)
    policy = {"pyscarcopula.api:size": {
        "owner": "buffer allocation", "reason": "float64 byte count",
        "test": "this test", "category": "adapter/DTO/validation",
        "rules": ["arithmetic"], "fingerprint": fingerprint(ast.parse(source).body[0]),
    }}
    assert audit_package(tmp_path, exceptions=policy)["violations"] == []
    path.write_text(source.replace("n * 8", "n * n"), encoding="utf-8")
    assert any(v["rule"] == "python-ownership-stale-allowlist"
               for v in audit_package(tmp_path, exceptions=policy)["violations"])


def test_python_ownership_inventory_and_import_graph_are_complete(tmp_path):
    from tools.check_python_ownership import audit_package

    _write(tmp_path, "pyscarcopula/__init__.py", "from . import api\n")
    _write(tmp_path, "pyscarcopula/api.py", "def run():\n    from . import helper\n")
    _write(tmp_path, "pyscarcopula/helper.py",
           "def f():\n    return 1\n"
           "def f():\n    return lambda x: x\n")
    _write(tmp_path, "pyscarcopula/unused.py", "def f(x):\n    return x * x\n")
    result = audit_package(tmp_path, exceptions={})
    keys = [entry["key"] for entry in result["entries"]]
    assert len(keys) == len(set(keys))
    modules = {item["module"]: item for item in result["import_graph"]["modules"]}
    assert modules["pyscarcopula.helper"]["import_path"]
    # No incoming edge is not a waiver: the module remains importable.
    assert not modules["pyscarcopula.unused"]["import_path"]
    assert modules["pyscarcopula.unused"]["wheel_importable"]
    assert any(v["path"].endswith("unused.py") for v in result["violations"])


def test_python_ownership_contrib_cannot_become_core_fallback(tmp_path):
    from tools.check_python_ownership import audit_package

    _write(tmp_path, "pyscarcopula/contrib/marginal.py",
           "def marginal(x):\n    return x ** 2\n")
    assert audit_package(tmp_path, exceptions={})["violations"] == []
    _write(tmp_path, "pyscarcopula/api.py",
           "from .contrib.marginal import marginal\n")
    assert any(v["rule"] == "python-ownership-contrib-boundary"
               for v in audit_package(tmp_path, exceptions={})["violations"])


def test_python_ownership_reviewed_symbols_are_exact():
    from tools.check_python_ownership import audit_package

    result = audit_package(ROOT)
    assert not result["unused_exception_keys"]
    assert not [v for v in result["violations"] if v["rule"].endswith("stale-allowlist")]
    entries = {e["key"]: e for e in result["entries"]}
    assert len(entries) == len(result["entries"])
    for entry in entries.values():
        if entry["exception"]:
            assert entry["exception"]["test"] == (
                "tests/test_cpp_architecture.py::test_python_ownership_reviewed_symbols_are_exact")
            assert not entry["unreviewed_signals"]


def test_remediated_fv1_fv2_python_owners_do_not_regress():
    from tools.check_python_ownership import audit_package

    result = audit_package(ROOT)
    migrated_modules = {
        "pyscarcopula.numerical.ou_kernels",
        "pyscarcopula.numerical.hermite_tm",
        "pyscarcopula.copula.multivariate.student_ppf_cache",
        "pyscarcopula.io",
    }
    migrated_symbols = {
        "pyscarcopula._native._descriptors:_set_factor",
        "pyscarcopula._native.scar_ou:sample_trajectory",
        "pyscarcopula._native.scar_ou:trajectory_from_innovations",
        "pyscarcopula._native.scar_ou:hermite_rule",
        "pyscarcopula._native.scar_ou:default_quad_order",
        "pyscarcopula._native.scar_ou:_kappa_dt",
        "pyscarcopula._native.model_policy:ou_kappa_dt",
        "pyscarcopula._native.model_policy:ou_auto_backend",
        "pyscarcopula._native.model_policy:ou_adaptive_spectral_basis_order",
        "pyscarcopula._native.model_policy:ou_resolve_quad_order",
        "pyscarcopula._native.jacobi:fixed_shape_rule",
        "pyscarcopula._native.jacobi:select_sparse_order",
        "pyscarcopula._native.jacobi:tau_to_parameter",
        "pyscarcopula._native.vine:_select_mcmc_density_algorithm",
        "pyscarcopula._native.vine:mcmc_default_steps",
        "pyscarcopula.numerical._scar_ou_config:select_auto_backend",
        "pyscarcopula.numerical._scar_ou_config:validate_cpp_config",
        "pyscarcopula.numerical.jacobi_sparse:compare_sparse_jacobi_corrections",
        "pyscarcopula.strategy.scar_tm:SCARTMStrategy._adaptive_spectral_basis_order",
        "pyscarcopula.strategy.scar_tm:SCARTMStrategy._kappa_dt",
        "pyscarcopula._native.multivariate:prepare_student_ppf_table",
        "pyscarcopula._native.multivariate:evaluate_student_ppf_table",
        "pyscarcopula._native.multivariate:interpolate_student_ppf_table",
        "pyscarcopula._utils:clip_unit",
        "pyscarcopula._utils:clip_pseudo_observations",
        "pyscarcopula._utils:clip_pseudo_observations_no_copy",
        "pyscarcopula._utils:clip_h_function_values",
        "pyscarcopula._utils:clip_rosenblatt_output",
        "pyscarcopula.copula.multivariate.conditional:sample_gaussian_conditional",
        "pyscarcopula.copula.multivariate.equicorr_prepared:EquicorrPreparedData.__post_init__",
        "pyscarcopula.copula.multivariate.gaussian:_validate_gaussian_fit_data",
        "pyscarcopula.copula.multivariate.student:_validate_student_fit_data",
        "pyscarcopula.numerical.jacobi_sparse:SparseJacobiTransition.__post_init__",
        "pyscarcopula.strategy.scar_tm:SCARTMStrategy._validate_final_fit",
        "pyscarcopula.strategy.scar_tm:SCARTMStrategy._fit_joint_static.validate_correlation",
        "pyscarcopula._types:MultivariateMLEResult.aic",
        "pyscarcopula._types:MultivariateMLEResult.bic",
        "pyscarcopula.copula.base:BivariateCopula.log_likelihood",
        "pyscarcopula.vine._rvine_dissmann:_beam_search_candidates",
        "pyscarcopula.vine._rvine_dissmann:_fit_score_levels",
        "pyscarcopula.vine._selection:_rotation_compatible",
        "pyscarcopula.vine._selection:_tau_for_itau",
        "pyscarcopula.vine._selection:select_best_copula",
        "pyscarcopula.vine._structure:<module>",
        "pyscarcopula.vine._structure:_build_next_tree",
        "pyscarcopula.vine._structure:_build_next_tree_conditional",
        "pyscarcopula.vine._structure:_build_tree_0",
        "pyscarcopula.vine._structure:_build_tree_0_conditional",
        "pyscarcopula.vine._structure:_dense_rank_matrix_no_ties",
        "pyscarcopula.vine._structure:_dense_ranks_no_ties",
        "pyscarcopula.vine._structure:_kendall_tau_from_dense_ranks",
        "pyscarcopula.vine._structure:_kendall_tau_value",
        "pyscarcopula.vine._vine_fit:_build_vine_edge_fit",
        "pyscarcopula.vine._vine_fit:_fit_tree_level",
        "pyscarcopula.vine.vine:VineCopula.aic",
        "pyscarcopula.vine.vine:VineCopula.bic",
        "pyscarcopula.vine.vine:VineCopula._apply_given_only_dynamic_updates_ordered",
        "pyscarcopula.vine.vine:VineCopula._compute_pseudo_obs",
        "pyscarcopula.vine.vine:VineCopula.fit",
        "pyscarcopula.vine.vine:VineCopula.log_likelihood",
        "pyscarcopula._native.vine:apply_given_only_dynamic_updates",
        "pyscarcopula._native.vine:conditional_trace",
        "pyscarcopula._native.vine:pseudo_observation_trace_supported",
        "pyscarcopula._native.vine:pseudo_observations",
    }
    retired_symbols = {
        "pyscarcopula.numerical.gof_blocks:forward_block_size",
        "pyscarcopula.numerical.gof_blocks:iter_forward_weight_block_arrays",
        "pyscarcopula.numerical.gof_blocks:iter_forward_weight_blocks",
        "pyscarcopula.vine._rvine_suffix:given_suffix_edge_observations_with_r",
    }
    entries = {entry["key"]: entry for entry in result["entries"]}
    assert migrated_symbols <= entries.keys()
    assert retired_symbols.isdisjoint(entries)
    assert "pyscarcopula._utils:linear_least_squares" not in entries
    assert not (ROOT / "pyscarcopula" / "numerical" / "gof_blocks.py").exists()
    assert "pyscarcopula.numerical.gof_blocks:" not in (
        ROOT / "tools" / "python_ownership_policy.py"
    ).read_text(encoding="utf-8")
    for entry in entries.values():
        if entry["module"] in migrated_modules or entry["key"] in migrated_symbols:
            assert not entry["unreviewed_signals"], entry["key"]


def test_python_ownership_artifact_guard_and_append_only(tmp_path):
    from tools.check_python_ownership import main

    root = tmp_path / "product"
    _write(root, "pyscarcopula/specimen.py", "def f(x):\n    return x\n")
    with pytest.raises(SystemExit):
        main(["--root", str(root), "--artifact-root", str(root / "capture")])
    assert not (root / "capture").exists()
    output = tmp_path / "external"
    assert main(["--root", str(root), "--artifact-root", str(output)]) == 0
    assert (output / "python_import_graph.json").is_file()
    with pytest.raises(SystemExit):
        main(["--root", str(root), "--artifact-root", str(output)])


def test_raw_extension_import_is_private_to_facade_loader(tmp_path):
    root = tmp_path
    _write(
        root,
        "setup.py",
        'Pybind11Extension("pyscarcopula._native._scar_cpp", [])\n',
    )
    _write(
        root,
        "pyscarcopula/_native/_extension.py",
        "import importlib\n"
        "module = importlib.import_module("
        "'pyscarcopula._native._scar_cpp')\n",
    )
    assert check_raw_extension_imports(root) == []

    bypass = _write(
        root,
        "pyscarcopula/api.py",
        "from pyscarcopula._native import _scar_cpp\n",
    )
    violations = check_raw_extension_imports(root)
    assert [(item.rule, item.path) for item in violations] == [
        ("raw-extension-import", bypass),
    ]


def test_removed_top_level_raw_extension_import_is_rejected(tmp_path):
    root = tmp_path
    _write(
        root,
        "setup.py",
        'Pybind11Extension("pyscarcopula._native._scar_cpp", [])\n',
    )
    loader = _write(
        root,
        "pyscarcopula/_native/_extension.py",
        "import importlib\n"
        "module = importlib.import_module('pyscarcopula._scar_cpp')\n",
    )
    violations = check_raw_extension_imports(root)
    assert any(
        item.path == loader and "removed raw extension path" in item.message
        for item in violations
    )


def test_registry_completeness_gate_detects_python_cpp_drift(tmp_path):
    relatives = (
        "pyscarcopula/_cpp/include/scar/copula/model_descriptor.hpp",
        "pyscarcopula/_cpp/include/scar/copula/capability.hpp",
        "pyscarcopula/_cpp/src/bindings/capability.cpp",
        "pyscarcopula/_native/registry.py",
    )
    for relative in relatives:
        _write(
            tmp_path,
            relative,
            (ROOT / relative).read_text(encoding="utf-8"),
        )
    assert check_registry_completeness(tmp_path) == []

    registry = tmp_path / "pyscarcopula/_native/registry.py"
    source = registry.read_text(encoding="utf-8")
    registry.write_text(
        source.replace(
            '    "StochasticStudent",\n    "Vine",\n)',
            '    "StochasticStudent",\n)',
            1,
        ),
        encoding="utf-8",
    )
    violations = check_registry_completeness(tmp_path)
    assert any(
        item.rule == "registry-completeness"
        and "missing=['Vine']" in item.message
        for item in violations
    )


def test_vine_production_python_backend_dispatch_is_rejected(tmp_path):
    root = _minimal_repository(tmp_path)
    _write(
        root,
        "pyscarcopula/vine/vine.py",
        "from pyscarcopula.numerical._rvine_backend import "
        "dispatch_rvine_backend\n",
    )

    assert "vine-native-boundary" in _rules(root)


def test_jacobi_python_sampling_kernel_is_rejected(tmp_path):
    root = _minimal_repository(tmp_path)
    _write(
        root,
        "pyscarcopula/numerical/jacobi_sampling.py",
        "@njit\ndef _lamperti_chunk_kernel():\n    return np.sin(0.0)\n",
    )

    assert "jacobi-native-sampling-ownership" in _rules(root)


def test_jacobi_numba_kernel_is_rejected_after_cleanup(tmp_path):
    root = _minimal_repository(tmp_path)
    _write(
        root,
        "pyscarcopula/numerical/jacobi_sparse.py",
        "from numba import njit\n\n@njit\ndef _sparse_to_dense():\n    pass\n",
    )

    assert "jacobi-python-cleanup" in _rules(root)


def test_jacobi_numpy_model_formula_is_rejected_after_cleanup(tmp_path):
    root = _minimal_repository(tmp_path)
    _write(
        root,
        "pyscarcopula/numerical/jacobi_tm.py",
        "import numpy as np\n\ndef transition(x):\n    return np.exp(-x)\n",
    )

    assert "jacobi-python-cleanup" in _rules(root)


def test_jacobi_strategy_legacy_numerical_dispatch_is_rejected(tmp_path):
    root = _minimal_repository(tmp_path)
    _write(
        root,
        "pyscarcopula/strategy/scar_jacobi.py",
        "from pyscarcopula.numerical.jacobi_tm import jacobi_matrix_loglik\n"
        "jacobi_native.PreparedScarJacobiEvaluator([], object())\n",
    )

    assert "jacobi-native-strategy-facade" in _rules(root)


def test_jacobi_strategy_module_alias_bypass_is_rejected(tmp_path):
    root = _minimal_repository(tmp_path)
    _write(
        root,
        "pyscarcopula/strategy/scar_jacobi.py",
        "from pyscarcopula.numerical import jacobi_tm\n"
        "jacobi_native.PreparedScarJacobiEvaluator([], object())\n",
    )

    assert "jacobi-native-strategy-facade" in _rules(root)


def test_jacobi_strategy_requires_actual_prepared_evaluator_call(tmp_path):
    root = _minimal_repository(tmp_path)
    _write(
        root,
        "pyscarcopula/strategy/scar_jacobi.py",
        "jacobi_native.PreparedScarJacobiEvaluator\n",
    )

    assert "jacobi-native-strategy-facade" in _rules(root)


def test_foundation_helpers_have_single_canonical_owners():
    include = ROOT / "pyscarcopula" / "_cpp" / "include" / "scar"
    source = ROOT / "pyscarcopula" / "_cpp" / "src"
    expected_headers = {
        "core/span.hpp",
        "core/matrix_view.hpp",
        "core/checked_arithmetic.hpp",
        "core/threading.hpp",
        "core/status.hpp",
        "core/result.hpp",
        "math/normal.hpp",
        "copula/model_descriptor.hpp",
        "copula/transforms.hpp",
        "copula/rotation.hpp",
    }
    assert all((include / path).is_file() for path in expected_headers)

    cpp_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(source.rglob("*.cpp"))
    )
    header_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(include.rglob("*.hpp"))
    )

    for removed_duplicate in (
        "validate_threads",
        "effective_threads",
        "checked_product",
    ):
        assert removed_duplicate not in cpp_text

    assert len(re.findall(r"\bdouble\s+normal_cdf\s*\(", cpp_text)) == 1
    assert len(re.findall(r"\bdouble\s+normal_quantile\s*\(", cpp_text)) == 1
    assert len(re.findall(
        r"\bdouble\s+regularized_gamma_p\s*\(", cpp_text)) == 1
    assert len(re.findall(r"\bdouble\s+log_gamma\s*\(", cpp_text)) == 1
    assert "::lgamma_r(value, &sign)" in (
        source / "math" / "gamma.cpp"
    ).read_text(encoding="utf-8")
    for path in sorted(source.rglob("*.cpp")):
        if path == source / "math" / "gamma.cpp":
            continue
        assert re.search(
            r"(?<![_\w])(?:std::|::)?lgamma\s*\(",
            path.read_text(encoding="utf-8"),
        ) is None, path
    assert len(re.findall(
        r"\bdouble\s+regularized_beta\s*\(", cpp_text)) == 1
    assert len(re.findall(r"\bdouble\s+softplus\s*\(", cpp_text)) == 1
    assert len(re.findall(
        r"\bdouble\s+inverse_softplus\s*\(", cpp_text)) == 1
    assert len(re.findall(
        r"\bdouble\s+logistic_unit\s*\(", cpp_text)) == 1
    assert len(re.findall(
        r"\bResult<std::size_t>\s+rosenblatt_output_size\s*\(",
        cpp_text,
    )) == 1
    assert "CopulaSpec::expected_dimension" not in cpp_text
    assert cpp_text.count(
        "TypedModelDescriptor CopulaSpec::model_descriptor()"
    ) == 1
    descriptor_text = (
        include / "copula" / "model_descriptor.hpp"
    ).read_text(encoding="utf-8")
    assert "class TypedModelDescriptor" in descriptor_text
    assert "int expected_dimension() const noexcept" in descriptor_text
    spec_text = (include / "copula" / "spec.hpp").read_text(encoding="utf-8")
    assert "TypedModelDescriptor model_descriptor() const noexcept;" in spec_text
    assert "int expected_dimension() const noexcept;" not in spec_text
    assert "struct DoubleView" not in header_text
    assert "using DoubleView = Span<const double>;" in (
        include / "core" / "span.hpp"
    ).read_text(encoding="utf-8")
    assert "n_threads > 256" not in cpp_text

    foundation_files = [
        *sorted((include / "core").glob("*.hpp")),
        *sorted((include / "math").glob("*.hpp")),
        include / "copula" / "transforms.hpp",
        include / "copula" / "rotation.hpp",
        source / "math" / "normal.cpp",
        source / "copula" / "transforms.cpp",
        source / "copula" / "rotation.cpp",
    ]
    forbidden_foundation_include = re.compile(
        r'#\s*include\s+"scar/(?:detail/|factor\.hpp|gas(?:_rvine)?\.hpp|'
        r'ou\.hpp|rvine(?:_plan)?\.hpp|copula\.hpp)'
    )
    for path in foundation_files:
        assert forbidden_foundation_include.search(
            path.read_text(encoding="utf-8")
        ) is None, path

    validation_text = (
        source / "scar_ou" / "validation.cpp"
    ).read_text(encoding="utf-8")
    assert "Result<std::size_t> rosenblatt_output_size(" in validation_text
    assert "result.failure.coordinate" in validation_text
    assert "return success(output_size);" in validation_text
    for name in ("gaussian_rosenblatt.cpp", "student_rosenblatt.cpp"):
        consumer = (source / "scar_ou" / name).read_text(encoding="utf-8")
        assert "const Result<std::size_t> output_shape" in consumer
        assert "output_shape.is_ok()" in consumer


@pytest.mark.parametrize("source", [
    "double local(double x) { return 0.5 * (1.0 + std::erf(x)); }\n",
    "double regularized_gamma_p(double a, double x) { return x / a; }\n",
    "double local(double x) { return std::lgamma(x); }\n",
    "double betacf(double a, double b, double x) { return a + b + x; }\n",
    "double local(double x) { return 1.0 / (1.0 + std::exp(-x)); }\n",
    "double local(double x) { return std::log1p(std::exp(-std::abs(x))) "
    "+ std::max(x, 0.0); }\n",
])
def test_foundation_formula_duplicate_gate_rejects_semantic_clones(
        tmp_path, source):
    root = _minimal_repository(tmp_path)
    _write(root, "pyscarcopula/_cpp/src/copula/duplicate.cpp", source)

    violations = CHECKER_MODULE.check_foundation_formula_duplicates(root)

    assert violations
    assert {item.rule for item in violations} == {
        "foundation-formula-duplicates"}


def test_python_exact_duplicate_gate_covers_cross_and_same_file_clones(
        tmp_path):
    root = _minimal_repository(tmp_path)
    body = (
        "def copied(value):\n"
        "    result = []\n"
        "    for item in value:\n"
        "        if item is not None:\n"
        "            result.append(item)\n"
        "    return tuple(result)\n"
    )
    _write(root, "pyscarcopula/first.py", body + body)
    _write(root, "pyscarcopula/second.py", body)

    violations = CHECKER_MODULE.check_python_exact_duplicates(root)

    assert len(violations) == 3
    assert {item.rule for item in violations} == {
        "python-exact-duplicates"}


def test_pair_copulas_are_vertical_and_prepared_once():
    include = ROOT / "pyscarcopula" / "_cpp" / "include" / "scar"
    source = ROOT / "pyscarcopula" / "_cpp" / "src" / "copula"
    manifest = include / "copula" / "pair" / "families.def"
    entries = re.findall(
        r"^SCAR_PAIR_FAMILY\(\s*([A-Za-z][A-Za-z0-9_]*)\s*,\s*"
        r"([a-z][a-z0-9_]*)\s*,",
        manifest.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert entries
    families = tuple(package for _, package in entries)

    assert not (include / "detail" / "copula.hpp").exists()
    assert (include / "copula" / "prepared_pair_kernel.hpp").is_file()
    for family in families:
        assert (include / "copula" / "pair" / f"{family}.hpp").is_file()
        assert (source / "pair" / f"{family}.cpp").is_file()

    registry = (source / "pair" / "runtime_registry.cpp").read_text(
        encoding="utf-8")
    assert registry.count("switch (family)") == 1
    assert registry.count(
        '#include "scar/copula/pair/families.def"') == 2

    kernel = (include / "copula" / "pair" / "kernel.hpp").read_text(
        encoding="utf-8")
    assert "PairSupportKernel" not in kernel
    assert "Rotation" not in kernel
    assert "Transform" not in kernel
    for family in families:
        implementation = (source / "pair" / f"{family}.cpp").read_text(
            encoding="utf-8")
        assert "_h_rotated" not in implementation
        assert "_h_inverse_rotated" not in implementation
        assert "scar/copula/rotation.hpp" not in implementation

    dispatch = (source / "dispatch.cpp").read_text(encoding="utf-8")
    for family in ("clayton", "gumbel", "frank", "joe"):
        assert re.search(rf"\b{family}_[A-Za-z0-9_]+", dispatch) is None
    assert "const scar::PreparedPairKernel kernel(spec);" in dispatch
    assert "is_pair_copula_family" not in dispatch

    core = (source / "core.cpp").read_text(encoding="utf-8")
    assert "const PreparedPairKernel kernel(spec);" in core
    assert core.count("scar_internal::copula_log_pdf_unrotated(") == 0
    assert "is_pair_copula_family" not in core

    binding = (
        ROOT / "pyscarcopula" / "_cpp" / "src" / "bindings" / "copula.cpp"
    ).read_text(encoding="utf-8")
    assert binding.count(
        '#include "scar/copula/pair/families.def"') == 1
    adapter = (
        ROOT / "pyscarcopula" / "_native" / "_descriptors.py"
    ).read_text(encoding="utf-8")
    rvine_adapter = (
        ROOT / "pyscarcopula" / "_native" / "vine.py"
    ).read_text(encoding="utf-8")
    for enum_name, _ in entries:
        assert f'.value("{enum_name}",' not in binding
        assert f"CopulaFamily.{enum_name}" not in adapter
    assert "_builtin_copula_types" not in rvine_adapter
    assert "__pyscarcopula_native_rvine__" not in rvine_adapter
    assert "from pyscarcopula.copula." not in rvine_adapter

    sources = (
        ROOT / "pyscarcopula" / "_cpp" / "build_support" / "sources.py"
    ).read_text(encoding="utf-8")
    assert "PAIR_FAMILY_SOURCES = _pair_family_sources()" in sources
    assert "*PAIR_FAMILY_SOURCES" in sources


def test_multivariate_models_are_vertical_and_typed():
    cpp = ROOT / "pyscarcopula" / "_cpp"
    include = cpp / "include" / "scar"
    source = cpp / "src" / "copula"
    multivariate_include = include / "copula" / "multivariate"
    multivariate_source = source / "multivariate"

    assert not (source / "multivariate.cpp").exists()
    assert not (source / "families" / "student.cpp").exists()
    assert not (include / "detail" / "copula" / "student.hpp").exists()

    for relative in (
        "correlation/dense.hpp",
        "correlation/factor.hpp",
        "gaussian/model.hpp",
        "gaussian/density.hpp",
        "gaussian/conditional.hpp",
        "equicorrelation/model.hpp",
        "equicorrelation/kernel.hpp",
        "student/model.hpp",
        "student/distribution.hpp",
        "student/quantile.hpp",
        "student/ppf_cache.hpp",
        "student/density.hpp",
        "student/conditional.hpp",
        "student/rosenblatt.hpp",
    ):
        assert (multivariate_include / relative).is_file()

    for relative in (
        "correlation/conditional.cpp",
        "correlation/dense.cpp",
        "correlation/factor.cpp",
        "equicorrelation/evaluator.cpp",
        "gaussian/density.cpp",
        "gaussian/conditional.cpp",
        "equicorrelation/model.cpp",
        "equicorrelation/kernel.cpp",
        "student/distribution.cpp",
        "student/density.cpp",
        "student/evaluator.cpp",
        "student/conditional.cpp",
        "student/factor_density.cpp",
        "student/factor_grid.cpp",
        "student/ppf_cache.cpp",
        "student/quantile.cpp",
        "student/rosenblatt.cpp",
    ):
        assert (multivariate_source / relative).is_file()

    dispatch = (multivariate_source / "dispatch.cpp").read_text(
        encoding="utf-8")
    assert len(dispatch.splitlines()) <= 100
    for model_implementation in (
        "StudentWorkspace",
        "EquicorrStats",
        "conditional_df",
        "parallel_for_blocks",
        "student_fill_",
        "equicorr_log_pdf_from_stats(",
        "normal_quantile",
        "cholesky",
    ):
        assert model_implementation not in dispatch

    conditional_engine = (
        multivariate_source / "correlation" / "conditional.cpp"
    ).read_text(encoding="utf-8")
    for model_specific in (
        "Student", "student_", "conditional_df", "chi_square",
    ):
        assert model_specific not in conditional_engine
    student_conditional = (
        multivariate_source / "student" / "conditional.cpp"
    ).read_text(encoding="utf-8")
    assert "student_conditional_scale" in student_conditional
    assert "conditional_df" in student_conditional

    spec = (include / "copula" / "spec.hpp").read_text(encoding="utf-8")
    for old_field in (
        "l_inv;", "log_det;", "ppf_n_obs;", "ppf_nodes;", "ppf_table;",
        "gaussian_z1_cache;", "equicorr_sum_cache;",
    ):
        assert old_field not in spec
    storage = (include / "copula" / "model_storage.hpp").read_text(
        encoding="utf-8")
    assert "std::variant<" in storage
    assert "student::DenseModelStorage" in storage
    assert "gaussian::FactorModelStorage" in storage

    factor = (
        multivariate_include / "correlation" / "factor.hpp"
    ).read_text(encoding="utf-8")
    assert "FactorCorrelationOperator" in factor
    assert "FactorStudent" not in factor

    for package in (
        multivariate_include / "gaussian",
        multivariate_source / "gaussian",
    ):
        for path in package.rglob("*"):
            if path.is_file():
                text = path.read_text(encoding="utf-8")
                assert "/student/" not in text
                assert "Student" not in text


def test_application_modules_use_prepared_copula_interfaces():
    cpp = ROOT / "pyscarcopula" / "_cpp"
    include = cpp / "include" / "scar"
    source = cpp / "src"

    emission_header = include / "copula" / "prepared_dynamic_emission.hpp"
    emission_source = source / "copula" / "prepared_dynamic_emission.cpp"
    assert (include / "copula" / "grid_values.hpp").is_file()
    assert emission_header.is_file()
    assert emission_source.is_file()
    emission_contract = emission_header.read_text(encoding="utf-8")
    assert "class PreparedDynamicEmission" in emission_contract
    assert "class PreparedDynamicEmissionWorkspace" in emission_contract
    assert "DynamicEmissionRowResult evaluate_parameter(" in emission_contract
    assert "bool observation_cache_compatible(" in emission_contract
    assert "copula/pair/" not in emission_contract
    assert "copula/multivariate/" not in emission_contract
    assert "detail/copula/" not in emission_contract

    gas = (source / "gas" / "evaluator.cpp").read_text(encoding="utf-8")
    assert "PreparedDynamicEmission" in gas
    assert '#include "scar/copula.hpp"' not in gas
    assert "detail/copula/dispatch.hpp" not in gas
    assert "copula/multivariate/" not in gas
    assert "copula/pair/" not in gas
    gas_header = (include / "gas.hpp").read_text(encoding="utf-8")
    assert "GasStateResult initial_state_prepared(" in gas_header
    assert "GasUpdateResult update_one_prepared(" in gas_header
    assert "GasUpdateResult update_observation_prepared(" in gas_header

    static_header = (include / "copula.hpp").read_text(encoding="utf-8")
    static_source = (source / "likelihood" / "static.cpp").read_text(
        encoding="utf-8")
    assert "PreparedDynamicEmission" in static_header
    assert "emission_" in static_header
    assert "emission_->evaluate_parameter(" in static_source

    ou_header = (include / "ou.hpp").read_text(encoding="utf-8")
    assert '#include "scar/copula.hpp"' not in ou_header
    assert "const PreparedDynamicEmission* prepared_emission_" in ou_header
    assert "PreparedDynamicEmission emission_;" in ou_header
    for name in (
        "evaluator.cpp",
        "likelihood.cpp",
        "prediction.cpp",
        "state_distribution.cpp",
        "transition.cpp",
        "validation.cpp",
    ):
        text = (source / "scar_ou" / name).read_text(encoding="utf-8")
        assert "PreparedDynamicEmission" in text, name
        assert "detail/copula/dispatch.hpp" not in text, name
        assert "copula/multivariate/" not in text, name
        assert "copula/pair/" not in text, name
        assert "factor_operator(" not in text, name
        assert "student_ppf_nodes(" not in text, name

    gradient = (source / "scar_ou" / "gradient.cpp").read_text(
        encoding="utf-8")
    assert "&prepared->compatibility_spec() == &copula" not in gradient
    assert "if (prepared != nullptr)" in gradient
    assert "const CopulaSpec& copula = emission.compatibility_spec();" in gradient

    vine_header = (include / "rvine.hpp").read_text(encoding="utf-8")
    assert "PreparedPairKernel kernel;" in vine_header
    assert "PreparedPairKernel transposed_kernel;" in vine_header
    for path in (source / "vine").glob("*.cpp"):
        text = path.read_text(encoding="utf-8")
        assert "detail/copula/dispatch.hpp" not in text, path
        assert "copula/multivariate/" not in text, path
        assert "copula/pair/" not in text, path

    composition = (source / "gas" / "rvine_sampler.cpp").read_text(
        encoding="utf-8")
    assert '#include "scar/gas_rvine.hpp"' in composition
    assert '#include "scar/rvine.hpp"' in composition
    assert '#include "scar/copula/prepared_dynamic_emission.hpp"' in composition
    assert "gas_emissions" in composition
    assert "gas_workspaces" in composition
    assert "update_one_prepared(" in composition
    assert "evaluator.update_one(" not in composition

    composition_header = (include / "gas_rvine.hpp").read_text(
        encoding="utf-8")
    assert '#include "scar/copula/spec.hpp"' in composition_header
    assert '#include "scar/copula.hpp"' not in composition_header


def test_public_cpp_api_is_domain_scoped_and_caller_neutral():
    cpp = ROOT / "pyscarcopula" / "_cpp"
    include = cpp / "include" / "scar"
    source = cpp / "src"

    contracts = {
        include / "static" / "result.hpp": "StaticObjectiveResult",
        include / "gas" / "result.hpp": "GasLogLikResult",
        include / "scar_ou" / "result.hpp": "ScarOuVectorResult",
        include / "copula" / "result.hpp": "MultivariateRowsResult",
        include / "vine" / "result.hpp": "SampleResult",
        include / "gas_rvine" / "result.hpp": "GasRvineSampleResult",
    }
    for path, marker in contracts.items():
        text = path.read_text(encoding="utf-8")
        assert marker in text
        assert "Status status" in text
        assert "FailureContext failure" in text
        assert re.search(r"\bint\s+status\b|\bfailure_index\b", text) is None

    umbrella_expectations = {
        include / "copula.hpp": (
            ('#include "scar/copula/result.hpp"',
             '#include "scar/static/result.hpp"'),
            ("struct MultivariateRowsResult", "struct StaticObjectiveResult")),
        include / "gas.hpp": (
            '#include "scar/gas/result.hpp"',
            "struct GasLogLikResult"),
        include / "ou.hpp": (
            '#include "scar/scar_ou/result.hpp"',
            "struct LogLikResult"),
        include / "rvine.hpp": (
            '#include "scar/vine/result.hpp"',
            "struct SampleResult"),
        include / "gas_rvine.hpp": (
            '#include "scar/gas_rvine/result.hpp"',
            "struct GasRvineSampleResult"),
    }
    for path, (required, forbidden) in umbrella_expectations.items():
        text = path.read_text(encoding="utf-8")
        required = required if isinstance(required, tuple) else (required,)
        forbidden = forbidden if isinstance(forbidden, tuple) else (forbidden,)
        assert all(marker in text for marker in required)
        assert all(marker not in text for marker in forbidden)

    copula_header = (include / "copula.hpp").read_text(encoding="utf-8")
    assert "copula/multivariate/" not in copula_header
    assert "copula/pair/" not in copula_header
    assert "copula/prepared_dynamic_emission.hpp" not in copula_header
    status_header = (include / "core" / "status.hpp").read_text(
        encoding="utf-8")
    assert "operator==(Status" not in status_header
    assert "operator!=(Status" not in status_header

    ou_header = (include / "ou.hpp").read_text(encoding="utf-8")
    assert "struct Workspace;" in ou_header
    assert "std::unique_ptr<Workspace>" in ou_header
    assert "ScarOuGridGradientWorkspace" not in ou_header
    assert "ScarOuSpectralGradientWorkspace" not in ou_header
    assert "ScarOuVectorResult predictive_mean_local_gh(" in ou_header
    assert "ScarOuVectorResult mixture_h_pair_auto(" in ou_header
    assert "ScarOuVectorResult predictive_mean(const OuParams& params)" in ou_header
    assert "ScarOuVectorResult mixture_h_pair(const OuParams& params)" in ou_header
    workspace = source / "scar_ou" / "gradient_workspace.hpp"
    assert workspace.is_file()
    assert "ScarOuSpectralGradientWorkspace" in workspace.read_text(
        encoding="utf-8")

    caller_specific = re.compile(
        r"\b(?:Python|pybind11|NumPy|PyObject)\b", re.IGNORECASE)
    compute_files = [
        *include.rglob("*.hpp"),
        *(path for path in source.rglob("*.cpp")
          if "bindings" not in path.relative_to(source).parts),
        *(path for path in source.rglob("*.hpp")
          if "bindings" not in path.relative_to(source).parts),
    ]
    for path in compute_files:
        assert caller_specific.search(path.read_text(encoding="utf-8")) is None


def test_checker_rejects_workspace_and_caller_leaks(tmp_path):
    cpp = "pyscarcopula/_cpp"
    include = f"{cpp}/include/scar"
    source = f"{cpp}/src"

    result_contracts = {
        "static/result.hpp": ("StaticObjectiveResult",),
        "gas/result.hpp": (
            "GasLogLikResult", "GasFilterResult", "GasUpdateResult",
            "GasStateResult", "GasPredictResult", "GasPathResult"),
        "scar_ou/result.hpp": (
            "ScarOuVectorResult", "LogLikResult", "GradLogLikResult",
            "StateDistribution", "SmoothedStateDistribution",
            "OuGridFilterResult",),
        "copula/result.hpp": (
            "MultivariateRowsResult", "MultivariateGridResult",
            "EquicorrPreparationResult"),
        "vine/result.hpp": (
            "SampleResult", "ConditionalSampleResult", "DensityResult",
            "RosenblattResult", "MCMCResult"),
        "gas_rvine/result.hpp": ("GasRvineSampleResult",),
    }
    for relative, result_names in result_contracts.items():
        _write(
            tmp_path,
            f"{include}/{relative}",
            "".join(
                f"struct {name} {{\n"
                "  Status status;\n"
                "  FailureContext failure;\n"
                "};\n"
                for name in result_names
            ),
        )

    _write(
        tmp_path,
        f"{include}/copula.hpp",
        '#include "scar/copula/result.hpp"\n'
        '#include "scar/static/result.hpp"\n',
    )
    _write(
        tmp_path,
        f"{include}/gas.hpp",
        '#include "scar/gas/result.hpp"\n',
    )
    _write(
        tmp_path,
        f"{include}/rvine.hpp",
        '#include "scar/vine/result.hpp"\n',
    )
    _write(
        tmp_path,
        f"{include}/gas_rvine.hpp",
        '#include "scar/gas_rvine/result.hpp"\n',
    )
    ou_header = _write(
        tmp_path,
        f"{include}/ou.hpp",
        '#include "scar/scar_ou/result.hpp"\n'
        "struct Workspace;\n"
        "std::unique_ptr<Workspace> workspace_;\n"
        "ScarOuVectorResult predictive_mean_local_gh();\n"
        "ScarOuVectorResult mixture_h_pair_auto();\n"
        "ScarOuVectorResult predictive_mean(const OuParams& params);\n"
        "ScarOuVectorResult mixture_h_pair(const OuParams& params);\n",
    )
    _write(
        tmp_path,
        f"{source}/scar_ou/gradient_workspace.hpp",
        "struct ScarOuSpectralGradientWorkspace {};\n",
    )

    assert CHECKER_MODULE.check_public_cpp_api(tmp_path) == []

    status_header = _write(
        tmp_path,
        f"{include}/core/status.hpp",
        "bool operator==(Status lhs, int rhs);\n",
    )
    violations = CHECKER_MODULE.check_public_cpp_api(tmp_path)
    assert any(
        "must not compare implicitly" in violation.message
        for violation in violations
    )
    status_header.write_text("#pragma once\n", encoding="utf-8")

    copula_header = tmp_path / f"{include}/copula.hpp"
    copula_contract = copula_header.read_text(encoding="utf-8")
    copula_header.write_text(
        copula_contract
        + '#include "scar/copula/multivariate/student/conditional.hpp"\n',
        encoding="utf-8",
    )
    violations = CHECKER_MODULE.check_public_cpp_api(tmp_path)
    assert any(
        "imports concrete implementation" in violation.message
        for violation in violations
    )
    copula_header.write_text(copula_contract, encoding="utf-8")

    ou_contract = ou_header.read_text(encoding="utf-8")
    ou_header.write_text(
        ou_contract
        + "class ScarOuEvaluator { public:\n"
        + "  void legacy(int& status);\nprivate:\n};\n",
        encoding="utf-8",
    )
    violations = CHECKER_MODULE.check_public_cpp_api(tmp_path)
    assert any(
        "legacy status out-parameter" in violation.message
        for violation in violations
    )
    ou_header.write_text(ou_contract, encoding="utf-8")

    vine_header = tmp_path / f"{include}/rvine.hpp"
    vine_contract = vine_header.read_text(encoding="utf-8")
    vine_header.write_text(
        vine_contract + "void legacy(std::int64_t& failure_row);\n",
        encoding="utf-8",
    )
    violations = CHECKER_MODULE.check_public_cpp_api(tmp_path)
    assert any(
        "failure out-parameters" in violation.message
        for violation in violations
    )
    vine_header.write_text(vine_contract, encoding="utf-8")

    ou_header.write_text(
        ou_header.read_text(encoding="utf-8")
        + "struct ScarOuGridGradientWorkspace {};\n",
        encoding="utf-8",
    )
    violations = CHECKER_MODULE.check_public_cpp_api(tmp_path)
    assert any(
        "exposes private workspace" in violation.message
        for violation in violations
    )

    _write(
        tmp_path,
        f"{source}/scar_ou/caller_leak.cpp",
        "// Python-specific computational contract.\n",
    )
    violations = CHECKER_MODULE.check_public_cpp_api(tmp_path)
    assert any(
        "caller-neutral" in violation.message for violation in violations
    )


def test_bindings_are_thin_and_domain_scoped():
    bindings = ROOT / "pyscarcopula/_cpp/src/bindings"
    assert not (bindings / "common.hpp").exists()
    for name in ("module.hpp", "array.hpp", "array.cpp"):
        assert (bindings / name).is_file()

    module = (bindings / "module.hpp").read_text(encoding="utf-8")
    assert 'scar/' not in module
    array = (bindings / "array.hpp").read_text(encoding="utf-8")
    for marker in (
            "CopulaSpec", "GridValues", "Gas", "OuBackend", "RVine",
            "Student", "Result"):
        assert marker not in array

    owners = {
        "copula.cpp": ("StaticObjectiveResult",),
        "multivariate.cpp": (
            "MultivariateRowsResult", "ConditionalSampleResult"),
        "gas.cpp": ("GasLogLikResult", "GasFilterResult"),
        "scar_ou.cpp": (
            "LogLikResult", "GradLogLikResult",
            "SmoothedStateDistribution", "OuGridFilterResult"),
        "factor.cpp": (
            "FactorStudentRowsResult", "FactorStudentJointResult",
            "FactorStudentGridResult"),
        "rvine.cpp": (
            "SampleResult", "ConditionalSampleResult", "DensityResult",
            "RosenblattResult", "MCMCResult"),
    }
    for owner, results in owners.items():
        source = (bindings / owner).read_text(encoding="utf-8")
        assert '#include "module.hpp"' in source
        for result in results:
            assert result in source

    for name in (
            "common.cpp", "parallel.cpp", "gas.cpp", "rvine.cpp",
            "scar_ou_types.cpp", "scar_ou.cpp"):
        source = (bindings / name).read_text(encoding="utf-8")
        assert 'copula/multivariate/student/' not in source

    factor = (bindings / "factor.cpp").read_text(encoding="utf-8")
    assert factor.count(
        'output["status"] = static_cast<int>(result.status);') >= 3
    assert "matrix_copy(" not in factor
    assert "vector_copy(" not in factor
    assert re.search(
        r"py::gil_scoped_release\s+\w+\s*;"
        r"(?:(?!^\s*\}).)*?\.(?:request|mutable_data|writeable)\s*\(",
        factor,
        re.MULTILINE | re.DOTALL,
    ) is None

    scar_ou = (bindings / "scar_ou.cpp").read_text(encoding="utf-8")
    for model_call in (
            "build_ou_grid(", "build_grid_transition_operator(",
            "forward_filter_emissions(", "backward_filter_emissions(",
            "smooth_state_emissions("):
        assert model_call not in scar_ou
    copula = (bindings / "copula.cpp").read_text(encoding="utf-8")
    assert '#include "scar/factor.hpp"' not in copula
    assert (
        '#include "scar/copula/multivariate/correlation/factor.hpp"'
        in copula)

    ou_header = (
        ROOT / "pyscarcopula/_cpp/include/scar/ou.hpp"
    ).read_text(encoding="utf-8")
    ou_result = (
        ROOT / "pyscarcopula/_cpp/include/scar/scar_ou/result.hpp"
    ).read_text(encoding="utf-8")
    assert "OuGridFilterResult filter_ou_grid_emissions(" in ou_header
    assert "struct OuGridFilterResult" in ou_result


def test_checker_rejects_umbrella_and_cross_domain_binding_leaks(
    tmp_path,
):
    root = _minimal_repository(tmp_path)
    bindings = root / "pyscarcopula/_cpp/src/bindings"
    _write(root, "pyscarcopula/_cpp/src/bindings/common.hpp", "#pragma once\n")
    _write(
        root,
        "pyscarcopula/_cpp/src/bindings/common.cpp",
        '#include "module.hpp"\n#include "scar/ou.hpp"\n',
    )
    _write(
        root,
        "pyscarcopula/_cpp/src/bindings/array.hpp",
        "struct CopulaSpec;\n",
    )
    _write(root, "pyscarcopula/_cpp/src/bindings/array.cpp", "")
    _write(
        root,
        "pyscarcopula/_cpp/src/bindings/gas.cpp",
        '#include "module.hpp"\n#include "scar/ou.hpp"\n',
    )

    violations = CHECKER_MODULE.check_thin_bindings(root)
    messages = {violation.message for violation in violations}
    assert "the umbrella bindings/common.hpp must not be restored" in messages
    assert any("shared binding helper imports" in message for message in messages)
    assert any("unrelated domain API" in message for message in messages)
    assert any("array/view conversion depends" in message for message in messages)
    assert bindings.is_dir()


def test_checker_rejects_policy_gil_and_model_logic_in_bindings(
    tmp_path,
):
    root = _minimal_repository(tmp_path)
    _write(root, "pyscarcopula/_cpp/src/bindings/array.hpp", "")
    _write(root, "pyscarcopula/_cpp/src/bindings/array.cpp", "")
    _write(
        root,
        "pyscarcopula/_cpp/src/bindings/factor.cpp",
        '#include "module.hpp"\n'
        "void bind() {\n"
        "    py::array_t<double> output(2);\n"
        "    {\n"
        "        py::gil_scoped_release release;\n"
        "        output.mutable_data();\n"
        "    }\n"
        "    FactorStudentRowsResult result;\n"
        "    if (result.failure.index >= 0) { throw 1; }\n"
        "    matrix_copy(values, 1, 1);\n"
        "}\n",
    )
    _write(
        root,
        "pyscarcopula/_cpp/src/bindings/scar_ou.cpp",
        '#include "module.hpp"\nvoid bind() { build_ou_grid(); }\n',
    )
    _write(
        root,
        "pyscarcopula/_cpp/src/bindings/copula.cpp",
        '#include "module.hpp"\n#include "scar/factor.hpp"\n',
    )

    messages = {
        violation.message
        for violation in CHECKER_MODULE.check_thin_bindings(root)
    }
    assert any("result-dependent policy" in message for message in messages)
    assert any("GIL is released" in message for message in messages)
    assert any("model orchestration" in message for message in messages)
    assert any("duplicates shared array" in message for message in messages)
    assert any("focused factor correlation" in message for message in messages)


def test_setup_build_path_does_not_mutate_path():
    source = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert 'os.environ["PATH"]' not in source
    assert "shutil.which" not in source


def test_gate4_workflow_covers_required_compilers_and_build_boundaries():
    source = (
        ROOT / ".github/workflows/parallel-release-gates.yml"
    ).read_text(encoding="utf-8")
    for configuration in (
        "linux-gcc-py310",
        "linux-gcc-py314",
        "linux-clang-py312",
        "windows-msvc-py312",
        "windows-mingw64-py312",
        "macos-arm64-clang-py312",
    ):
        assert source.count(f"name: {configuration}") == 1

    assert "msystem: MINGW64" in source
    assert "install: mingw-w64-x86_64-gcc" in source
    assert "python tools/build_cpp_tests.py --force -j 4" in source
    assert "python tools/check_python_ownership.py" in source
    assert "python -m pyscarcopula._native.smoke" in source
    assert "Run full non-benchmark suite against wheel" in source
    assert source.index("Verify Python-free C++ build boundary") < source.index(
        "Build strict wheel")


def test_release_workflows_automate_sanitizers_and_wheel_smoke():
    release = (
        ROOT / ".github/workflows/parallel-release-gates.yml"
    ).read_text(encoding="utf-8")
    assert "Run architecture contracts" in release
    assert "Run Python-free ASan and UBSan executable" in release
    assert "--sanitize address-undefined" in release
    assert "Run Python-free ThreadSanitizer executable" in release
    assert "--sanitize thread" in release
    assert release.count("tools/build_cpp_tests.py --force -j 4") >= 3
    assert "python -m pytest -n 4 -q" in release
    assert "tests/test_cpp_architecture.py tests/test_gate1_benchmarks.py" in release
    assert "-j 8" not in release
    assert "-n 8" not in release


def test_gate1_manifest_covers_full_fv6_performance_matrix():
    manifest = json.loads(
        (ROOT / "benchmarks/gate1_manifest_v2.json").read_text(
            encoding="utf-8"))
    cases = manifest["cases"]
    ids = [case["id"] for case in cases]

    assert manifest["manifest_id"] == "pyscarcopula-gate1-v2"
    assert len(ids) == len(set(ids))
    assert manifest["thread_values"] == [1, 2, 4, "physical"]
    policy = manifest["protocol"]["regression_policy"]
    assert policy["maximum_runtime_ratio"] == 2.0
    assert policy["maximum_python_allocation_count_ratio"] == 2.0
    assert policy["maximum_python_peak_memory_ratio"] == 2.0
    assert policy["maximum_process_peak_rss_ratio"] == 2.0
    assert policy["maximum_parallel_scaling_loss_ratio"] == 2.0
    assert policy["require_checksum_match"] is True
    assert policy["require_domain_diagnostics_match"] is True

    models = {case["model"] for case in cases}
    assert {
        "pair",
        "dense_gaussian",
        "factor_gaussian",
        "gaussian_shrinkage",
        "gaussian_cholesky",
        "dense_student",
        "factor_student",
        "equicorr_gaussian",
        "gas_pair",
        "gas_equicorr",
        "gas_dense_student",
        "gas_stochastic_student_factor",
        "scar_ou_pair",
        "scar_ou_equicorr",
        "scar_ou_stochastic_student_dense",
        "scar_ou_stochastic_student_factor",
        "scar_jacobi_pair",
        "student_ppf",
        "correlation",
        "cvine_static",
        "rvine_static",
        "rvine_gas",
        "rvine_scar_ou",
        "rvine_scar_jacobi",
    } <= models
    assert {case.get("topology", "dvine") for case in cases
            if case["runner"].startswith("vine_")} == {
                "cvine", "dvine", "rvine"}
    assert {case.get("dynamic") for case in cases
            if case["runner"] == "vine_dynamic_rosenblatt"} == {
                "gas", "scar_ou", "scar_jacobi"}
    assert {case["mode"] for case in cases
            if case["model"] == "scar_ou_pair"} >= {
                "cold_preparation", "prepared_repeated"}
    assert {case["mode"] for case in cases
            if case["model"] == "scar_jacobi_pair"} >= {
                "cold_preparation", "prepared_repeated"}


def test_gate1_runner_requires_external_append_only_artifacts(tmp_path):
    from tools import run_gate1_benchmarks as gate1

    assert gate1._required_thread_capacity([
        {"n_threads": 1},
        {"n_threads": 2},
        {"n_threads": 4},
        {"n_threads": "physical"},
    ]) == 4

    with pytest.raises(SystemExit, match="outside the product repository"):
        gate1._artifact_paths(SimpleNamespace(
            artifact_root=ROOT, output=None, summary=None))

    artifact_root, output, summary = gate1._artifact_paths(SimpleNamespace(
        artifact_root=tmp_path / "run", output=None, summary=None))
    assert artifact_root == (tmp_path / "run").resolve()
    assert output == artifact_root / "gate1_candidate.json"
    assert summary == artifact_root / "performance_summary.md"
    output.write_text("immutable", encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        gate1._artifact_paths(SimpleNamespace(
            artifact_root=artifact_root, output=None, summary=None))

    wheels = (ROOT / ".github/workflows/wheels.yml").read_text(
        encoding="utf-8")
    assert 'CIBW_BUILD: "cp310-* cp311-* cp312-* cp313-* cp314-*"' in wheels
    assert "python -m pyscarcopula._native.smoke" in wheels


def test_stateless_scar_bindings_release_gil_after_array_validation():
    source = (
        ROOT / "pyscarcopula/_cpp/src/bindings/scar_ou.cpp"
    ).read_text(encoding="utf-8")

    assert "with_observation_view_without_gil" in source
    assert source.count("observation_view_from_array(") == 2
    helper = source.index("with_observation_view_without_gil")
    view = source.index("observation_view_from_array(", helper)
    release = source.index("py::gil_scoped_release release", view)
    assert helper < view < release


def test_rvine_sample_binding_keeps_arrays_alive_and_releases_gil():
    source = (
        ROOT / "pyscarcopula/_cpp/src/bindings/rvine.cpp"
    ).read_text(encoding="utf-8")

    request = source.index('"rvine_sample"')
    release = source.index("py::gil_scoped_release release", request)
    native_call = source.index("scar::rvine::sample(", release)
    result_dict = source.index("py::dict diagnostics", native_call)
    assert request < release < native_call < result_dict
    for name in ("scalar_parameters", "row_parameters", "uniforms"):
        assert name in source[request:release]


def test_rvine_conditional_binding_keeps_arrays_alive_and_releases_gil():
    source = (
        ROOT / "pyscarcopula/_cpp/src/bindings/rvine.cpp"
    ).read_text(encoding="utf-8")

    request = source.index('"rvine_conditional_sample"')
    release = source.index("py::gil_scoped_release release", request)
    native_call = source.index("scar::rvine::conditional_sample(", release)
    result_dict = source.index("py::dict diagnostics", native_call)
    assert request < release < native_call < result_dict
    for name in (
            "scalar_parameters",
            "row_parameters",
            "given_values",
            "uniforms"):
        assert name in source[request:release]


def test_rvine_density_binding_keeps_arrays_alive_and_releases_gil():
    source = (
        ROOT / "pyscarcopula/_cpp/src/bindings/rvine.cpp"
    ).read_text(encoding="utf-8")

    request = source.index('"rvine_log_pdf_rows"')
    release = source.index("py::gil_scoped_release release", request)
    native_call = source.index("scar::rvine::log_pdf_rows(", release)
    result_dict = source.index("py::dict diagnostics", native_call)
    assert request < release < native_call < result_dict
    for name in (
            "scalar_parameters", "row_parameters", "observations"):
        assert name in source[request:release]


def test_rvine_rosenblatt_binding_keeps_arrays_alive_and_releases_gil():
    source = (
        ROOT / "pyscarcopula/_cpp/src/bindings/rvine.cpp"
    ).read_text(encoding="utf-8")

    request = source.index('"rvine_rosenblatt_transform"')
    release = source.index("py::gil_scoped_release release", request)
    native_call = source.index(
        "scar::rvine::rosenblatt_transform(", release)
    result_dict = source.index("py::dict diagnostics", native_call)
    assert request < release < native_call < result_dict
    for name in (
            "scalar_parameters", "row_parameters", "observations"):
        assert name in source[request:release]


def test_dynamic_rvine_rosenblatt_binding_releases_gil():
    source = (
        ROOT / "pyscarcopula/_cpp/src/bindings/rvine.cpp"
    ).read_text(encoding="utf-8")

    request = source.index('"dynamic_rvine_rosenblatt_transform"')
    release = source.index("py::gil_scoped_release release", request)
    native_call = source.index(
        "scar::dynamic_rvine_rosenblatt_transform(", release)
    result_dict = source.index("py::dict diagnostics", native_call)
    assert request < release < native_call < result_dict
    for name in (
            "scalar_parameters", "row_parameters", "observations"):
        assert name in source[request:release]


def test_dense_student_binding_keeps_arrays_alive_and_releases_gil():
    source = (
        ROOT / "pyscarcopula/_cpp/src/bindings/multivariate.cpp"
    ).read_text(encoding="utf-8")

    request = source.index('"dense_student_rosenblatt_transform"')
    release = source.index("py::gil_scoped_release release", request)
    native_call = source.index("scar::student_rosenblatt_dense(", release)
    result_dict = source.index(
        "return multivariate_rosenblatt_result_to_dict(result);",
        native_call,
    )
    helper = source.index("py::dict multivariate_rosenblatt_result_to_dict(")
    assert helper < request < release < native_call < result_dict
    for name in ("correlation_view", "df_view", "observations"):
        assert name in source[request:release]


def test_rvine_mcmc_binding_keeps_arrays_alive_and_releases_gil():
    source = (
        ROOT / "pyscarcopula/_cpp/src/bindings/rvine.cpp"
    ).read_text(encoding="utf-8")

    request = source.index("const auto mcmc_binding")
    release = source.index("py::gil_scoped_release release", request)
    native_call = source.index("scar::rvine::mcmc_chunk(", release)
    result_dict = source.index("py::dict diagnostics", native_call)
    assert request < release < native_call < result_dict
    for name in (
            "scalar_parameters",
            "row_parameters",
            "given_values",
            "current_state",
            "current_log_pdf",
            "proposal_uniforms",
            "acceptance_uniforms"):
        assert name in source[request:release]


@pytest.mark.parametrize(
    ("relative", "content", "expected_rule"),
    [
        (
            "pyscarcopula/_cpp/src/gas/bad.cpp",
            '#include "scar/ou.hpp"\n',
            "gas-independent-of-ou",
        ),
        (
            "pyscarcopula/_cpp/src/math/bad.cpp",
            '#include "scar/detail/safety.hpp"\n',
            "foundation-independent-of-models",
        ),
        (
            "pyscarcopula/_cpp/src/copula/bad.cpp",
            '#include "scar/gas.hpp"\n',
            "copula-independent-of-gas",
        ),
        (
            "pyscarcopula/_cpp/src/copula/families/bad.cpp",
            '#include "scar/detail/scar_ou/grid.hpp"\n',
            "families-independent-of-ou",
        ),
        (
            "pyscarcopula/_cpp/src/vine/bad.cpp",
            '#include "scar/gas_rvine.hpp"\n',
            "rvine-independent-of-dynamic-models",
        ),
        (
            "pyscarcopula/_cpp/src/gas/evaluator.cpp",
            '#include "scar/copula/multivariate/student/density.hpp"\n',
            "prepared-application-modules",
        ),
        (
            "pyscarcopula/_cpp/src/scar_ou/likelihood.cpp",
            '#include "scar/detail/copula/dispatch.hpp"\n',
            "prepared-application-modules",
        ),
        (
            "pyscarcopula/_cpp/src/copula/bad.cpp",
            '#include "scar/rvine.hpp"\n',
            "prepared-application-modules",
        ),
        (
            "pyscarcopula/_cpp/src/scar_ou/gradient.cpp",
            "&prepared->compatibility_spec() == &copula;\n",
            "prepared-application-modules",
        ),
        (
            "pyscarcopula/_cpp/include/scar/gas_rvine.hpp",
            '#include "scar/copula.hpp"\n',
            "prepared-application-modules",
        ),
        (
            "pyscarcopula/_cpp/src/gas/rvine_sampler.cpp",
            "evaluator.update_one();\n",
            "prepared-application-modules",
        ),
        (
            "pyscarcopula/_cpp/include/scar/detail/internal.hpp",
            "#pragma once\n",
            "removed-internal-header",
        ),
    ],
)
def test_forbidden_dependencies_produce_clear_rule(
    tmp_path,
    relative,
    content,
    expected_rule,
):
    root = _minimal_repository(tmp_path)
    _write(root, relative, content)
    if relative.endswith(".cpp"):
        source = Path(relative).relative_to(
            "pyscarcopula/_cpp/src").as_posix()
        if source.startswith("bindings/"):
            _set_manifest(root, (), ("bindings/module.cpp", source))
        else:
            _set_manifest(root, (source,))
    assert expected_rule in _rules(root)


@pytest.mark.parametrize(
    ("relative", "content"),
    [
        (
            ".github/workflows/rvine.yml",
            "env:\n  PYSCARCOPULA_TEST_RVINE_BACKEND: python_executor\n",
        ),
        (
            "pyscarcopula/numerical/student_gof.py",
            "from numba import njit\n",
        ),
        (
            "tests/reference/rvine_runtime.py",
            "from pyscarcopula._native import vine\n",
        ),
        (
            "tests/conftest.py",
            "install_reference_oracles()\n",
        ),
        (
            "ARCHITECTURE.md",
            "Production uses multivariate_native.py.\n",
        ),
    ],
)
def test_retired_vine_surfaces_produce_clear_rule(
        tmp_path, relative, content):
    root = _minimal_repository(tmp_path)
    _write(root, relative, content)
    assert "mandatory-vine-dispatch" in _rules(root)


@pytest.mark.parametrize(
    ("relative", "content"),
    [
        ("pyscarcopula/copula/_protocol.py", "class CopulaProtocol: pass\n"),
        ("pyscarcopula/vine/cvine.py", "class CVineCopula: pass\n"),
        ("pyscarcopula/vine/_conditional_cvine.py", "def sample(): pass\n"),
        ("pyscarcopula/numerical/tm_grid.py", "class TMGrid: pass\n"),
        ("pyscarcopula/__init__.py", "CopulaCapabilities = object\n"),
    ],
)
def test_removed_compatibility_surfaces_produce_clear_rule(
        tmp_path, relative, content):
    root = _minimal_repository(tmp_path)
    _write(root, relative, content)
    assert "removed-compatibility-cleanup" in _rules(root)


def test_source_manifest_detects_unlisted_cpp(tmp_path):
    root = _minimal_repository(tmp_path)
    _write(root, "pyscarcopula/_cpp/src/gas/unlisted.cpp")
    assert "source-manifest" in _rules(root)


def test_module_entrypoint_rejects_binding_implementation(tmp_path):
    root = _minimal_repository(tmp_path)
    module = root / "pyscarcopula/_cpp/src/bindings/module.cpp"
    module.write_text(
        module.read_text(encoding="utf-8")
        + 'module.def("unexpected", [] { return 1; });\n',
        encoding="utf-8",
    )
    assert "minimal-module-entrypoint" in _rules(root)


def test_public_header_cycle_is_rejected(tmp_path):
    root = _minimal_repository(tmp_path)
    _write(
        root,
        "pyscarcopula/_cpp/include/scar/copula.hpp",
        '#include "scar/gas.hpp"\n',
    )
    _write(
        root,
        "pyscarcopula/_cpp/include/scar/gas.hpp",
        '#include "scar/copula.hpp"\n',
    )
    assert "public-header-cycle" in _rules(root)


@pytest.mark.parametrize(
    "include",
    [
        '#include "scar/gas.hpp"\n',
        "#include <scar/gas.hpp>\n",
        "#include<scar/gas.hpp>\n",
    ],
)
def test_disallowed_logical_target_dependency_is_rejected(tmp_path, include):
    root = _minimal_repository(tmp_path)
    _write(
        root,
        "pyscarcopula/_cpp/src/scar_ou/bad.cpp",
        include,
    )
    _set_manifest(root, ("scar_ou/bad.cpp",))

    violations = CHECKER_MODULE.check_target_dependency_graph(root)
    assert len(violations) == 1
    assert violations[0].rule == "target-dependency-graph"
    assert "'scar_ou' may not depend on 'gas'" in violations[0].message


def test_domain_module_include_cycle_is_rejected(tmp_path):
    root = _minimal_repository(tmp_path)
    _write(
        root,
        "pyscarcopula/_cpp/src/gas/cycle.cpp",
        '#include "scar/ou.hpp"\n',
    )
    _write(
        root,
        "pyscarcopula/_cpp/src/scar_ou/cycle.cpp",
        '#include "scar/gas.hpp"\n',
    )
    _set_manifest(root, ("gas/cycle.cpp", "scar_ou/cycle.cpp"))

    violations = CHECKER_MODULE.check_domain_module_cycles(root)
    assert len(violations) == 1
    assert violations[0].rule == "domain-module-cycle"
    assert "gas -> scar_ou -> gas" in violations[0].message


@pytest.mark.parametrize(
    "content",
    [
        '#include <pybind11/pybind11.h>\n',
        '#include <Python.h>\n',
        '#include <numpy/arrayobject.h>\n',
        "PyObject* callback = nullptr;\n",
        "auto value = py::none();\n",
    ],
)
def test_python_dependencies_are_rejected_from_compute_boundary(
    tmp_path, content,
):
    root = _minimal_repository(tmp_path)
    _write(root, "pyscarcopula/_cpp/src/copula/bad.cpp", content)
    _set_manifest(root, ("copula/bad.cpp",))
    assert "python-free-compute-boundary" in _rules(root)


def test_binding_source_cannot_be_declared_as_compute(tmp_path):
    root = _minimal_repository(tmp_path)
    _set_manifest(root, ("bindings/module.cpp",), ())
    assert "source-manifest" in _rules(root)


def test_setup_must_consume_both_canonical_manifests(tmp_path):
    root = _minimal_repository(tmp_path)
    (root / "setup.py").write_text(
        "SCAR_COMPUTE_SOURCES = manifest.SCAR_COMPUTE_SOURCES\n",
        encoding="utf-8",
    )
    assert "source-manifest" in _rules(root)
