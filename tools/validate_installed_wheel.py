"""Validate an installed release wheel without importing the source package.

The script is intentionally standalone: it is executed from an external
working directory by cibuildwheel and by the release workflows.  The product
repository supplies this driver, but every ``pyscarcopula`` import must resolve
to wheel-installed files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.abc
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import zipfile


EXPECTED_NATIVE_IDS = (
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

REMOVED_MODULES = (
    "pyscarcopula._scar_cpp",
    "pyscarcopula.copula._protocol",
    "pyscarcopula.numerical.auto_tm",
    "pyscarcopula.numerical.mc_native",
    "pyscarcopula.numerical.mc_samplers",
    "pyscarcopula.numerical.tm_grid",
    "pyscarcopula.strategy.scar_mc",
    "pyscarcopula.vine._conditional_cvine",
    "pyscarcopula.vine.cvine",
)

REMOVED_PUBLIC_NAMES = (
    "CVineCopula",
    "TMGrid",
    "CopulaCapabilities",
    "CopulaProtocol",
    "CommonCopulaProtocol",
    "BivariateCopulaProtocol",
    "MultivariateCopulaProtocol",
)

REMOVED_METHODS = (
    "scar-p-ou",
    "SCAR-P-OU",
    "scar_p_ou",
    "scarpou",
    "scar-m-ou",
    "SCAR-M-OU",
    "scar_m_ou",
    "scarmou",
)

PAIR_PARITY = (
    0.5658059626806888,
    1.1157079598117075,
    0.6952745770609072,
)

GAUSSIAN_PARITY = (
    (0.579259709439103, 0.1612500919669241, 0.5401919073440096),
    (0.691462461274013, 0.6686475698127168, 0.22664696189770903),
)

STUDENT_PARITY = (
    (0.6054118881436016, 0.10795587036023158, 0.5537878930325462),
    (0.6549801159940957, 0.6362622691500188, 0.27659141330204645),
)

OU_PARITY = (
    0.6431196783135578,
    (-0.015784099202033674, 0.5453999619169039, 0.07317368797511314),
)

JACOBI_PARITY = (
    1.399496961158348,
    (0.04313200547369751, 7.7832153396844745, 0.1890116052992734),
)


def _inside(root: Path, target: Path) -> bool:
    return target == root or root in target.parents


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_allclose(actual, expected, *, atol: float, label: str) -> None:
    import numpy as np

    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    if actual_array.shape != expected_array.shape or not np.allclose(
        actual_array,
        expected_array,
        rtol=0.0,
        atol=atol,
    ):
        raise RuntimeError(
            f"{label} parity failed: actual={actual_array!r}, "
            f"expected={expected_array!r}, atol={atol}"
        )


def _wheel_record(wheel: Path) -> dict:
    wheel = wheel.resolve()
    if not wheel.is_file():
        raise RuntimeError(f"wheel does not exist: {wheel}")
    with zipfile.ZipFile(wheel) as archive:
        members = tuple(sorted(archive.namelist()))
        extensions = tuple(
            name
            for name in members
            if name.startswith("pyscarcopula/_native/_scar_cpp.")
            and name.lower().endswith((".pyd", ".so", ".dylib"))
        )
        if len(extensions) != 1:
            raise RuntimeError(
                "wheel must contain exactly one namespaced native extension; "
                f"found {extensions!r}"
            )
        binary = archive.read(extensions[0])
    return {
        "file": wheel.name,
        "sha256": _sha256(wheel),
        "files": len(members),
        "extension": extensions[0],
        "extension_sha256": hashlib.sha256(binary).hexdigest(),
    }


def _assert_wheel_import(source_root: Path) -> dict:
    import pyscarcopula
    from pyscarcopula._native import _extension

    source_root = source_root.resolve()
    package_path = Path(pyscarcopula.__file__).resolve()
    extension_path = Path(_extension.load().__file__).resolve()
    for label, path in (
        ("package", package_path),
        ("native extension", extension_path),
    ):
        if _inside(source_root, path):
            raise RuntimeError(
                f"source-tree leakage: {label} resolved inside "
                f"{source_root}: {path}"
            )

    distribution = metadata.distribution("pyscarcopula")
    if distribution.read_text("WHEEL") is None:
        raise RuntimeError("installed distribution has no WHEEL metadata")
    return {
        "package": str(package_path),
        "extension": str(extension_path),
        "distribution_version": distribution.version,
    }


def _registry_completeness() -> dict:
    from pyscarcopula import (
        BivariateGaussianCopula,
        ClaytonCopula,
        EquicorrGaussianCopula,
        FrankCopula,
        GaussianCopula,
        GumbelCopula,
        IndependentCopula,
        JoeCopula,
        StochasticStudentCopula,
        StudentCopula,
        VineCopula,
    )
    from pyscarcopula._native.registry import (
        native_id_for,
        registered_model_types,
    )

    models = (
        IndependentCopula(),
        ClaytonCopula(),
        FrankCopula(),
        GumbelCopula(),
        JoeCopula(),
        BivariateGaussianCopula(),
        GaussianCopula(d=3),
        StudentCopula(d=3),
        EquicorrGaussianCopula(3),
        StochasticStudentCopula(3),
        VineCopula(),
    )
    registered = registered_model_types()
    if tuple(type(model) for model in models) != registered:
        raise RuntimeError("installed Python native registry is incomplete")
    native_ids = tuple(native_id_for(model) for model in models)
    if native_ids != EXPECTED_NATIVE_IDS:
        raise RuntimeError(
            f"installed native ids differ: {native_ids!r}"
        )
    return {
        "registered_types": [model_type.__name__ for model_type in registered],
        "native_ids": list(native_ids),
    }


def _representative_native_cases() -> tuple[dict, dict]:
    import numpy as np

    from pyscarcopula import ClaytonCopula, IndependentCopula, VineCopula
    from pyscarcopula._native import (
        _descriptors,
        _extension,
        jacobi,
        scar_ou,
    )
    from pyscarcopula.numerical._scar_ou_config import AutoTMConfig

    module = _extension.load()
    observations = np.array(
        [[0.2, 0.7], [0.4, 0.6], [0.8, 0.3]], dtype=np.float64
    )
    spec = _descriptors.make_copula_ops_spec(module, ClaytonCopula())
    pair = np.asarray(
        module.copula_pdf(spec, observations, np.full(3, 1.2)),
        dtype=np.float64,
    )
    _assert_allclose(pair, PAIR_PARITY, atol=2e-14, label="pair native")

    correlation = np.array(
        [[1.0, 0.3, -0.1], [0.3, 1.0, 0.2], [-0.1, 0.2, 1.0]],
        dtype=np.float64,
    )
    normal_draws = np.array(
        [[0.2, -1.1, 0.4], [0.5, 0.3, -0.8]], dtype=np.float64
    )
    gaussian_result = dict(module.multivariate_gaussian_sample_from_normals(
        correlation, normal_draws, 1
    ))
    student_result = dict(
        module.multivariate_student_sample_from_normal_uniforms(
            correlation,
            np.array([6.0], dtype=np.float64),
            normal_draws,
            np.array([0.2, 0.8], dtype=np.float64),
            1,
        )
    )
    if gaussian_result["status"] != module.SCAR_OK:
        raise RuntimeError("direct native Gaussian case failed")
    if student_result["status"] != module.SCAR_OK:
        raise RuntimeError("direct native Student case failed")
    gaussian = np.asarray(gaussian_result["values"], dtype=np.float64)
    student = np.asarray(student_result["values"], dtype=np.float64)
    _assert_allclose(
        gaussian, GAUSSIAN_PARITY, atol=2e-14, label="Gaussian native"
    )
    _assert_allclose(
        student, STUDENT_PARITY, atol=2e-12, label="Student native"
    )

    dynamic_observations = np.array(
        [[0.2, 0.7], [0.6, 0.3], [0.4, 0.8], [0.75, 0.25]],
        dtype=np.float64,
    )
    ou_config = AutoTMConfig(
        K=16,
        adaptive=False,
        max_K=16,
        transition_method="local",
        gh_order=3,
        n_threads=1,
    )
    ou_objective, ou_gradient = scar_ou.neg_loglik_with_grad(
        1.2,
        0.1,
        0.7,
        dynamic_observations,
        ClaytonCopula(),
        ou_config,
    )
    _assert_allclose(
        [ou_objective], [OU_PARITY[0]], atol=2e-10, label="OU objective"
    )
    _assert_allclose(
        ou_gradient, OU_PARITY[1], atol=2e-9, label="OU gradient"
    )

    jacobi_objective, jacobi_gradient = (
        jacobi.PreparedScarJacobiEvaluator(
            dynamic_observations,
            ClaytonCopula(),
            basis_order=4,
            quad_order=16,
            gh_order=3,
            transition_method="local_fixed",
        ).neg_loglik_with_grad(1.2, 0.4, 0.25)
    )
    _assert_allclose(
        [jacobi_objective],
        [JACOBI_PARITY[0]],
        atol=2e-10,
        label="Jacobi objective",
    )
    _assert_allclose(
        jacobi_gradient,
        JACOBI_PARITY[1],
        atol=2e-8,
        label="Jacobi gradient",
    )

    vine_observations = np.array(
        [
            [0.2, 0.4, 0.7],
            [0.3, 0.6, 0.8],
            [0.7, 0.2, 0.5],
            [0.8, 0.7, 0.3],
            [0.4, 0.8, 0.2],
            [0.6, 0.3, 0.9],
        ],
        dtype=np.float64,
    )
    vine = VineCopula.dvine(
        3, order=[0, 1, 2], candidates=[IndependentCopula]
    ).fit(vine_observations, method="mle")
    original_vine_density = module.rvine_log_pdf_rows
    vine_calls = []

    def traced_vine_density(*args, **kwargs):
        vine_calls.append(1)
        return original_vine_density(*args, **kwargs)

    module.rvine_log_pdf_rows = traced_vine_density
    try:
        vine_log_likelihood = vine.log_likelihood(vine_observations)
    finally:
        module.rvine_log_pdf_rows = original_vine_density
    if vine_log_likelihood != 0.0 or vine_calls != [1]:
        raise RuntimeError(
            "representative vine case did not make exactly one native "
            "density call"
        )

    values = {
        "pair_pdf": pair.tolist(),
        "gaussian_samples": gaussian.tolist(),
        "student_samples": student.tolist(),
        "ou_objective": float(ou_objective),
        "ou_gradient": np.asarray(ou_gradient).tolist(),
        "jacobi_objective": float(jacobi_objective),
        "jacobi_gradient": np.asarray(jacobi_gradient).tolist(),
        "vine_log_likelihood": float(vine_log_likelihood),
    }
    context = {
        "correlation": correlation,
        "dynamic_observations": dynamic_observations,
        "ou_config": ou_config,
        "vine": vine,
        "vine_observations": vine_observations,
    }
    return values, context


def _ownership_sentinel(context: dict) -> dict:
    import numpy as np
    from scipy import linalg as scipy_linalg
    from scipy import special as scipy_special

    from pyscarcopula import ClaytonCopula, GaussianCopula, StudentCopula
    from pyscarcopula._native import jacobi, scar_ou

    correlation = context["correlation"]
    dynamic_observations = context["dynamic_observations"]
    gaussian = GaussianCopula(d=3, R=correlation)
    gaussian.corr = correlation.copy()
    student = StudentCopula(d=3, R=correlation)
    student._correlation = correlation.copy()
    student.df = 6.0

    patched = []

    def forbidden(*args, **kwargs):
        raise AssertionError("forbidden Python numerical owner was called")

    targets = (
        (np.linalg, (
            "cholesky", "det", "eigvals", "eigvalsh", "eigh", "inv",
            "lstsq", "pinv", "slogdet", "solve", "svd",
        )),
        (scipy_linalg, (
            "cholesky", "det", "eigh", "inv", "lu", "solve", "svd",
        )),
        (scipy_special, (
            "betainc", "betaincinv", "gammaln", "ndtr", "ndtri",
            "roots_hermite", "roots_hermitenorm", "roots_jacobi", "stdtrit",
        )),
    )
    try:
        for owner, names in targets:
            for name in names:
                if hasattr(owner, name):
                    original = getattr(owner, name)
                    setattr(owner, name, forbidden)
                    patched.append((owner, name, original))

        operations = {
            "pair": ClaytonCopula().pdf([0.2], [0.7], 1.2),
            "gaussian": gaussian.log_likelihood(
                np.array([[0.2, 0.4, 0.7], [0.7, 0.6, 0.3]])
            ),
            "student": student.log_likelihood(
                np.array([[0.2, 0.4, 0.7], [0.7, 0.6, 0.3]])
            ),
            "ou": scar_ou.neg_loglik_with_grad(
                1.2,
                0.1,
                0.7,
                dynamic_observations,
                ClaytonCopula(),
                context["ou_config"],
            )[0],
            "jacobi": jacobi.PreparedScarJacobiEvaluator(
                dynamic_observations,
                ClaytonCopula(),
                basis_order=4,
                quad_order=16,
                gh_order=3,
                transition_method="local_fixed",
            ).neg_loglik_with_grad(1.2, 0.4, 0.25)[0],
            "vine": context["vine"].log_likelihood(
                context["vine_observations"]
            ),
        }
    finally:
        for owner, name, original in reversed(patched):
            setattr(owner, name, original)

    for name, value in operations.items():
        if not np.all(np.isfinite(np.asarray(value, dtype=np.float64))):
            raise RuntimeError(f"ownership sentinel operation {name} failed")
    return {
        "patched_symbols": len(patched),
        "operations": sorted(operations),
    }


def _removal_contracts() -> dict:
    import numpy as np

    import pyscarcopula
    from pyscarcopula import ClaytonCopula, load_model
    from pyscarcopula._native.errors import NativeUnsupported
    from pyscarcopula._native.registry import registry_entry_for
    from pyscarcopula.strategy._base import get_strategy

    for module_name in REMOVED_MODULES:
        if importlib.util.find_spec(module_name) is not None:
            raise RuntimeError(f"removed module is importable: {module_name}")

    namespaces = (
        pyscarcopula,
        pyscarcopula.copula,
        pyscarcopula.numerical,
        pyscarcopula.vine,
    )
    for namespace in namespaces:
        for name in REMOVED_PUBLIC_NAMES:
            if hasattr(namespace, name):
                raise RuntimeError(
                    f"removed public name remains: {namespace.__name__}.{name}"
                )

    for method in REMOVED_METHODS:
        try:
            get_strategy(method)
        except ValueError as error:
            if "Unknown method" not in str(error):
                raise
        else:
            raise RuntimeError(f"removed strategy alias remains: {method}")

    callback_calls = []

    class CallbackCopula(ClaytonCopula):
        def pdf(self, *args, **kwargs):
            callback_calls.append(1)
            return np.ones(1, dtype=np.float64)

    try:
        registry_entry_for(CallbackCopula())
    except NativeUnsupported as error:
        if "exact registered" not in str(error):
            raise
    else:
        raise RuntimeError("custom copula subclass entered the native registry")
    if callback_calls:
        raise RuntimeError("custom copula callback executed during rejection")

    payloads = (
        {
            "format": "pyscarcopula-model",
            "class": "pyscarcopula.vine.cvine.CVineCopula",
            "include_data": False,
            "state": {"class": "pyscarcopula.vine.cvine.CVineCopula"},
        },
        {
            "format": "pyscarcopula-model",
            "class": "pyscarcopula.copula.clayton.ClaytonCopula",
            "include_data": False,
            "state": {"method": "SCAR-P-OU"},
        },
        {
            "format": "pyscarcopula-model",
            "class": "pyscarcopula.copula.clayton.ClaytonCopula",
            "include_data": False,
            "state": {"method": "SCAR-M-OU"},
        },
    )
    import_calls = []
    original_import_module = importlib.import_module

    def forbidden_import(name, package=None):
        import_calls.append((name, package))
        raise AssertionError("removed persistence payload attempted an import")

    with tempfile.TemporaryDirectory(prefix="pyscarcopula-wheel-removal-") as tmp:
        importlib.import_module = forbidden_import
        try:
            for index, payload in enumerate(payloads):
                path = Path(tmp) / f"removed-{index}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                try:
                    load_model(path)
                except ValueError as error:
                    if "no migration execution path" not in str(error):
                        raise
                else:
                    raise RuntimeError(
                        "removed persistence payload was unexpectedly loaded"
                    )
        finally:
            importlib.import_module = original_import_module
    if import_calls:
        raise RuntimeError(
            "removed persistence payload reached dynamic import: "
            f"{import_calls!r}"
        )
    return {
        "removed_modules": list(REMOVED_MODULES),
        "removed_public_names": list(REMOVED_PUBLIC_NAMES),
        "removed_method_aliases": list(REMOVED_METHODS),
        "legacy_payloads_rejected_before_import": len(payloads),
        "custom_callbacks_executed": len(callback_calls),
    }


def _loaded_package_boundary(source_root: Path) -> dict:
    from pyscarcopula._native.smoke import installed_distribution_boundary

    source_root = source_root.resolve()
    leaking = []
    for name, module in sorted(sys.modules.items()):
        if name != "pyscarcopula" and not name.startswith("pyscarcopula."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        path = Path(module_file).resolve()
        if _inside(source_root, path):
            leaking.append(f"{name}={path}")
    if leaking:
        raise RuntimeError(
            "source-tree modules were loaded during wheel validation: "
            + "; ".join(leaking)
        )
    result = installed_distribution_boundary()
    if not result.get("wheel_metadata"):
        raise RuntimeError("distribution boundary was not read from a wheel")
    return result


def _parallel_runtime_contract(extension_sha256: str) -> dict:
    """Prove the installed binary's default and explicit thread behavior."""
    import numpy as np

    from pyscarcopula import FactorCorrelation, NumericalConfig
    from pyscarcopula._native import _extension

    module = _extension.load()
    if _sha256(Path(module.__file__)) != extension_sha256:
        raise RuntimeError(
            "parallel runtime probe loaded a different native extension"
        )
    if NumericalConfig().n_threads != 1:
        raise RuntimeError("installed default n_threads is not one")

    loadings = np.arange(16, dtype=np.float64).reshape(8, 2) / 100.0
    values = np.arange(512, dtype=np.float64).reshape(64, 8) / 97.0
    operator = FactorCorrelation(loadings).prepare()

    module._parallel_runtime_shutdown()
    try:
        initial = dict(module._parallel_runtime_info())
        sequential = operator.solve(values)
        after_default = dict(module._parallel_runtime_info())
        if initial["initialized"] or after_default["initialized"]:
            raise RuntimeError(
                "default n_threads=1 initialized the native thread pool"
            )
        if (
                after_default["batches_submitted"] != 0
                or after_default["tasks_submitted"] != 0):
            raise RuntimeError(
                "default n_threads=1 submitted native queued work"
            )

        parallel = operator.solve(values, n_threads=2)
        after_parallel = dict(module._parallel_runtime_info())
        if not np.array_equal(parallel, sequential):
            raise RuntimeError(
                "installed explicit parallel call changed its result"
            )
        if (
                not after_parallel["initialized"]
                or after_parallel["owner_pid"] != os.getpid()
                or after_parallel["worker_count"] != 2
                or after_parallel["batches_submitted"] != 1
                or after_parallel["tasks_submitted"] != 2):
            raise RuntimeError(
                "installed explicit parallel call did not execute one "
                "two-runner batch"
            )
        result_sha256 = hashlib.sha256(
            np.ascontiguousarray(parallel).tobytes()
        ).hexdigest()
    finally:
        stopped = dict(module._parallel_runtime_shutdown())
    if stopped["initialized"]:
        raise RuntimeError("installed parallel runtime did not shut down")

    return {
        "extension_sha256": extension_sha256,
        "default_n_threads": 1,
        "default_call": {
            "runtime_initialized": after_default["initialized"],
            "batches_submitted": after_default["batches_submitted"],
            "tasks_submitted": after_default["tasks_submitted"],
        },
        "parallel_call": {
            "requested_n_threads": 2,
            "runtime_initialized": after_parallel["initialized"],
            "owner_pid_matches": after_parallel["owner_pid"] == os.getpid(),
            "worker_count": after_parallel["worker_count"],
            "batches_submitted": after_parallel["batches_submitted"],
            "tasks_submitted": after_parallel["tasks_submitted"],
        },
        "result_sha256": result_sha256,
        "shutdown_initialized": stopped["initialized"],
    }


class _RejectNumba(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numba" or fullname.startswith("numba."):
            raise AssertionError("core wheel validation must not import numba")
        return None


def validate(source_root: Path, wheel: Path, configuration: str) -> dict:
    blocker = _RejectNumba()
    sys.meta_path.insert(0, blocker)
    try:
        return _validate_without_numba(source_root, wheel, configuration)
    finally:
        sys.meta_path.remove(blocker)


def _validate_without_numba(source_root: Path, wheel: Path, configuration: str) -> dict:
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise RuntimeError(f"source root does not exist: {source_root}")

    import_boundary = _assert_wheel_import(source_root)
    wheel_record = _wheel_record(wheel)
    installed_binary_sha256 = _sha256(Path(import_boundary["extension"]))
    if installed_binary_sha256 != wheel_record["extension_sha256"]:
        raise RuntimeError(
            "installed extension bytes differ from the validated wheel: "
            f"{installed_binary_sha256} != "
            f"{wheel_record['extension_sha256']}"
        )
    import_boundary["extension_sha256"] = installed_binary_sha256
    registry = _registry_completeness()
    native_cases, context = _representative_native_cases()
    ownership = _ownership_sentinel(context)
    removal = _removal_contracts()
    distribution = _loaded_package_boundary(source_root)
    parallel_runtime = _parallel_runtime_contract(installed_binary_sha256)

    return {
        "schema_version": 2,
        "record_type": "installed_wheel_validation",
        "status": "passed",
        "configuration": configuration,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "import_boundary": import_boundary,
        "wheel": wheel_record,
        "registry": registry,
        "representative_native_cases": native_cases,
        "ownership_sentinel": ownership,
        "removal_contracts": removal,
        "distribution_boundary": distribution,
        "parallel_runtime_contract": parallel_runtime,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--configuration", default="local-wheel")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)

    source_root = arguments.source_root.resolve()
    if arguments.output is not None:
        output = arguments.output.resolve()
        if _inside(source_root, output):
            parser.error("--output must be outside the product repository")
        if output.exists():
            parser.error("refusing to overwrite existing wheel evidence")
    else:
        output = None

    report = validate(source_root, arguments.wheel, arguments.configuration)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
