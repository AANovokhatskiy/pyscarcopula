"""Capture and verify numerical goldens for the C++ refactor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

try:
    from tools.benchmark_cpp_refactor import ROOT, _source_digest
except ImportError:  # Direct ``python tools/capture_...py`` execution.
    from benchmark_cpp_refactor import ROOT, _source_digest


DEFAULT_OUTPUT = (
    ROOT / "tests" / "fixtures" / "cpp_refactor_goldens_v1.json"
)
TRANSFORMS = ("softplus", "xtanh", "exp", "logistic")
ROTATIONS = (0, 90, 180, 270)
CROSS_PLATFORM_RTOL = 5e-12
CROSS_PLATFORM_ATOL = 5e-12
REDUCTION_CROSS_PLATFORM_RTOL = 0.0


def _platform_key() -> dict[str, str]:
    import pyscarcopula._scar_cpp as extension

    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python_compiler": platform.python_compiler(),
        "native_compiler": str(extension.__cpp_compiler__),
        "extension_suffix": "".join(Path(extension.__file__).suffixes),
    }


def _case_matrix():
    from pyscarcopula import (
        BivariateGaussianCopula,
        ClaytonCopula,
        FrankCopula,
        GumbelCopula,
        IndependentCopula,
        JoeCopula,
    )

    for family_name, factory in (
        ("clayton", ClaytonCopula),
        ("gumbel", GumbelCopula),
        ("joe", JoeCopula),
    ):
        for transform in TRANSFORMS:
            for rotation in ROTATIONS:
                yield (
                    f"pair.{family_name}.{transform}.r{rotation}",
                    factory(rotate=rotation, transform_type=transform),
                    family_name,
                    transform,
                    rotation,
                )
    for transform in TRANSFORMS:
        yield (
            f"pair.frank.{transform}.r0",
            FrankCopula(transform_type=transform),
            "frank",
            transform,
            0,
        )
    yield (
        "pair.gaussian.gaussian_tanh.r0",
        BivariateGaussianCopula(),
        "gaussian",
        "gaussian_tanh",
        0,
    )
    yield (
        "pair.independent.identity.r0",
        IndependentCopula(),
        "independent",
        "identity",
        0,
    )


def _parameter(family: str) -> float:
    return {
        "clayton": 0.8,
        "frank": 2.0,
        "gumbel": 1.5,
        "joe": 1.6,
        "gaussian": -0.35,
        "independent": 0.0,
    }[family]


def _serialize(values: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if np.any(~np.isfinite(array)):
        raise RuntimeError("golden case returned non-finite values")
    flat = array.ravel()
    return {
        "dtype": "float64",
        "shape": list(array.shape),
        "hex": [float(value).hex() for value in flat],
        "values": [float(value) for value in flat],
    }


def build_cases() -> list[dict[str, Any]]:
    u1 = np.array([1e-8, 0.07, 0.23, 0.51, 0.83, 1.0 - 1e-8])
    u2 = np.array([0.91, 0.72, 0.44, 0.29, 0.11, 0.63])
    quantiles = np.array([0.03, 0.19, 0.41, 0.66, 0.88, 0.97])
    latent = np.array([-2.0, -0.7, 0.0, 0.4, 1.7])
    grid = np.array([-1.4, -0.3, 0.8, 1.9])
    observations = np.column_stack((u1, u2))
    cases = []
    for case_id, copula, family, transform, rotation in _case_matrix():
        parameter = _parameter(family)
        transformed = copula.transform(latent)
        first_h, second_h = copula.h_pair(u1, u2, parameter)
        pdf_grid, gradient_grid = copula.pdf_and_grad_on_grid_batch(
            observations, grid)
        outputs = {
            "transform": _serialize(transformed),
            "inverse_transform": _serialize(copula.inv_transform(transformed)),
            "dtransform": _serialize(copula.dtransform(latent)),
            "pdf": _serialize(copula.pdf(u1, u2, parameter)),
            "log_pdf": _serialize(copula.log_pdf(u1, u2, parameter)),
            "dlog_pdf_dr": _serialize(
                copula.dlog_pdf_dr(u1, u2, parameter)),
            "h_first": _serialize(first_h),
            "h_second": _serialize(second_h),
            "h_inverse": _serialize(
                copula.h_inverse(quantiles, u2, parameter)),
            "pdf_grid": _serialize(pdf_grid),
            "gradient_grid": _serialize(gradient_grid),
        }
        if family != "independent":
            outputs["tau_to_param"] = _serialize(copula.tau_to_param(
                np.array([0.08, 0.27, 0.61])))
            outputs["param_to_tau"] = _serialize(copula.param_to_tau(
                np.array([parameter])))
        cases.append({
            "id": case_id,
            "family": family,
            "transform": transform,
            "rotation": rotation,
            "parameter": parameter,
            "cross_platform_rtol": CROSS_PLATFORM_RTOL,
            "cross_platform_atol": CROSS_PLATFORM_ATOL,
            "outputs": outputs,
        })
    return cases


def build_reduction_cases() -> list[dict[str, Any]]:
    """Capture thread-specific values where parallel reduction order matters."""
    from pyscarcopula._native import static as static_likelihood

    dimension = 12
    correlation = np.full((dimension, dimension), 0.15, dtype=np.float64)
    np.fill_diagonal(correlation, 1.0)
    observations = np.random.default_rng(2026082390).uniform(
        0.01, 0.99, size=(768, dimension))
    outputs = {}
    for n_threads in (1, 2, 4, 8):
        value = static_likelihood.prepare_gaussian(
            correlation, observations, n_threads=n_threads
        ).log_likelihood(0.0)
        outputs[f"n_threads_{n_threads}"] = _serialize([value])
    return [{
        "id": "reduction.static_gaussian.d12.n768",
        "contract": (
            "Thread-specific values are exact on the reference toolchain. "
            "Cross-thread equality is not part of the current contract; the "
            "locked cross-platform envelope is 5e-12."
        ),
        "cross_platform_rtol": REDUCTION_CROSS_PLATFORM_RTOL,
        "cross_platform_atol": CROSS_PLATFORM_ATOL,
        "outputs": outputs,
    }]


def build_payload() -> dict[str, Any]:
    import pyscarcopula

    return {
        "schema_version": 1,
        "fixture_id": "cpp-architecture-refactor-goldens-v1",
        "source_commit": _git_commit(),
        "compute_source_sha256": _source_digest(),
        "pyscarcopula_version": pyscarcopula.__version__,
        "platform_key": _platform_key(),
        "comparison_contract": {
            "same_platform_key": "bitwise float.hex equality",
            "different_platform_key": (
                "locked per-case rtol/atol; tolerances may not be widened "
                "during the architecture refactor"
            ),
        },
        "cases": build_cases(),
        "reduction_cases": build_reduction_cases(),
    }


def _git_commit() -> str | None:
    try:
        import subprocess

        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _array(serialized: dict[str, Any], key: str) -> np.ndarray:
    values = serialized[key]
    return np.asarray(values, dtype=np.float64).reshape(serialized["shape"])


def check_fixture(path: Path) -> None:
    expected = json.loads(path.read_text(encoding="utf-8"))
    actual = build_payload()
    for metadata_key in (
            "schema_version", "fixture_id", "pyscarcopula_version",
            "comparison_contract"):
        if expected.get(metadata_key) != actual.get(metadata_key):
            raise AssertionError(f"fixture {metadata_key} changed")
    exact = expected["platform_key"] == actual["platform_key"]
    for section in ("cases", "reduction_cases"):
        expected_cases = {
            case["id"]: case for case in expected[section]
        }
        actual_cases = {case["id"]: case for case in actual[section]}
        if expected_cases.keys() != actual_cases.keys():
            missing = sorted(expected_cases.keys() - actual_cases.keys())
            extra = sorted(actual_cases.keys() - expected_cases.keys())
            raise AssertionError(
                f"{section} mismatch; missing={missing}, extra={extra}")
        for case_id, expected_case in expected_cases.items():
            actual_case = actual_cases[case_id]
            for metadata_key in (
                    "family", "transform", "rotation", "parameter", "contract",
                    "cross_platform_rtol", "cross_platform_atol"):
                if (
                        expected_case.get(metadata_key)
                        != actual_case.get(metadata_key)):
                    raise AssertionError(
                        f"{case_id}: {metadata_key} changed")
            if expected_case["outputs"].keys() != actual_case["outputs"].keys():
                raise AssertionError(f"{case_id}: operation set changed")
            for operation, expected_output in expected_case["outputs"].items():
                actual_output = actual_case["outputs"][operation]
                if expected_output["shape"] != actual_output["shape"]:
                    raise AssertionError(f"{case_id}/{operation}: shape changed")
                if exact:
                    if expected_output["hex"] != actual_output["hex"]:
                        raise AssertionError(
                            f"{case_id}/{operation}: bitwise output changed")
                else:
                    np.testing.assert_allclose(
                        _array(actual_output, "values"),
                        _array(expected_output, "values"),
                        rtol=float(actual_case["cross_platform_rtol"]),
                        atol=float(actual_case["cross_platform_atol"]),
                        err_msg=f"{case_id}/{operation}",
                    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output = args.output.resolve()
    if args.check:
        check_fixture(output)
        print(f"golden fixture is unchanged: {output}")
        return 0
    payload = build_payload()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
