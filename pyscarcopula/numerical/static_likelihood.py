"""Native static likelihood and scalar-parameter objective adapter."""

from __future__ import annotations

import numpy as np

from pyscarcopula.numerical import _cpp_copula, _cpp_extension
from pyscarcopula.numerical._cpp_extension import (
    CppError,
    LEGACY_CPP_STATUS_EXCEPTION_POLICY,
    raise_for_status,
)


class StaticLikelihoodEvaluator:
    """Own one native evaluator and its reusable prepared observation state."""

    def __init__(self, copula, u, *, n_threads=1):
        module = _cpp_extension.load()
        from pyscarcopula.copula.multivariate.equicorr_prepared import (
            EquicorrPreparedData,
        )
        if isinstance(u, EquicorrPreparedData):
            if int(getattr(copula, "d", -1)) != u.dimension:
                raise ValueError(
                    "prepared dimension does not match copula dimension")
            spec = _cpp_copula.make_static_likelihood_spec(
                module, copula, u=None)
            self._initialize_prepared(
                module, spec, u, n_threads)
            return
        observations = np.ascontiguousarray(
            np.asarray(u, dtype=np.float64))
        spec = _cpp_copula.make_static_likelihood_spec(
            module, copula, u=observations)
        self._initialize(module, spec, observations, n_threads)

    def _initialize(self, module, spec, observations, n_threads):
        from pyscarcopula.numerical.multivariate_native import (
            _validated_n_threads,
        )
        self._module = module
        self._native = module.StaticCopulaEvaluator(
            spec, observations, _validated_n_threads(n_threads))
        raise_for_status(
            int(self._native.status),
            "static likelihood evaluator initialization",
            exception_policy=LEGACY_CPP_STATUS_EXCEPTION_POLICY,
        )

    def _initialize_prepared(self, module, spec, prepared, n_threads):
        from pyscarcopula.numerical.multivariate_native import (
            _validated_n_threads,
        )
        self._module = module
        self._native = module.StaticCopulaEvaluator(
            spec,
            prepared.sum_z,
            prepared.sum_z2,
            _validated_n_threads(n_threads),
        )
        raise_for_status(
            int(self._native.status),
            "static likelihood prepared-statistics initialization",
            exception_policy=LEGACY_CPP_STATUS_EXCEPTION_POLICY,
        )

    @classmethod
    def from_spec(cls, module, spec, u, *, n_threads=1):
        instance = cls.__new__(cls)
        observations = np.ascontiguousarray(
            np.asarray(u, dtype=np.float64))
        instance._initialize(module, spec, observations, n_threads)
        return instance

    def result(self, parameter: float) -> dict:
        return dict(self._native.objective(float(parameter)))

    def value_result(self, parameter: float) -> dict:
        return dict(self._native.objective_value(float(parameter)))

    def joint_result(self, parameter: float) -> dict:
        return dict(
            self._native.objective_with_correlation_gradient(
                float(parameter)))

    def objective_and_gradient(
            self, parameter: float, *, fail_value: float = 1e10):
        result = self.result(parameter)
        if result["status"] != 0:
            return float(fail_value), np.array([0.0], dtype=np.float64)
        value = float(result["negative_log_likelihood"])
        gradient = float(result["negative_gradient"])
        if not np.isfinite(value) or not np.isfinite(gradient):
            return float(fail_value), np.array([0.0], dtype=np.float64)
        return value, np.array([gradient], dtype=np.float64)

    def objective_and_joint_gradient(
            self, parameter: float, *, fail_value: float = 1e10):
        """Return scalar-parameter and native correlation derivatives."""
        result = self.joint_result(parameter)
        if result["status"] != 0:
            return (
                float(fail_value),
                np.array([0.0], dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )
        value = float(result["negative_log_likelihood"])
        parameter_gradient = float(result["negative_gradient"])
        correlation_gradient = np.asarray(
            result["negative_correlation_gradient"],
            dtype=np.float64,
        )
        if (
            not np.isfinite(value)
            or not np.isfinite(parameter_gradient)
            or np.any(~np.isfinite(correlation_gradient))
        ):
            return (
                float(fail_value),
                np.array([0.0], dtype=np.float64),
                np.zeros_like(correlation_gradient),
            )
        return (
            value,
            np.array([parameter_gradient], dtype=np.float64),
            correlation_gradient,
        )

    def gaussian_objective_and_gradient(
            self, correlation, *, fail_value: float = 1e10):
        """Evaluate a dense Gaussian trial using owned cached normal scores."""
        spec = _cpp_copula.make_gaussian_static_spec(
            self._module, correlation)
        result = dict(
            self._native.gaussian_objective_with_correlation_gradient(spec))
        gradient = np.asarray(
            result["negative_correlation_gradient"], dtype=np.float64)
        value = float(result["negative_log_likelihood"])
        if (
                result["status"] != 0
                or not np.isfinite(value)
                or np.any(~np.isfinite(gradient))):
            return float(fail_value), np.zeros_like(gradient)
        return value, gradient

    def log_pdf_rows(self, parameter: float) -> np.ndarray:
        values = np.asarray(
            self._native.log_pdf_rows(float(parameter)),
            dtype=np.float64,
        )
        if np.any(~np.isfinite(values)):
            raise CppError("C++ static likelihood returned non-finite rows")
        return values

    def log_likelihood(self, parameter: float) -> float:
        result = self.value_result(parameter)
        value = float(result["negative_log_likelihood"])
        if result["status"] != 0 or not np.isfinite(value):
            raise CppError(
                "C++ static likelihood reduction failed with "
                f"status={result['status']}, "
                f"failure_index={result['failure_index']}")
        return -value


def supported(copula) -> bool:
    return _cpp_copula.supported_for_static_likelihood(copula)


def prepare(copula, u, *, n_threads=1) -> StaticLikelihoodEvaluator:
    return StaticLikelihoodEvaluator(copula, u, n_threads=n_threads)


def prepare_student(correlation, u, *, n_threads=1) -> StaticLikelihoodEvaluator:
    module = _cpp_extension.load()
    spec = _cpp_copula.make_student_static_spec(module, correlation)
    return StaticLikelihoodEvaluator.from_spec(
        module, spec, u, n_threads=n_threads)


def prepare_gaussian(correlation, u, *, n_threads=1) -> StaticLikelihoodEvaluator:
    module = _cpp_extension.load()
    spec = _cpp_copula.make_gaussian_static_spec(module, correlation)
    return StaticLikelihoodEvaluator.from_spec(
        module, spec, u, n_threads=n_threads)
