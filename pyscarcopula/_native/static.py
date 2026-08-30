"""Native static likelihood and scalar-parameter objective adapter."""

from __future__ import annotations

import numpy as np
from pyscarcopula.numerical._arrays import as_float64_array

from pyscarcopula._native import _descriptors, _extension, model_policy
from pyscarcopula._native.errors import (
    NativeError,
    NATIVE_ADAPTER_STATUS_EXCEPTION_POLICY,
    raise_for_status,
)


class StaticLikelihoodEvaluator:
    """Own one native evaluator and its reusable prepared observation state."""

    def __init__(self, copula, u, *, n_threads=1):
        module = _extension.load()
        from pyscarcopula.copula.multivariate.equicorr_prepared import (
            EquicorrPreparedData,
        )
        if isinstance(u, EquicorrPreparedData):
            if int(getattr(copula, "d", -1)) != u.dimension:
                raise ValueError(
                    "prepared dimension does not match copula dimension")
            spec = _descriptors.make_static_likelihood_spec(
                module, copula, u=None)
            self._initialize_prepared(
                module, spec, u, n_threads)
            return
        observations = np.ascontiguousarray(
            as_float64_array(u, name="u"))
        spec = _descriptors.make_static_likelihood_spec(
            module, copula, u=observations)
        self._initialize(module, spec, observations, n_threads)

    def _initialize(self, module, spec, observations, n_threads):
        from pyscarcopula._native.multivariate import _validated_n_threads
        self._module = module
        self._native = module.StaticCopulaEvaluator(
            spec, observations, _validated_n_threads(n_threads))
        raise_for_status(
            int(self._native.status),
            "static likelihood evaluator initialization",
            exception_policy=NATIVE_ADAPTER_STATUS_EXCEPTION_POLICY,
        )

    def _initialize_prepared(self, module, spec, prepared, n_threads):
        from pyscarcopula._native.multivariate import _validated_n_threads
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
            exception_policy=NATIVE_ADAPTER_STATUS_EXCEPTION_POLICY,
        )

    @classmethod
    def from_spec(cls, module, spec, u, *, n_threads=1):
        instance = cls.__new__(cls)
        observations = np.ascontiguousarray(
            as_float64_array(u, name="u"))
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
        """Evaluate an optimizer trial, penalizing native numerical failures."""
        try:
            return self.validated_objective_and_gradient(parameter)
        except FloatingPointError as error:
            return model_policy.optimizer_numerical_failure_evaluation_for_size(
                error, 1, fail_value)

    def validated_objective_and_gradient(self, parameter: float):
        """Evaluate the actual objective and gradient without a failure penalty."""
        result = self.result(parameter)
        raise_for_status(result, "static objective gradient")
        value = float(result["negative_log_likelihood"])
        gradient = float(result["negative_gradient"])
        if not np.isfinite(value) or not np.isfinite(gradient):
            raise FloatingPointError(
                "C++ static objective gradient returned non-finite values")
        return value, np.array([gradient], dtype=np.float64)

    def transformed_objective_and_gradient(
            self, optimizer_parameter: float, *, fail_value: float = 1e10):
        """Evaluate an equicorrelation objective in native raw coordinates."""
        result = dict(
            self._native.transformed_objective(float(optimizer_parameter)))
        if result["status"] != 0:
            try:
                raise_for_status(result, "transformed static objective gradient")
            except FloatingPointError as error:
                return model_policy.optimizer_numerical_failure_evaluation_for_size(
                    error, 1, fail_value)
        value = float(result["negative_log_likelihood"])
        gradient = float(result["negative_gradient"])
        if not np.isfinite(value) or not np.isfinite(gradient):
            raise FloatingPointError(
                "C++ transformed static objective gradient returned "
                "non-finite values")
        return value, np.array([gradient], dtype=np.float64)

    def objective_and_joint_gradient(
            self, parameter: float, *, fail_value: float = 1e10):
        """Return scalar-parameter and native correlation derivatives."""
        result = self.joint_result(parameter)
        correlation_gradient = np.asarray(
            result["negative_correlation_gradient"],
            dtype=np.float64,
        )
        if result["status"] != 0:
            try:
                raise_for_status(result, "static joint objective gradient")
            except FloatingPointError as error:
                value, parameter_gradient = (
                    model_policy
                    .optimizer_numerical_failure_evaluation_for_size(
                        error, 1, fail_value))
                _, correlation_gradient = (
                    model_policy
                    .optimizer_numerical_failure_evaluation_for_size(
                        error, correlation_gradient.size, fail_value))
                return value, parameter_gradient, correlation_gradient
        value = float(result["negative_log_likelihood"])
        parameter_gradient = float(result["negative_gradient"])
        if (
            not np.isfinite(value)
            or not np.isfinite(parameter_gradient)
            or np.any(~np.isfinite(correlation_gradient))
        ):
            raise FloatingPointError(
                "C++ static joint objective gradient returned non-finite values")
        return (
            value,
            np.array([parameter_gradient], dtype=np.float64),
            correlation_gradient,
        )

    def gaussian_objective_and_gradient(
            self, correlation, *, fail_value: float = 1e10):
        """Evaluate a dense Gaussian trial using owned cached normal scores."""
        spec = _descriptors.make_gaussian_static_spec(
            self._module, correlation)
        result = dict(
            self._native.gaussian_objective_with_correlation_gradient(spec))
        gradient = np.asarray(
            result["negative_correlation_gradient"], dtype=np.float64)
        value = float(result["negative_log_likelihood"])
        if result["status"] != 0:
            try:
                raise_for_status(result, "Gaussian static objective gradient")
            except FloatingPointError as error:
                return model_policy.optimizer_numerical_failure_evaluation_for_size(
                    error, gradient.size, fail_value)
        if not np.isfinite(value) or np.any(~np.isfinite(gradient)):
            raise FloatingPointError(
                "C++ Gaussian static objective gradient returned non-finite values")
        return value, gradient

    def log_pdf_rows(self, parameter: float) -> np.ndarray:
        values = np.asarray(
            self._native.log_pdf_rows(float(parameter)),
            dtype=np.float64,
        )
        if np.any(~np.isfinite(values)):
            raise NativeError("C++ static likelihood returned non-finite rows")
        return values

    def log_likelihood(self, parameter: float) -> float:
        result = self.value_result(parameter)
        value = float(result["negative_log_likelihood"])
        raise_for_status(result, "static likelihood reduction")
        if not np.isfinite(value):
            raise FloatingPointError(
                "C++ static likelihood reduction returned a non-finite value")
        return -value


def supported(copula) -> bool:
    return _descriptors.supported_for_static_likelihood(copula)


def prepare(copula, u, *, n_threads=1) -> StaticLikelihoodEvaluator:
    return StaticLikelihoodEvaluator(copula, u, n_threads=n_threads)


def prepare_student(correlation, u, *, n_threads=1) -> StaticLikelihoodEvaluator:
    module = _extension.load()
    spec = _descriptors.make_student_static_spec(module, correlation)
    return StaticLikelihoodEvaluator.from_spec(
        module, spec, u, n_threads=n_threads)


def prepare_gaussian(correlation, u, *, n_threads=1) -> StaticLikelihoodEvaluator:
    module = _extension.load()
    spec = _descriptors.make_gaussian_static_spec(module, correlation)
    return StaticLikelihoodEvaluator.from_spec(
        module, spec, u, n_threads=n_threads)


def prepare_factor_gaussian(operator, u, *, n_threads=1) -> StaticLikelihoodEvaluator:
    module = _extension.load()
    spec = _descriptors.make_factor_gaussian_static_spec(module, operator)
    return StaticLikelihoodEvaluator.from_spec(
        module, spec, u, n_threads=n_threads)
