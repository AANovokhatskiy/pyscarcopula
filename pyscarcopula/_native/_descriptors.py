"""Python-copula mapping for the bundled C++ extension.

This module is the single place where Python copula classes are translated to
the pybind11 ``CopulaSpec`` structure. SCAR-TM-OU kernels intentionally have a
stricter support matrix than the shared point/grid copula operations.
"""

from __future__ import annotations

import weakref

import numpy as np

from pyscarcopula._native import _extension
from pyscarcopula._native.errors import NativeUnsupported


_STUDENT_SPEC_CACHE = weakref.WeakKeyDictionary()


def _native_model_id(copula) -> str:
    """Validate exact registration and return the stable native model id."""
    from pyscarcopula._native.registry import native_id_for

    return native_id_for(copula)


def _set_student_ppf_cache(spec, cache) -> None:
    """Copy a Python Student PPF cache into owning C++ storage."""
    nodes = np.ascontiguousarray(cache.ppf_nodes, dtype=np.float64)
    if cache.ppf_table is None:
        # Preserve the covered df interval even when the observation table is
        # over budget.  The native kernel can still select its controlled
        # large-df asymptotic above the final node; values within the interval
        # use the exact-quantile fallback.
        spec.ppf_nodes = nodes.tolist()
        spec.ppf_n_obs = int(cache.u_shape[0])
        return
    table = np.ascontiguousarray(cache.ppf_table, dtype=np.float64)
    spec.set_student_ppf_cache(nodes, table)


def _transform_name(copula) -> str:
    return str(getattr(copula, "_transform_type", "")).lower()


def _native_archimedean_transform(module, transform_name):
    mapping = {
        "softplus": module.Transform.Softplus,
        "xtanh": module.Transform.XTanh,
        "exp": module.Transform.Exponential,
        "logistic": module.Transform.Logistic,
    }
    try:
        return mapping[transform_name]
    except KeyError as exc:
        raise NativeUnsupported(
            f"Unsupported Archimedean transform: {transform_name!r}"
        ) from exc


def _native_pair_family_name(copula) -> str | None:
    """Return the pair family for an exact registered built-in type."""
    try:
        native_id = _native_model_id(copula)
    except NativeUnsupported:
        return None
    return {
        "Independent": "Independent",
        "Clayton": "Clayton",
        "Frank": "Frank",
        "Gumbel": "Gumbel",
        "Joe": "Joe",
        "BivariateGaussian": "Gaussian",
    }.get(native_id)


def _ensure_native_pair_copula(copula) -> str:
    family_name = _native_pair_family_name(copula)
    if family_name is None:
        name = getattr(copula, "name", type(copula).__name__)
        raise NativeUnsupported(
            f"C++ pair kernels do not support {name}; an exact registered "
            "native pair model is required"
        )
    if int(getattr(copula, "rotate", 0)) not in (0, 90, 180, 270):
        raise NativeUnsupported(
            f"Invalid rotation for native pair family {family_name}"
        )
    return family_name


def _make_native_pair_spec(module, copula):
    family_name = _ensure_native_pair_copula(copula)
    try:
        family = getattr(module.CopulaFamily, family_name)
    except AttributeError as exc:
        raise NativeUnsupported(
            f"Native pair family {family_name!r} is not registered"
        ) from exc

    spec = module._default_pair_copula_spec(family)
    spec.rotation = {
        0: module.Rotation.R0,
        90: module.Rotation.R90,
        180: module.Rotation.R180,
        270: module.Rotation.R270,
    }[int(getattr(copula, "rotate", 0))]

    transform_name = _transform_name(copula)
    if (
            transform_name
            and spec.transform != module.Transform.GaussianTanh):
        spec.transform = _native_archimedean_transform(
            module, transform_name)
    return spec


def supported_for_scar_ou(copula) -> bool:
    """Return whether ``copula`` can use C++ SCAR-TM-OU kernels."""
    try:
        ensure_supported_for_scar_ou(copula)
    except NativeUnsupported:
        return False
    return True


def supported_for_copula_ops(copula) -> bool:
    """Return whether ``copula`` can use shared C++ copula operations."""
    try:
        ensure_supported_for_copula_ops(copula)
    except NativeUnsupported:
        return False
    return True


def supported_for_gas(copula) -> bool:
    """Return whether ``copula`` can use the C++ GAS evaluator."""
    try:
        ensure_supported_for_gas(copula)
    except NativeUnsupported:
        return False
    return True


def supported_for_static_likelihood(copula) -> bool:
    """Return whether ``copula`` has a native static likelihood kernel."""
    try:
        ensure_supported_for_static_likelihood(copula)
    except NativeUnsupported:
        return False
    return True


def ensure_supported_for_static_likelihood(copula) -> None:
    """Validate support for native static likelihood evaluation."""
    _native_model_id(copula)
    from pyscarcopula.copula.multivariate.equicorr import (
        EquicorrGaussianCopula,
    )
    from pyscarcopula.copula.multivariate.gaussian import GaussianCopula
    from pyscarcopula.copula.multivariate.stochastic_student import (
        StochasticStudentCopula,
    )
    from pyscarcopula.copula.multivariate.student import StudentCopula

    if isinstance(copula, GaussianCopula):
        if getattr(copula, "corr_mode", "dense") == "factor":
            try:
                copula.correlation_operator_
            except (AttributeError, ValueError) as exc:
                raise NativeUnsupported(
                    "GaussianCopula requires initialized "
                    "factor correlation") from exc
            return
        if copula.corr is None:
            raise NativeUnsupported("GaussianCopula requires initialized corr")
        return
    if isinstance(copula, StudentCopula):
        if copula.shape is None:
            raise NativeUnsupported("StudentCopula requires initialized shape")
        return
    if isinstance(copula, (EquicorrGaussianCopula, StochasticStudentCopula)):
        if (
                isinstance(copula, StochasticStudentCopula)
                and copula.corr_mode == "factor"):
            try:
                copula.correlation_operator_
            except (AttributeError, ValueError) as exc:
                raise NativeUnsupported(
                    "StochasticStudentCopula requires initialized "
                    "factor correlation") from exc
            return
        if (
                isinstance(copula, StochasticStudentCopula)
                and getattr(copula, "R", None) is None):
            raise NativeUnsupported(
                "StochasticStudentCopula requires initialized R")
        return
    ensure_supported_for_copula_ops(copula)


def _set_factor(spec, correlation) -> None:
    correlation = np.asarray(correlation, dtype=np.float64)
    factor = np.linalg.cholesky(correlation)
    spec.dim = int(correlation.shape[0])
    spec.l_inv = np.linalg.inv(factor).reshape(-1).tolist()
    spec.log_det = float(2.0 * np.sum(np.log(np.diag(factor))))


def make_student_static_spec(module, correlation):
    """Build a fixed-correlation Student spec for static likelihood."""
    spec = module.CopulaSpec()
    spec.family = module.CopulaFamily.Student
    spec.rotation = module.Rotation.R0
    spec.transform = module.Transform.Softplus
    spec.offset = 2.0
    _set_factor(spec, correlation)
    return spec


def make_gaussian_static_spec(module, correlation):
    """Build a fixed-correlation Gaussian spec for static likelihood."""
    spec = module.CopulaSpec()
    spec.family = module.CopulaFamily.MultivariateGaussian
    spec.rotation = module.Rotation.R0
    spec.transform = module.Transform.GaussianTanh
    spec.offset = 0.0
    _set_factor(spec, correlation)
    return spec


def make_factor_gaussian_static_spec(module, operator):
    """Build a multivariate Gaussian spec with compact factor correlation."""
    spec = module.CopulaSpec()
    spec.family = module.CopulaFamily.MultivariateGaussian
    spec.rotation = module.Rotation.R0
    spec.transform = module.Transform.GaussianTanh
    spec.offset = 0.0
    spec.dim = int(operator.dimension)
    spec.correlation_kind = module.CorrelationKind.Factor
    spec.factor_correlation = operator._native
    spec.log_det = float(operator.logdet)
    return spec


def make_static_likelihood_spec(module, copula, u=None):
    """Build a C++ spec for static objective and likelihood reductions."""
    ensure_supported_for_static_likelihood(copula)
    from pyscarcopula.copula.multivariate.equicorr import (
        EquicorrGaussianCopula,
    )
    from pyscarcopula.copula.multivariate.gaussian import GaussianCopula
    from pyscarcopula.copula.multivariate.stochastic_student import (
        StochasticStudentCopula,
    )
    from pyscarcopula.copula.multivariate.student import StudentCopula

    if isinstance(copula, EquicorrGaussianCopula):
        return make_gas_spec(module, copula, u=u)
    if isinstance(copula, StochasticStudentCopula):
        return make_spec(module, copula, u=None)
    if isinstance(copula, GaussianCopula):
        if getattr(copula, "corr_mode", "dense") == "factor":
            return make_factor_gaussian_static_spec(
                module, copula.correlation_operator_)
        return make_gaussian_static_spec(module, copula.corr)
    if isinstance(copula, StudentCopula):
        return make_student_static_spec(module, copula.shape)
    return make_copula_ops_spec(module, copula)


def ensure_supported_for_scar_ou(copula) -> None:
    """Validate that ``copula`` is implemented for C++ SCAR-TM-OU."""
    _native_model_id(copula)
    if _native_pair_family_name(copula) is not None:
        _ensure_native_pair_copula(copula)
        return

    try:
        from pyscarcopula.copula.multivariate.equicorr import EquicorrGaussianCopula
        from pyscarcopula.copula.multivariate.stochastic_student import StochasticStudentCopula
    except ImportError as exc:
        raise NativeUnsupported("Required copula classes are not importable") from exc

    if isinstance(copula, EquicorrGaussianCopula):
        return
    if isinstance(copula, StochasticStudentCopula):
        if getattr(copula, "corr_mode", None) == "factor":
            try:
                copula.correlation_operator_
            except (AttributeError, ValueError) as exc:
                raise NativeUnsupported(
                    "StochasticStudentCopula requires initialized "
                    "factor correlation") from exc
            return
        if getattr(copula, "R", None) is None:
            raise NativeUnsupported("StochasticStudentCopula requires initialized R")
        return

    name = getattr(copula, "name", type(copula).__name__)
    raise NativeUnsupported(
        "C++ SCAR-OU kernels require a registered native pair family, "
        "EquicorrGaussianCopula, or StochasticStudentCopula; "
        f"got {name}"
    )


def ensure_supported_for_copula_ops(copula) -> None:
    """Validate that ``copula`` is implemented by the C++ copula core."""
    _ensure_native_pair_copula(copula)


def ensure_supported_for_gas(copula) -> None:
    """Validate support for the built-in C++ GAS evaluator."""
    _native_model_id(copula)
    try:
        from pyscarcopula.copula.multivariate.equicorr import (
            EquicorrGaussianCopula,
        )
        from pyscarcopula.copula.multivariate.stochastic_student import (
            StochasticStudentCopula,
        )
    except ImportError as exc:
        raise NativeUnsupported(
            "Required multivariate copula classes are not importable"
        ) from exc

    if isinstance(copula, EquicorrGaussianCopula):
        return
    if isinstance(copula, StochasticStudentCopula):
        if getattr(copula, "corr_mode", None) == "factor":
            try:
                copula.correlation_operator_
            except (AttributeError, ValueError) as exc:
                raise NativeUnsupported(
                    "StochasticStudentCopula requires initialized "
                    "factor correlation") from exc
            return
        if getattr(copula, "R", None) is None:
            raise NativeUnsupported(
                "StochasticStudentCopula requires initialized R")
        return

    try:
        ensure_supported_for_copula_ops(copula)
    except NativeUnsupported as exc:
        name = getattr(copula, "name", type(copula).__name__)
        raise NativeUnsupported(
            "C++ bivariate GAS supports Clayton, Gumbel, Joe with rotations, "
            "Frank rotate=0, BivariateGaussian rotate=0, Independent, "
            "while multivariate GAS supports EquicorrGaussianCopula and "
            "StochasticStudentCopula; "
            f"got {name}"
        ) from exc


def make_copula_ops_spec(module, copula):
    """Build a C++ ``CopulaSpec`` for shared point/grid operations."""
    return _make_native_pair_spec(module, copula)


def make_gas_spec(module, copula, u=None, *, use_student_cache=True):
    """Build a C++ ``CopulaSpec`` for the GAS evaluator."""
    ensure_supported_for_gas(copula)
    from pyscarcopula.copula.multivariate.equicorr import (
        EquicorrGaussianCopula,
    )
    from pyscarcopula.copula.multivariate.stochastic_student import (
        StochasticStudentCopula,
    )

    if isinstance(copula, StochasticStudentCopula):
        return make_spec(
            module,
            copula,
            u=u if use_student_cache else None,
        )
    if isinstance(copula, EquicorrGaussianCopula):
        spec = module.CopulaSpec()
        spec.family = module.CopulaFamily.EquicorrGaussian
        spec.rotation = module.Rotation.R0
        spec.transform = module.Transform.GaussianTanh
        spec.offset = 0.0
        spec.dim = int(copula.d)
        return spec
    return make_copula_ops_spec(module, copula)


def make_multivariate_transform_spec(module, copula):
    """Build the minimal native spec needed by multivariate transforms."""
    _native_model_id(copula)
    from pyscarcopula.copula.multivariate.equicorr import (
        EquicorrGaussianCopula,
    )
    from pyscarcopula.copula.multivariate.stochastic_student import (
        StochasticStudentCopula,
    )

    spec = module.CopulaSpec()
    spec.rotation = module.Rotation.R0
    if isinstance(copula, EquicorrGaussianCopula):
        spec.family = module.CopulaFamily.EquicorrGaussian
        spec.transform = module.Transform.GaussianTanh
        spec.offset = 0.0
        spec.dim = int(copula.d)
        return spec
    if isinstance(copula, StochasticStudentCopula):
        spec.family = module.CopulaFamily.Student
        spec.transform = module.Transform.Softplus
        spec.offset = float(copula._df_offset)
        spec.dim = int(copula.d)
        return spec
    raise NativeUnsupported(
        f"Unsupported multivariate copula: {type(copula).__name__}")


def make_multivariate_spec(module, copula, cache=None):
    """Build a native dynamic multivariate spec with an optional PPF cache."""
    _native_model_id(copula)
    state_lock = getattr(copula, "_state_lock", None)
    if state_lock is None:
        return _make_multivariate_spec_unlocked(module, copula, cache)
    with state_lock:
        return _make_multivariate_spec_unlocked(module, copula, cache)


def _make_multivariate_spec_unlocked(module, copula, cache=None):
    from pyscarcopula.copula.multivariate.equicorr import (
        EquicorrGaussianCopula,
    )
    from pyscarcopula.copula.multivariate.stochastic_student import (
        StochasticStudentCopula,
    )

    if isinstance(copula, EquicorrGaussianCopula):
        return make_multivariate_transform_spec(module, copula)
    if not isinstance(copula, StochasticStudentCopula):
        raise NativeUnsupported(
            f"Unsupported multivariate copula: {type(copula).__name__}")
    if copula.corr_mode == "factor":
        spec = make_multivariate_transform_spec(module, copula)
        spec.correlation_kind = module.CorrelationKind.Factor
        spec.factor_correlation = (
            copula.correlation_operator_._native)
        spec.factor_dimension_tile = int(copula.factor_tile_size)
        spec.log_det = float(copula.correlation_operator_.logdet)
        return spec
    if copula.R is None:
        raise NativeUnsupported(
            "StochasticStudentCopula requires initialized R")

    corr_version = int(copula._corr_cache_version)
    if cache is not None:
        cached_spec = _STUDENT_SPEC_CACHE.get(copula)
        if (
                cached_spec is not None
                and cached_spec[0] == cache.version
                and cached_spec[1] == corr_version):
            return cached_spec[2]

    spec = make_multivariate_transform_spec(module, copula)
    spec.l_inv = np.asarray(
        copula._L_inv, dtype=np.float64).reshape(-1).tolist()
    spec.log_det = float(copula._log_det)
    if cache is not None:
        _set_student_ppf_cache(spec, cache)
        _STUDENT_SPEC_CACHE[copula] = (
            cache.version, corr_version, spec)
    return spec


def make_spec(module, copula, u=None):
    """Build a C++ ``CopulaSpec`` for SCAR-TM-OU kernels."""
    state_lock = getattr(copula, "_state_lock", None)
    if state_lock is None:
        return _make_spec_unlocked(module, copula, u)
    with state_lock:
        return _make_spec_unlocked(module, copula, u)


def _make_spec_unlocked(module, copula, u=None):
    ensure_supported_for_scar_ou(copula)
    if _native_pair_family_name(copula) is not None:
        return _make_native_pair_spec(module, copula)

    spec = module.CopulaSpec()
    spec.rotation = {
        0: module.Rotation.R0,
        90: module.Rotation.R90,
        180: module.Rotation.R180,
        270: module.Rotation.R270,
    }[int(getattr(copula, "rotate", 0))]

    from pyscarcopula.copula.multivariate.equicorr import EquicorrGaussianCopula
    from pyscarcopula.copula.multivariate.stochastic_student import StochasticStudentCopula

    if isinstance(copula, EquicorrGaussianCopula):
        spec.family = module.CopulaFamily.EquicorrGaussian
        spec.rotation = module.Rotation.R0
        spec.transform = module.Transform.GaussianTanh
        spec.offset = 0.0
        spec.dim = int(copula.d)
    elif isinstance(copula, StochasticStudentCopula):
        if copula.corr_mode == "factor":
            spec.family = module.CopulaFamily.Student
            spec.rotation = module.Rotation.R0
            spec.transform = module.Transform.Softplus
            spec.offset = float(copula._df_offset)
            spec.dim = int(copula.d)
            spec.correlation_kind = module.CorrelationKind.Factor
            spec.factor_correlation = (
                copula.correlation_operator_._native)
            spec.factor_dimension_tile = int(
                copula.factor_tile_size)
            spec.log_det = float(
                copula.correlation_operator_.logdet)
            return spec
        cache = None
        cached_spec = None
        corr_version = int(copula._corr_cache_version)
        if u is not None:
            u_array = np.asarray(u)
            if u_array.ndim != 2 or u_array.shape[1] != int(copula.d):
                raise ValueError(
                    "u dimension does not match StochasticStudentCopula.d")
            cache = copula.prepare_emission_cache(u)
            cached_spec = _STUDENT_SPEC_CACHE.get(copula)
            if (
                    cached_spec is not None
                    and cached_spec[0] == cache.version
                    and cached_spec[1] == corr_version):
                return cached_spec[2]

        if (
                cache is not None
                and cached_spec is not None
                and cached_spec[0] == cache.version):
            spec = cached_spec[2]
        spec.family = module.CopulaFamily.Student
        spec.rotation = module.Rotation.R0
        spec.transform = module.Transform.Softplus
        spec.offset = float(copula._df_offset)
        spec.dim = int(copula.d)
        spec.l_inv = np.asarray(
            copula._L_inv, dtype=np.float64).reshape(-1).tolist()
        spec.log_det = float(copula._log_det)
        if cache is not None:
            if cached_spec is None or cached_spec[0] != cache.version:
                _set_student_ppf_cache(spec, cache)
            _STUDENT_SPEC_CACHE[copula] = (
                cache.version, corr_version, spec)
    else:
        raise NativeUnsupported(f"Unsupported copula: {type(copula).__name__}")
    return spec
