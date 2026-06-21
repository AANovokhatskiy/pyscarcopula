"""Python-copula mapping for the bundled C++ extension.

This module is the single place where Python copula classes are translated to
the pybind11 ``CopulaSpec`` structure. SCAR-TM-OU kernels intentionally have a
stricter support matrix than the shared point/grid copula operations.
"""

from __future__ import annotations

import weakref

import numpy as np

from pyscarcopula.numerical import _cpp_extension
from pyscarcopula.numerical._cpp_extension import CppUnsupported


_STUDENT_SPEC_CACHE = weakref.WeakKeyDictionary()


def _set_student_ppf_cache(spec, cache) -> None:
    """Copy a Python Student PPF cache into owning C++ storage."""
    nodes = np.ascontiguousarray(cache.ppf_nodes, dtype=np.float64)
    table = np.ascontiguousarray(cache.ppf_table, dtype=np.float64)
    spec.set_student_ppf_cache(nodes, table)


def _transform_name(copula) -> str:
    return str(getattr(copula, "_transform_type", "")).lower()


def supported_for_scar_ou(copula) -> bool:
    """Return whether ``copula`` can use C++ SCAR-TM-OU kernels."""
    try:
        ensure_supported_for_scar_ou(copula)
    except CppUnsupported:
        return False
    return _cpp_extension.available()


def supported_for_copula_ops(copula) -> bool:
    """Return whether ``copula`` can use shared C++ copula operations."""
    try:
        ensure_supported_for_copula_ops(copula)
    except CppUnsupported:
        return False
    return _cpp_extension.available()


def supported_for_gas(copula) -> bool:
    """Return whether ``copula`` can use the C++ GAS evaluator."""
    try:
        ensure_supported_for_gas(copula)
    except CppUnsupported:
        return False
    return _cpp_extension.available()


def supported_for_static_likelihood(copula) -> bool:
    """Return whether ``copula`` has a native static likelihood kernel."""
    try:
        ensure_supported_for_static_likelihood(copula)
    except CppUnsupported:
        return False
    return _cpp_extension.available()


def supported_for_mc(copula) -> bool:
    """Return whether ``copula`` has native SCAR-MC trajectory density."""
    try:
        ensure_supported_for_mc(copula)
    except CppUnsupported:
        return False
    return _cpp_extension.available()


def ensure_supported_for_mc(copula) -> None:
    """Validate support for native SCAR-MC trajectory density."""
    from pyscarcopula.copula.multivariate.stochastic_student import (
        StochasticStudentCopula,
    )

    if isinstance(copula, StochasticStudentCopula):
        if copula.R is None:
            raise CppUnsupported(
                "StochasticStudentCopula requires initialized R")
        return
    ensure_supported_for_copula_ops(copula)


def ensure_supported_for_static_likelihood(copula) -> None:
    """Validate support for native static likelihood evaluation."""
    from pyscarcopula.copula.multivariate.equicorr import (
        EquicorrGaussianCopula,
    )
    from pyscarcopula.copula.multivariate.gaussian import GaussianCopula
    from pyscarcopula.copula.multivariate.stochastic_student import (
        StochasticStudentCopula,
    )
    from pyscarcopula.copula.multivariate.student import StudentCopula

    if isinstance(copula, GaussianCopula):
        if copula.corr is None:
            raise CppUnsupported("GaussianCopula requires initialized corr")
        return
    if isinstance(copula, StudentCopula):
        if copula.shape is None:
            raise CppUnsupported("StudentCopula requires initialized shape")
        return
    if isinstance(copula, (EquicorrGaussianCopula, StochasticStudentCopula)):
        if (
                isinstance(copula, StochasticStudentCopula)
                and getattr(copula, "R", None) is None):
            raise CppUnsupported(
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
        return make_gaussian_static_spec(module, copula.corr)
    if isinstance(copula, StudentCopula):
        return make_student_static_spec(module, copula.shape)
    return make_copula_ops_spec(module, copula)


def ensure_supported_for_scar_ou(copula) -> None:
    """Validate that ``copula`` is implemented for C++ SCAR-TM-OU."""
    try:
        from pyscarcopula.copula.clayton import ClaytonCopula
        from pyscarcopula.copula.elliptical import BivariateGaussianCopula
        from pyscarcopula.copula.independent import IndependentCopula
        from pyscarcopula.copula.multivariate.equicorr import EquicorrGaussianCopula
        from pyscarcopula.copula.multivariate.stochastic_student import StochasticStudentCopula
        from pyscarcopula.copula.frank import FrankCopula
        from pyscarcopula.copula.gumbel import GumbelCopula
        from pyscarcopula.copula.joe import JoeCopula
    except ImportError as exc:
        raise CppUnsupported("Required copula classes are not importable") from exc

    archimedean_types = (ClaytonCopula, GumbelCopula, FrankCopula, JoeCopula)
    if (
            isinstance(copula, archimedean_types)
            and _transform_name(copula) in {"softplus", "xtanh"}):
        if isinstance(copula, FrankCopula) and int(getattr(copula, "rotate", 0)) != 0:
            pass
        else:
            return
    if isinstance(copula, IndependentCopula):
        return
    if isinstance(copula, BivariateGaussianCopula):
        return
    if isinstance(copula, EquicorrGaussianCopula):
        return
    if isinstance(copula, StochasticStudentCopula):
        if getattr(copula, "R", None) is None:
            raise CppUnsupported("StochasticStudentCopula requires initialized R")
        return

    name = getattr(copula, "name", type(copula).__name__)
    transform = _transform_name(copula) or "<unknown>"
    raise CppUnsupported(
        "C++ SCAR-OU kernels currently support only "
        f"Clayton, Gumbel, Frank, Joe with softplus/xtanh transforms, "
        f"IndependentCopula, BivariateGaussianCopula, "
        f"EquicorrGaussianCopula, and StochasticStudentCopula; got {name} "
        f"with transform={transform}"
    )


def ensure_supported_for_copula_ops(copula) -> None:
    """Validate that ``copula`` is implemented by the C++ copula core."""
    try:
        from pyscarcopula.copula.clayton import ClaytonCopula
        from pyscarcopula.copula.elliptical import BivariateGaussianCopula
        from pyscarcopula.copula.frank import FrankCopula
        from pyscarcopula.copula.gumbel import GumbelCopula
        from pyscarcopula.copula.independent import IndependentCopula
        from pyscarcopula.copula.joe import JoeCopula
    except ImportError as exc:
        raise CppUnsupported("Required copula classes are not importable") from exc

    if isinstance(copula, IndependentCopula):
        return
    if isinstance(copula, FrankCopula):
        if int(getattr(copula, "rotate", 0)) == 0:
            return
    elif isinstance(copula, (ClaytonCopula, GumbelCopula, JoeCopula)):
        return
    elif isinstance(copula, BivariateGaussianCopula):
        if int(getattr(copula, "rotate", 0)) == 0:
            return

    name = getattr(copula, "name", type(copula).__name__)
    raise CppUnsupported(
        "C++ copula operations currently support Clayton, Gumbel, "
        f"Joe with rotations, Frank rotate=0, BivariateGaussian rotate=0, "
        f"and Independent; got {name}"
    )


def ensure_supported_for_gas(copula) -> None:
    """Validate support for the built-in C++ GAS evaluator."""
    try:
        from pyscarcopula.copula.multivariate.equicorr import (
            EquicorrGaussianCopula,
        )
        from pyscarcopula.copula.multivariate.stochastic_student import (
            StochasticStudentCopula,
        )
    except ImportError as exc:
        raise CppUnsupported(
            "Required multivariate copula classes are not importable"
        ) from exc

    if isinstance(copula, EquicorrGaussianCopula):
        return
    if isinstance(copula, StochasticStudentCopula):
        if getattr(copula, "R", None) is None:
            raise CppUnsupported(
                "StochasticStudentCopula requires initialized R")
        return

    try:
        ensure_supported_for_copula_ops(copula)
    except CppUnsupported as exc:
        name = getattr(copula, "name", type(copula).__name__)
        raise CppUnsupported(
            "C++ bivariate GAS supports Clayton, Gumbel, Joe with rotations, "
            "Frank rotate=0, BivariateGaussian rotate=0, Independent, "
            "while multivariate GAS supports EquicorrGaussianCopula and "
            "StochasticStudentCopula; "
            f"got {name}"
        ) from exc


def make_copula_ops_spec(module, copula):
    """Build a C++ ``CopulaSpec`` for shared point/grid operations."""
    ensure_supported_for_copula_ops(copula)
    spec = module.CopulaSpec()
    spec.rotation = {
        0: module.Rotation.R0,
        90: module.Rotation.R90,
        180: module.Rotation.R180,
        270: module.Rotation.R270,
    }[int(getattr(copula, "rotate", 0))]

    from pyscarcopula.copula.clayton import ClaytonCopula
    from pyscarcopula.copula.elliptical import BivariateGaussianCopula
    from pyscarcopula.copula.frank import FrankCopula
    from pyscarcopula.copula.gumbel import GumbelCopula
    from pyscarcopula.copula.independent import IndependentCopula
    from pyscarcopula.copula.joe import JoeCopula

    transform = _transform_name(copula)
    if isinstance(copula, IndependentCopula):
        spec.family = module.CopulaFamily.Independent
        spec.transform = module.Transform.Softplus
        spec.offset = 0.0
    elif isinstance(copula, ClaytonCopula):
        spec.family = module.CopulaFamily.Clayton
        spec.transform = (
            module.Transform.XTanh if transform == "xtanh"
            else module.Transform.Softplus
        )
        spec.offset = 0.0001
    elif isinstance(copula, GumbelCopula):
        spec.family = module.CopulaFamily.Gumbel
        spec.transform = (
            module.Transform.XTanh if transform == "xtanh"
            else module.Transform.Softplus
        )
        spec.offset = 1.0001
    elif isinstance(copula, FrankCopula):
        spec.family = module.CopulaFamily.Frank
        spec.transform = (
            module.Transform.XTanh if transform == "xtanh"
            else module.Transform.Softplus
        )
        spec.offset = 0.0001
    elif isinstance(copula, JoeCopula):
        spec.family = module.CopulaFamily.Joe
        spec.transform = (
            module.Transform.XTanh if transform == "xtanh"
            else module.Transform.Softplus
        )
        spec.offset = 1.0001
    elif isinstance(copula, BivariateGaussianCopula):
        spec.family = module.CopulaFamily.Gaussian
        spec.rotation = module.Rotation.R0
        spec.transform = module.Transform.GaussianTanh
        spec.offset = 0.0
    else:
        raise CppUnsupported(f"Unsupported copula: {type(copula).__name__}")
    return spec


def make_mc_spec(module, copula, u=None):
    """Build a native spec for SCAR-MC trajectory density evaluation."""
    ensure_supported_for_mc(copula)
    from pyscarcopula.copula.multivariate.stochastic_student import (
        StochasticStudentCopula,
    )

    if isinstance(copula, StochasticStudentCopula):
        return make_spec(module, copula, u=u)
    return make_copula_ops_spec(module, copula)


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
    raise CppUnsupported(
        f"Unsupported multivariate copula: {type(copula).__name__}")


def make_multivariate_spec(module, copula, cache=None):
    """Build a native dynamic multivariate spec with an optional PPF cache."""
    from pyscarcopula.copula.multivariate.equicorr import (
        EquicorrGaussianCopula,
    )
    from pyscarcopula.copula.multivariate.stochastic_student import (
        StochasticStudentCopula,
    )

    if isinstance(copula, EquicorrGaussianCopula):
        return make_multivariate_transform_spec(module, copula)
    if not isinstance(copula, StochasticStudentCopula):
        raise CppUnsupported(
            f"Unsupported multivariate copula: {type(copula).__name__}")
    if copula.R is None:
        raise CppUnsupported(
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
    ensure_supported_for_scar_ou(copula)
    spec = module.CopulaSpec()
    spec.rotation = {
        0: module.Rotation.R0,
        90: module.Rotation.R90,
        180: module.Rotation.R180,
        270: module.Rotation.R270,
    }[int(getattr(copula, "rotate", 0))]

    from pyscarcopula.copula.clayton import ClaytonCopula
    from pyscarcopula.copula.elliptical import BivariateGaussianCopula
    from pyscarcopula.copula.independent import IndependentCopula
    from pyscarcopula.copula.multivariate.equicorr import EquicorrGaussianCopula
    from pyscarcopula.copula.multivariate.stochastic_student import StochasticStudentCopula
    from pyscarcopula.copula.frank import FrankCopula
    from pyscarcopula.copula.gumbel import GumbelCopula
    from pyscarcopula.copula.joe import JoeCopula

    transform = _transform_name(copula)
    if isinstance(copula, IndependentCopula):
        spec.family = module.CopulaFamily.Independent
        spec.rotation = module.Rotation.R0
        spec.transform = module.Transform.Softplus
        spec.offset = 0.0
    elif isinstance(copula, ClaytonCopula):
        spec.family = module.CopulaFamily.Clayton
        spec.transform = (
            module.Transform.XTanh
            if transform == "xtanh"
            else module.Transform.Softplus)
        spec.offset = 0.0001
    elif isinstance(copula, GumbelCopula):
        spec.family = module.CopulaFamily.Gumbel
        spec.transform = (
            module.Transform.XTanh
            if transform == "xtanh"
            else module.Transform.Softplus)
        spec.offset = 1.0001
    elif isinstance(copula, FrankCopula):
        spec.family = module.CopulaFamily.Frank
        spec.transform = (
            module.Transform.XTanh
            if transform == "xtanh"
            else module.Transform.Softplus)
        spec.offset = 0.0001
    elif isinstance(copula, JoeCopula):
        spec.family = module.CopulaFamily.Joe
        spec.transform = (
            module.Transform.XTanh
            if transform == "xtanh"
            else module.Transform.Softplus)
        spec.offset = 1.0001
    elif isinstance(copula, BivariateGaussianCopula):
        spec.family = module.CopulaFamily.Gaussian
        spec.rotation = module.Rotation.R0
        spec.transform = module.Transform.GaussianTanh
        spec.offset = 0.0
    elif isinstance(copula, EquicorrGaussianCopula):
        spec.family = module.CopulaFamily.EquicorrGaussian
        spec.rotation = module.Rotation.R0
        spec.transform = module.Transform.GaussianTanh
        spec.offset = 0.0
        spec.dim = int(copula.d)
    elif isinstance(copula, StochasticStudentCopula):
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
        raise CppUnsupported(f"Unsupported copula: {type(copula).__name__}")
    return spec
