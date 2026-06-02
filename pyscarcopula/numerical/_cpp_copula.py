"""Python-copula mapping for the optional C++ extension.

This module is the single place where Python copula classes are translated to
the pybind11 ``CopulaSpec`` structure.  SCAR-TM-OU kernels intentionally have a
stricter support matrix than pointwise copula h/h_inverse kernels because the
OU backend also needs parameter transforms and grid density derivatives.
"""

from __future__ import annotations

from pyscarcopula.numerical import _cpp_extension
from pyscarcopula.numerical._cpp_extension import CppUnsupported


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
    """Return whether ``copula`` can use C++ h/h_inverse kernels."""
    try:
        ensure_supported_for_copula_ops(copula)
    except CppUnsupported:
        return False
    return _cpp_extension.available()


def ensure_supported_for_scar_ou(copula) -> None:
    """Validate that ``copula`` is implemented for C++ SCAR-TM-OU."""
    try:
        from pyscarcopula.copula.clayton import ClaytonCopula
        from pyscarcopula.copula.elliptical import BivariateGaussianCopula
        from pyscarcopula.copula.frank import FrankCopula
        from pyscarcopula.copula.gumbel import GumbelCopula
        from pyscarcopula.copula.joe import JoeCopula
    except ImportError as exc:
        raise CppUnsupported("Required copula classes are not importable") from exc

    archimedean_types = (ClaytonCopula, GumbelCopula, FrankCopula, JoeCopula)
    if isinstance(copula, archimedean_types) and _transform_name(copula) == "softplus":
        if isinstance(copula, FrankCopula) and int(getattr(copula, "rotate", 0)) != 0:
            pass
        else:
            return
    if isinstance(copula, BivariateGaussianCopula):
        return

    name = getattr(copula, "name", type(copula).__name__)
    transform = _transform_name(copula) or "<unknown>"
    raise CppUnsupported(
        "C++ SCAR-OU kernels currently support only "
        f"Clayton, Gumbel, Frank, Joe with softplus transform, "
        f"and BivariateGaussianCopula; got {name} "
        f"with transform={transform}"
    )


def ensure_supported_for_copula_ops(copula) -> None:
    """Validate that ``copula`` is implemented for C++ h/h_inverse ops."""
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
        "C++ copula h/h_inverse kernels currently support Clayton, Gumbel, "
        f"Joe with rotations, Frank rotate=0, BivariateGaussian rotate=0, "
        f"and Independent; got {name}"
    )


def make_copula_ops_spec(module, copula):
    """Build a C++ ``CopulaSpec`` for pointwise h/h_inverse kernels."""
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


def make_spec(module, copula):
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
    from pyscarcopula.copula.frank import FrankCopula
    from pyscarcopula.copula.gumbel import GumbelCopula
    from pyscarcopula.copula.joe import JoeCopula

    if isinstance(copula, ClaytonCopula):
        spec.family = module.CopulaFamily.Clayton
        spec.transform = module.Transform.Softplus
        spec.offset = 0.0001
    elif isinstance(copula, GumbelCopula):
        spec.family = module.CopulaFamily.Gumbel
        spec.transform = module.Transform.Softplus
        spec.offset = 1.0001
    elif isinstance(copula, FrankCopula):
        spec.family = module.CopulaFamily.Frank
        spec.transform = module.Transform.Softplus
        spec.offset = 0.0001
    elif isinstance(copula, JoeCopula):
        spec.family = module.CopulaFamily.Joe
        spec.transform = module.Transform.Softplus
        spec.offset = 1.0001
    elif isinstance(copula, BivariateGaussianCopula):
        spec.family = module.CopulaFamily.Gaussian
        spec.rotation = module.Rotation.R0
        spec.transform = module.Transform.GaussianTanh
        spec.offset = 0.0
    else:
        raise CppUnsupported(f"Unsupported copula: {type(copula).__name__}")
    return spec
