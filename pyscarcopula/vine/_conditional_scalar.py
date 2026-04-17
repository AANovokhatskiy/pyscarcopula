"""Scalar copula fast paths for R-vine conditional sampling."""

from __future__ import annotations

import numpy as np

from pyscarcopula.copula.clayton import (
    ClaytonCopula, _clayton_h, _clayton_h_inv, _clayton_pdf,
)
from pyscarcopula.copula.elliptical import (
    BivariateGaussianCopula,
    _gauss_h_inv_numba,
    _gauss_h_numba,
    _gauss_log_pdf_numba,
)
from pyscarcopula.copula.frank import (
    FrankCopula, _frank_h, _frank_h_inv, _frank_pdf,
)
from pyscarcopula.copula.gumbel import (
    GumbelCopula, _gumbel_h, _gumbel_h_inverse_newton, _gumbel_pdf,
)
from pyscarcopula.copula.independent import IndependentCopula
from pyscarcopula.copula.joe import (
    JoeCopula, _joe_h, _joe_h_inverse_newton, _joe_pdf,
)


_FAMILY_FALLBACK = -1
_FAMILY_INDEPENDENT = 0
_FAMILY_CLAYTON = 1
_FAMILY_FRANK = 2
_FAMILY_GUMBEL = 3
_FAMILY_JOE = 4
_FAMILY_GAUSSIAN = 5


def _copula_family_id(copula):
    if isinstance(copula, IndependentCopula):
        return _FAMILY_INDEPENDENT
    if isinstance(copula, ClaytonCopula):
        return _FAMILY_CLAYTON
    if isinstance(copula, FrankCopula):
        return _FAMILY_FRANK
    if isinstance(copula, GumbelCopula):
        return _FAMILY_GUMBEL
    if isinstance(copula, JoeCopula):
        return _FAMILY_JOE
    if isinstance(copula, BivariateGaussianCopula):
        return _FAMILY_GAUSSIAN
    return _FAMILY_FALLBACK


def _scalar_array(value):
    return np.array([value], dtype=np.float64)


def _scratch_arrays():
    return (
        np.empty(1, dtype=np.float64),
        np.empty(1, dtype=np.float64),
        np.empty(1, dtype=np.float64),
    )


def _set_scratch(work, u, v, r):
    ua, va, ra = work
    ua[0] = u
    va[0] = v
    ra[0] = r
    return ua, va, ra


def _clip_scalar(value):
    eps = 1e-10
    if value < eps:
        return eps
    if value > 1.0 - eps:
        return 1.0 - eps
    return float(value)


def _unrotated_pdf_scalar(copula, u, v, r):
    ua = _scalar_array(u)
    va = _scalar_array(v)
    ra = _scalar_array(r)
    family_id = _copula_family_id(copula)
    return _unrotated_pdf_scalar_fast(family_id, copula, ua, va, ra)


def _unrotated_pdf_scalar_fast(family_id, copula, ua, va, ra):
    if family_id == _FAMILY_INDEPENDENT:
        return 1.0
    if family_id == _FAMILY_CLAYTON:
        return float(_clayton_pdf(ua, va, ra)[0])
    if family_id == _FAMILY_FRANK:
        return float(_frank_pdf(ua, va, ra)[0])
    if family_id == _FAMILY_GUMBEL:
        return float(_gumbel_pdf(ua, va, ra)[0])
    if family_id == _FAMILY_JOE:
        return float(_joe_pdf(ua, va, ra)[0])
    if family_id == _FAMILY_GAUSSIAN:
        return float(np.exp(_gauss_log_pdf_numba(ua, va, ra))[0])
    return float(copula.pdf(ua, va, ra)[0])


def _unrotated_h_scalar(copula, u, v, r):
    ua = _scalar_array(u)
    va = _scalar_array(v)
    ra = _scalar_array(r)
    family_id = _copula_family_id(copula)
    return _unrotated_h_scalar_fast(family_id, copula, ua, va, ra)


def _unrotated_h_scalar_fast(family_id, copula, ua, va, ra):
    if family_id == _FAMILY_INDEPENDENT:
        return float(ua[0])
    if family_id == _FAMILY_CLAYTON:
        return float(_clayton_h(ua, va, ra)[0])
    if family_id == _FAMILY_FRANK:
        return float(_frank_h(ua, va, ra)[0])
    if family_id == _FAMILY_GUMBEL:
        return float(_gumbel_h(ua, va, ra)[0])
    if family_id == _FAMILY_JOE:
        return float(_joe_h(ua, va, ra)[0])
    if family_id == _FAMILY_GAUSSIAN:
        return float(_gauss_h_numba(ua, va, ra)[0])
    return float(copula.h_unrotated(ua, va, ra)[0])


def _unrotated_h_inverse_scalar(copula, u, v, r):
    ua = _scalar_array(u)
    va = _scalar_array(v)
    ra = _scalar_array(r)
    family_id = _copula_family_id(copula)
    return _unrotated_h_inverse_scalar_fast(family_id, copula, ua, va, ra)


def _unrotated_h_inverse_scalar_fast(family_id, copula, ua, va, ra):
    if family_id == _FAMILY_INDEPENDENT:
        return float(ua[0])
    if family_id == _FAMILY_CLAYTON:
        return float(_clayton_h_inv(ua, va, ra)[0])
    if family_id == _FAMILY_FRANK:
        return float(_frank_h_inv(ua, va, ra)[0])
    if family_id == _FAMILY_GUMBEL:
        return float(_gumbel_h_inverse_newton(ua, va, ra)[0])
    if family_id == _FAMILY_JOE:
        return float(_joe_h_inverse_newton(ua, va, ra)[0])
    if family_id == _FAMILY_GAUSSIAN:
        return float(_gauss_h_inv_numba(ua, va, ra)[0])
    return float(copula.h_inverse_unrotated(ua, va, ra)[0])


def _copula_pdf_scalar(copula, u, v, r):
    rot = copula.rotate
    if rot == 0:
        u0, v0 = u, v
    elif rot == 90:
        u0, v0 = 1.0 - u, v
    elif rot == 180:
        u0, v0 = 1.0 - u, 1.0 - v
    else:
        u0, v0 = u, 1.0 - v
    return _unrotated_pdf_scalar(copula, u0, v0, r)


def _copula_h_scalar(copula, u, v, r):
    rot = copula.rotate
    if rot == 0:
        return _unrotated_h_scalar(copula, u, v, r)
    if rot == 90:
        return 1.0 - _unrotated_h_scalar(copula, 1.0 - u, v, r)
    if rot == 180:
        return 1.0 - _unrotated_h_scalar(copula, 1.0 - u, 1.0 - v, r)
    return _unrotated_h_scalar(copula, u, 1.0 - v, r)


def _copula_h_inverse_scalar(copula, u, v, r):
    rot = copula.rotate
    if rot == 0:
        return _unrotated_h_inverse_scalar(copula, u, v, r)
    if rot == 90:
        return 1.0 - _unrotated_h_inverse_scalar(copula, 1.0 - u, v, r)
    if rot == 180:
        return 1.0 - _unrotated_h_inverse_scalar(copula, 1.0 - u, 1.0 - v, r)
    return _unrotated_h_inverse_scalar(copula, u, 1.0 - v, r)


def _copula_pdf_meta(family_id, rot, copula, u, v, r, work=None):
    if rot == 0:
        u0, v0 = u, v
    elif rot == 90:
        u0, v0 = 1.0 - u, v
    elif rot == 180:
        u0, v0 = 1.0 - u, 1.0 - v
    else:
        u0, v0 = u, 1.0 - v
    if work is not None:
        ua, va, ra = _set_scratch(work, u0, v0, r)
        return _unrotated_pdf_scalar_fast(family_id, copula, ua, va, ra)
    return _unrotated_pdf_scalar_fast(
        family_id, copula, _scalar_array(u0), _scalar_array(v0),
        _scalar_array(r))


def _copula_h_meta(family_id, rot, copula, u, v, r, work=None):
    if rot == 0:
        if work is not None:
            ua, va, ra = _set_scratch(work, u, v, r)
            return _unrotated_h_scalar_fast(family_id, copula, ua, va, ra)
        return _unrotated_h_scalar_fast(
            family_id, copula, _scalar_array(u), _scalar_array(v),
            _scalar_array(r))
    if rot == 90:
        if work is not None:
            ua, va, ra = _set_scratch(work, 1.0 - u, v, r)
            return 1.0 - _unrotated_h_scalar_fast(
                family_id, copula, ua, va, ra)
        return 1.0 - _unrotated_h_scalar_fast(
            family_id, copula, _scalar_array(1.0 - u), _scalar_array(v),
            _scalar_array(r))
    if rot == 180:
        if work is not None:
            ua, va, ra = _set_scratch(work, 1.0 - u, 1.0 - v, r)
            return 1.0 - _unrotated_h_scalar_fast(
                family_id, copula, ua, va, ra)
        return 1.0 - _unrotated_h_scalar_fast(
            family_id, copula, _scalar_array(1.0 - u), _scalar_array(1.0 - v),
            _scalar_array(r))
    if work is not None:
        ua, va, ra = _set_scratch(work, u, 1.0 - v, r)
        return _unrotated_h_scalar_fast(family_id, copula, ua, va, ra)
    return _unrotated_h_scalar_fast(
        family_id, copula, _scalar_array(u), _scalar_array(1.0 - v),
        _scalar_array(r))


def _copula_h_inverse_meta(family_id, rot, copula, u, v, r, work=None):
    if rot == 0:
        if work is not None:
            ua, va, ra = _set_scratch(work, u, v, r)
            return _unrotated_h_inverse_scalar_fast(
                family_id, copula, ua, va, ra)
        return _unrotated_h_inverse_scalar_fast(
            family_id, copula, _scalar_array(u), _scalar_array(v),
            _scalar_array(r))
    if rot == 90:
        if work is not None:
            ua, va, ra = _set_scratch(work, 1.0 - u, v, r)
            return 1.0 - _unrotated_h_inverse_scalar_fast(
                family_id, copula, ua, va, ra)
        return 1.0 - _unrotated_h_inverse_scalar_fast(
            family_id, copula, _scalar_array(1.0 - u), _scalar_array(v),
            _scalar_array(r))
    if rot == 180:
        if work is not None:
            ua, va, ra = _set_scratch(work, 1.0 - u, 1.0 - v, r)
            return 1.0 - _unrotated_h_inverse_scalar_fast(
                family_id, copula, ua, va, ra)
        return 1.0 - _unrotated_h_inverse_scalar_fast(
            family_id, copula, _scalar_array(1.0 - u), _scalar_array(1.0 - v),
            _scalar_array(r))
    if work is not None:
        ua, va, ra = _set_scratch(work, u, 1.0 - v, r)
        return _unrotated_h_inverse_scalar_fast(
            family_id, copula, ua, va, ra)
    return _unrotated_h_inverse_scalar_fast(
        family_id, copula, _scalar_array(u), _scalar_array(1.0 - v),
        _scalar_array(r))
