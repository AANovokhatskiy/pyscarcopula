import numpy as np
import pytest

from pyscarcopula.copula.clayton import ClaytonCopula
from pyscarcopula.copula.elliptical import BivariateGaussianCopula
from pyscarcopula.copula.frank import FrankCopula
from pyscarcopula.copula.gumbel import GumbelCopula
from pyscarcopula.copula.joe import JoeCopula
from pyscarcopula.numerical.gas_filter import (
    gas_filter,
    _gas_score,
    _gas_unit_dlog_dr,
    _gas_unit_log_pdf,
)


def _gas_filter_reference(omega, gamma, beta, u, copula, score_eps=1e-4):
    T = len(u)
    g_path = np.empty(T)
    r_path = np.empty(T)
    total_logL = 0.0

    if abs(beta) < 1.0 - 1e-8:
        g_t = omega / (1.0 - beta)
    else:
        g_t = omega

    for t in range(T):
        g_path[t] = g_t
        r_t = float(copula.transform(np.array([g_t]))[0])
        r_path[t] = r_t

        u1 = u[t:t + 1, 0]
        u2 = u[t:t + 1, 1]
        ll_t = float(copula.log_pdf(u1, u2, np.array([r_t]))[0])
        if not np.isfinite(ll_t):
            return g_path, r_path, -1e10

        total_logL += ll_t

        if t < T - 1:
            s_t = _gas_score(
                u1, u2, g_t, r_t, ll_t, copula, 'unit', score_eps)
            if not np.isfinite(s_t):
                return g_path, r_path, -1e10

            s_t = np.clip(s_t, -100.0, 100.0)
            g_t = omega + beta * g_t + gamma * s_t
            g_t = np.clip(g_t, -50.0, 50.0)

    return g_path, r_path, total_logL


@pytest.mark.parametrize(
    "copula",
    [
        BivariateGaussianCopula(),
        ClaytonCopula(rotate=0),
        ClaytonCopula(rotate=90),
        ClaytonCopula(rotate=180),
        ClaytonCopula(rotate=270),
        ClaytonCopula(transform_type='xtanh'),
        FrankCopula(),
        FrankCopula(transform_type='xtanh'),
        GumbelCopula(rotate=0),
        GumbelCopula(rotate=90),
        GumbelCopula(rotate=180),
        GumbelCopula(rotate=270),
        GumbelCopula(transform_type='xtanh'),
        JoeCopula(rotate=0),
        JoeCopula(rotate=90),
        JoeCopula(rotate=180),
        JoeCopula(rotate=270),
        JoeCopula(transform_type='xtanh'),
    ],
)
def test_gas_filter_unit_numba_matches_reference(copula):
    rng = np.random.default_rng(12345)
    u = rng.uniform(0.01, 0.99, size=(80, 2))
    params = (0.04, 0.7, 0.55)

    got = gas_filter(*params, u, copula, scaling='unit')
    expected = _gas_filter_reference(*params, u, copula)

    assert np.allclose(got[0], expected[0], rtol=1e-10, atol=1e-10)
    assert np.allclose(got[1], expected[1], rtol=1e-10, atol=1e-10)
    assert got[2] == pytest.approx(expected[2], rel=1e-10, abs=1e-10)


def test_gas_unit_dispatch_rejects_unknown_family():
    u1 = np.array([0.4], dtype=np.float64)
    u2 = np.array([0.6], dtype=np.float64)
    r = np.array([1.5], dtype=np.float64)

    with pytest.raises(ValueError, match="Unsupported GAS copula family"):
        _gas_unit_log_pdf(999, u1, u2, r)

    with pytest.raises(ValueError, match="Unsupported GAS copula family"):
        _gas_unit_dlog_dr(999, u1, u2, r)
