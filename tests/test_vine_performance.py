"""Optional performance regression checks for vine conditional prediction."""

import os
import time

import numpy as np
import pandas as pd
import pytest

from pyscarcopula._utils import pobs
from pyscarcopula.vine.rvine import RVineCopula


def _example_u():
    crypto_prices = pd.read_csv("data/crypto_prices.csv", index_col=0, sep=";")
    tickers = [
        "BTC-USD",
        "ETH-USD",
        "BNB-USD",
        "ADA-USD",
        "XRP-USD",
        "DOGE-USD",
    ]
    returns = np.log(crypto_prices[tickers] / crypto_prices[tickers].shift(1))
    return pobs(returns[1:251].values)


def _skip_unless_enabled():
    if os.environ.get("PYSCA_RUN_BENCHMARKS") != "1":
        pytest.skip("set PYSCA_RUN_BENCHMARKS=1 to run benchmark checks")


@pytest.mark.data
@pytest.mark.benchmark
def test_rvine_mle_conditional_grid_predict_speed_smoke():
    _skip_unless_enabled()
    u = _example_u()
    vine = RVineCopula()
    vine.fit(u, method="mle")

    t0 = time.perf_counter()
    out = vine.predict(
        1000,
        u=u,
        given={0: 0.2, 3: 0.8},
        horizon="next",
        quad_order=4,
        conditional_method="grid",
    )
    elapsed = time.perf_counter() - t0

    assert out.shape == (1000, 6)
    assert elapsed < 2.0


@pytest.mark.data
@pytest.mark.benchmark
def test_rvine_scar_conditional_grid_cached_predict_speed_smoke():
    _skip_unless_enabled()
    u = _example_u()
    vine = RVineCopula()
    vine.fit(u, method="scar-tm-ou")
    vine.predict(
        10,
        u=u,
        given={0: 0.2, 3: 0.8},
        horizon="next",
        quad_order=4,
        conditional_method="grid",
    )

    t0 = time.perf_counter()
    out = vine.predict(
        1000,
        u=u,
        given={0: 0.2, 3: 0.8},
        horizon="next",
        quad_order=4,
        conditional_method="grid",
    )
    elapsed = time.perf_counter() - t0

    assert out.shape == (1000, 6)
    assert elapsed < 2.0
