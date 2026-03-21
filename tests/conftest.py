"""Shared fixtures for pyscarcopula tests."""
import pytest
import numpy as np
import pandas as pd
from pyscarcopula.utils import pobs
from pyscarcopula import (
    GumbelCopula, ClaytonCopula, FrankCopula, JoeCopula,
    IndependentCopula, CVineCopula, GaussianCopula, StudentCopula,
    BivariateGaussianCopula,
)


# ═══════════════════════════════════════════════════════════
# Copula instances for parametrized tests
# ═══════════════════════════════════════════════════════════

ARCHIMEDEAN_COPULAS = [
    (GumbelCopula, 0, 2.0),
    (GumbelCopula, 180, 2.0),
    (ClaytonCopula, 0, 2.0),
    (ClaytonCopula, 180, 2.0),
    (FrankCopula, 0, 5.0),
    (JoeCopula, 0, 2.0),
    (JoeCopula, 180, 2.0),
]

ARCHIMEDEAN_IDS = [
    f"{cls.__name__}-{rot}" for cls, rot, _ in ARCHIMEDEAN_COPULAS
]


@pytest.fixture(params=ARCHIMEDEAN_COPULAS, ids=ARCHIMEDEAN_IDS)
def archimedean(request):
    """Yield (copula_instance, test_r_value) for each Archimedean copula."""
    cls, rot, r = request.param
    return cls(rotate=rot), r


# ═══════════════════════════════════════════════════════════
# Data fixtures
# ═══════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def crypto_data():
    """Load crypto dataset, return pseudo-obs (T, 2)."""
    df = pd.read_csv("data/crypto_prices.csv", index_col=0, sep=";")
    prices = df[["BTC-USD", "ETH-USD"]].dropna().values
    lr = np.diff(np.log(prices), axis=0)
    return pobs(lr)


@pytest.fixture(scope="session")
def crypto_data_6d():
    """Load 6-crypto dataset, return pseudo-obs (250, 6)."""
    df = pd.read_csv("data/crypto_prices.csv", index_col=0, sep=";")
    tickers = ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "XRP-USD", "DOGE-USD"]
    prices = df[tickers].dropna().values
    lr = np.diff(np.log(prices), axis=0)[:250]
    return pobs(lr)


@pytest.fixture
def random_u2():
    """Random pseudo-obs (200, 2)."""
    rng = np.random.default_rng(42)
    return pobs(rng.standard_normal((200, 2)))


@pytest.fixture
def independent_u2():
    """Independent uniform data (500, 2)."""
    rng = np.random.default_rng(123)
    return rng.uniform(0, 1, (500, 2))
