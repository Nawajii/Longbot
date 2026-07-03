import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _dates(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="D").date


@pytest.fixture
def cointegrated_prices():
    """Synthetic pair where B is a random walk and A = beta*B plus a
    mean-reverting (OU-like) noise term -- genuinely cointegrated."""
    rng = np.random.default_rng(42)
    n = 400
    beta_true = 1.3
    log_b = np.cumsum(rng.normal(0, 0.02, n))

    noise = np.zeros(n)
    phi = 0.9  # mean reversion speed of the spread
    for i in range(1, n):
        noise[i] = phi * noise[i - 1] + rng.normal(0, 0.01)

    log_a = beta_true * log_b + 0.5 + noise
    prices = pd.DataFrame({
        "A": np.exp(log_a),
        "B": np.exp(log_b),
    }, index=_dates(n))
    return prices


@pytest.fixture
def independent_prices():
    """Two unrelated random walks -- should generally fail cointegration."""
    rng = np.random.default_rng(7)
    n = 400
    log_a = np.cumsum(rng.normal(0, 0.02, n))
    log_b = np.cumsum(rng.normal(0, 0.02, n))
    prices = pd.DataFrame({
        "A": np.exp(log_a),
        "B": np.exp(log_b),
    }, index=_dates(n))
    return prices
