import numpy as np
import pandas as pd
import pytest

from longbot import indicators as ind


def _ohlcv(close, vol=None):
    close = pd.Series(close, dtype=float)
    high = close * 1.01
    low = close * 0.99
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(vol if vol is not None else np.ones(len(close)) * 1000.0)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2023-01-01", periods=len(close), freq="h"),
    )


def test_ema_matches_pandas_ewm():
    s = pd.Series(np.arange(1, 51, dtype=float))
    got = ind.ema(s, 10)
    want = s.ewm(span=10, adjust=False, min_periods=10).mean()
    pd.testing.assert_series_equal(got, want)


def test_rsi_all_gains_is_100():
    s = pd.Series(np.arange(1, 60, dtype=float))  # strictly rising
    r = ind.rsi(s, 14).dropna()
    assert (r > 99.0).all()


def test_atr_positive_and_warms_up():
    df = _ohlcv(np.linspace(100, 120, 60))
    a = ind.atr(df, 14)
    assert a.iloc[:13].isna().all()      # warmup NaNs
    assert (a.dropna() > 0).all()


def test_slope_sign():
    rising = pd.Series(np.arange(0, 30, dtype=float))
    falling = pd.Series(np.arange(30, 0, -1, dtype=float))
    assert ind.slope(rising, 10).dropna().iloc[-1] > 0
    assert ind.slope(falling, 10).dropna().iloc[-1] < 0


def test_relative_volume_centers_on_one():
    vol = np.ones(50) * 1000.0
    rv = ind.relative_volume(pd.Series(vol), 20).dropna()
    assert np.allclose(rv, 1.0)


def test_validate_ohlcv_rejects_unsorted():
    df = _ohlcv(np.linspace(100, 110, 10))
    with pytest.raises(ValueError):
        ind.validate_ohlcv(df.iloc[::-1])
