"""Hand-rolled technical indicators (pandas / numpy only, no TA-Lib).

Design rules
------------
* Every function takes a pandas Series (or an OHLCV DataFrame) and returns a
  result aligned to the *same* index, so it slots into a backtest without
  re-indexing surprises.
* **No lookahead.** Each value at bar `i` is computed only from bars `<= i`.
  We use trailing rolling windows and exponential smoothing only; nothing here
  ever peeks at a future bar. (The backtest still has to be careful about *when*
  it reads these values — see backtest.py — but the indicators themselves are
  honest.)
* Early bars where a window is not yet full are returned as NaN. Callers must
  treat NaN as "not enough history, do not trade".

The OHLCV DataFrame convention used throughout the project: columns named
``open``, ``high``, ``low``, ``close``, ``volume`` (lowercase), indexed by
timestamp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


# --------------------------------------------------------------------------- #
# Moving averages
# --------------------------------------------------------------------------- #
def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average.

    Uses the standard ``2/(period+1)`` smoothing factor. ``adjust=False`` makes
    it a true recursive EMA (each value depends only on the prior EMA and the
    current price) which is what a live bot would compute bar by bar.
    """
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period, min_periods=period).mean()


def _rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (a.k.a. RMA / modified moving average).

    Equivalent to an EMA with ``alpha = 1/period``. Used by RSI, ATR and ADX,
    which Welles Wilder originally defined with this smoothing.
    """
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def slope(series: pd.Series, period: int) -> pd.Series:
    """Per-bar slope of ``series`` over a trailing ``period`` window.

    Returned in price units per bar via a least-squares fit. Positive means the
    line is rising. Used to check that EMA200 is sloping *up*, not just that
    price is above it.
    """
    x = np.arange(period, dtype=float)
    x_mean = x.mean()
    x_centered = x - x_mean
    denom = (x_centered**2).sum()

    def _fit(window: np.ndarray) -> float:
        if np.isnan(window).any():
            return np.nan
        y_mean = window.mean()
        return float((x_centered * (window - y_mean)).sum() / denom)

    return series.rolling(window=period, min_periods=period).apply(_fit, raw=True)


# --------------------------------------------------------------------------- #
# Momentum / oscillators
# --------------------------------------------------------------------------- #
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder). Range 0..100.

    >70 conventionally "overbought", <30 "oversold". In this project we do NOT
    treat low RSI as a buy signal on its own — it is used to confirm a pullback
    has *cooled* inside an established uptrend, and to flag exhaustion at peaks.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = _rma(gain, period)
    avg_loss = _rma(loss, period)
    rs = avg_gain / avg_loss
    out = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss == 0 the RS is +inf -> RSI 100; pandas handles that, but make
    # the all-gains / all-losses edges explicit and NaN-safe.
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(avg_gain != 0, out.where(avg_loss == 0, 0.0))
    return out


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line and histogram.

    macd = EMA(fast) - EMA(slow); signal = EMA(macd, signal); hist = macd - signal.
    """
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "hist": hist}
    )


def stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
) -> pd.DataFrame:
    """Stochastic oscillator (%K and %D). Range 0..100."""
    low_min = df["low"].rolling(k_period, min_periods=k_period).min()
    high_max = df["high"].rolling(k_period, min_periods=k_period).max()
    rng = (high_max - low_min).replace(0.0, np.nan)
    raw_k = 100.0 * (df["close"] - low_min) / rng
    k = raw_k.rolling(smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(d_period, min_periods=d_period).mean()
    return pd.DataFrame({"k": k, "d": d})


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R. Range -100..0 (closer to 0 = stronger / more overbought)."""
    high_max = df["high"].rolling(period, min_periods=period).max()
    low_min = df["low"].rolling(period, min_periods=period).min()
    rng = (high_max - low_min).replace(0.0, np.nan)
    return -100.0 * (high_max - df["close"]) / rng


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index.

    Uses mean absolute deviation (the classic CCI definition), not standard
    deviation. ~±100 marks the conventional normal band.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    ma = tp.rolling(period, min_periods=period).mean()
    mad = tp.rolling(period, min_periods=period).apply(
        lambda w: np.mean(np.abs(w - w.mean())), raw=True
    )
    return (tp - ma) / (0.015 * mad)


# --------------------------------------------------------------------------- #
# Volatility / range
# --------------------------------------------------------------------------- #
def true_range(df: pd.DataFrame) -> pd.Series:
    """True range = max(high-low, |high-prev_close|, |low-prev_close|)."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder). Absolute price units.

    This is the volatility ruler the Trade Setter uses to place stops, so that
    a $0.50 stop on a $5 coin and a $500 stop on a $50k coin represent the same
    *risk in volatility terms*.
    """
    return _rma(true_range(df), period)


def bollinger(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands plus %b and bandwidth.

    %b = where price sits inside the bands (0 = lower, 1 = upper).
    bandwidth = (upper-lower)/mid, a squeeze/expansion measure.
    """
    mid = sma(close, period)
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower).replace(0.0, np.nan)
    pct_b = (close - lower) / width
    bandwidth = (upper - lower) / mid
    return pd.DataFrame(
        {"upper": upper, "mid": mid, "lower": lower, "pct_b": pct_b, "bandwidth": bandwidth}
    )


# --------------------------------------------------------------------------- #
# Trend strength / direction
# --------------------------------------------------------------------------- #
def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index with +DI / -DI (Wilder). Range 0..100.

    ADX measures *how strong* a trend is regardless of direction; +DI vs -DI
    tells you the direction. We use ADX as a trend-quality input, not a trigger.
    """
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    tr = true_range(df)
    atr_n = _rma(tr, period)
    plus_di = 100.0 * _rma(plus_dm, period) / atr_n
    minus_di = 100.0 * _rma(minus_dm, period) / atr_n

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx_line = _rma(dx, period)
    return pd.DataFrame({"adx": adx_line, "plus_di": plus_di, "minus_di": minus_di})


def aroon(df: pd.DataFrame, period: int = 25) -> pd.DataFrame:
    """Aroon Up / Down / Oscillator. Range 0..100 (osc -100..100).

    Measures how recently the highest high / lowest low occurred within the
    window — a clean way to see a young, healthy uptrend vs a stalling one.
    """

    def _periods_since_max(w: np.ndarray) -> float:
        return float(len(w) - 1 - int(np.argmax(w)))

    def _periods_since_min(w: np.ndarray) -> float:
        return float(len(w) - 1 - int(np.argmin(w)))

    win = period + 1  # Aroon looks back `period` bars => window of period+1
    since_high = df["high"].rolling(win, min_periods=win).apply(_periods_since_max, raw=True)
    since_low = df["low"].rolling(win, min_periods=win).apply(_periods_since_min, raw=True)
    up = 100.0 * (period - since_high) / period
    down = 100.0 * (period - since_low) / period
    return pd.DataFrame({"up": up, "down": down, "oscillator": up - down})


# --------------------------------------------------------------------------- #
# Volume
# --------------------------------------------------------------------------- #
def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume. A cumulative volume line that rises on up-closes.

    We read its *direction* (is OBV confirming price?) not its absolute level.
    """
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def vwap(df: pd.DataFrame, period: int | None = None) -> pd.Series:
    """Volume-Weighted Average Price.

    Crypto trades 24/7 with no daily session to anchor to, so by default this
    is a *rolling* VWAP over ``period`` bars. Pass ``period=None`` for a
    cumulative (anchored-from-start) VWAP.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    if period is None:
        return pv.cumsum() / df["volume"].cumsum()
    pv_sum = pv.rolling(period, min_periods=period).sum()
    vol_sum = df["volume"].rolling(period, min_periods=period).sum().replace(0.0, np.nan)
    return pv_sum / vol_sum


def relative_volume(volume: pd.Series, period: int = 20) -> pd.Series:
    """Relative volume = current bar volume / average volume over `period`.

    >1 means heavier-than-usual participation. Used as the *independent volume
    confirmation* required before a setup is actionable.
    """
    avg = volume.rolling(period, min_periods=period).mean().replace(0.0, np.nan)
    return volume / avg


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #
def validate_ohlcv(df: pd.DataFrame) -> None:
    """Raise if `df` is not a well-formed OHLCV frame."""
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV frame missing columns: {missing}")
    if not df.index.is_monotonic_increasing:
        raise ValueError("OHLCV index must be sorted ascending by time (no lookahead).")
    if df.index.has_duplicates:
        raise ValueError("OHLCV index has duplicate timestamps.")
