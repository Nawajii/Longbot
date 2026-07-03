"""Multi-scale equilibrium gate — the 24h (daily) higher-timeframe filter.

Hypothesis: requiring agreement ACROSS timeframes isolates higher-quality
entries. The existing engine is the 4h "entry structure" (a located AT_NODE
pullback). This module adds the 24h "trend + macro equilibrium" scale, and the
two are combined with a plain AND: trade only when the daily says uptrend AND
price sits at/above a daily-scale equilibrium support, AND the 4h shows the
located pullback.

IMPORTANT — this is still, mechanically, a FILTER on the existing entry engine.
A filter can only REMOVE trades; it cannot manufacture edge. The test is whether
multi-scale *alignment* isolates a subset that happens to carry genuine edge; the
base-rate expectation is that it does not.

NO-LOOKAHEAD (sacred): the daily gate at a 4h bar uses only COMPLETED daily
candles. The in-progress daily candle (which is built from 4h bars including ones
at/after the current bar) is never read — we shift the daily signal forward by
one day so a 4h bar on day D sees only day D-1's finished daily data. There is a
unit test proving a 4h bar's gate value is unchanged by future bars.
"""

from __future__ import annotations

import pandas as pd

from . import indicators as ind
from . import volume_profile as vp


def daily_ohlcv(df_4h: pd.DataFrame) -> pd.DataFrame:
    """Resample a 4h OHLCV frame to daily (UTC-day) candles."""
    d = df_4h.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return d.dropna()


def daily_gate_series(
    df_4h: pd.DataFrame,
    ema_fast: int = 50,
    ema_slow: int = 200,
    slope_bars: int = 20,
    vp_lookback: int = 120,
) -> pd.Series:
    """Boolean per-4h-bar gate: daily uptrend AND price >= daily equilibrium support.

    * Daily trend: daily close > EMA(fast) > EMA(slow) with EMA(slow) rising.
    * Daily equilibrium: daily close at/above the daily volume-profile value-area
      low (macro support) — i.e. not hanging below the daily equilibrium zone.

    Returns a boolean Series aligned to ``df_4h.index``. No-lookahead: each 4h bar
    reads only the prior COMPLETED daily candle (daily signal shifted by 1 day).
    """
    d = daily_ohlcv(df_4h)
    close = d["close"]

    ema_f = ind.ema(close, ema_fast)
    ema_s = ind.ema(close, ema_slow)
    slope = ind.slope(ema_s, slope_bars)
    trend_ok = (close > ema_f) & (ema_f > ema_s) & (slope > 0)

    loc = vp.compute_location_features(d, lookback=vp_lookback)
    equilibrium_ok = close >= loc["vp_val"]        # at/above the daily value-area low

    daily_gate = (trend_ok & equilibrium_ok).fillna(False)

    # No-lookahead: a 4h bar on day D may use only day D-1's COMPLETED daily bar.
    daily_gate = daily_gate.shift(1).fillna(False)

    # Map daily -> 4h by forward-fill (each 4h bar takes the most recent daily row,
    # which now holds the prior completed day's value).
    gate_4h = daily_gate.reindex(df_4h.index, method="ffill").fillna(False)
    return gate_4h.astype(bool)
