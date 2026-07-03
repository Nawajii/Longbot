"""
Regime bucketing, reused from the prior single-asset project: BTC price
relative to its own 200-period EMA. bull = BTC close > EMA200, else bear.

This is used only for reporting (regime breakdown per fold), not for gating
entries -- the entire point of testing a cointegration pairs strategy is
that it should NOT need a regime filter to survive the bear.
"""

from __future__ import annotations

import pandas as pd

from . import config


def compute_btc_regime(btc_close: pd.Series) -> pd.Series:
    """Given a BTC daily close series (date index), return a same-indexed
    Series of 'bull' / 'bear' labels based on close vs EMA(span=200).
    The EMA at date t only uses data up to and including t (pandas .ewm is
    causal/no-lookahead by construction)."""
    ema = btc_close.ewm(span=config.REGIME_EMA_SPAN, adjust=False).mean()
    regime = pd.Series(
        ["bull" if c > e else "bear" for c, e in zip(btc_close, ema)],
        index=btc_close.index,
        name="regime",
    )
    return regime


def label_dates(dates: pd.Index, regime_series: pd.Series) -> list[str]:
    """Look up the regime label for each date in `dates`. Dates without a
    matching BTC bar (shouldn't happen once data is aligned) map to
    'unknown'."""
    return [regime_series.get(d, "unknown") for d in dates]
