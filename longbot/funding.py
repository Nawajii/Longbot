"""Binance USDT-M perpetual funding-rate fetch, cache, and no-lookahead
alignment onto a spot OHLCV bar grid.

DATA REALITY (see README): Binance does not offer free historical
liquidation or long-history open-interest data. The funding rate
(``/fapi/v1/fundingRate``) is the one freely available, deep-history proxy
for aggregate leveraged positioning, printed every 8 hours back to each
perpetual's listing (~2020 for most majors). This module fetches and
caches it exactly like ``run_real_backtest.fetch_or_cache`` caches spot
klines, and provides the causal (no-lookahead) machinery needed to use it
as a trailing-window percentile signal.

No-lookahead is the whole point of this file:
  * a 4h bar at time t is aligned to the funding print with the LATEST
    fundingTime <= t -- never a print stamped after the bar (see
    ``align_to_bars``, a backward ``merge_asof``).
  * the percentile rank and trailing median of a funding print are computed
    using only that print and the ones before it (a trailing rolling
    window) -- never a later print.
"""

from __future__ import annotations

import os
import sys
import time as _time

import numpy as np
import pandas as pd

from run_real_backtest import _http_get_json

FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
FUNDING_INTERVAL_MS = 8 * 60 * 60 * 1000  # Binance perp funding prints every 8h


def fetch_funding_rate_range(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Page ``/fapi/v1/fundingRate`` for ALL prints between `start` and `end`.

    Returns a DataFrame indexed by ``funding_time`` (UTC) with one column
    ``funding_rate``. If the symbol has no USDT-M perpetual (bad symbol,
    never listed as a perp), Binance returns an error payload and this
    returns an EMPTY frame rather than raising -- callers use that to skip
    the pair and report why, per the prompt's "skip any without a perp"
    requirement.
    """
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int((pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.utcnow()).timestamp() * 1000)
    frames: list[pd.DataFrame] = []
    cursor = start_ms
    calls = 0
    while cursor < end_ms:
        url = f"{FUNDING_URL}?symbol={symbol}&startTime={cursor}&endTime={end_ms}&limit=1000"
        try:
            raw = _http_get_json(url)
        except Exception:
            return pd.DataFrame(columns=["funding_rate"]).rename_axis("funding_time")
        if isinstance(raw, dict) and raw.get("code"):
            # e.g. {"code":-1121,"msg":"Invalid symbol."} -- no perp for this asset.
            return pd.DataFrame(columns=["funding_rate"]).rename_axis("funding_time")
        if not raw:
            break
        frames.append(pd.DataFrame(raw))
        last_time = int(raw[-1]["fundingTime"])
        calls += 1
        if calls % 10 == 0:
            print(f"[info] {symbol} funding: fetched ~{calls * 1000} prints...", file=sys.stderr)
        if len(raw) < 1000:
            break
        cursor = last_time + 1
        _time.sleep(0.25)

    if not frames:
        return pd.DataFrame(columns=["funding_rate"]).rename_axis("funding_time")

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"fundingTime": "funding_time", "fundingRate": "funding_rate"})
    df["funding_time"] = pd.to_datetime(df["funding_time"], unit="ms")
    df["funding_rate"] = df["funding_rate"].astype(float)
    df = df.drop_duplicates("funding_time").sort_values("funding_time").set_index("funding_time")
    return df[["funding_rate"]]


def fetch_funding_or_cache(
    symbol: str, start: str, end: str | None = None,
    cache_dir: str = "data", refresh: bool = False,
) -> pd.DataFrame:
    """Cache wrapper matching ``run_real_backtest.fetch_or_cache``'s convention:
    ``cache_dir/{symbol}_funding.csv``. An empty cached file means "checked,
    no perp/funding history" -- it is cached too, so a symbol without a
    perp is not re-queried on every run."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{symbol}_funding.csv")
    if os.path.exists(path) and not refresh:
        df = pd.read_csv(path)
        if df.empty or "funding_time" not in df.columns:
            return pd.DataFrame(columns=["funding_rate"]).rename_axis("funding_time")
        df["funding_time"] = pd.to_datetime(df["funding_time"])
        return df.set_index("funding_time")

    df = fetch_funding_rate_range(symbol, start, end)
    df.reset_index().to_csv(path, index=False)
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) if end else (df.index.max() if not df.empty else lo)
    if df.empty:
        return df
    return df[(df.index >= lo) & (df.index <= hi)]


# --------------------------------------------------------------------------- #
# no-lookahead alignment
# --------------------------------------------------------------------------- #
def align_to_bars(funding_df: pd.DataFrame, bar_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward-fill every column of `funding_df` (indexed by funding_time)
    onto `bar_index`, using the LAST COMPLETED print at or before each bar
    -- i.e. a backward "as of" join. A bar can never see a funding print
    timestamped after it. Bars before the first funding print get NaN.
    """
    cols = list(funding_df.columns)
    if funding_df.empty:
        return pd.DataFrame({c: np.nan for c in cols}, index=bar_index)

    # merge_asof requires BOTH keys to share the exact same datetime64
    # resolution. Bar timestamps and funding timestamps can end up at
    # different resolutions depending on how each was parsed (a fresh
    # fetch via `unit="ms"` vs. a CSV round-trip's string inference, for
    # example), even though they represent the same instants -- so
    # normalize both merge keys to a common resolution (ns) right before
    # the merge. The function's return value is still reindexed against
    # the caller's original `bar_index` untouched, so this is purely an
    # internal merge-key normalization, not a change to what gets aligned.
    common_res = "datetime64[ns]"
    bars = pd.DataFrame({
        "bar_time": pd.DatetimeIndex(bar_index).astype(common_res),
    }).sort_values("bar_time")
    fr = funding_df.sort_index().reset_index()  # index is always named "funding_time" (see fetch_* above)
    fr["funding_time"] = pd.DatetimeIndex(fr["funding_time"]).astype(common_res)

    merged = pd.merge_asof(bars, fr, left_on="bar_time", right_on="funding_time", direction="backward")
    merged = merged.set_index("bar_time")
    return merged[cols].reindex(pd.DatetimeIndex(bar_index).astype(common_res)).set_axis(bar_index)


def funding_percentile_rank(funding_rate: pd.Series, window: int) -> pd.Series:
    """Trailing, causal percentile rank of each print within its own
    trailing `window` prints (inclusive of itself, exclusive of anything
    after it). Range [0, 1]. NaN until `window` prints of history exist."""

    def _pct_rank(w: np.ndarray) -> float:
        return float((w <= w[-1]).sum()) / len(w)

    return funding_rate.rolling(window, min_periods=window).apply(_pct_rank, raw=True)


def funding_trailing_median(funding_rate: pd.Series, window: int) -> pd.Series:
    """Trailing, causal median funding rate over the same window convention."""
    return funding_rate.rolling(window, min_periods=window).median()


def build_funding_features(funding_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Attach causal pct_rank + trailing_median columns to a funding_time-
    indexed DataFrame (columns computed BEFORE alignment, on the funding
    series' own native ~8h timeline, which is both more correct -- windows
    are in funding-print units, not bar units -- and far cheaper)."""
    if funding_df.empty:
        return pd.DataFrame(columns=["funding_rate", "pct_rank", "trailing_median"])
    out = funding_df.copy()
    out["pct_rank"] = funding_percentile_rank(out["funding_rate"], window)
    out["trailing_median"] = funding_trailing_median(out["funding_rate"], window)
    return out
