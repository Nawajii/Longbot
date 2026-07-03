"""
Deep Binance kline fetch with CSV caching.

Design reused from the prior (retired) single-asset project: paginate the
public /api/v3/klines endpoint 1000 bars at a time, cache to CSV, and on
subsequent runs only fetch bars newer than what's already cached.

This module makes real network calls to api.binance.com. It is exercised
here only through unit tests with a stubbed transport -- the actual fetch
must be run in an environment with outbound network access (this repo's
sandbox does not have one; see README "Reproduce").
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_cache")
MAX_BARS_PER_REQUEST = 1000
REQUEST_SLEEP_SEC = 0.25
MAX_RETRIES = 5

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]

# Tickers that were renamed on Binance. Each entry maps a logical asset to
# an ordered list of Binance symbols to fetch and splice together
# chronologically (earlier symbol's history first, then the successor's).
# MATIC -> POL migration (Sept 2024): MATICUSDT klines are frozen at the
# migration date; POLUSDT continues the same USDT market 1:1 from there.
SYMBOL_ALIASES: dict[str, list[str]] = {
    "MATIC": ["MATICUSDT", "POLUSDT"],
}


def _quote_symbol(asset: str) -> list[str]:
    """Resolve a logical asset name to the ordered list of Binance symbols
    that together make up its continuous USDT price history."""
    if asset in SYMBOL_ALIASES:
        return SYMBOL_ALIASES[asset]
    return [f"{asset}USDT"]


def _cache_path(binance_symbol: str) -> str:
    return os.path.join(CACHE_DIR, f"{binance_symbol}_{_interval()}.csv")


def _interval() -> str:
    from . import config
    return config.BINANCE_INTERVAL


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _fetch_klines_raw(binance_symbol: str, start_ms: int, end_ms: int) -> list[list]:
    """Paginate /api/v3/klines from start_ms to end_ms (inclusive-ish)."""
    out: list[list] = []
    cursor = start_ms
    from . import config
    interval = config.BINANCE_INTERVAL

    while cursor < end_ms:
        params = {
            "symbol": binance_symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": MAX_BARS_PER_REQUEST,
        }
        for attempt in range(MAX_RETRIES):
            resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
            if resp.status_code == 200:
                break
            if resp.status_code in (429, 418):
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            if resp.status_code == 400:
                # Symbol likely doesn't exist (e.g. pre-rebrand ticker on an
                # exchange that never listed it, or a delisted/typo'd pair).
                return out
            resp.raise_for_status()
        else:
            resp.raise_for_status()

        batch = resp.json()
        if not batch:
            break
        out.extend(batch)
        last_open_time = batch[-1][0]
        if len(batch) < MAX_BARS_PER_REQUEST:
            break
        cursor = last_open_time + 1
        time.sleep(REQUEST_SLEEP_SEC)

    return out


def _load_cache(binance_symbol: str) -> pd.DataFrame | None:
    path = _cache_path(binance_symbol)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df


def _save_cache(binance_symbol: str, df: pd.DataFrame) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(_cache_path(binance_symbol), index=False)


def fetch_symbol_klines(binance_symbol: str, start: str, refresh_tail: bool = True) -> pd.DataFrame:
    """Fetch (with CSV cache) all klines for one raw Binance symbol from
    `start` (YYYY-MM-DD) to now. Returns a DataFrame indexed by open_time
    (UTC) with numeric OHLCV columns, or an empty DataFrame if the symbol
    does not exist on Binance (e.g. a post-rebrand ticker queried before
    the rebrand happened)."""
    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    now_dt = datetime.now(timezone.utc)

    cached = _load_cache(binance_symbol)
    if cached is not None and not cached.empty:
        last_cached = cached["open_time"].max()
        # Refetch the last cached bar too, in case it was partial when cached.
        fetch_from = last_cached if refresh_tail else last_cached + pd.Timedelta(days=1)
        fetch_from_ms = _ms(fetch_from.to_pydatetime())
    else:
        cached = None
        fetch_from_ms = _ms(start_dt)

    end_ms = _ms(now_dt)
    if fetch_from_ms >= end_ms:
        new_df = pd.DataFrame(columns=KLINE_COLUMNS)
    else:
        raw = _fetch_klines_raw(binance_symbol, fetch_from_ms, end_ms)
        new_df = pd.DataFrame(raw, columns=KLINE_COLUMNS) if raw else pd.DataFrame(columns=KLINE_COLUMNS)
        if not new_df.empty:
            new_df["open_time"] = pd.to_datetime(new_df["open_time"], unit="ms", utc=True)

    if cached is not None and not new_df.empty:
        combined = pd.concat([cached, new_df], ignore_index=True)
    elif cached is not None:
        combined = cached
    else:
        combined = new_df

    if combined.empty:
        return combined

    combined = combined.drop_duplicates(subset="open_time", keep="last")
    combined = combined.sort_values("open_time").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        combined[col] = combined[col].astype(float)

    _save_cache(binance_symbol, combined)
    return combined


def get_symbol_series(asset: str, start: str = None) -> pd.Series:
    """Return a continuous daily close-price series (UTC date index) for a
    logical asset, splicing renamed tickers (e.g. MATIC->POL) end to end."""
    from . import config
    start = start or config.DATA_START

    symbols = _quote_symbol(asset)
    pieces = []
    for sym in symbols:
        df = fetch_symbol_klines(sym, start)
        if df.empty:
            continue
        s = df.set_index("open_time")["close"]
        s.index = s.index.date
        pieces.append(s)

    if not pieces:
        return pd.Series(dtype=float, name=asset)

    combined = pieces[0]
    for nxt in pieces[1:]:
        # Successor symbol wins on overlapping dates (it's the live market).
        combined = pd.concat([combined[~combined.index.isin(nxt.index)], nxt])
    combined = combined.sort_index()
    combined.name = asset
    combined.index.name = "date"
    return combined


def get_pair_prices(asset_a: str, asset_b: str, start: str = None) -> pd.DataFrame:
    """Inner-joined daily close price DataFrame for two assets, columns
    named asset_a and asset_b."""
    sa = get_symbol_series(asset_a, start)
    sb = get_symbol_series(asset_b, start)
    df = pd.concat([sa, sb], axis=1, join="inner")
    df.columns = [asset_a, asset_b]
    return df.sort_index()
