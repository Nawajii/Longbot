"""Run the backtest on real Binance history (or a CSV, or synthetic data).

Market data (klines) is FREE and PUBLIC — no API key needed. Three sources:

    # real Binance klines (needs outbound network access to api.binance.com)
    python run_real_backtest.py --symbol BTCUSDT --timeframe 1h --limit 1000

    # a CSV you downloaded yourself (columns: time,open,high,low,close,volume)
    python run_real_backtest.py --csv data/BTCUSDT_1h.csv --timeframe 1h

    # synthetic data, just to prove the plumbing runs offline (NOT an edge test)
    python run_real_backtest.py --synthetic

Optionally pass --regime-csv / --regime-symbol (BTC) to enable the regime gate,
and --compare-timeframes to run several timeframes head-to-head.

IMPORTANT: a backtest on one pair / one period is a hypothesis test, not proof.
Run several pairs and timeframes, and read the ASSUMPTIONS block every time.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

import numpy as np
import pandas as pd

import backtest as bt
from longbot import scoring_engine as se

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000,
       "2h": 7_200_000, "4h": 14_400_000, "1d": 86_400_000}


# --------------------------------------------------------------------------- #
# data sources
# --------------------------------------------------------------------------- #
def _http_get_json(url: str):
    """GET a URL and parse JSON, preferring `requests` (uses certifi, so it
    avoids the classic macOS 'SSL: CERTIFICATE_VERIFY_FAILED' with plain
    urllib) and falling back to the stdlib if requests isn't installed.
    """
    try:
        import requests  # type: ignore

        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except ImportError:
        import ssl

        ctx = ssl.create_default_context()
        try:  # use certifi's CA bundle if available — most reliable on macOS
            import certifi  # type: ignore

            ctx.load_verify_locations(certifi.where())
        except ImportError:
            pass
        with urllib.request.urlopen(url, timeout=30, context=ctx) as resp:  # noqa: S310
            return json.loads(resp.read().decode())


def fetch_binance_klines(symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
    """Fetch up to `limit` klines from Binance's public REST endpoint.

    No API key needed — klines are free and public. Uses a plain HTTPS GET
    (via `requests`/`urllib`), so it does NOT require ccxt. `limit` is capped at
    1000 by the endpoint; for longer history, use CSVs from data.binance.vision.
    """
    url = f"{BINANCE_KLINES_URL}?symbol={symbol}&interval={interval}&limit={min(limit, 1000)}"
    raw = _http_get_json(url)
    if isinstance(raw, dict) and raw.get("code"):  # Binance error payload
        raise RuntimeError(f"Binance API error for {symbol}: {raw}")
    df = pd.DataFrame(raw, columns=[
        "time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbb", "tbq", "ignore",
    ])
    df = df[["time", "open", "high", "low", "close", "volume"]].astype(
        {"open": float, "high": float, "low": float, "close": float, "volume": float}
    )
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df.set_index("time").sort_index()


def fetch_binance_klines_range(
    symbol: str, interval: str, start: str, end: str | None = None
) -> pd.DataFrame:
    """Page Binance for ALL klines between `start` and `end` (dates, e.g.
    "2020-01-01"). The endpoint returns 1000 bars max per call, so this loops
    forward until it reaches the end, assembling years of history. Free, no key.
    """
    import time as _time

    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int((pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.utcnow()).timestamp() * 1000)
    step = _MS[interval]
    frames: list[pd.DataFrame] = []
    cursor = start_ms
    calls = 0
    while cursor < end_ms:
        url = (f"{BINANCE_KLINES_URL}?symbol={symbol}&interval={interval}"
               f"&startTime={cursor}&endTime={end_ms}&limit=1000")
        raw = _http_get_json(url)
        if isinstance(raw, dict) and raw.get("code"):
            raise RuntimeError(f"Binance API error for {symbol}: {raw}")
        if not raw:
            break
        frames.append(pd.DataFrame(raw, columns=[
            "time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "tbb", "tbq", "ignore"]))
        last_open = raw[-1][0]
        cursor = last_open + step          # advance past the last bar we got
        calls += 1
        if calls % 10 == 0:
            print(f"[info] {symbol} {interval}: fetched ~{calls * 1000} bars...", file=sys.stderr)
        if len(raw) < 1000:                # reached the most recent data
            break
        _time.sleep(0.25)                  # be polite to the public endpoint
    if not frames:
        raise RuntimeError(f"no data returned for {symbol} {interval} from {start}")
    df = pd.concat(frames, ignore_index=True)
    df = df[["time", "open", "high", "low", "close", "volume"]].astype(
        {"open": float, "high": float, "low": float, "close": float, "volume": float})
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df.drop_duplicates("time").set_index("time").sort_index()


def fetch_or_cache(
    symbol: str, interval: str, start: str, end: str | None = None,
    cache_dir: str = "data", refresh: bool = False,
) -> pd.DataFrame:
    """Fetch a deep history range, caching to ``cache_dir/{symbol}_{interval}.csv``.

    First run downloads and caches; later runs load the CSV so the experiment is
    reproducible and does not re-hit the network. Pass ``refresh=True`` to force
    a re-download. Returns the cached data sliced to [start, end].
    """
    import os

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{symbol}_{interval}.csv")
    if os.path.exists(path) and not refresh:
        df = load_csv(path)
    else:
        df = fetch_binance_klines_range(symbol, interval, start, end)
        df.to_csv(path)
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) if end else df.index.max()
    return df[(df.index >= lo) & (df.index <= hi)]


def load_csv(path: str) -> pd.DataFrame:
    """Load OHLCV from a CSV. Handles three common shapes:

    1. our --save-csv output (header row, datetime in the first column);
    2. any CSV with a time/open/high/low/close/volume header;
    3. Binance's *headerless* monthly dumps from data.binance.vision
       (12 columns: open_time(ms), open, high, low, close, volume, ...).
    """
    with open(path) as f:
        first_line = f.readline().lower()
    has_header = "open" in first_line and "close" in first_line

    if has_header:
        df = pd.read_csv(path)
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "time" not in df.columns:
            df = df.rename(columns={df.columns[0]: "time"})
    else:
        # headerless Binance kline dump — first 6 cols are what we need
        df = pd.read_csv(path, header=None)
        cols = ["time", "open", "high", "low", "close", "volume"]
        df = df.iloc[:, : len(cols)]
        df.columns = cols

    if pd.api.types.is_numeric_dtype(df["time"]):
        # epoch could be ms or microseconds depending on the dump; detect by magnitude
        unit = "us" if df["time"].iloc[0] > 1e15 else "ms"
        df["time"] = pd.to_datetime(df["time"], unit=unit)
    else:
        df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")[bt.ind.OHLCV_COLUMNS].sort_index()


def make_synthetic(n: int = 1500, seed: int = 7) -> pd.DataFrame:
    """A trending-with-pullbacks random walk. ONLY for proving the code runs.

    This has no predictive content and tells you NOTHING about real edge —
    it exists so you can watch the whole pipeline execute without network.
    """
    rng = np.random.default_rng(seed)
    drift = 0.0006                       # gentle uptrend
    shocks = rng.normal(drift, 0.015, n)
    # inject periodic pullbacks so the pullback hunter has something to chew on
    for k in range(0, n, 120):
        shocks[k : k + 8] -= 0.02
    price = 100.0 * np.exp(np.cumsum(shocks))
    high = price * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = price * (1 - np.abs(rng.normal(0, 0.006, n)))
    open_ = np.concatenate([[price[0]], price[:-1]])
    vol = rng.lognormal(10, 0.4, n)
    idx = pd.date_range("2023-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": price, "volume": vol}, index=idx
    )


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_one(df: pd.DataFrame, timeframe: str, capital: float,
            regime_df: pd.DataFrame | None, entry_valid_bars: int = 3) -> None:
    params = bt.BacktestParams(
        starting_capital=capital, timeframe=timeframe, entry_valid_bars=entry_valid_bars,
    )
    result = bt.run_backtest(df, params, regime_df=regime_df)
    # buy-the-peak baseline: same engine, enter on the trap verdict instead
    peak = bt.run_backtest(
        df, params, entry_verdicts=(se.Verdict.PEAK_EXTENDED,), regime_df=regime_df
    )
    span = f"{df.index[0].date()} -> {df.index[-1].date()}"
    print(f"[data] {len(df)} bars, {span}")
    print(bt.format_report(result, df, peak_baseline=peak))


def _get_data(symbol: str, timeframe: str, args) -> pd.DataFrame:
    """Resolve a symbol+timeframe to OHLCV using whichever source the flags imply."""
    if args.start:
        return fetch_binance_klines_range(symbol, timeframe, args.start, args.end)
    return fetch_binance_klines(symbol, timeframe, args.limit)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--timeframe", default="1h", choices=list(_MS))
    ap.add_argument("--limit", type=int, default=1000,
                    help="bars to fetch (max 1000) when --start is NOT given")
    ap.add_argument("--start", help="fetch ALL history from this date, e.g. 2020-01-01 "
                                    "(pages past the 1000-bar cap — use this for a real test)")
    ap.add_argument("--end", help="end date for --start range (default: now)")
    ap.add_argument("--capital", type=float, default=1000.0)
    ap.add_argument("--entry-valid-bars", type=int, default=3,
                    help="how many bars the buy-stop waits before it is a MISS (tunable)")
    ap.add_argument("--save-csv", help="also save the fetched OHLCV to this CSV for re-use")
    ap.add_argument("--csv", help="load OHLCV from this CSV instead of fetching")
    ap.add_argument("--regime-csv", help="BTC OHLCV CSV to enable the regime gate")
    ap.add_argument("--regime-symbol", help="fetch this symbol as the regime gate (e.g. BTCUSDT)")
    ap.add_argument("--synthetic", action="store_true",
                    help="use synthetic data (offline smoke test only)")
    ap.add_argument("--compare-timeframes", nargs="*", default=None,
                    help="fetch & compare these timeframes head-to-head (real data only)")
    args = ap.parse_args()

    regime_df = None
    if args.regime_csv:
        regime_df = load_csv(args.regime_csv)
    elif args.regime_symbol:
        regime_df = _get_data(args.regime_symbol, args.timeframe, args)

    if args.synthetic:
        run_one(make_synthetic(), "1h-synthetic", args.capital, regime_df, args.entry_valid_bars)
        return

    if args.csv:
        run_one(load_csv(args.csv), args.timeframe, args.capital, regime_df, args.entry_valid_bars)
        return

    if args.compare_timeframes:
        for tf in args.compare_timeframes:
            print(f"\n########## {args.symbol} @ {tf} ##########")
            run_one(_get_data(args.symbol, tf, args), tf, args.capital,
                    regime_df, args.entry_valid_bars)
        return

    df = _get_data(args.symbol, args.timeframe, args)
    if args.save_csv:
        df.to_csv(args.save_csv)
        print(f"[info] saved {len(df)} bars to {args.save_csv}")
    run_one(df, args.timeframe, args.capital, regime_df, args.entry_valid_bars)


if __name__ == "__main__":
    main()
