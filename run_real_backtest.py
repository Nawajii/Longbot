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


def load_csv(path: str) -> pd.DataFrame:
    """Load OHLCV from a CSV with columns time,open,high,low,close,volume."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "time" not in df.columns:
        df = df.rename(columns={df.columns[0]: "time"})
    # accept ms epoch or ISO strings
    if np.issubdtype(df["time"].dtype, np.number):
        df["time"] = pd.to_datetime(df["time"], unit="ms")
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
            regime_df: pd.DataFrame | None) -> None:
    params = bt.BacktestParams(starting_capital=capital, timeframe=timeframe)
    result = bt.run_backtest(df, params, regime_df=regime_df)
    # buy-the-peak baseline: same engine, enter on the trap verdict instead
    peak = bt.run_backtest(
        df, params, entry_verdicts=(se.Verdict.PEAK_EXTENDED,), regime_df=regime_df
    )
    print(bt.format_report(result, df, peak_baseline=peak))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--timeframe", default="1h", choices=list(_MS))
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--capital", type=float, default=1000.0)
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
        regime_df = fetch_binance_klines(args.regime_symbol, args.timeframe, args.limit)

    if args.synthetic:
        df = make_synthetic()
        run_one(df, "1h-synthetic", args.capital, regime_df)
        return

    if args.csv:
        df = load_csv(args.csv)
        run_one(df, args.timeframe, args.capital, regime_df)
        return

    if args.compare_timeframes:
        for tf in args.compare_timeframes:
            print(f"\n########## {args.symbol} @ {tf} ##########")
            df = fetch_binance_klines(args.symbol, tf, args.limit)
            run_one(df, tf, args.capital, regime_df)
        return

    df = fetch_binance_klines(args.symbol, args.timeframe, args.limit)
    run_one(df, args.timeframe, args.capital, regime_df)


if __name__ == "__main__":
    main()
