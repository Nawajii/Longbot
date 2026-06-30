"""Scenario sweep: run the backtest across many pairs and named regime windows,
then aggregate ALL trades into one honest summary.

Why named windows? A long-only pullback strategy will *look* great in a bull
market and do nothing in a bear. Splitting results by regime (BULL / BEAR /
CHOP) is the only way to tell a real edge from "it was just a bull market".

Usage (run where Binance is reachable — klines are free, no key needed):

    python run_scenarios.py
    python run_scenarios.py --timeframe 4h --entry-valid-bars 6
    python run_scenarios.py --pairs BTCUSDT ETHUSDT SOLUSDT

It prints each per-pair report as it goes, then a COMBINED SUMMARY: total
trades, overall win rate, overall expectancy (R), and a per-scenario breakdown.

Note on warmup: each window is fetched with ~300 extra bars of lead-in so the
EMA200 and other slow indicators are warm by the window's start date; trades
are still overwhelmingly inside the requested window.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import backtest as bt
from longbot import scoring_engine as se
from run_real_backtest import _MS, fetch_binance_klines_range

# (start, end) inclusive-ish date windows the user asked for.
DEFAULT_SCENARIOS: dict[str, tuple[str, str]] = {
    "BULL": ("2023-11-01", "2024-03-01"),
    "BEAR": ("2022-04-01", "2022-08-01"),
    "CHOP": ("2023-05-01", "2023-09-01"),
}

DEFAULT_PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
    "AVAXUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT", "ATOMUSDT", "UNIUSDT",
    "ETCUSDT", "XLMUSDT", "TRXUSDT", "NEARUSDT", "AAVEUSDT", "ALGOUSDT",
]

WARMUP_BARS = 300          # lead-in so slow indicators are warm at window start
MIN_USABLE_BARS = 260      # below this we skip the pair/window as insufficient


@dataclass
class RunRecord:
    scenario: str
    pair: str
    trades: list[bt.Trade] = field(default_factory=list)
    total_return_pct: float = float("nan")
    buy_hold_pct: float = float("nan")
    skipped: str | None = None     # reason, if this run was skipped


def _buffered_start(start: str, timeframe: str, bars: int = WARMUP_BARS) -> str:
    ms = bars * _MS[timeframe]
    return (pd.Timestamp(start, tz="UTC") - pd.Timedelta(milliseconds=ms)).strftime("%Y-%m-%d")


def _fetch_window(symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    """Fetch a window plus warmup lead-in. Raises on no data."""
    return fetch_binance_klines_range(symbol, timeframe, _buffered_start(start, timeframe), end)


def run_sweep(
    pairs: list[str],
    timeframe: str,
    scenarios: dict[str, tuple[str, str]],
    capital: float = 1000.0,
    entry_valid_bars: int = 3,
    regime_symbol: str = "BTCUSDT",
) -> list[RunRecord]:
    params = bt.BacktestParams(
        starting_capital=capital, timeframe=timeframe, entry_valid_bars=entry_valid_bars
    )
    records: list[RunRecord] = []

    for scen, (start, end) in scenarios.items():
        # fetch the regime series (BTC) once per window
        try:
            regime_df = _fetch_window(regime_symbol, timeframe, start, end)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {scen}: could not fetch regime {regime_symbol} ({exc}); "
                  f"running without regime gate.")
            regime_df = None

        for pair in pairs:
            rec = RunRecord(scenario=scen, pair=pair)
            try:
                df = _fetch_window(pair, timeframe, start, end)
            except Exception as exc:  # noqa: BLE001
                rec.skipped = f"fetch failed ({exc})"
                records.append(rec)
                print(f"[skip] {scen} {pair}: {rec.skipped}")
                continue

            if len(df) < MIN_USABLE_BARS:
                rec.skipped = f"insufficient data ({len(df)} bars < {MIN_USABLE_BARS})"
                records.append(rec)
                print(f"[skip] {scen} {pair}: {rec.skipped}")
                continue

            res = bt.run_backtest(df, params, regime_df=regime_df)
            peak = bt.run_backtest(
                df, params, entry_verdicts=(se.Verdict.PEAK_EXTENDED,), regime_df=regime_df
            )
            rec.trades = res.trades
            rec.total_return_pct = res.total_return_pct
            rec.buy_hold_pct = bt.buy_and_hold_return_pct(df, params)
            records.append(rec)

            span = f"{df.index[0].date()} -> {df.index[-1].date()}"
            print(f"\n########## {scen}  {pair}  ({timeframe}, {len(df)} bars, {span}) ##########")
            print(bt.format_report(res, df, peak_baseline=peak))

    return records


# --------------------------------------------------------------------------- #
# aggregation (pure function so it is unit-testable without network)
# --------------------------------------------------------------------------- #
def _agg(trades: list[bt.Trade]) -> dict:
    if not trades:
        return {"n": 0, "win_rate": float("nan"), "expectancy_r": float("nan")}
    rs = [t.r_multiple for t in trades if not np.isnan(t.r_multiple)]
    wins = sum(1 for t in trades if t.pnl > 0)
    return {
        "n": len(trades),
        "win_rate": wins / len(trades),
        "expectancy_r": float(np.mean(rs)) if rs else float("nan"),
    }


def summarize(records: list[RunRecord], scenarios: dict[str, tuple[str, str]]) -> str:
    executed = [r for r in records if r.skipped is None]
    skipped = [r for r in records if r.skipped is not None]
    all_trades = [t for r in executed for t in r.trades]

    def _pct(x: float) -> str:
        return "n/a" if (x is None or np.isnan(x)) else f"{x * 100:.1f}%"

    overall = _agg(all_trades)
    lines = [
        "=" * 74,
        " COMBINED SUMMARY — all trades aggregated across every pair & window",
        "=" * 74,
        f" runs executed: {len(executed)}    skipped (insufficient data): {len(skipped)}",
        f" TOTAL trades : {overall['n']}",
        f" overall win rate  : {_pct(overall['win_rate'])}",
        f" overall expectancy: {overall['expectancy_r']:.3f} R per trade",
        "-" * 74,
        " PER-SCENARIO BREAKDOWN:",
        f"   {'scenario':<8}{'window':<26}{'trades':>8}{'win rate':>12}{'expectancy':>14}",
    ]
    for scen, (start, end) in scenarios.items():
        s_trades = [t for r in executed if r.scenario == scen for t in r.trades]
        a = _agg(s_trades)
        win = _pct(a["win_rate"])
        exp = "n/a" if np.isnan(a["expectancy_r"]) else f"{a['expectancy_r']:.3f} R"
        lines.append(
            f"   {scen:<8}{start+'..'+end:<26}{a['n']:>8}{win:>12}{exp:>14}"
        )
    lines += ["-" * 74]

    # honesty check: how often did the strategy beat buy & hold, by scenario?
    for scen in scenarios:
        srecs = [r for r in executed if r.scenario == scen]
        if not srecs:
            continue
        beat = sum(1 for r in srecs if r.total_return_pct > r.buy_hold_pct)
        avg_edge = np.nanmean([r.total_return_pct - r.buy_hold_pct for r in srecs])
        lines.append(
            f"   {scen}: beat buy&hold in {beat}/{len(srecs)} pairs "
            f"(avg edge {avg_edge:+.2f} pts) — note: in a down window, 'beating' "
            f"often just means it sat in cash."
        )
    lines += [
        "=" * 74,
        " READ THIS: positive expectancy in BULL but flat/negative in BEAR & CHOP",
        " is the classic 'no real edge, just long a bull market' signature. A real",
        " edge shows positive expectancy that survives CHOP and limits damage in BEAR,",
        " net of the fees & slippage in each report's ASSUMPTIONS block.",
        "=" * 74,
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="*", default=DEFAULT_PAIRS)
    ap.add_argument("--timeframe", default="4h", choices=list(_MS))
    ap.add_argument("--capital", type=float, default=1000.0)
    ap.add_argument("--entry-valid-bars", type=int, default=3)
    ap.add_argument("--regime-symbol", default="BTCUSDT")
    args = ap.parse_args()

    records = run_sweep(
        args.pairs, args.timeframe, DEFAULT_SCENARIOS,
        capital=args.capital, entry_valid_bars=args.entry_valid_bars,
        regime_symbol=args.regime_symbol,
    )
    print("\n" + summarize(records, DEFAULT_SCENARIOS))


if __name__ == "__main__":
    main()
