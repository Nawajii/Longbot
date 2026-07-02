"""DEEP-HISTORY exit grid (2019 -> now) — pre-registered, binding.

Same engine, same 5 exit variants as run_exit_grid.py (TRAIL_2 control, TRAIL_4,
TRAIL_6, TIME_12, and now TARGET_2R = SLC's fixed 2R take-profit), on IDENTICAL
AT_NODE located entries with an IDENTICAL 2xATR node-anchored initial stop. The
ONLY difference between variants is the exit.

The change here is DATA, not strategy: continuous 4h history from 2019-01-01 to
now, so every regime bucket finally has a large sample instead of the ~2.5-month
windows that kept returning n<30. Trades are bucketed by BTC's actual regime at
entry (BULL = BTC above a rising EMA200, BEAR = below a falling EMA200, else
CHOP), so we can see whether any edge survives OUTSIDE bull markets.

PRE-REGISTERED SUCCESS BAR (binding, fixed before results):
    A variant is a real finding ONLY if ALL hold on the full deep-history sweep:
      * aggregate expectancy >= +0.15 R (net of fees + slippage), AND
      * NOT positive-only-in-BULL (must hold up across the continuous history),
      * AND >= 200 trades aggregate.
    Below the bar = NOT SUPPORTED. No sweeping of ATR/VP/R/thresholds, no "one
    more value". One pre-registered grid, one run, binding verdict.

Data is cached under data/ so re-runs don't re-download. Run where Binance is
reachable:  python run_deep_grid.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import backtest as bt
from longbot import indicators as ind
from longbot import scoring_engine as se
from longbot import volume_profile as vp
from run_exit_grid import VARIANTS, _metrics, _p, _r
from run_real_backtest import fetch_or_cache
from run_scenarios import DEFAULT_PAIRS

MIN_CONCLUSIVE = 30
BAR_EXPECTANCY = 0.15
BAR_MIN_TRADES = 200          # deep history should easily clear this
REGIMES = ("BULL", "BEAR", "CHOP")


def btc_regime_labels(btc_df: pd.DataFrame) -> pd.Series:
    """Per-bar BTC regime: BULL / BEAR / CHOP (no lookahead — trailing EMA+slope)."""
    close = btc_df["close"]
    ema = ind.ema(close, 200)
    slope = ind.slope(ema, 20)
    label = pd.Series("CHOP", index=btc_df.index)
    label[(close > ema) & (slope > 0)] = "BULL"
    label[(close < ema) & (slope < 0)] = "BEAR"
    return label


def _loc_features_cached(symbol: str, timeframe: str, df: pd.DataFrame,
                         cache_dir: str, refresh: bool) -> pd.DataFrame:
    """Volume-profile location features, cached (deep VP is the slow part)."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{symbol}_{timeframe}_loc.csv")
    if os.path.exists(path) and not refresh:
        loc = pd.read_csv(path, index_col=0, parse_dates=True)
        if len(loc) == len(df) and loc.index.equals(df.index):
            for c in ("vp_at_node", "vp_in_void", "vp_in_value_area"):
                if c in loc:
                    # robust across dtypes: "False"/"0"/False -> False (astype(bool)
                    # on the string "False" would wrongly be True).
                    loc[c] = loc[c].astype(str).str.lower().isin(("true", "1"))
            return loc
    loc = vp.compute_location_features(df)
    loc.to_csv(path)
    return loc


def run_deep(pairs, timeframe, start, end, capital, entry_valid_bars,
             regime_symbol, cache_dir, refresh):
    located_sp = se.ScoringParams(use_location=True)
    trades = {v: [] for v in VARIANTS}
    by_regime = {v: {r: [] for r in REGIMES} for v in VARIANTS}
    worst_dd = {v: 0.0 for v in VARIANTS}
    bh_returns, coverage = [], []
    executed = skipped = 0

    print(f"[data] fetching regime {regime_symbol} {timeframe} {start}->now (cached in {cache_dir}/)")
    regime_df = fetch_or_cache(regime_symbol, timeframe, start, end, cache_dir, refresh)
    regime_label = btc_regime_labels(regime_df)

    for pair in pairs:
        try:
            df = fetch_or_cache(pair, timeframe, start, end, cache_dir, refresh)
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            print(f"[skip] {pair}: fetch failed ({exc})")
            continue
        if len(df) < 300:
            skipped += 1
            print(f"[skip] {pair}: only {len(df)} bars total")
            continue
        executed += 1
        first = df.index[0].date()
        coverage.append((pair, first, len(df)))
        # how much of 2019->now this pair actually covered (listing date)
        missing = "" if first.year <= 2019 else f"  (listed {first}, earlier history N/A)"
        print(f"[run] {pair:<9} {len(df):>6} bars  from {first}{missing}")

        loc = _loc_features_cached(pair, timeframe, df, cache_dir, refresh)
        labels_on_pair = regime_label.reindex(df.index, method="ffill")

        bh_returns.append(bt.buy_and_hold_return_pct(df, bt.BacktestParams()))
        for name, over in VARIANTS.items():
            p = bt.BacktestParams(starting_capital=capital, timeframe=timeframe,
                                  entry_valid_bars=entry_valid_bars, **over)
            res = bt.run_backtest(df, p, scoring_params=located_sp,
                                  regime_df=regime_df, location_features=loc)
            trades[name] += res.trades
            for t in res.trades:
                lab = labels_on_pair.get(t.entry_time, "CHOP")
                by_regime[name].setdefault(lab, []).append(t)
            if not np.isnan(res.max_drawdown_pct):
                worst_dd[name] = min(worst_dd[name], res.max_drawdown_pct)

    return dict(trades=trades, by_regime=by_regime, worst_dd=worst_dd,
                executed=executed, skipped=skipped, coverage=coverage,
                bh_returns=bh_returns)


def _verdict(name, R):
    m = _metrics(R["trades"][name])
    per = {r: _metrics(R["by_regime"][name][r])["exp"] for r in REGIMES}
    bull, bear, chop = per["BULL"], per["BEAR"], per["CHOP"]
    if m["n"] < BAR_MIN_TRADES:
        return False, f"only {m['n']} trades (< {BAR_MIN_TRADES})"
    if not (m["exp"] >= BAR_EXPECTANCY):
        return False, f"expectancy {m['exp']:+.3f} R (< +{BAR_EXPECTANCY})"
    not_bull_only = (
        (not np.isnan(bear) and bear >= 0) or (not np.isnan(chop) and chop >= 0)
        or (not np.isnan(bull) and bull >= BAR_EXPECTANCY
            and (np.isnan(bear) or bear > -0.05) and (np.isnan(chop) or chop > -0.05))
    )
    if not not_bull_only:
        return False, "positive-only-in-BULL (fails regime-robustness)"
    return True, f"expectancy {m['exp']:+.3f} R on {m['n']} trades, robust across regimes"


def format_deep(R) -> str:
    L = ["=" * 96,
         " DEEP-HISTORY EXIT GRID (2019 -> now, 4h) — pre-registered, binding",
         "-" * 96,
         " SUCCESS BAR (all required): expectancy >= +0.15 R  AND  >= 200 trades  AND  not BULL-only.",
         "=" * 96,
         f" pairs run: {R['executed']}   skipped: {R['skipped']}"]
    short = [(p, d) for (p, d, _n) in R["coverage"] if d.year > 2019]
    if short:
        L.append(" partial-history pairs (listed after 2019, earlier data genuinely N/A): "
                 + ", ".join(f"{p}@{d}" for p, d in short))
    bh = [x for x in R["bh_returns"] if not np.isnan(x)]
    if bh:
        L.append(f" baseline buy&hold over the window: median {np.median(bh):+.0f}% across pairs "
                 f"(long-only in-and-out will trail this in a secular bull; that is NOT a strike "
                 f"against it, and 'beating' it in a down stretch just means sitting in cash).")
    L += ["-" * 96,
          f" {'variant':<10}{'trades':>7}{'win%':>7}{'avgWinR':>9}{'avgLossR':>9}"
          f"{'exp R':>8}{'PF':>6}{'bars':>7}{'givebk':>8}{'wDD%':>7}"]
    for name in VARIANTS:
        m = _metrics(R["trades"][name])
        pf = "inf" if m["profit_factor"] == float("inf") else (
            "n/a" if np.isnan(m["profit_factor"]) else f"{m['profit_factor']:.2f}")
        gb = "n/a" if np.isnan(m["giveback"]) else f"{m['giveback']:+.2f}"
        L.append(f" {name:<10}{m['n']:>7}{_p(m['win']):>7}{_r(m['avg_win']):>9}{_r(m['avg_loss']):>9}"
                 f"{_r(m['exp']):>8}{pf:>6}{m['bars']:>7.1f}{gb:>8}{R['worst_dd'][name]:>7.1f}")
    L += ["-" * 96, " EXPECTANCY BY BTC REGIME AT ENTRY (the honesty check):",
          f"   {'variant':<10}" + "".join(f"{r:>20}" for r in REGIMES)]
    for name in VARIANTS:
        cells = []
        for r in REGIMES:
            m = _metrics(R["by_regime"][name][r])
            flag = "" if m["n"] >= MIN_CONCLUSIVE else "*"
            cells.append(f"{_r(m['exp'])} (n={m['n']}){flag}")
        L.append(f"   {name:<10}" + "".join(f"{c:>20}" for c in cells))
    L.append("   * = < 30 trades in cell: NOT CONCLUSIVE")

    L += ["-" * 96, " BINDING VERDICT (against the pre-registered bar):"]
    winners = []
    for name in VARIANTS:
        ok, why = _verdict(name, R)
        L.append(f"   {name}: {'PASS' if ok else 'fail'} — {why}")
        if ok:
            winners.append((name, _metrics(R["trades"][name])["exp"]))
    L += ["=" * 96]
    if winners:
        best = max(winners, key=lambda x: x[1])
        L.append(f" SUPPORTED: '{best[0]}' clears the bar (expectancy {best[1]:+.3f} R) on deep history.")
    else:
        L.append(" NOT SUPPORTED: No exit variant cleared the bar on deep history. Across blind,")
        L.append(" located, exit-grid and now SLC's 2R target on 2019->now data, the pullback/SLC")
        L.append(" engine shows no regime-robust edge net of costs. The engine is retired.")
    L += ["=" * 96]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="*", default=DEFAULT_PAIRS)
    ap.add_argument("--timeframe", default="4h")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--capital", type=float, default=1000.0)
    ap.add_argument("--entry-valid-bars", type=int, default=3)
    ap.add_argument("--regime-symbol", default="BTCUSDT")
    ap.add_argument("--cache-dir", default="data")
    ap.add_argument("--refresh", action="store_true", help="force re-download / recompute cache")
    args = ap.parse_args()

    print(__doc__.split("Data is cached")[0])
    R = run_deep(args.pairs, args.timeframe, args.start, args.end, args.capital,
                 args.entry_valid_bars, args.regime_symbol, args.cache_dir, args.refresh)
    print("\n" + format_deep(R))


if __name__ == "__main__":
    main()
