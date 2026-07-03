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
import time

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

# The regime LABEL must be INDEPENDENT of the entry gate, or it is tautological.
# The entry gate uses EMA200 on the trading timeframe (~33 days on 4h); a trade
# only fires when BTC is above a *rising* EMA200 — which is the gate's own BULL
# condition, so labelling with the same measure forces every trade to "BULL".
# The macro cycle uses a far longer trend (~200 DAYS), so a trade taken during a
# 2022 bear-market rally (fine gate briefly open) is correctly tagged BEAR.
MACRO_EMA_BARS = 1200         # ~200 days on 4h — the macro bull/bear trend line
MACRO_SLOPE_BARS = 180        # ~30 days — slope of that macro trend


def btc_regime_labels(btc_df: pd.DataFrame,
                      ema_bars: int = MACRO_EMA_BARS,
                      slope_bars: int = MACRO_SLOPE_BARS) -> pd.Series:
    """Per-bar BTC MACRO regime: BULL / BEAR / CHOP (no lookahead — trailing EMA+slope).

    Deliberately uses a long (~200-day) trend, INDEPENDENT of the strategy's
    entry gate, so bear-cycle trades are counted as BEAR instead of collapsing
    into a tautological all-BULL label.
    """
    close = btc_df["close"]
    ema = ind.ema(close, ema_bars)
    slope = ind.slope(ema, slope_bars)
    label = pd.Series("CHOP", index=btc_df.index)
    label[(close > ema) & (slope > 0)] = "BULL"
    label[(close < ema) & (slope < 0)] = "BEAR"
    return label


def regime_bar_proof(regime_label: pd.Series) -> tuple[str, bool]:
    """Print regime bar counts + a 2022=bear sanity check. Returns (text, sane)."""
    total = len(regime_label)
    counts = {r: int((regime_label == r).sum()) for r in REGIMES}
    lines = [" REGIME BAR-COUNT PROOF (full BTC series, macro ~200-day trend):"]
    for r in REGIMES:
        lines.append(f"   {r:<5}: {counts[r]:>6} bars ({100*counts[r]/total:.1f}%)")
    idx = regime_label.index
    def _frac(lo, hi, reg):
        m = regime_label[(idx >= lo) & (idx < hi)]
        return float((m == reg).mean()) * 100 if len(m) else float("nan")
    b22 = _frac("2022-04-01", "2022-12-01", "BEAR")
    u21 = _frac("2021-01-01", "2021-12-01", "BULL")
    u23 = _frac("2023-06-01", "2024-06-01", "BULL")
    lines += [
        f"   sanity: 2022 Apr-Nov = {b22:.0f}% BEAR (expect predominantly BEAR)",
        f"           2021        = {u21:.0f}% BULL (expect predominantly BULL)",
        f"           2023H2-2024 = {u23:.0f}% BULL (expect predominantly BULL)",
    ]
    # sane if all three regimes are represented and 2022 reads mostly bear
    sane = all(counts[r] > total * 0.03 for r in REGIMES) and (b22 >= 50)
    if not sane:
        lines.append("   !! REGIME PROOF NOT SANE — do not trust the per-regime verdict below.")
    return "\n".join(lines), sane


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
             regime_symbol, cache_dir, refresh, variant_names=None, entry_gate_fn=None):
    """Deep sweep. Optionally restrict to `variant_names` and AND an extra
    per-pair entry gate `entry_gate_fn(df_4h) -> bool Series` (e.g. the
    multi-scale daily gate). Caller guarantees the gate is no-lookahead.
    """
    located_sp = se.ScoringParams(use_location=True)
    run_variants = {k: VARIANTS[k] for k in variant_names} if variant_names else dict(VARIANTS)
    trades = {v: [] for v in run_variants}
    by_regime = {v: {r: [] for r in REGIMES} for v in run_variants}
    worst_dd = {v: 0.0 for v in run_variants}
    bh_returns, coverage = [], []
    executed = skipped = 0

    print(f"[data] fetching regime {regime_symbol} {timeframe} {start}->now (cached in {cache_dir}/)")
    regime_df = fetch_or_cache(regime_symbol, timeframe, start, end, cache_dir, refresh)
    regime_label = btc_regime_labels(regime_df)
    proof_text, proof_sane = regime_bar_proof(regime_label)
    print("\n" + proof_text + "\n")

    network_skipped = []
    for pair in pairs:
        # per-pair retry pass on top of the per-request retries in the fetcher;
        # cached (already-downloaded) pairs load instantly and never re-hit net.
        df, err = None, None
        for attempt in range(3):
            try:
                df = fetch_or_cache(pair, timeframe, start, end, cache_dir, refresh)
                break
            except Exception as exc:  # noqa: BLE001
                err = exc
                if attempt < 2:
                    time.sleep(3 * (2 ** attempt))
        if df is None:
            skipped += 1
            network_skipped.append(pair)
            print(f"[skip] {pair}: fetch failed after retries ({err})")
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
        gate = entry_gate_fn(df) if entry_gate_fn is not None else None

        bh_returns.append(bt.buy_and_hold_return_pct(df, bt.BacktestParams()))
        for name, over in run_variants.items():
            p = bt.BacktestParams(starting_capital=capital, timeframe=timeframe,
                                  entry_valid_bars=entry_valid_bars, **over)
            res = bt.run_backtest(df, p, scoring_params=located_sp,
                                  regime_df=regime_df, location_features=loc,
                                  entry_gate=gate)
            trades[name] += res.trades
            for t in res.trades:
                lab = labels_on_pair.get(t.entry_time, "CHOP")
                by_regime[name].setdefault(lab, []).append(t)
            if not np.isnan(res.max_drawdown_pct):
                worst_dd[name] = min(worst_dd[name], res.max_drawdown_pct)

    return dict(trades=trades, by_regime=by_regime, worst_dd=worst_dd,
                executed=executed, skipped=skipped, coverage=coverage,
                bh_returns=bh_returns, proof_text=proof_text, proof_sane=proof_sane,
                network_skipped=network_skipped, requested=len(pairs),
                regime_label=regime_label)


def _verdict(name, R):
    """Return (status, reason) where status is PASS | FAIL | UNPROVEN.

    The regime-robustness clause requires a CONCLUSIVE (n>=30) non-bull cell that
    is >= 0. If BEAR and CHOP are both too thin to judge, robustness is UNPROVEN
    (NOT a free pass) — because the strategy's own gate confines it to bull.
    """
    m = _metrics(R["trades"][name])
    stats = {r: _metrics(R["by_regime"][name][r]) for r in REGIMES}
    if m["n"] < BAR_MIN_TRADES:
        return "FAIL", f"only {m['n']} trades (< {BAR_MIN_TRADES})"
    if not (m["exp"] >= BAR_EXPECTANCY):
        return "FAIL", f"expectancy {m['exp']:+.3f} R (< +{BAR_EXPECTANCY})"

    nonbull = [(r, stats[r]) for r in ("BEAR", "CHOP")]
    conclusive = [(r, s) for r, s in nonbull if s["n"] >= MIN_CONCLUSIVE]
    if not conclusive:
        return "UNPROVEN", (f"BEAR/CHOP samples too thin "
                            f"(BEAR n={stats['BEAR']['n']}, CHOP n={stats['CHOP']['n']}; "
                            f"< {MIN_CONCLUSIVE}) — the gate confines trading to bull, so "
                            f"regime-robustness is UNPROVEN")
    positive = [r for r, s in conclusive if s["exp"] >= 0]
    if positive:
        return "PASS", (f"expectancy {m['exp']:+.3f} R on {m['n']} trades; "
                        f"holds up in {', '.join(positive)} (>=0, n>=30)")
    worst = min(conclusive, key=lambda x: x[1]["exp"])
    return "FAIL", (f"positive-only-in-BULL: {worst[0]} expectancy "
                    f"{worst[1]['exp']:+.3f} R (n={worst[1]['n']})")


def format_deep(R) -> str:
    L = ["=" * 96,
         " DEEP-HISTORY EXIT GRID (2019 -> now, 4h) — pre-registered, binding",
         "-" * 96,
         " SUCCESS BAR (all required): expectancy >= +0.15 R  AND  >= 200 trades  AND  not BULL-only.",
         "=" * 96,
         R.get("proof_text", ""),
         "-" * 96,
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
    winners, unproven = [], []
    for name in VARIANTS:
        status, why = _verdict(name, R)
        L.append(f"   {name}: {status} — {why}")
        if status == "PASS":
            winners.append((name, _metrics(R["trades"][name])["exp"]))
        elif status == "UNPROVEN":
            unproven.append(name)
    L += ["=" * 96]
    net_skipped = R.get("network_skipped", [])
    if net_skipped:
        L.append(f" VERDICT WITHHELD — INCOMPLETE SAMPLE: {len(net_skipped)} pair(s) failed to")
        L.append(f" download ({', '.join(net_skipped)}). The >=200-trade bar cannot be judged on a")
        L.append(" partial universe. Re-run (cached pairs load instantly; only the missing ones")
        L.append(" re-fetch) until 'pairs run' equals the full requested set, THEN read the verdict.")
    elif not R.get("proof_sane", True):
        L.append(" REGIME PROOF NOT SANE — the per-regime split is untrustworthy; verdict WITHHELD.")
    elif winners:
        best = max(winners, key=lambda x: x[1])
        L.append(f" SUPPORTED: '{best[0]}' clears the bar (expectancy {best[1]:+.3f} R) AND holds up")
        L.append(" outside bull markets with a conclusive (n>=30) BEAR/CHOP sample.")
    elif unproven:
        L.append(" NOT SUPPORTED: variants clear +0.15 R and >=200 trades, but regime-robustness is")
        L.append(f" UNPROVEN ({', '.join(unproven)}): the strategy's own gate confines it to bull, so")
        L.append(" BEAR/CHOP samples are too thin to show the edge survives outside a bull market.")
        L.append(" A bull-only edge you cannot verify outside bull is not a deployable edge — retired.")
    else:
        L.append(" NOT SUPPORTED: No exit variant cleared the bar. Once BEAR/CHOP trades are counted")
        L.append(" correctly, the edge is positive only in bull markets. The pullback/SLC engine is retired.")
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
