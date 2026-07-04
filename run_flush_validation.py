"""LEVERAGE-FLUSH MEAN REVERSION (long-only) -- pre-registered walk-forward validation.

THE IDEA: in crypto perpetuals, a heavily-long-leveraged crowd pays rising
funding to hold. A downward nudge can trigger cascading forced liquidations
-- margin calls dump into thin liquidity and price OVERSHOOTS. That selling
is mechanical, not informational, so price tends to snap back once
liquidations exhaust. We buy the flush: long-only spot, betting on the
mechanical overshoot reverting. The edge (if real) is direction-agnostic --
it trades overshoots, not trend -- which is exactly what the retired
trend-following pullback engine lacked.

DATA REALITY: Binance does not offer free historical liquidation or deep
open-interest data. Positioning/leverage is proxied by the FUNDING RATE
(real, full history, ~2020-> now). The liquidation cascade itself is
proxied by its OBSERVABLE FOOTPRINT in spot OHLCV: a violent range/volume
spike with a long lower wick and a strong close (absorption) -- the
mechanical-overshoot signature distinguishing a tradeable flush from a
real, information-driven waterfall that keeps falling.

FIXED PARAMETERS (pre-registered, see longbot/flush_engine.FlushParams):
  spring loaded : funding pct_rank >= 0.85 of its trailing 90-day distribution
  flush trigger : range >= 2x trailing-20avg AND volume >= 2x trailing-20avg
                  AND lower wick >= 1.5x body AND close in upper third of range
  entry         : next bar's open, after BOTH conditions confirmed on a CLOSED bar
  target        : +1.5 ATR (frozen at signal bar) OR funding <= its trailing median
  stop          : -2.0 ATR (frozen at signal bar) -- MANDATORY falling-knife protection
  time stop     : 30 bars
  costs         : 0.10%/side fee, 0.05%/side slippage (same as the retired pullback engine)

BINDING SUCCESS BAR (fixed BEFORE results; see `evaluate_bar` below):
  * aggregate OOS expectancy >= +0.15 R net of costs, AND
  * >= 100 OOS trades aggregate, AND
  * 2022 bear-fold expectancy >= 0 on a conclusive (>=30) sample (if <30,
    reported explicitly as inconclusive rather than papered over), AND
  * not outlier-driven: removing the top 5 trades leaves expectancy > +0.05 R, AND
  * not one-year-only: edge appears in >= 2 separate OOS folds (>=30 trades, exp>0)
Any failure -> NOT VALIDATED, retired. No re-tuning against OOS results.

Reuses proven infrastructure from this repo: fetch_or_cache (deep Binance
kline fetch + CSV cache), btc_regime_labels (BULL/BEAR/CHOP macro regime,
independent of any entry gate), and the fee/slippage constants. The flush
signal/entry/exit logic itself (longbot/flush_engine.py) is a NEW, separate
module -- it does not import or depend on scoring_engine.py / trade_setter.py,
the retired pullback engine's brain.

Run where Binance (spot + futures funding) is reachable:
    python run_flush_validation.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from longbot import flush_engine as fe
from longbot import funding as fund
from run_deep_grid import btc_regime_labels, regime_bar_proof, REGIMES
from run_real_backtest import fetch_or_cache
from run_scenarios import DEFAULT_PAIRS

MIN_CONCLUSIVE = 30
BAR_MIN_AGG_EXP = 0.15
BAR_MIN_TRADES = 100
BAR_MIN_BEAR_EXP = 0.0
CONCENTRATION_TOP_N = 5
BAR_MIN_EXP_AFTER_REMOVAL = 0.05
MIN_FOLDS_POSITIVE = 2

FOLDS = ("2021", "2022", "2023", "2024", "2025+")
CRITICAL_BEAR_FOLD = "2022"


def _fold_of(ts) -> str:
    y = ts.year
    if y < 2021:
        return "pre-2021"
    if y >= 2025:
        return "2025+"
    return str(y)


def _r(x) -> str:
    return "n/a" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:+.3f}"


def _p(x) -> str:
    return "n/a" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x * 100:.1f}"


def _metrics(trades: list[fe.FlushTrade]) -> dict:
    if not trades:
        return dict(n=0, win=float("nan"), exp=float("nan"), pf=float("nan"),
                    bars=float("nan"), max_dd_r=float("nan"))
    rs = [t.r_multiple for t in trades if not np.isnan(t.r_multiple)]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)
    pf = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else float("nan"))
    ordered = [t.r_multiple for t in sorted(trades, key=lambda t: t.exit_time) if not np.isnan(t.r_multiple)]
    cum = np.cumsum(ordered) if ordered else np.array([0.0])
    peak = np.maximum.accumulate(cum)
    max_dd_r = float((cum - peak).min()) if ordered else float("nan")
    return dict(
        n=len(rs), win=len(wins) / len(rs) if rs else float("nan"),
        exp=float(np.mean(rs)) if rs else float("nan"), pf=pf,
        bars=float(np.mean([t.bars_held for t in trades])), max_dd_r=max_dd_r,
    )


def _regime_split(trades: list[fe.FlushTrade]) -> dict:
    out = {r: [] for r in REGIMES}
    for t in trades:
        if t.regime in out:
            out[t.regime].append(t)
    return out


# --------------------------------------------------------------------------- #
# data + run
# --------------------------------------------------------------------------- #
MIN_HISTORY_BARS = 400   # formation needs funding_window_prints (~270 8h-prints) of history to warm up


def run_all(pairs, timeframe, start, end, cache_dir, refresh, params: fe.FlushParams):
    print(f"[data] fetching regime BTCUSDT {timeframe} {start}->now (cached in {cache_dir}/)")
    regime_df = fetch_or_cache("BTCUSDT", timeframe, start, end, cache_dir, refresh)
    regime_label = btc_regime_labels(regime_df)
    proof_text, proof_sane = regime_bar_proof(regime_label)
    print("\n" + proof_text + "\n")

    per_pair = {}
    no_perp = []
    insufficient = []

    for pair in pairs:
        try:
            df = fetch_or_cache(pair, timeframe, start, end, cache_dir, refresh)
        except Exception as exc:  # noqa: BLE001
            insufficient.append((pair, f"spot fetch failed ({exc})"))
            continue
        if len(df) < MIN_HISTORY_BARS:
            insufficient.append((pair, f"only {len(df)} spot bars (< {MIN_HISTORY_BARS})"))
            continue

        funding_raw = fund.fetch_funding_or_cache(pair, start, end, cache_dir, refresh)
        if funding_raw.empty:
            no_perp.append(pair)
            continue
        funding_feat = fund.build_funding_features(funding_raw, params.funding_window_prints)
        if funding_feat["pct_rank"].notna().sum() == 0:
            insufficient.append((pair, f"only {len(funding_raw)} funding prints "
                                       f"(< {params.funding_window_prints} needed to warm up the percentile)"))
            continue

        aligned = fund.align_to_bars(funding_feat, df.index)
        result = fe.run_flush_backtest(df, aligned, params)
        for t in result.trades:
            t.entry_year = t.entry_time.year
            t.pair = pair
            t.regime = regime_label.reindex(df.index, method="ffill").get(t.entry_time, "CHOP")

        per_pair[pair] = {
            "result": result, "n_bars": len(df),
            "history": f"{df.index[0].date()} -> {df.index[-1].date()}",
        }

    return dict(per_pair=per_pair, no_perp=no_perp, insufficient=insufficient,
                regime_label=regime_label, proof_text=proof_text, proof_sane=proof_sane)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def all_trades(R) -> list:
    out = []
    for p in R["per_pair"].values():
        out.extend(p["result"].trades)
    return out


def format_per_pair(R) -> list[str]:
    L = [" PER-PAIR:", f"   {'pair':<10}{'bars':>7}{'spring-loaded':>15}{'flush-trig':>12}"
                       f"{'armed':>7}{'trades':>8}{'exp R':>8}   history"]
    for pair, p in R["per_pair"].items():
        res = p["result"]
        m = _metrics(res.trades)
        L.append(f"   {pair:<10}{p['n_bars']:>7}{res.spring_loaded_bars:>15}{res.flush_trigger_bars:>12}"
                 f"{res.armed_signals:>7}{m['n']:>8}{_r(m['exp']):>8}   {p['history']}")
    if R["no_perp"]:
        L.append(f"   skipped (no perp / no funding history): {', '.join(R['no_perp'])}")
    if R["insufficient"]:
        L.append("   skipped (insufficient history):")
        for pair, why in R["insufficient"]:
            L.append(f"     {pair}: {why}")
    return L


def format_per_fold(R) -> list[str]:
    trades = all_trades(R)
    L = [" PER-FOLD (out-of-sample walk-forward, strategy fixed, no per-fold refitting):",
         f"   {'fold':<9}{'trades':>7}{'win%':>7}{'expR':>8}{'PF':>6}{'maxDD_R':>9}   "
         f"{'BULL':>18}{'BEAR':>18}{'CHOP':>18}"]
    fold_metrics = {}
    for f in FOLDS:
        ft = [t for t in trades if _fold_of(t.entry_time) == f]
        m = _metrics(ft)
        fold_metrics[f] = m
        reg = _regime_split(ft)
        pf = "inf" if m["pf"] == float("inf") else ("n/a" if np.isnan(m["pf"]) else f"{m['pf']:.2f}")
        cells = []
        for rg in REGIMES:
            rm = _metrics(reg[rg])
            flag = "" if rm["n"] >= MIN_CONCLUSIVE else "*"
            cells.append(f"{_r(rm['exp'])}(n={rm['n']}){flag}")
        L.append(f"   {f:<9}{m['n']:>7}{_p(m['win']):>7}{_r(m['exp']):>8}{pf:>6}"
                 f"{m['max_dd_r']:>9.2f}   " + "".join(f"{c:>18}" for c in cells))
    pre = [t for t in trades if _fold_of(t.entry_time) == "pre-2021"]
    if pre:
        m = _metrics(pre)
        L.append(f"   (pre-2021 funding-history warm-up, excluded from folds: {m['n']} trades, exp {_r(m['exp'])})")
    L.append("   * = < 30 trades in cell: NOT CONCLUSIVE")
    return L, fold_metrics


def format_concentration(trades: list[fe.FlushTrade]) -> tuple[list[str], dict]:
    rs = sorted((t.r_multiple for t in trades if not np.isnan(t.r_multiple)), reverse=True)
    L = [" CONCENTRATION CHECK (top-5-removed):"]
    if not rs:
        L.append("   no trades")
        return L, {"passes": False, "n": 0}
    before = float(np.mean(rs))
    n_remove = min(CONCENTRATION_TOP_N, len(rs))
    remainder = rs[n_remove:]
    after = float(np.mean(remainder)) if remainder else float("-inf")
    passes = after > BAR_MIN_EXP_AFTER_REMOVAL
    L.append(f"   trades: {len(rs)}, removed top {n_remove}")
    L.append(f"   expectancy before: {_r(before)} R   after: {_r(after)} R "
             f"(bar: > +{BAR_MIN_EXP_AFTER_REMOVAL} R) -> {'PASS' if passes else 'FAIL'}")
    return L, {"passes": passes, "before": before, "after": after, "n": len(rs)}


def format_mae_mfe(trades: list[fe.FlushTrade]) -> list[str]:
    """Descriptive diagnostic only -- see longbot/flush_engine.py docstring.
    Does not change any entry/exit/stop/threshold."""
    L = [" MAE / MFE DIAGNOSTIC (descriptive only -- not a trading rule):"]
    if not trades:
        return L + ["   no trades"]

    all_mfe = [t.mfe_r for t in trades if not np.isnan(t.mfe_r)]
    all_mae = [t.mae_r for t in trades if not np.isnan(t.mae_r)]
    if all_mfe:
        q = np.percentile(all_mfe, [25, 50, 75])
        L.append(f"   MFE (all trades, R): p25 {q[0]:+.2f}  median {q[1]:+.2f}  p75 {q[2]:+.2f}")
    if all_mae:
        q = np.percentile(all_mae, [25, 50, 75])
        L.append(f"   MAE (all trades, R): p25 {q[0]:+.2f}  median {q[1]:+.2f}  p75 {q[2]:+.2f}")

    stopped = [t for t in trades if t.exit_reason == "stop"]
    if stopped:
        mfe_of_losers = [t.mfe_r for t in stopped if not np.isnan(t.mfe_r)]
        near_target = sum(1 for t in stopped if not np.isnan(t.mfe_r) and t.mfe_r >= 0.75 * (t.target_initial - t.entry_price) / (t.entry_price - t.stop_initial))
        L.append(f"   stopped-out losers: {len(stopped)}")
        if mfe_of_losers:
            L.append(f"     of these, MFE before stopping: median {np.median(mfe_of_losers):+.2f} R "
                     f"({near_target}/{len(stopped)} got >=75% of the way to target before reversing)")
        post = [t.post_stop_mfe_r for t in stopped if t.post_stop_mfe_r is not None]
        if post:
            recovered = sum(1 for x in post if x > 0)
            L.append(f"     tracked past the stop ({len(post)} with data): {recovered}/{len(post)} recovered "
                     f"back above entry within the time-stop horizon; median post-stop MFE {np.median(post):+.2f} R")
            L.append("     -> " + (
                "many recovered: the hard stop may be cutting mechanical overshoots too early for their own volatility."
                if recovered > len(post) / 2 else
                "most kept falling: the hard stop is correctly cutting information-driven crashes, not mechanical noise."
            ))

    winners = [t for t in trades if t.r_multiple > 0]
    if winners:
        overrun = [t.mfe_r - (t.target_initial - t.entry_price) / (t.entry_price - t.stop_initial)
                  for t in winners if not np.isnan(t.mfe_r) and t.entry_price > t.stop_initial]
        if overrun:
            L.append(f"   winners: MFE beyond the +1.5 ATR target, median {np.median(overrun):+.2f} R "
                     f"({'the fixed target is leaving R on the table' if np.median(overrun) > 0.25 else 'target roughly captures the reversion move'})")
    return L


def evaluate_bar(R, fold_metrics: dict, concentration: dict) -> dict:
    trades = all_trades(R)
    overall = _metrics(trades)
    bear = fold_metrics[CRITICAL_BEAR_FOLD]
    bear_conclusive = bear["n"] >= MIN_CONCLUSIVE

    good_folds = [f for f in FOLDS if fold_metrics[f]["n"] >= MIN_CONCLUSIVE and fold_metrics[f]["exp"] > 0]

    checks = {
        "aggregate_expectancy_ge_0.15R": (not np.isnan(overall["exp"])) and overall["exp"] >= BAR_MIN_AGG_EXP,
        "trade_count_ge_100": overall["n"] >= BAR_MIN_TRADES,
        "2022_bear_conclusive_and_non_negative": bear_conclusive and bear["exp"] >= BAR_MIN_BEAR_EXP,
        "not_outlier_driven": concentration.get("passes", False),
        "edge_in_ge_2_folds": len(good_folds) >= MIN_FOLDS_POSITIVE,
    }
    validated = all(checks.values())
    return dict(validated=validated, checks=checks, overall=overall,
               bear=bear, bear_conclusive=bear_conclusive, good_folds=good_folds)


def format_report(R, params: fe.FlushParams) -> str:
    L = [__doc__.split("Reuses proven infrastructure")[0], "=" * 100]
    L += format_per_pair(R)
    L += ["-" * 100]
    fold_lines, fold_metrics = format_per_fold(R)
    L += fold_lines
    trades = all_trades(R)
    L += ["-" * 100]
    conc_lines, conc = format_concentration(trades)
    L += conc_lines
    L += ["-" * 100]
    L += format_mae_mfe(trades)

    verdict = evaluate_bar(R, fold_metrics, conc)
    L += ["-" * 100, " AGGREGATE OOS vs. PRE-REGISTERED BAR:"]
    o = verdict["overall"]
    L.append(f"   trades: {o['n']} (bar >= {BAR_MIN_TRADES})   expectancy: {_r(o['exp'])} R (bar >= +{BAR_MIN_AGG_EXP})")
    L.append(f"   win%: {_p(o['win'])}  PF: {o['pf']:.2f}  max DD: {o['max_dd_r']:.2f} R"
             if not np.isnan(o['pf']) else f"   win%: {_p(o['win'])}")
    b = verdict["bear"]
    conclusive_note = "" if verdict["bear_conclusive"] else "  ** NOT CONCLUSIVE (n<30) **"
    L.append(f"   2022 bear fold: n={b['n']}, expectancy {_r(b['exp'])} (bar >= 0){conclusive_note}")
    L.append(f"   positive on >=30-trade folds: {verdict['good_folds']} (need >= {MIN_FOLDS_POSITIVE})")

    L += ["-" * 100, " CHECKS:"]
    for name, ok in verdict["checks"].items():
        L.append(f"   {name}: {'PASS' if ok else 'FAIL'}")

    L += ["=" * 100]
    if not R["per_pair"]:
        L.append(" VERDICT WITHHELD -- no pair produced usable data (spot + funding history). "
                 "Check network access and re-run.")
    elif verdict["validated"]:
        L.append(" VALIDATED: the mechanical-overshoot edge held up out-of-sample, survived the 2022 bear")
        L.append(" fold on a conclusive sample, is not a handful of outliers, and appears in >=2 separate")
        L.append(" OOS folds. The MAE/MFE diagnostic above explains the mechanism -- read it, not just the R.")
    else:
        L.append(" NOT VALIDATED. Reasons:")
        for name, ok in verdict["checks"].items():
            if not ok:
                L.append(f"   - {name} failed")
        if not verdict["checks"]["2022_bear_conclusive_and_non_negative"] and not verdict["bear_conclusive"]:
            L.append("   NOTE: 2022 was inconclusive (too few trades), not proven to fail -- but the bar")
            L.append("   requires a CONCLUSIVE non-negative bear fold, and an inconclusive sample does not")
            L.append("   satisfy that. This is reported as a failure of proof, not proof of failure.")
        L.append(" Strategy retired as specified. No re-tuning against these OOS results.")
    L += ["=" * 100]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="*", default=DEFAULT_PAIRS)
    ap.add_argument("--timeframe", default="4h")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--cache-dir", default="data")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    params = fe.FlushParams(timeframe=args.timeframe)
    R = run_all(args.pairs, args.timeframe, args.start, args.end, args.cache_dir, args.refresh, params)
    print(format_report(R, params))


if __name__ == "__main__":
    main()
