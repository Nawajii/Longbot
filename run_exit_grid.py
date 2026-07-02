"""EXIT-VARIANT GRID — the final, pre-registered experiment for this engine.

HYPOTHESIS (registered before results):
    On AT_NODE pullback entries, a wider or different exit style materially lifts
    expectancy — meaning the post-entry moves were there and our exit was
    breaking them. If no exit variant lifts expectancy, the moves are genuinely
    too small net of costs, and the pullback engine is retired.

BINDING SUCCESS BAR (fixed in advance):
    A variant is a real finding ONLY if ALL hold:
      * aggregate expectancy >= +0.15 R across the full sweep, AND
      * NOT positive-only-in-BULL  (>= 0 in at least one of BEAR/CHOP, OR BULL
        strength with only small negative <= 0.05 R elsewhere), AND
      * >= 60 trades aggregate.
    Anything below = NOT SUPPORTED. No re-runs, no new variants after results.

Entries are IDENTICAL across all variants: the located engine (location gate ON,
AT_NODE-only setups) with the same buy-stop entry, sizing, fees, slippage and
no-lookahead rules, and the same 2xATR node-anchored INITIAL hard stop. The ONLY
thing that differs is the exit. (Because exits change how long a position is
held, later entries can differ slightly via occupancy — noted in the report.)

Run where Binance is reachable:  python run_exit_grid.py --timeframe 4h
"""

from __future__ import annotations

import argparse

import numpy as np

import backtest as bt
from longbot import scoring_engine as se
from longbot import volume_profile as vp
from run_scenarios import DEFAULT_PAIRS, DEFAULT_SCENARIOS, MIN_USABLE_BARS, _fetch_window

MIN_CONCLUSIVE = 30
BAR_EXPECTANCY = 0.15
BAR_MIN_TRADES = 60

# Shared base: identical entries + identical 2xATR node-anchored INITIAL stop.
_BASE = dict(use_node_stop=True, stop_atr_mult=2.0, stop_atr_floor_mult=2.0)

# The four registered variants — ONLY the exit differs.
VARIANTS: dict[str, dict] = {
    "TRAIL_2": {**_BASE, "exit_style": "TRAIL", "trail_atr_mult": 3.0},   # control (current)
    "TRAIL_4": {**_BASE, "exit_style": "TRAIL", "trail_atr_mult": 4.0},
    "TRAIL_6": {**_BASE, "exit_style": "TRAIL", "trail_atr_mult": 6.0},
    "TIME_12": {**_BASE, "exit_style": "TIME", "time_exit_bars": 12},
}


def _metrics(trades) -> dict:
    if not trades:
        return dict(n=0, win=float("nan"), avg_win=float("nan"), avg_loss=float("nan"),
                    exp=float("nan"), bars=float("nan"), giveback=float("nan"))
    rs = [t.r_multiple for t in trades if not np.isnan(t.r_multiple)]
    wins = [t.r_multiple for t in trades if t.pnl > 0]
    losses = [t.r_multiple for t in trades if t.pnl <= 0]
    gb = [t.giveback_r for t in trades if not np.isnan(t.giveback_r)]
    return dict(
        n=len(trades),
        win=len(wins) / len(trades),
        avg_win=float(np.mean(wins)) if wins else float("nan"),
        avg_loss=float(np.mean(losses)) if losses else float("nan"),
        exp=float(np.mean(rs)) if rs else float("nan"),
        bars=float(np.mean([t.bars_held for t in trades])),
        giveback=float(np.mean(gb)) if gb else float("nan"),
    )


def run_grid(pairs, timeframe, scenarios, capital, entry_valid_bars, regime_symbol):
    located_sp = se.ScoringParams(use_location=True)
    trades = {v: [] for v in VARIANTS}
    by_scen = {v: {s: [] for s in scenarios} for v in VARIANTS}
    worst_dd = {v: 0.0 for v in VARIANTS}
    executed = skipped = 0

    for scen, (start, end) in scenarios.items():
        try:
            regime_df = _fetch_window(regime_symbol, timeframe, start, end)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {scen}: no regime data ({exc}); ungated.")
            regime_df = None

        for pair in pairs:
            try:
                df = _fetch_window(pair, timeframe, start, end)
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                print(f"[skip] {scen} {pair}: fetch failed ({exc})")
                continue
            if len(df) < MIN_USABLE_BARS:
                skipped += 1
                print(f"[skip] {scen} {pair}: insufficient data ({len(df)} bars)")
                continue
            executed += 1
            loc = vp.compute_location_features(df)     # once; identical entries for all variants

            counts = {}
            for name, over in VARIANTS.items():
                p = bt.BacktestParams(starting_capital=capital, timeframe=timeframe,
                                      entry_valid_bars=entry_valid_bars, **over)
                res = bt.run_backtest(df, p, scoring_params=located_sp,
                                      regime_df=regime_df, location_features=loc)
                trades[name] += res.trades
                by_scen[name][scen] += res.trades
                dd = res.max_drawdown_pct
                if not np.isnan(dd):
                    worst_dd[name] = min(worst_dd[name], dd)
                counts[name] = len(res.trades)
            print(f"[run] {scen:<4} {pair:<9} " +
                  "  ".join(f"{k}={counts[k]:>2}" for k in VARIANTS))

    return dict(trades=trades, by_scen=by_scen, worst_dd=worst_dd,
                executed=executed, skipped=skipped, scenarios=scenarios)


def _verdict(name, R) -> tuple[bool, str]:
    m = _metrics(R["trades"][name])
    scen = R["scenarios"]
    per = {s: _metrics(R["by_scen"][name][s])["exp"] for s in scen}
    bull = per.get("BULL", float("nan"))
    bear = per.get("BEAR", float("nan"))
    chop = per.get("CHOP", float("nan"))

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
        return False, "positive-only-in-BULL (fails the regime-robustness clause)"
    return True, f"expectancy {m['exp']:+.3f} R on {m['n']} trades, robust across regimes"


def format_grid(R) -> str:
    scen = R["scenarios"]
    L = ["=" * 92,
         " EXIT-VARIANT GRID — pre-registered, binding",
         "-" * 92,
         " HYPOTHESIS: on AT_NODE pullbacks, a wider/different exit materially lifts expectancy;",
         "   if none does, the moves are too small net of costs and the engine is retired.",
         " SUCCESS BAR (all required): expectancy >= +0.15 R AND >= 60 trades AND not BULL-only.",
         "=" * 92,
         f" runs executed: {R['executed']}   skipped: {R['skipped']}   "
         f"(entries identical across variants; trade counts differ only via holding-time occupancy)",
         "-" * 92,
         f" {'variant':<9}{'trades':>7}{'win%':>7}{'avgWinR':>9}{'avgLossR':>9}"
         f"{'exp R':>8}{'bars':>7}{'giveback':>9}{'worstDD%':>9}"]
    for name in VARIANTS:
        m = _metrics(R["trades"][name])
        gb = "n/a" if np.isnan(m["giveback"]) else f"{m['giveback']:+.2f}"
        L.append(
            f" {name:<9}{m['n']:>7}{_p(m['win']):>7}{_r(m['avg_win']):>9}{_r(m['avg_loss']):>9}"
            f"{_r(m['exp']):>8}{m['bars']:>7.1f}{gb:>9}{R['worst_dd'][name]:>9.1f}")
    L += ["-" * 92, " PER-SCENARIO EXPECTANCY (R):",
          f"   {'variant':<9}" + "".join(f"{s:>22}" for s in scen)]
    for name in VARIANTS:
        cells = []
        for s in scen:
            m = _metrics(R["by_scen"][name][s])
            flag = "" if m["n"] >= MIN_CONCLUSIVE else "*"
            cells.append(f"{_r(m['exp'])} (n={m['n']}){flag}")
        L.append(f"   {name:<9}" + "".join(f"{c:>22}" for c in cells))
    L.append("   * = < 30 trades in cell: NOT CONCLUSIVE")

    # binding verdict
    L += ["-" * 92, " BINDING VERDICT (against the pre-registered bar):"]
    winners = []
    for name in VARIANTS:
        ok, why = _verdict(name, R)
        L.append(f"   {name}: {'PASS' if ok else 'fail'} — {why}")
        if ok:
            winners.append((name, _metrics(R["trades"][name])["exp"]))
    L += ["=" * 92]
    if winners:
        best = max(winners, key=lambda x: x[1])
        L.append(f" SUPPORTED: '{best[0]}' clears the bar (expectancy {best[1]:+.3f} R). "
                 f"The exit WAS truncating winners.")
    else:
        L.append(" NOT SUPPORTED: No exit variant rescued expectancy; the post-entry moves are")
        L.append(" too small net of costs; the pullback engine is retired.")
    L += ["=" * 92]
    return "\n".join(L)


def _p(x):
    return "n/a" if (x is None or np.isnan(x)) else f"{x*100:.1f}"


def _r(x):
    return "n/a" if (x is None or np.isnan(x)) else f"{x:+.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="*", default=DEFAULT_PAIRS)
    ap.add_argument("--timeframe", default="4h")
    ap.add_argument("--capital", type=float, default=1000.0)
    ap.add_argument("--entry-valid-bars", type=int, default=3)
    ap.add_argument("--regime-symbol", default="BTCUSDT")
    args = ap.parse_args()

    print(__doc__.split("Run where")[0])
    R = run_grid(args.pairs, args.timeframe, DEFAULT_SCENARIOS,
                 args.capital, args.entry_valid_bars, args.regime_symbol)
    print("\n" + format_grid(R))


if __name__ == "__main__":
    main()
