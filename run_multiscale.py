"""FINAL TEST of the trend-following/pullback line — multi-scale equilibrium.

THE IDEA: require agreement across timeframes. The 4h located AT_NODE pullback
(entry structure) is ANDed with a 24h daily gate (trend + macro equilibrium):
trade only when the daily is in an uptrend AND price is at/above the daily
value-area support, AND the 4h shows the located pullback. Exits: the two prior
leaders, TRAIL_6 and TARGET_2R. Long-only spot, one position, fees 0.10%/side +
slippage 0.05%, no-lookahead on BOTH scales (the daily gate reads only completed
daily candles).

HONEST NOTE: this multi-scale gate is still, mechanically, a FILTER on the
existing entry engine. Filters remove trades; they do not create edge. Two prior
strategy families (single-scale pullback, cointegration) were retired after
failing walk-forward validation. This checks whether multi-scale ALIGNMENT
isolates a subset with genuine edge; the base-rate expectation is that it does
not. This is the FINAL test of this line — pass or fail, it is retired after.

BINDING BAR (fixed before results; VALIDATED only if ALL hold on OOS walk-forward
2022/2023/2024/2025+):
  * aggregate OOS expectancy >= +0.15 R net of costs, AND
  * >= 100 OOS trades (if alignment is so restrictive it can't reach 100 over 6
    years x 20 pairs, it trades too rarely to be viable), AND
  * 2022 bear fold >= 0 on a conclusive (>=30) sample — OR, if <30 bear trades,
    near-zero/small-positive (it SAT OUT rather than BLED; stated explicitly), AND
  * not outlier-driven: removing the top 5 trades leaves expectancy > +0.05 R, AND
  * not bull-only: the edge appears in >= 2 separate OOS folds.
  Any failure -> NOT VALIDATED -> the trend-following line is retired.

Reuses the deep walk-forward harness and cached CSVs. Run where Binance is
reachable:  python run_multiscale.py
"""

from __future__ import annotations

import argparse

import numpy as np

import run_deep_grid as D
import run_validation as V
from longbot import multiscale as ms
from run_scenarios import DEFAULT_PAIRS

VARIANTS_TESTED = ["TRAIL_6", "TARGET_2R"]
OOS_MIN_TRADES = 100          # this test's bar (lower than the 200 of the general gate)
OOS_MIN_EXP = 0.15
FOLDS = V.FOLDS
MIN_CONCLUSIVE = V.MIN_CONCLUSIVE


def _entry_gate_fn(df):
    return ms.daily_gate_series(df)


def _fold_report(R, name):
    regime_label = R["regime_label"]
    all_tr = R["trades"][name]
    folds = {f: [t for t in all_tr if V._fold_of(t.entry_time) == f] for f in FOLDS}
    oos = [t for t in all_tr if V._fold_of(t.entry_time) != "IS"]
    L = [f"\n {name}",
         f"   {'fold':<8}{'trades':>7}{'win%':>7}{'expR':>8}{'PF':>6}{'maxDD_R':>9}"
         f"   {'BULL':>15}{'BEAR':>15}{'CHOP':>15}"]
    for f in list(FOLDS) + ["OOS-agg"]:
        tr = oos if f == "OOS-agg" else folds[f]
        m = V._metrics(tr)
        pf = ("inf" if m["profit_factor"] == float("inf")
              else "n/a" if np.isnan(m["profit_factor"]) else f"{m['profit_factor']:.2f}")
        reg = V._regime_split(tr, regime_label)
        rc = []
        for rg in V.REGIMES:
            rm = V._metrics(reg[rg])
            rc.append(f"{V._r(rm['exp'])}(n={rm['n']}){'' if rm['n'] >= MIN_CONCLUSIVE else '*'}")
        L.append(f"   {f:<8}{m['n']:>7}{V._p(m['win']):>7}{V._r(m['exp']):>8}{pf:>6}"
                 f"{V._dd_r(tr):>9.2f}   " + "".join(f"{c:>15}" for c in rc))
    return L, folds, oos


def _verdict(R, name):
    _lines, folds, oos = _fold_report(R, name)
    m = V._metrics(oos)
    notes, ok = [], True

    ok_exp = (not np.isnan(m["exp"])) and m["exp"] >= OOS_MIN_EXP
    ok_n = m["n"] >= OOS_MIN_TRADES
    notes.append(f"OOS expectancy {V._r(m['exp'])} (>=+{OOS_MIN_EXP}): {'ok' if ok_exp else 'FAIL'}")
    notes.append(f"OOS trades {m['n']} (>={OOS_MIN_TRADES}): {'ok' if ok_n else 'FAIL'}")
    ok = ok and ok_exp and ok_n

    m22 = V._metrics(folds["2022"])
    if m22["n"] >= MIN_CONCLUSIVE:
        ok22 = m22["exp"] >= 0
        notes.append(f"2022 bear {V._r(m22['exp'])} on n={m22['n']} (>=0): {'ok' if ok22 else 'FAIL'}")
    else:
        # <30 bear trades: acceptable only if it SAT OUT (near-zero / small positive)
        ok22 = np.isnan(m22["exp"]) or m22["exp"] >= -0.05
        notes.append(f"2022 bear n={m22['n']} (<30) -> sat out; expectancy {V._r(m22['exp'])} "
                     f"(need >=-0.05, not bleeding): {'ok' if ok22 else 'FAIL (it BLED)'}")
    ok = ok and ok22

    good = [f for f in FOLDS if V._metrics(folds[f])["n"] >= MIN_CONCLUSIVE
            and V._metrics(folds[f])["exp"] > 0]
    ok_multi = len(good) >= 2
    notes.append(f"positive on >=2 conclusive folds {good}: {'ok' if ok_multi else 'FAIL'}")
    ok = ok and ok_multi

    _r2lines, ok_r2, r2notes = V.rung2(R, name)
    notes += r2notes
    ok = ok and ok_r2
    return ok, notes


def format_multiscale(R) -> str:
    L = [__doc__.split("Reuses the deep")[0]]
    if R.get("network_skipped"):
        L += ["=" * 96,
              f" WITHHELD — INCOMPLETE SAMPLE: {len(R['network_skipped'])} pair(s) failed "
              f"({', '.join(R['network_skipped'])}). Re-run until all pairs are in.", "=" * 96]
        return "\n".join(L)
    if not R.get("proof_sane", True):
        return "\n".join(L + [" REGIME PROOF NOT SANE — verdict WITHHELD."])

    bh = [x for x in R["bh_returns"] if not np.isnan(x)]
    L += ["#" * 96, " WALK-FORWARD OUT-OF-SAMPLE (multi-scale gate ON)", "#" * 96]
    if bh:
        L.append(f" baseline buy&hold: median {np.median(bh):+.0f}% across pairs "
                 f"(not credited as edge — a filtered long-only strat sitting out a fall is not skill).")
    verdicts = {}
    for name in VARIANTS_TESTED:
        lines, _f, _o = _fold_report(R, name)
        L += lines
    L.append("   * = < 30 trades in cell: NOT CONCLUSIVE")

    L += ["\n" + "#" * 96, " RUNG 2 — outlier / concentration check (full history)", "#" * 96]
    for name in VARIANTS_TESTED:
        r2lines, _ok, _n = V.rung2(R, name)
        L += r2lines

    L += ["\n" + "=" * 96, " BINDING VERDICT (multi-scale, final test of this line):"]
    for name in VARIANTS_TESTED:
        ok, notes = _verdict(R, name)
        verdicts[name] = ok
        L.append(f"\n   {name}: {'VALIDATED' if ok else 'NOT VALIDATED'}")
        L += [f"     - {n}" for n in notes]
    L += ["=" * 96]
    if any(verdicts.values()):
        winners = [k for k, v in verdicts.items() if v]
        L.append(f" VALIDATED: {', '.join(winners)} cleared the bar out-of-sample. Multi-scale")
        L.append(" alignment isolated a subset that survived unseen years and is not outlier-driven.")
        L.append(" Next gate is survivorship-robust testing + paper trading — NOT live capital.")
    else:
        L.append(" NOT VALIDATED. Requiring multi-scale agreement did not isolate a real edge —")
        L.append(" it is the same no-edge/beta pattern with fewer trades. As pre-agreed, the")
        L.append(" trend-following / pullback line is RETIRED. No further variants.")
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
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    R = D.run_deep(args.pairs, args.timeframe, args.start, args.end, args.capital,
                   args.entry_valid_bars, args.regime_symbol, args.cache_dir, args.refresh,
                   variant_names=VARIANTS_TESTED, entry_gate_fn=_entry_gate_fn)
    print("\n" + format_multiscale(R))


if __name__ == "__main__":
    main()
