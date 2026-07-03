"""VALIDATION Gate 1 — walk-forward out-of-sample + trade-distribution decomposition.

This is VALIDATION, not optimization. The strategy is FIXED (located AT_NODE
entries, 2xATR node-anchored initial stop, the 5 exit variants, same fees/
slippage/VP config/no-lookahead). Nothing here changes it. We only change HOW we
look: read performance chronologically forward on unseen years, and dissect the
trade distribution to see whether any edge is real or a handful of outliers.

THE ONE RULE: the out-of-sample data is touched ONCE and the result BELIEVED —
pass or fail. No re-tuning against it, no "one more value". A failed gate = the
strategy is retired.

  Rung 1 — Walk-forward out-of-sample by calendar year (2022, 2023, 2024,
     2025->now), each a separate unseen fold. 2022 is the critical bear fold: a
     real edge must not bleed there. We do NOT refit per fold — the strategy is
     fixed; we just read it forward in time.

  Rung 2 — Trade-distribution decomposition for TRAIL_6 (headline) and TARGET_2R
     (runner-up) on the full history: R-multiple distribution, and a
     concentration check — if removing the top 5 winners flips expectancy to
     <= 0 (or below +0.05 R), the "edge" is a few outliers, not a durable edge.

BINDING VERDICT (both must hold):
  Rung 1: OOS aggregate expectancy >= +0.15 R on >= 200 OOS trades, AND 2022 does
          not bleed (expectancy >= 0 on a conclusive >=30 sample), AND the edge
          appears in more than one fold (>=2 folds positive on >=30 trades).
  Rung 2: removing the top 5 trades leaves expectancy > +0.05 R, AND the top
          winners are not all from the 2020-2021 mania.
  Either fails -> NOT VALIDATED (retired, or edge = outlier artifact).

Reuses the deep sweep (all 20 pairs, 2019->now 4h, cached). Run where Binance is
reachable:  python run_validation.py
"""

from __future__ import annotations

import argparse

import numpy as np

import run_deep_grid as D
from run_exit_grid import VARIANTS, _metrics
from run_scenarios import DEFAULT_PAIRS

MIN_CONCLUSIVE = 30
OOS_MIN_TRADES = 200
OOS_MIN_EXP = 0.15
REMOVE_TOP = 5
RUNG2_MIN_EXP = 0.05
REGIMES = ("BULL", "BEAR", "CHOP")
FOLDS = ("2022", "2023", "2024", "2025+")     # out-of-sample folds (2019-2021 = design era)


def _fold_of(ts) -> str:
    y = ts.year
    if y <= 2021:
        return "IS"            # in-sample / design era — excluded from OOS
    if y >= 2025:
        return "2025+"
    return str(y)


def _dd_r(trades) -> float:
    """Max peak-to-trough drawdown of the cumulative-R curve (trades by exit time)."""
    if not trades:
        return float("nan")
    rs = [t.r_multiple for t in sorted(trades, key=lambda t: t.exit_time)
          if not np.isnan(t.r_multiple)]
    if not rs:
        return float("nan")
    cum = np.cumsum(rs)
    peak = np.maximum.accumulate(cum)
    return float((cum - peak).min())


def _regime_split(trades, regime_label):
    out = {r: [] for r in REGIMES}
    for t in trades:
        lab = regime_label.asof(t.entry_time)
        if lab in out:
            out[lab].append(t)
    return out


def _r(x):
    return "n/a" if (x is None or np.isnan(x)) else f"{x:+.3f}"


def _p(x):
    return "n/a" if (x is None or np.isnan(x)) else f"{x*100:.1f}"


# --------------------------------------------------------------------------- #
# Rung 1
# --------------------------------------------------------------------------- #
def rung1(R):
    regime_label = R["regime_label"]
    lines = ["#" * 96, " RUNG 1 — WALK-FORWARD OUT-OF-SAMPLE (folds: 2022, 2023, 2024, 2025+)", "#" * 96]
    per_variant = {}

    for name in R["trades"]:
        all_tr = R["trades"][name]
        folds = {f: [t for t in all_tr if _fold_of(t.entry_time) == f] for f in FOLDS}
        oos = [t for t in all_tr if _fold_of(t.entry_time) != "IS"]
        per_variant[name] = {"folds": folds, "oos": oos}

        lines.append(f"\n {name}")
        lines.append(f"   {'fold':<8}{'trades':>7}{'win%':>7}{'expR':>8}{'PF':>6}{'maxDD_R':>9}"
                     f"   {'BULL':>16}{'BEAR':>16}{'CHOP':>16}")
        for f in list(FOLDS) + ["OOS-agg"]:
            tr = oos if f == "OOS-agg" else folds[f]
            m = _metrics(tr)
            pf = ("inf" if m["profit_factor"] == float("inf")
                  else "n/a" if np.isnan(m["profit_factor"]) else f"{m['profit_factor']:.2f}")
            reg = _regime_split(tr, regime_label)
            rcells = []
            for rg in REGIMES:
                rm = _metrics(reg[rg])
                flag = "" if rm["n"] >= MIN_CONCLUSIVE else "*"
                rcells.append(f"{_r(rm['exp'])}(n={rm['n']}){flag}")
            lines.append(f"   {f:<8}{m['n']:>7}{_p(m['win']):>7}{_r(m['exp']):>8}{pf:>6}"
                         f"{_dd_r(tr):>9.2f}   " + "".join(f"{c:>16}" for c in rcells))
    lines.append("   * = < 30 trades in cell: NOT CONCLUSIVE")
    return lines, per_variant


def rung1_checks(name, per_variant):
    """Return (passed: bool, notes: list[str]) for one variant's Rung-1 criteria."""
    pv = per_variant[name]
    oos_m = _metrics(pv["oos"])
    notes = []
    ok_exp = oos_m["exp"] >= OOS_MIN_EXP if not np.isnan(oos_m["exp"]) else False
    ok_n = oos_m["n"] >= OOS_MIN_TRADES
    notes.append(f"OOS expectancy {_r(oos_m['exp'])} (need >=+{OOS_MIN_EXP}) : {'ok' if ok_exp else 'FAIL'}")
    notes.append(f"OOS trades {oos_m['n']} (need >={OOS_MIN_TRADES}) : {'ok' if ok_n else 'FAIL'}")

    # 2022 bear fold must not bleed, on a conclusive sample
    m2022 = _metrics(pv["folds"]["2022"])
    if m2022["n"] < MIN_CONCLUSIVE:
        ok_2022 = False
        notes.append(f"2022 bear fold n={m2022['n']} (< {MIN_CONCLUSIVE}) : NOT CONCLUSIVE -> FAIL")
    else:
        ok_2022 = m2022["exp"] >= 0
        notes.append(f"2022 bear fold expectancy {_r(m2022['exp'])} (need >=0) : {'ok' if ok_2022 else 'FAIL'}")

    # edge in more than one fold (>=2 folds positive on >=30 trades)
    good_folds = [f for f in FOLDS
                  if _metrics(pv["folds"][f])["n"] >= MIN_CONCLUSIVE
                  and _metrics(pv["folds"][f])["exp"] > 0]
    ok_multi = len(good_folds) >= 2
    notes.append(f"positive on >=2 conclusive folds : {good_folds} : {'ok' if ok_multi else 'FAIL'}")

    return (ok_exp and ok_n and ok_2022 and ok_multi), notes


# --------------------------------------------------------------------------- #
# Rung 2
# --------------------------------------------------------------------------- #
def rung2(R, name):
    trades = [t for t in R["trades"][name] if not np.isnan(t.r_multiple)]
    rs = np.array(sorted(t.r_multiple for t in trades))
    n = len(rs)
    lines = [f"\n {name}  (n={n})"]
    if n == 0:
        return lines + ["   no trades"], False, []
    q = np.percentile(rs, [0, 25, 50, 75, 100])
    lines.append(f"   R distribution: min {q[0]:+.2f}  p25 {q[1]:+.2f}  median {q[2]:+.2f}  "
                 f"p75 {q[3]:+.2f}  max {q[4]:+.2f}")
    exp_all = float(rs.mean())

    top_sorted = sorted(trades, key=lambda t: t.r_multiple, reverse=True)
    top5 = top_sorted[:5]
    top10 = top_sorted[:10]
    lines.append("   top 5 winners (R @ year): "
                 + ", ".join(f"{t.r_multiple:+.1f}@{t.entry_time.year}" for t in top5))

    # Concentration against GROSS profit (sum of winning R) — always bounded and
    # interpretable, unlike a ratio to tiny/negative NET R.
    gross_win = sum(r for r in rs if r > 0)
    top5_win = sum(t.r_multiple for t in top5 if t.r_multiple > 0)
    top10_win = sum(t.r_multiple for t in top10 if t.r_multiple > 0)
    if gross_win > 0:
        lines.append(f"   share of GROSS profit from top 5: {100*top5_win/gross_win:.0f}%   "
                     f"top 10: {100*top10_win/gross_win:.0f}%")

    exp_ex5 = float(rs[:-REMOVE_TOP].mean()) if n > REMOVE_TOP else float("nan")
    lines.append(f"   expectancy: all {exp_all:+.3f} R  ->  removing top {REMOVE_TOP}: "
                 f"{exp_ex5:+.3f} R  (need > +{RUNG2_MIN_EXP})")

    mania = sum(1 for t in top10 if t.entry_time.year in (2020, 2021))
    lines.append(f"   top-10 winners from 2020-2021 mania: {mania}/10 "
                 f"({10-mania}/10 from 2022 onward)")

    ok_outlier = (not np.isnan(exp_ex5)) and exp_ex5 > RUNG2_MIN_EXP
    ok_years = mania < 10           # not ALL from the mania (literal criterion)
    notes = [
        f"remove-top-{REMOVE_TOP} expectancy {exp_ex5:+.3f} R (> +{RUNG2_MIN_EXP}) : "
        f"{'ok' if ok_outlier else 'FAIL'}",
        f"winners not all 2020-2021 ({mania}/10 mania) : {'ok' if ok_years else 'FAIL'}"
        + ("  [WARNING: mania-heavy]" if mania >= 7 else ""),
    ]
    return lines, (ok_outlier and ok_years), notes


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def validate(R) -> str:
    L = [__doc__.split("Reuses the deep")[0]]
    if R.get("network_skipped"):
        L.append("=" * 96)
        L.append(f" VALIDATION WITHHELD — INCOMPLETE SAMPLE: {len(R['network_skipped'])} pair(s) "
                 f"failed to download ({', '.join(R['network_skipped'])}).")
        L.append(" Re-run until all pairs are included (cache makes it fast), THEN validate.")
        L.append("=" * 96)
        return "\n".join(L)
    if not R.get("proof_sane", True):
        L.append(" REGIME PROOF NOT SANE — the per-regime split is untrustworthy; validation WITHHELD.")
        return "\n".join(L)

    r1_lines, per_variant = rung1(R)
    L += r1_lines

    # Rung-1 verdict is driven by the headline variant TRAIL_6.
    r1_pass, r1_notes = rung1_checks("TRAIL_6", per_variant)
    L += ["\n RUNG-1 CRITERIA (headline TRAIL_6):"] + [f"   - {n}" for n in r1_notes]

    L += ["\n" + "#" * 96,
          " RUNG 2 — TRADE-DISTRIBUTION DECOMPOSITION (full history)", "#" * 96]
    r2_pass = {}
    for name in ("TRAIL_6", "TARGET_2R"):
        lines, ok, notes = rung2(R, name)
        L += lines + [f"   -> {x}" for x in notes]
        r2_pass[name] = ok

    # Rung-2 verdict is driven by the headline TRAIL_6.
    L += ["\n" + "=" * 96, " BINDING VERDICT:"]
    L.append(f"   Rung 1 (walk-forward OOS): {'PASS' if r1_pass else 'FAIL'}")
    L.append(f"   Rung 2 (not outlier-driven, TRAIL_6): {'PASS' if r2_pass['TRAIL_6'] else 'FAIL'}")
    L += ["=" * 96]
    if r1_pass and r2_pass["TRAIL_6"]:
        L.append(" VALIDATED: the edge survived unseen years AND is not a handful of outliers.")
        L.append(" It has earned the right to proceed to further validation (robustness, then")
        L.append(" paper trading) — NOT to live capital. (Survivorship bias in the 20-pair universe")
        L.append(" remains an open caveat for the next gate.)")
    else:
        L.append(" NOT VALIDATED.")
        if not r1_pass:
            L.append(" Rung 1 failed: the edge did not hold up out-of-sample (thinned out, bled in the")
            L.append(" 2022 bear, or was carried by a single year). The strategy is retired.")
        if not r2_pass["TRAIL_6"]:
            L.append(" Rung 2 failed: the apparent edge is an artifact of a few outlier winners")
            L.append(" (likely the 2020-2021 mania). Removing them collapses it. Not a durable edge.")
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
                   args.entry_valid_bars, args.regime_symbol, args.cache_dir, args.refresh)
    print("\n" + validate(R))


if __name__ == "__main__":
    main()
