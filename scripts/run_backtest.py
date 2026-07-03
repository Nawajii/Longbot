#!/usr/bin/env python3
"""
One-command entry point: fetch/cache data, run the walk-forward backtest
across the pre-registered pair universe, print the report, and write it
(plus a trade-level CSV) to results/.

    python scripts/run_backtest.py

Requires outbound network access to api.binance.com on first run (data is
then cached in data_cache/ as CSV; subsequent runs only fetch new bars).
"""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pairs_pipeline import config, report, walkforward

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def print_preamble():
    print("=" * 78)
    print("LONG-ONLY COINTEGRATION PAIRS STRATEGY -- pre-registered walk-forward run")
    print("=" * 78)
    print("""
IDEA: two assets sharing the same fundamental driver (e.g. two L1 tokens)
tend to move together (cointegration, Engle-Granger). When their price
relationship diverges due to a liquidity event rather than new information,
there is no informational reason for the gap, so capital tends to pull it
back. This edge is regime-agnostic BY DESIGN -- it trades the relationship
between two assets, not market direction.

LONG-ONLY WEAKNESS (stated up front, not hidden): true pairs trading is
market-neutral -- long the underperformer AND short the outperformer. This
system is long-only (halal constraint: no shorting/futures/margin), so it
can only take the long leg. It is therefore NOT market-neutral and still
carries market exposure: if both assets fall together, this strategy can
lose money even while the spread itself is reverting correctly. This
materially weakens the edge versus textbook stat-arb -- we test it anyway,
honestly, within the constraint.
""")
    print("PAIR UNIVERSE (fixed, economically motivated, not data-mined):")
    for a, b in config.PAIR_UNIVERSE:
        print(f"  {a}/{b}")
    print()
    print("BINDING SUCCESS BAR (fixed before results):")
    print(f"  - aggregate OOS expectancy >= +{config.MIN_AGGREGATE_EXPECTANCY_R} R net of costs")
    print(f"  - >= {config.MIN_OOS_TRADES} OOS trades aggregate")
    print(f"  - {config.CRITICAL_BEAR_FOLD} bear fold expectancy >= 0 R on a conclusive "
          f"(>= {config.MIN_BEAR_FOLD_SAMPLE}) sample")
    print(f"  - removing top {config.CONCENTRATION_TOP_N_REMOVED} trades leaves expectancy > "
          f"+{config.MIN_EXPECTANCY_AFTER_REMOVAL_R} R")
    print("  Any failure -> NOT VALIDATED. No re-tuning against OOS results.")
    print("=" * 78)
    print()


def write_trades_csv(results, path):
    fields = [
        "pair", "direction", "entry_date", "entry_price", "entry_z",
        "exit_date", "exit_price", "exit_reason", "days_held",
        "return_pct", "r_multiple", "entry_year", "regime",
        "formation_beta", "formation_adf_pvalue",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for pr in results.values():
            if pr.skipped:
                continue
            for t in pr.trades:
                w.writerow({
                    "pair": f"{t.pair[0]}/{t.pair[1]}",
                    "direction": t.direction,
                    "entry_date": t.entry_date,
                    "entry_price": t.entry_price,
                    "entry_z": t.entry_z,
                    "exit_date": t.exit_date,
                    "exit_price": t.exit_price,
                    "exit_reason": t.exit_reason,
                    "days_held": t.days_held,
                    "return_pct": t.return_pct,
                    "r_multiple": t.r_multiple,
                    "entry_year": t.entry_year,
                    "regime": t.regime,
                    "formation_beta": t.formation.beta,
                    "formation_adf_pvalue": t.formation.adf_pvalue,
                })


def main():
    print_preamble()
    print("Fetching/caching data and running walk-forward backtest ...")
    results = walkforward.run_all()

    md = report.render_markdown(results)
    print(md)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "report.md")
    trades_path = os.path.join(RESULTS_DIR, "trades.csv")
    with open(report_path, "w") as f:
        f.write(md)
    write_trades_csv(results, trades_path)

    print(f"\nWrote {report_path}")
    print(f"Wrote {trades_path}")

    verdict = report.evaluate_bar(results)
    sys.exit(0 if verdict["validated"] else 1)


if __name__ == "__main__":
    main()
