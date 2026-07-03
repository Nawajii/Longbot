"""
Reporting: per-pair, per-fold, aggregate, concentration check, and the
binding VALIDATED / NOT VALIDATED verdict against the pre-registered bar
in config.py. This module only reads results; it must never feed back into
config.py or engine.py (no re-tuning against what it prints).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from .walkforward import PairResult


@dataclass
class FoldStats:
    label: str
    n_trades: int
    win_pct: float
    expectancy_r: float
    profit_factor: float
    max_drawdown_r: float


def _drawdown_r(r_multiples: list[float]) -> float:
    if not r_multiples:
        return 0.0
    equity = np.cumsum(r_multiples)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(dd.min())


def _fold_stats(label: str, trades: list) -> FoldStats:
    if not trades:
        return FoldStats(label, 0, 0.0, 0.0, 0.0, 0.0)
    r = [t.r_multiple for t in trades]
    wins = [x for x in r if x > 0]
    losses = [x for x in r if x <= 0]
    win_pct = 100.0 * len(wins) / len(r)
    expectancy = float(np.mean(r))
    gross_win = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0
    return FoldStats(label, len(r), win_pct, expectancy, pf, _drawdown_r(r))


def all_trades(results: dict) -> list:
    out = []
    for pr in results.values():
        if not pr.skipped:
            out.extend(pr.trades)
    return out


def per_pair_table(results: dict) -> list[dict]:
    rows = []
    for (a, b), pr in results.items():
        if pr.skipped:
            rows.append({
                "pair": f"{a}/{b}", "status": "SKIPPED", "reason": pr.skip_reason,
                "coint_freq_pct": None, "n_trades": 0, "expectancy_r": None,
            })
            continue
        coint_freq = (100.0 * pr.n_formation_cointegrated / pr.n_formation_tested) if pr.n_formation_tested else 0.0
        stats = _fold_stats(f"{a}/{b}", pr.trades)
        rows.append({
            "pair": f"{a}/{b}",
            "status": "OK",
            "history": f"{pr.history_start} -> {pr.history_end} ({pr.n_bars} bars)",
            "coint_freq_pct": round(coint_freq, 1),
            "n_formation_tested": pr.n_formation_tested,
            "n_trades": stats.n_trades,
            "win_pct": round(stats.win_pct, 1),
            "expectancy_r": round(stats.expectancy_r, 3),
            "profit_factor": round(stats.profit_factor, 2) if np.isfinite(stats.profit_factor) else "inf",
        })
    return rows


def per_fold_table(results: dict) -> list[dict]:
    trades = all_trades(results)
    years = sorted({t.entry_year for t in trades})
    rows = []
    for y in years:
        yt = [t for t in trades if t.entry_year == y]
        stats = _fold_stats(str(y), yt)
        bull = [t for t in yt if t.regime == "bull"]
        bear = [t for t in yt if t.regime == "bear"]
        rows.append({
            "fold": y,
            "n_trades": stats.n_trades,
            "win_pct": round(stats.win_pct, 1),
            "expectancy_r": round(stats.expectancy_r, 3),
            "profit_factor": round(stats.profit_factor, 2) if np.isfinite(stats.profit_factor) else "inf",
            "max_dd_r": round(stats.max_drawdown_r, 2),
            "n_bull_entries": len(bull),
            "n_bear_entries": len(bear),
            "is_critical_bear_fold": y == config.CRITICAL_BEAR_FOLD,
        })
    return rows


def concentration_check(results: dict) -> dict:
    trades = all_trades(results)
    if not trades:
        return {"n_trades": 0, "expectancy_before": None, "expectancy_after_removal": None, "passes": False}
    r = sorted((t.r_multiple for t in trades), reverse=True)
    before = float(np.mean(r))
    n_remove = min(config.CONCENTRATION_TOP_N_REMOVED, len(r))
    remainder = r[n_remove:]
    after = float(np.mean(remainder)) if remainder else float("-inf")
    return {
        "n_trades": len(r),
        "n_removed": n_remove,
        "expectancy_before": round(before, 3),
        "expectancy_after_removal": round(after, 3),
        "passes": after > config.MIN_EXPECTANCY_AFTER_REMOVAL_R,
    }


def evaluate_bar(results: dict) -> dict:
    """The binding, pre-registered VALIDATED / NOT VALIDATED verdict.
    Every threshold here is read from config.py, fixed before any run."""
    trades = all_trades(results)
    overall = _fold_stats("aggregate", trades)

    bear_trades = [t for t in trades if t.entry_year == config.CRITICAL_BEAR_FOLD]
    bear_stats = _fold_stats(str(config.CRITICAL_BEAR_FOLD), bear_trades)
    bear_sample_conclusive = len(bear_trades) >= config.MIN_BEAR_FOLD_SAMPLE

    conc = concentration_check(results)

    checks = {
        "aggregate_expectancy_ge_bar": overall.expectancy_r >= config.MIN_AGGREGATE_EXPECTANCY_R,
        "trade_count_ge_bar": overall.n_trades >= config.MIN_OOS_TRADES,
        "bear_fold_sample_conclusive": bear_sample_conclusive,
        "bear_fold_expectancy_ge_zero": (
            bear_sample_conclusive and bear_stats.expectancy_r >= config.MIN_BEAR_FOLD_EXPECTANCY_R
        ),
        "not_outlier_driven": conc["passes"],
    }
    validated = all(checks.values())

    return {
        "validated": validated,
        "checks": checks,
        "aggregate": {
            "n_trades": overall.n_trades,
            "win_pct": round(overall.win_pct, 1),
            "expectancy_r": round(overall.expectancy_r, 3),
            "profit_factor": round(overall.profit_factor, 2) if np.isfinite(overall.profit_factor) else "inf",
            "max_dd_r": round(overall.max_drawdown_r, 2),
        },
        "bear_fold": {
            "year": config.CRITICAL_BEAR_FOLD,
            "n_trades": bear_stats.n_trades,
            "expectancy_r": round(bear_stats.expectancy_r, 3),
            "conclusive_sample": bear_sample_conclusive,
        },
        "concentration": conc,
        "bar": {
            "min_aggregate_expectancy_r": config.MIN_AGGREGATE_EXPECTANCY_R,
            "min_oos_trades": config.MIN_OOS_TRADES,
            "min_bear_fold_sample": config.MIN_BEAR_FOLD_SAMPLE,
            "min_bear_fold_expectancy_r": config.MIN_BEAR_FOLD_EXPECTANCY_R,
            "concentration_top_n_removed": config.CONCENTRATION_TOP_N_REMOVED,
            "min_expectancy_after_removal_r": config.MIN_EXPECTANCY_AFTER_REMOVAL_R,
        },
    }


def render_markdown(results: dict) -> str:
    lines = []
    lines.append("# longbot-pairs -- walk-forward OOS results\n")

    lines.append("## Pair universe results\n")
    lines.append("| Pair | Status | Cointegrated (formation windows) | Trades | Win% | Expectancy (R) | PF | Notes |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in per_pair_table(results):
        if row["status"] == "SKIPPED":
            lines.append(f"| {row['pair']} | SKIPPED | - | - | - | - | - | {row['reason']} |")
        else:
            lines.append(
                f"| {row['pair']} | OK | {row['coint_freq_pct']}% ({row['n_formation_tested']} tested) | "
                f"{row['n_trades']} | {row['win_pct']}% | {row['expectancy_r']} | {row['profit_factor']} | "
                f"{row['history']} |"
            )

    lines.append("\n## Per calendar-year OOS fold\n")
    lines.append("| Fold | Trades | Win% | Expectancy (R) | PF | Max DD (R) | Bull entries | Bear entries |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in per_fold_table(results):
        marker = " **(critical bear fold)**" if row["is_critical_bear_fold"] else ""
        lines.append(
            f"| {row['fold']}{marker} | {row['n_trades']} | {row['win_pct']}% | {row['expectancy_r']} | "
            f"{row['profit_factor']} | {row['max_dd_r']} | {row['n_bull_entries']} | {row['n_bear_entries']} |"
        )

    verdict = evaluate_bar(results)
    lines.append("\n## Concentration check (top-5-removed)\n")
    c = verdict["concentration"]
    lines.append(f"- Trades: {c['n_trades']}, removed top {c.get('n_removed', 0)}")
    lines.append(f"- Expectancy before removal: {c['expectancy_before']} R")
    lines.append(f"- Expectancy after removal: {c['expectancy_after_removal']} R "
                 f"(bar: > {config.MIN_EXPECTANCY_AFTER_REMOVAL_R} R) -> "
                 f"{'PASS' if c['passes'] else 'FAIL'}")

    lines.append("\n## Aggregate OOS vs. pre-registered bar\n")
    a = verdict["aggregate"]
    lines.append(f"- Aggregate trades: {a['n_trades']} (bar: >= {config.MIN_OOS_TRADES})")
    lines.append(f"- Aggregate expectancy: {a['expectancy_r']} R (bar: >= {config.MIN_AGGREGATE_EXPECTANCY_R} R)")
    lines.append(f"- Win%: {a['win_pct']}%, Profit factor: {a['profit_factor']}, Max DD: {a['max_dd_r']} R")

    b = verdict["bear_fold"]
    lines.append(f"\n- {b['year']} bear fold trades: {b['n_trades']} "
                 f"(bar: >= {config.MIN_BEAR_FOLD_SAMPLE} for a conclusive sample)")
    lines.append(f"- {b['year']} bear fold expectancy: {b['expectancy_r']} R (bar: >= {config.MIN_BEAR_FOLD_EXPECTANCY_R} R)")

    lines.append("\n## Checks\n")
    for name, passed in verdict["checks"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")

    lines.append(f"\n## VERDICT: {'VALIDATED' if verdict['validated'] else 'NOT VALIDATED'}\n")
    return "\n".join(lines)
