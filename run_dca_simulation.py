"""ADAPTIVE DCA (volatility harvesting) -- an accumulation simulator, NOT a
trade backtester. There are no entries/exits/stops/expectancy, and NOTHING
is ever sold. This prints an honest comparison, not a pass/fail "edge"
verdict -- see FRAMING QUESTIONS below.

THE IDEA (behavioral, not an alpha edge): retail investors systematically
buy during euphoria and sell during panic, which is why the average
investor underperforms the assets they hold. This strategy does the
opposite on the buy side only: it buys MORE on red days, skips buying on
euphoric green days, and never sells. It is not timing tops/bottoms; it is
mechanically harvesting a documented behavioral tendency (buy-low
discipline) that persists precisely because it's a behavior, not a priced
inefficiency. Long-only, spot, never-sell -> trivially halal (no shorting,
riba, leverage, or even a sell decision).

THE PRE-REGISTERED LADDER (fixed -- see longbot/accumulation.LadderParams):
    r > +5%           -> $0   (skip -- euphoria)
    -2% <= r <= +5%    -> $5   (base)
    -5% <  r <  -2%     -> $7
    -8% <  r <= -5%      -> $10
    -12% < r <= -8%       -> $15
    r <= -12%              -> $25
  0.10% spot buy fee on every purchase. No slippage modelled at this size.

THREE APPROACHES COMPARED (same asset, same period):
    1. Adaptive DCA           -- the ladder above.
    2. Flat DCA                -- constant $5/day, same execution days as
                                   adaptive (the honest, like-for-like fight).
    3. Lump-sum                -- adaptive's TOTAL deployed capital, all on
                                   day 1 (context benchmark, NOT the fair
                                   fight -- lump-sum tends to win when markets
                                   rise more often than they fall).
    (+ Flat DCA, same total capital as adaptive, spread evenly across
       adaptive's ACTUAL buy days -- isolates variable SIZING from the fact
       that adaptive deploys a different total amount than flat.)

FRAMING (NOT a trading pass/fail -- read plainly, do not force a verdict):
    1. Did Adaptive achieve a LOWER average cost basis than Flat? By how much?
    2. Did Adaptive achieve a better money-weighted return (XIRR) than Flat,
       net of fees?
    3. Was the advantage concentrated in the volatile/BEAR regime, as the
       thesis predicts?
    4. Is the advantage large enough to justify the added complexity over
       just running Flat DCA?
Losing to lump-sum in a rising market is NOT a failure -- a dip-buyer that
holds cash for dips will trail a strategy that was never holding cash. That
is the strategy working as designed.

ASSETS: BTCUSDT and ETHUSDT (majors -- the never-sell "will this asset
survive and eventually rise" assumption is defensible here) plus ETCUSDT
as a CAUTIONARY alt: it peaked near $176 in 2021 and has spent years since
grinding along a small fraction of that high, with no new ATH. It is
included specifically to demonstrate this strategy's core UNMANAGED risk:
"buy more on the way down" is a genuine disaster if the asset never
recovers, since this strategy has no mechanism to ever stop buying or
sell. Report it as a cautionary result, not a code failure.

Reuses this repo's proven fetch_or_cache (deep Binance kline fetch + CSV
cache) and btc_regime_labels (BULL/BEAR/CHOP). The simulator itself
(longbot/accumulation.py, longbot/dca_metrics.py) is a new, independent
module -- it does not import scoring_engine.py / trade_setter.py /
flush_engine.py, none of which apply to a strategy that never exits.

Run where Binance is reachable:  python run_dca_simulation.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from longbot import accumulation as acc
from longbot import dca_metrics as m
from run_deep_grid import btc_regime_labels, REGIMES
from run_real_backtest import fetch_or_cache

DEFAULT_ASSETS = ["BTCUSDT", "ETHUSDT", "ETCUSDT"]
CAUTIONARY_ALT = "ETCUSDT"
DEFAULT_START = "2017-01-01"
FOLD_YEARS = None   # printed dynamically per asset (whatever years are present)


def _money(x: float) -> str:
    return "n/a" if (x is None or np.isnan(x)) else f"${x:,.2f}"


def _pct(x: float) -> str:
    return "n/a" if (x is None or np.isnan(x)) else f"{x * 100:+.1f}%"


def _row(label: str, result: acc.DCAResult, df: pd.DataFrame) -> dict:
    final_price = float(df["close"].iloc[-1])
    values = m.portfolio_value_series(result, df)
    return dict(
        label=label,
        deployed=result.total_deployed,
        units=result.total_units,
        cost_basis=result.avg_cost_basis,
        terminal=result.terminal_value(final_price),
        xirr=m.result_xirr(result, df.index[-1], final_price),
        max_dd=m.max_drawdown_pct(values),
    )


def run_asset(symbol: str, timeframe: str, start: str, end: str | None,
             cache_dir: str, refresh: bool, regime_label: pd.Series) -> str:
    L = [f"\n{'=' * 100}", f" {symbol}", "=" * 100]

    df = fetch_or_cache(symbol, timeframe, start, end, cache_dir, refresh)
    if len(df) < 400:
        L.append(f" SKIPPED: only {len(df)} bars (< 400) -- insufficient history.")
        return "\n".join(L)

    L.append(f" history: {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} daily bars)")

    adaptive = acc.simulate_adaptive_dca(df, symbol)
    buy_indices = [b.execution_index for b in adaptive.buys]
    flat = acc.simulate_flat_dca(df, symbol, execution_indices=acc.executable_indices(len(df)))
    flat_same_capital = acc.simulate_flat_same_capital(
        df, symbol, total_capital=adaptive.total_deployed, buy_indices=buy_indices,
    )
    lump = acc.simulate_lump_sum(df, symbol, total_amount=adaptive.total_deployed)

    rows = [
        _row("Adaptive DCA", adaptive, df),
        _row("Flat DCA ($5/day, same days)", flat, df),
        _row("Flat DCA (same $ as adaptive, adaptive's buy-days)", flat_same_capital, df),
        _row("Lump-sum (adaptive's total, day 1)", lump, df),
    ]
    L.append("\n APPROACH COMPARISON:")
    L.append(f"   {'approach':<52}{'deployed':>12}{'units':>12}{'avg cost':>12}"
             f"{'terminal':>14}{'XIRR':>9}{'maxDD':>9}")
    for r in rows:
        L.append(f"   {r['label']:<52}{_money(r['deployed']):>12}{r['units']:>12.5f}"
                 f"{_money(r['cost_basis']):>12}{_money(r['terminal']):>14}"
                 f"{_pct(r['xirr']):>9}{_pct(r['max_dd']):>9}")

    # per-year deployment + cost basis, adaptive vs flat
    adaptive_by_year = m.deployed_by_bucket(adaptive, m.year_bucket)
    flat_by_year = m.deployed_by_bucket(flat, m.year_bucket)
    adaptive_basis_by_year = m.cost_basis_by_bucket(adaptive, m.year_bucket)
    flat_basis_by_year = m.cost_basis_by_bucket(flat, m.year_bucket)
    years = sorted(set(adaptive_by_year) | set(flat_by_year))
    L.append("\n PER-YEAR DEPLOYMENT + COST BASIS (adaptive vs flat):")
    L.append(f"   {'year':<6}{'adaptive $':>12}{'flat $':>12}{'adaptive basis':>16}{'flat basis':>14}")
    for y in years:
        L.append(f"   {y:<6}{_money(adaptive_by_year.get(y, 0.0)):>12}{_money(flat_by_year.get(y, 0.0)):>12}"
                 f"{_money(adaptive_basis_by_year.get(y, float('nan'))):>16}"
                 f"{_money(flat_basis_by_year.get(y, float('nan'))):>14}")

    # per-regime deployment + cost basis (BTC macro regime, reused/shared across assets)
    aligned_regime = regime_label.reindex(df.index, method="ffill").fillna("CHOP")
    regime_fn = m.regime_bucket_fn(aligned_regime)
    adaptive_by_regime = m.deployed_by_bucket(adaptive, regime_fn)
    flat_by_regime = m.deployed_by_bucket(flat, regime_fn)
    adaptive_basis_by_regime = m.cost_basis_by_bucket(adaptive, regime_fn)
    flat_basis_by_regime = m.cost_basis_by_bucket(flat, regime_fn)
    L.append("\n PER-REGIME (BTC macro BULL/BEAR/CHOP) DEPLOYMENT + COST BASIS:")
    L.append(f"   {'regime':<8}{'adaptive $':>12}{'flat $':>12}{'adaptive basis':>16}{'flat basis':>14}")
    for rg in REGIMES:
        L.append(f"   {rg:<8}{_money(adaptive_by_regime.get(rg, 0.0)):>12}{_money(flat_by_regime.get(rg, 0.0)):>12}"
                 f"{_money(adaptive_basis_by_regime.get(rg, float('nan'))):>16}"
                 f"{_money(flat_basis_by_regime.get(rg, float('nan'))):>14}")

    # framing questions, answered from the actual numbers above
    a_row, f_row = rows[0], rows[1]
    basis_edge_pct = ((f_row["cost_basis"] - a_row["cost_basis"]) / f_row["cost_basis"]
                      if f_row["cost_basis"] not in (0, None) and not np.isnan(f_row["cost_basis"]) else float("nan"))
    xirr_edge = (a_row["xirr"] - f_row["xirr"]) if not (np.isnan(a_row["xirr"]) or np.isnan(f_row["xirr"])) else float("nan")

    bear_adaptive = adaptive_by_regime.get("BEAR", 0.0)
    bear_flat = flat_by_regime.get("BEAR", 0.0)
    bear_share_adaptive = bear_adaptive / adaptive.total_deployed if adaptive.total_deployed > 0 else float("nan")
    bear_share_flat = bear_flat / flat.total_deployed if flat.total_deployed > 0 else float("nan")

    L.append("\n FRAMING READ (not a pass/fail -- see module docstring):")
    L.append(f"   1. cost basis: adaptive {_money(a_row['cost_basis'])} vs flat {_money(f_row['cost_basis'])}"
             f"  ({_pct(basis_edge_pct)} {'lower' if basis_edge_pct > 0 else 'higher'} for adaptive)")
    L.append(f"   2. XIRR: adaptive {_pct(a_row['xirr'])} vs flat {_pct(f_row['xirr'])}"
             f"  (edge: {_pct(xirr_edge)})")
    L.append(f"   3. capital deployed in BEAR: adaptive {_pct(bear_share_adaptive)} of its total vs "
             f"flat {_pct(bear_share_flat)} of its total"
             f"  ({'more concentrated in BEAR' if bear_share_adaptive > bear_share_flat else 'not more concentrated in BEAR'})")
    worth_it = (not np.isnan(basis_edge_pct)) and basis_edge_pct > 0.02 and (not np.isnan(xirr_edge)) and xirr_edge > 0
    L.append(f"   4. worth the complexity over flat DCA? {'YES -- a meaningful, consistent edge on both cost basis and XIRR.' if worth_it else 'MARGINAL/NO on these numbers -- flat DCA captures most of the benefit with far less complexity.'}")

    if symbol == CAUTIONARY_ALT:
        L.append(f"\n CAUTIONARY NOTE ({symbol}): this asset is included specifically because it fell hard")
        L.append(f"   from its all-time high and never recovered. A never-sell, buy-more-on-dips strategy")
        L.append(f"   has NO mechanism to protect against this -- it just keeps averaging down into a")
        L.append(f"   permanently impaired asset. Read the cost-basis/terminal-value numbers above with")
        L.append(f"   that risk in mind: a low cost basis is worthless if the asset itself never recovers.")

    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", nargs="*", default=DEFAULT_ASSETS)
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=None)
    ap.add_argument("--cache-dir", default="data")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    print(__doc__.split("Reuses this repo's proven")[0])

    print(f"[data] fetching regime BTCUSDT {args.timeframe} {args.start}->now (cached in {args.cache_dir}/)")
    regime_df = fetch_or_cache("BTCUSDT", args.timeframe, args.start, args.end, args.cache_dir, args.refresh)
    # btc_regime_labels' defaults (1200/180 bars) are calibrated for 4h bars
    # (~200-day EMA, ~30-day slope). On daily bars, 1 bar = 1 day, so the
    # equivalent day-counts are passed directly instead of the 4h defaults.
    if args.timeframe == "1d":
        regime_label = btc_regime_labels(regime_df, ema_bars=200, slope_bars=30)
    else:
        regime_label = btc_regime_labels(regime_df)

    for symbol in args.assets:
        print(run_asset(symbol, args.timeframe, args.start, args.end, args.cache_dir, args.refresh, regime_label))


if __name__ == "__main__":
    main()
