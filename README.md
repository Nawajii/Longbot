# longbot-pairs

A long-only cointegration pairs-trading strategy, backtested with a
pre-registered, walk-forward, out-of-sample methodology. This is a *new*
project, not the prior single-asset pullback bot -- that project was
rigorously retired after failing walk-forward validation in the 2022 bear
(it had no real edge: it only made money in bull markets, i.e. it was pure
beta wearing a strategy's clothes).

## The idea

**Cointegration pairs trading.** Two assets driven by the same fundamentals
(e.g. two L1 tokens) tend to move together. Sometimes their price
relationship diverges temporarily -- a liquidity event, not new information
-- and because there's no informational reason for the gap, real capital
tends to pull them back together. Betting on that reversion is grounded in
**cointegration** (Engle-Granger), not chart patterns. This edge is
**regime-agnostic by design**: it trades the *relationship between two
assets*, not market direction, so it should (if real) work in bear and
chop, not just bull. That's the whole thesis, and the success bar below
requires it explicitly (a bear-market fold with expectancy >= 0 is
non-negotiable).

## The honest weakness (long-only)

True pairs trading is **market-neutral**: long the underperformer AND short
the outperformer, which cancels market risk. This system is **long-only**
(a religious/halal constraint: no shorting, no futures, no margin). It can
only take the long leg -- buy whichever asset is the relative
*underperformer* and bet it catches up. This means:

- We are **not market-neutral**. We still carry market exposure.
- If both assets fall together, we can lose even while the spread is
  reverting exactly as predicted.
- This materially weakens the strategy versus textbook stat-arb.

We test it anyway, honestly, within the constraint, and the bar below is
the only thing that decides whether it survives.

## Pair universe (fixed, economically motivated -- not data-mined)

Testing all `C(20,2)` combinations of a coin list and keeping whichever
happen to look cointegrated is the multiple-comparisons trap: ~10 of ~190
random pairs will show p<0.05 by pure chance even if none are truly
related. Pre-selecting by economic kinship is what makes a pass meaningful.
Only these 10 pairs are ever tested:

| Pair | Why |
|---|---|
| ETH/BTC | the two majors |
| SOL/AVAX | competing high-throughput L1s |
| AVAX/DOT | L1 smart-contract peers |
| ADA/DOT | L1 peers, similar era |
| NEAR/SOL | L1 peers |
| LTC/BCH | BTC-derived payment coins |
| UNI/AAVE | DeFi blue chips |
| LINK/ETH | ETH-beta infrastructure |
| MATIC(POL)/ETH | ETH-scaling, ETH-beta (ticker spliced across the 2024 MATIC->POL rebrand) |
| ATOM/DOT | interoperability L1s |

If a pair lacks sufficient overlapping Binance history it is skipped, and
the report says exactly why (see `report.py::per_pair_table`, `status:
SKIPPED`).

## Binding success bar (fixed before any result is seen)

The strategy is **VALIDATED** only if ALL of the following hold on the
out-of-sample walk-forward run:

- aggregate OOS expectancy >= **+0.15 R** net of costs
- >= **100 trades** OOS aggregate (if it can't reach 100 trades across 6
  years and 10 pairs, the honest conclusion is "trades too rarely to
  matter," reported as such, not fudged)
- **regime-agnostic**: expectancy >= 0 in the **2022 bear** fold on a
  conclusive sample (>= 30 trades) -- non-negotiable, it's the entire
  reason this strategy was chosen over the retired one
- **not outlier-driven**: removing the top 5 trades leaves expectancy >
  **+0.05 R**

Any single failure => **NOT VALIDATED**, reported plainly. No re-tuning
thresholds/windows/pairs against OOS results -- one pre-registered run,
read once, believed. All of these numbers live in `pairs_pipeline/config.py`
and nowhere else.

## Design decisions made explicit (things the prompt left open)

- **Timeframe: daily, not 4h.** The formation window is specified as "90
  days" -- daily bars make that exactly 90 observations with no rounding.
  Daily closes also avoid the strong intraday autocorrelation/microstructure
  noise that biases ADF tests on 4h crypto data, and match the resolution
  used in the pairs-trading literature.
- **Roll cadence: daily re-estimation.** The formation window's *length*
  (90d) is fixed by the prompt; how often it's re-estimated is not. This
  implementation refits a trailing 90-day formation window every day a pair
  is flat -- the standard convention -- and then **freezes** that specific
  fit's beta/mean/std for the entire lifetime of any trade it opens (never
  re-estimated mid-trade). Because the window is always strictly prior to
  the day it's used for, every single day of this multi-year run is
  out-of-sample by construction; "per calendar-year fold" reporting slices
  that continuous OOS stream by the entry year of each trade, rather than
  doing one single train/test split.
- **One position at a time, per pair (not globally).** The prompt says
  "one position at a time (matches constraint and keeps it debuggable)."
  We read that as: each of the 10 pairs runs its own independent
  single-slot state machine (no pyramiding, no doubling up on the same
  pair), but different pairs can be open simultaneously. This is required
  for the success bar's own arithmetic to be satisfiable -- the bar
  explicitly reasons about reaching >=100 trades "across 6 years and 10
  pairs," which only scales with pair count if pairs trade independently.
  It is also still "debuggable": each pair is a simple, independent, fully
  cash-settled spot state machine with no margin or cross-pair netting.
- **1R = entry-to-stop spread distance, translated into an actual price.**
  We hold a position in one asset, not the spread itself, so "R" has to be
  expressed in that asset's price terms to be a meaningful, realized P&L
  unit. We take the entry-day price of the *other* leg as fixed and ask:
  what price would the *held* asset need to reach for z to hit the stop
  threshold (|z| = 3.5)? That implied price, versus the actual entry
  price, is 1R. See `engine.py::_implied_stop_price`.
- **Execution timing.** Signal and fill both use the same daily bar's
  close; fees (0.10%/side) + slippage (0.05%/side) are the disclosed model
  of execution cost/uncertainty. This is a standard daily-bar backtest
  simplification, distinct from the no-lookahead hazard around formation
  *parameters*, which is structurally prevented (see below).
- **Scope**: this is a per-trade expectancy/R study, not a capital-
  constrained portfolio equity simulation. With up to 10 pairs able to be
  open simultaneously, real position sizing/leverage-across-concurrent-
  trades would need separate capital allocation rules before this could be
  traded live -- that is explicitly out of scope here.

## No-lookahead (the #1 correctness risk)

Every entry decision made on day *t* uses ONLY a formation window fit on
data strictly before *t* (`cointegration.py::formation_for_index`, rows
`[i-90, i)`, never row `i`). Once a trade opens, its beta/mean/std are
frozen for its whole life -- never re-estimated with newer data.

This is not just asserted, it's tested mechanically in
`tests/test_no_lookahead.py`:
- truncating the price history right after a decision day produces an
  identical formation result to using the full history
- **replacing the entire future with different random data** does not
  change a past formation result at all
- mutating *only* the decision day's own price does not change that day's
  formation result (since it isn't in the window)
- replaying the full engine on a truncated history vs. the full history
  produces byte-identical trades for everything that completed before the
  truncation point

## Project layout

```
pairs_pipeline/
  config.py         pre-registered constants (universe, thresholds, bar)
  data_fetch.py      Binance kline fetch + CSV cache (reused design)
  regime.py          BTC 200-EMA bull/bear tagging (reused design)
  cointegration.py   formation-window OLS + ADF, the no-lookahead choke point
  engine.py          per-pair day-by-day entry/exit state machine
  walkforward.py     runs every pair, tags trades by fold year + regime
  report.py          per-pair / per-fold / aggregate tables, concentration
                      check, VALIDATED / NOT VALIDATED verdict
tests/               unit tests, esp. no-lookahead proofs
scripts/run_backtest.py   the one command to reproduce a full run
data_cache/          fetched OHLCV, CSV-cached (gitignored, regenerated)
results/             report.md + trades.csv from the last run (gitignored)
```

## Reproduce

Requires outbound network access to `api.binance.com` (this repo was built
and committed from a sandboxed environment with no such access -- the code
has not been executed against real data; do not trust any numbers that
aren't freshly generated by running this yourself).

```bash
pip install -r requirements.txt
python -m pytest tests/ -v          # no network needed -- all synthetic data
python scripts/run_backtest.py      # fetches/caches Binance data, runs the walk-forward backtest
```

Output: the full report prints to stdout, and is also written to
`results/report.md` and `results/trades.csv`. The process exits 0 if
VALIDATED, 1 if NOT VALIDATED.

First run fetches and caches ~14 symbols of daily history back to each
symbol's Binance listing date (`data_cache/*.csv`); subsequent runs only
fetch new bars.

## Reading the verdict

If it fails the bear fold, it fails -- regardless of how good the aggregate
number looks. No reinterpretation, no softening, no re-tuning and re-
running. That's the entire point of pre-registering the bar before looking
at results.
