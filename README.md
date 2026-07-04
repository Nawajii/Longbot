# Longbot — a long-only "pullback hunter" for Binance spot

A single-user, **long-only spot** crypto bot. It looks for one thing: a coin in a
healthy uptrend that has just *paused and cooled off* (a pullback), and buys the
turn back up — never the falling knife, never the overheated peak. One position
at a time. No shorting, no futures, no margin, no leverage — ever (a deliberate,
non-negotiable constraint).

> **Status: RETIRED.** The pullback/trend-following engine below failed
> walk-forward out-of-sample validation (see `run_validation.py`) — its
> apparent edge did not survive being read forward through unseen years and
> did not hold up in the 2022 bear: it was pure beta wearing a strategy's
> clothes, not a real edge. It is kept in this repo for reference and because
> the new strategy below reuses its proven backtest infrastructure. **Do not
> trade the pullback/scoring_engine logic.** See "Leverage-flush mean
> reversion" further down for the current, active strategy under test.

---

## Plain-language tour of what's here

| File | What it does | In plain words |
|---|---|---|
| `longbot/indicators.py` | Hand-rolled EMA, RSI, MACD, ADX, ATR, Bollinger, Stochastic, Williams %R, CCI, OBV, VWAP, Aroon, relative volume. | The measuring instruments. No TA-Lib — just pandas/numpy, so nothing is a black box. |
| `longbot/scoring_engine.py` | Four independent layers → one `classify_setup()` verdict. | The brain. Decides "buy candidate" vs "overheated peak — skip". |
| `longbot/trade_setter.py` | `build_ticket()` → exact entry, stop, size, R-levels. | Turns "this one" into an exact, placeable order with the risk pre-measured. |
| `longbot/pair_picker.py` | The 5-stage funnel (regime → liquidity → trend → pullback → rank). | Scans many coins, throws out everything unsafe, ranks what's left, picks one. |
| `backtest.py` | Bar-by-bar simulator with no lookahead, fees, slippage, fill modelling. | The truth machine. Tells you if any of this would have worked. |
| `run_real_backtest.py` | Fetches Binance klines (or a CSV, or synthetic) and runs the backtest. | The button you press to run the experiment. |
| `longbot/volume_profile.py` | Rolling volume-at-price: POC, Value Area, HVN/LVN, `location()`. | The **location** gear — is the pullback resting on real support, or hanging in a void? |
| `longbot/config.py` | Settings + secrets via `.env`. | Where keys and risk limits live (keys never go in git). |

### The one rule that makes this strategy different
**Trend and momentum are read in *opposite* directions.**
- Strong trend **+ exhausted/overbought momentum + price stretched** = `PEAK_EXTENDED` → **skip**. This is the trap most beginners fall into (buying because "number go up").
- Strong trend **+ cooled momentum + a shallow pullback that held support**, with volume confirming = `PULLBACK_IN_UPTREND` → **the candidate**.

Momentum is never added in as "more bullish = better". A high RSI is treated as a
*danger* signal near highs, not a buy signal.

---

## How to run the backtest

```bash
pip install -r requirements.txt

# 1) Prove the plumbing runs with no network (NOT an edge test):
python run_real_backtest.py --synthetic

# 2) The real experiment — needs network access to api.binance.com:
python run_real_backtest.py --symbol BTCUSDT --timeframe 1h --limit 1000
python run_real_backtest.py --symbol ETHUSDT --timeframe 4h --limit 1000

# 3) If your network blocks Binance, download klines to a CSV elsewhere and:
python run_real_backtest.py --csv data/BTCUSDT_1h.csv --timeframe 1h

# Enable the regime gate (only trade when BTC is risk-on):
python run_real_backtest.py --symbol ETHUSDT --timeframe 1h --regime-symbol BTCUSDT

# Fetch YEARS of history (pages past the 1000-bar cap) for a real test:
python run_real_backtest.py --symbol BTCUSDT --timeframe 4h --start 2020-01-01 --regime-symbol BTCUSDT

# Multi-pair x multi-regime sweep (BULL/BEAR/CHOP) with an aggregated summary:
python run_scenarios.py --timeframe 4h
```

`run_scenarios.py` runs the default 20 pairs across three named windows
(BULL / BEAR / CHOP), skips pairs with insufficient history, prints each
per-pair report, then an aggregated summary: total trades, overall win rate,
overall expectancy (R), and a per-scenario breakdown. This is the test that
separates a real edge from "it was just a bull market".

### The Volume Profile "location" layer + experiment
The blind pullback engine knew direction, timing and risk but not **location** —
it would buy a pullback without checking whether real support sat beneath it.
`volume_profile.py` adds that gear: a rolling **volume-at-price** histogram whose
peaks (High-Volume Nodes) are equilibrium/support zones and whose valleys
(Low-Volume Nodes) are voids. It is inserted as a toggle between the trend and
momentum layers, and it can re-anchor the stop just beyond the supporting node
(with an ATR floor so the stop is never absurdly tight).

> **Approximation, stated plainly:** true volume-at-price needs tick data. We
> only have OHLCV candles, so each candle's volume is spread *evenly* across the
> price bins its high–low range spans. That is the standard OHLCV approximation
> and nothing more — it cannot see intrabar structure. The profile at bar `i`
> uses only bars `≤ i` (there is a unit test proving no lookahead).

```bash
# The pre-registered location experiment (primary tag test + secondary A/B):
python run_location_test.py --timeframe 4h
```
It prints the hypothesis, then: **(primary)** every blind-engine pullback tagged
`AT_NODE` vs `IN_VOID` with expectancy/win-rate/count for each — does location
*discriminate*? — and **(secondary)** the full new gearbox (location gate +
node stops) head-to-head against the blind engine, aggregate and per scenario.
Cells with <30 trades are flagged NOT CONCLUSIVE. Fixed VP config, no sweeping.

### The exit-variant grid (final engine experiment)
`run_exit_grid.py` asks whether the *exit* was truncating winners. On identical
located (AT_NODE) entries and an identical 2×ATR node-anchored initial stop, it
runs four exits — `TRAIL_2` (trail 3×ATR, control), `TRAIL_4`, `TRAIL_6`, and
`TIME_12` (exit at market after 12 bars) — over the same 20×3×4h sweep, so any
difference is attributable to the exit alone (a unit test proves the entries are
identical). It reports trades / win% / avg win R / avg loss R / expectancy /
bars held / giveback (peak→exit in R) / worst drawdown, per-scenario expectancy,
and a **binding** verdict against a success bar fixed in advance (expectancy
≥ +0.15 R, ≥ 60 trades, not positive-only-in-BULL).

```bash
python run_exit_grid.py --timeframe 4h
```

### Multi-scale equilibrium gate (final test of the trend line)
`longbot/multiscale.py` + `run_multiscale.py` add a **24h (daily) trend + macro
equilibrium gate** ANDed onto the 4h located entry: trade only when the daily is
in an uptrend AND price is at/above the daily value-area support, AND the 4h
shows the located pullback. Exits are the two prior leaders (TRAIL_6, TARGET_2R).

> **Honest note:** this multi-scale gate is still, mechanically, a **filter** on
> the existing entry engine. A filter can only *remove* trades — it cannot create
> edge. The test is whether multi-scale *alignment* isolates a subset that
> happens to carry genuine edge; the base-rate expectation is that it does not.
> No-lookahead is enforced on **both** scales — the daily gate reads only
> *completed* daily candles (shifted one day), proven by a unit test.

```bash
python run_multiscale.py            # deep walk-forward, binding verdict
```

The report **always** prints its assumptions (fees, slippage, risk %, stop
distances) and compares your result against **buy-and-hold** and a
**buy-the-peak** baseline. If the strategy can't beat buy-and-hold net of costs,
that is the finding — and the honest conclusion is to stop, not to deploy.

CSV format: columns `time,open,high,low,close,volume` (`time` as ms-epoch or ISO).

---

## How honest is the backtest? (the assumptions, stated plainly)

- **No lookahead.** A decision on a closed candle `i` uses only candles `≤ i`.
  An order placed at the close of `i` can fill no earlier than `i+1`.
- **A signal is not a trade.** The default entry is a *buy-stop above the
  pullback high* — it only fills if price actually rises through it within a few
  bars. Orders that never trigger are counted as **misses**. (In testing, miss
  rates were high — that is a real, important cost of this entry style.)
- **Costs always applied:** a taker fee on both sides + slippage against us on
  every fill.
- **Pessimistic intrabar rule:** while in a trade, each bar is checked for a
  stop-out *before* the trailing stop is allowed to ratchet up — so one bar can
  never both save you and stop you. We assume the worse ordering.

### Things that will bite you, stated up front
- **The edge is unproven.** Everything here is a hypothesis until a multi-pair,
  multi-timeframe real-data backtest says otherwise.
- **At small capital, fees and the ~$10 minimum order dominate.** A $200 account
  paying 0.1% per side on small, frequent trades can bleed out on costs alone.
  The trade setter *flags* trades that can't clear the minimum rather than
  silently shrinking your risk.
- **Synthetic data proves nothing about profit** — only that the code runs.
- **In a strong bull market, buy-and-hold often beats this.** An in-and-out
  strategy with a high miss rate gives up a lot of the trend. That's a feature
  of the honesty, not a bug in the report.

---

## Architecture (the 5 layers + circuit breaker)

1. **Pair Picker** *(built)* — deterministic funnel that picks one pair.
2. **Trade Setter** *(built)* — exact entry/stop/size ticket.
3. **Mitigator** *(designed, not yet live)* — trails the stop up-only and places
   an exchange-native stop as a dead-man's switch.
4. **Monitor** *(not yet built)* — scheduling, logging to SQLite, restarts.
5. **Coroner** *(not yet built)* — Gemini Flash writes a post-mortem on losing
   trades and reports **to you**. It never feeds back into an auto-retuning loop.
6. **Risk Governor** *(designed in `config.py`, not yet wired)* — a dumb hard
   kill-switch: halts on max daily loss, max drawdown, N consecutive losses, or
   an API fault.

**No LLM ever decides a trade.** Selection, entry, and exit are all deterministic
Python. The LLM lives only in the post-trade Coroner.

---

## What's NOT built yet (honest backlog, in build order)
- Phase 2 finish: run the **real-data** backtest across several pairs/timeframes
  and decide go/no-go. *(Blocked in the current dev sandbox because outbound
  access to `api.binance.com` is denied by network policy — run it where Binance
  is reachable, or feed it CSVs.)*
- Phase 3: paper/testnet execution (`ccxt` pointed at `testnet.binance.vision`).
- Phase 4: Monitor + Risk Governor + Coroner + Telegram alerts.
- Phase 5: live, human-in-the-loop (bot suggests, you confirm) — only after a
  proven edge and only at capital where fees are a small fraction of returns.

## Security (do this before any live key touches the bot)
- Binance API key with **trade enabled, withdrawal DISABLED**.
- **IP-whitelist** the key to the machine running the bot.
- Keys in `.env` / a secrets store — **never** committed. `.env` is gitignored.

## Tests
```bash
python -m pytest -q
```

---

## Leverage-flush mean reversion (long-only) — the CURRENT strategy under test

A new, separate strategy module (`longbot/funding.py`, `longbot/flush_engine.py`,
`run_flush_validation.py`) that does **not** import or depend on
`scoring_engine.py` / `trade_setter.py` — the retired pullback engine's brain.
It reuses only this repo's proven *infrastructure*: `fetch_or_cache` (deep
Binance kline fetch + CSV cache), `btc_regime_labels` (BULL/BEAR/CHOP macro
regime), and the 0.10%/side fee + 0.05%/side slippage cost model.

### The idea, and its data reality

**Thesis:** in crypto perpetuals, a heavily long-leveraged crowd pays rising
funding to hold. A downward nudge can trigger cascading forced liquidations —
margin calls dump into thin liquidity and price *overshoots*. That selling is
mechanical, not informational, so price tends to snap back once liquidations
exhaust. We buy the flush, long-only spot, betting on the mechanical overshoot
reverting. Unlike the retired pullback engine, this edge is supposed to be
**direction-agnostic** — it trades overshoots, not trend — so it must not bleed
in a bear market. That's the whole test.

**Data reality (do not fight this):** Binance does not offer free historical
liquidation data, and open interest history is capped at ~30 days. What *is*
freely available with deep history is the **funding rate**
(`/fapi/v1/fundingRate`, 8-hourly, back to ~2020 for most perps). So:
- **Positioning/leverage signal = funding rate** (real, full history).
- **Flush trigger = the price/volume footprint of a liquidation cascade**,
  read off the spot klines we already fetch: a violent range + volume spike
  with a long lower wick and a strong close (absorption) — the signature that
  distinguishes a mechanical overshoot that gets bought back from a real,
  information-driven crash that keeps falling.

We never trade the perp itself, never short, never use leverage. Funding is
used **only as a signal**; every position is a plain spot long. This keeps the
halal, long-only constraint fully intact.

### Fixed parameters (pre-registered — see `longbot/flush_engine.FlushParams`)

| Condition | Rule |
|---|---|
| Spring loaded | funding's trailing 90-day percentile rank ≥ 0.85 (top 15%) |
| Flush trigger | true range ≥ 2× trailing-20 avg **AND** volume ≥ 2× trailing-20 avg **AND** lower wick ≥ 1.5× body **AND** close in the upper third of the bar's range |
| Entry | next bar's **open**, after both conditions are confirmed on a *closed* bar |
| Target | +1.5×ATR (ATR frozen at the signal bar) **OR** funding ≤ its own trailing median |
| Stop (mandatory) | −2.0×ATR (frozen at the signal bar) — the falling-knife protection |
| Time stop | 30 bars |
| Costs | 0.10%/side fee + 0.05%/side slippage, same as the retired engine |

### No-lookahead, on two axes

- **Price side:** the flush candle's wick/close-location is only fully known
  at that candle's *close*, so entry is always the **next** bar's open — never
  the signal bar itself. Trailing range/volume baselines use the 20 bars
  *strictly before* the candle (`shift(1).rolling(20)`), so a bar can never
  inflate its own baseline.
- **Funding side:** a bar is aligned to the **last completed** funding print at
  or before it (`longbot/funding.align_to_bars`, a backward `merge_asof` — a
  bar can never see a print stamped after it). The percentile rank and
  trailing median of a print use only that print and the ones before it.
- Both are proven mechanically in `tests/test_no_lookahead.py`-style truncation
  tests: `tests/test_funding_alignment.py` and the
  `test_full_engine_truncation_invariance` test in `tests/test_flush_engine.py`
  show that truncating (or replacing) the future never changes a past decision.

### Universe

The perp-listed subset of the existing 20-pair `DEFAULT_PAIRS` universe
(`run_scenarios.DEFAULT_PAIRS`). Availability is checked **at run time**, not
guessed offline: any symbol with no USDT-M perpetual (no funding history) is
skipped and reported by name, as is any symbol with too little combined
spot+funding history to warm up the 90-day percentile window.

### Binding success bar (fixed before any result is seen)

VALIDATED only if **all** of the following hold on the out-of-sample
walk-forward run (folds: 2021, 2022, 2023, 2024, 2025+; strategy fixed, no
per-fold refitting):

- aggregate OOS expectancy ≥ **+0.15 R** net of costs
- ≥ **100 OOS trades** aggregate (if flush signals are too rare to reach that
  across the universe and years, the honest conclusion is "trades too rarely
  to be viable," reported as such)
- **2022 bear-fold expectancy ≥ 0** on a conclusive (≥30 trade) sample — a
  thin sample is reported as *inconclusive*, not quietly treated as a pass
- **not outlier-driven:** removing the top 5 trades leaves expectancy > **+0.05 R**
- **not one-year-only:** the edge appears in ≥ 2 separate OOS folds

Any single failure ⇒ **NOT VALIDATED**, reported plainly, no re-tuning against
the OOS results.

### The MAE/MFE diagnostic (descriptive only — does not feed back into the strategy)

For every trade, `run_flush_validation.py` reports Maximum Adverse/Favorable
Excursion in R, and specifically: how far stopped-out losers got toward target
before reversing (is the stop too tight?), whether stopped-out trades tracked
*past* the stop recovered or kept falling (the direct empirical read on
whether the hard stop is cutting mechanical noise or real crashes), and how far
winners ran past the fixed +1.5×ATR target (is it leaving R on the table?).
This turns a pass/fail verdict into a diagnosis of *why*.

### Reproduce

```bash
pip install -r requirements.txt
python -m pytest tests/ -v                 # no network needed — synthetic + hand-built OHLCV/funding
python run_flush_validation.py             # needs api.binance.com (spot) + fapi.binance.com (funding)
```

This was built and tested in a sandboxed environment with no outbound network
access — it has **not** been run against real data. Do not trust any numbers
that are not freshly generated by running it yourself. Reads the bar
mechanically: if 2022 bleeds, or the edge only shows up in one fold, it is
**NOT VALIDATED** regardless of how the aggregate number looks.
Covers the indicators, the PEAK-vs-PULLBACK logic, the position sizing /
min-notional rules, and the backtest's funnel/fee accounting.
