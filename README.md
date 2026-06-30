# Longbot — a long-only "pullback hunter" for Binance spot

A single-user, **long-only spot** crypto bot. It looks for one thing: a coin in a
healthy uptrend that has just *paused and cooled off* (a pullback), and buys the
turn back up — never the falling knife, never the overheated peak. One position
at a time. No shorting, no futures, no margin, no leverage — ever (a deliberate,
non-negotiable constraint).

> **Status: Phase 1–2 foundation built; edge UNPROVEN.**
> The brain, the order math, and an honest backtest exist and are tested. Whether
> the strategy actually makes money is an open question that only a real-data
> backtest can answer. **Do not risk real money** until that backtest shows an
> edge over simply buying and holding, *after* fees and slippage.

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
Covers the indicators, the PEAK-vs-PULLBACK logic, the position sizing /
min-notional rules, and the backtest's funnel/fee accounting.
