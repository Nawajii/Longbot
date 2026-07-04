"""Leverage-flush mean reversion (long-only) -- a NEW strategy module.

Deliberately independent of scoring_engine.py / trade_setter.py (the
retired pullback engine's brain). This file reuses only the project's
proven CONVENTIONS -- no-lookahead discipline, the fee/slippage constants,
ATR/true-range/volume helpers from indicators.py -- not its logic.

THE ECONOMIC THESIS: in crypto perpetuals, when the crowd is heavily
leveraged long, funding rises (longs pay shorts) and positioning becomes a
loaded spring. A downward nudge triggers cascading forced liquidations --
margin calls dump into thin liquidity and price OVERSHOOTS. Because that
selling is mechanical, not informational, price tends to snap back once
liquidations exhaust. We buy the flush, long-only, spot only.

THE DATA REALITY: Binance does not offer free historical liquidation or
deep open-interest data. Positioning is proxied by the funding rate
(longbot/funding.py, real, full history). The liquidation cascade itself
is proxied by its observable footprint in spot OHLCV: a violent range/
volume spike with a long lower wick and a strong close (absorption). See
README for the full writeup.

NO-LOOKAHEAD, made concrete:
  * "spring loaded" (funding percentile) and "flush trigger" (candle
    shape) are both evaluated on bar i using only data with timestamp<=i.
  * the flush candle's wick/close-location is only fully known at that
    candle's CLOSE, so entry is on bar i+1's OPEN -- one bar after the
    signal, never the signal bar itself.
  * ATR and the funding trailing-median used for exits are FROZEN at the
    signal bar (i) and never re-estimated mid-trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from longbot import indicators as ind


@dataclass(frozen=True)
class FlushParams:
    # costs -- same values as backtest.BacktestParams (0.10%/side fee, 0.05%/side slippage)
    fee_pct: float = 0.001
    slippage_pct: float = 0.0005

    # positioning / leverage signal (funding rate)
    funding_window_prints: int = 270      # ~90 days of 8h prints (3/day * 90)
    funding_pct_threshold: float = 0.85    # "top 15%" of the trailing distribution
    spring_loaded_lookback_bars: int = 6    # "within the last 24h" on 4h bars (24/4)

    # flush (liquidation-cascade proxy) candle trigger
    trailing_avg_window: int = 20
    flush_range_mult: float = 2.0
    flush_volume_mult: float = 2.0
    flush_wick_body_mult: float = 1.5
    flush_close_location_min: float = 2.0 / 3.0   # close in the upper third of the bar's range

    # exits
    atr_period: int = 14
    target_atr_mult: float = 1.5
    stop_atr_mult: float = 2.0
    time_stop_bars: int = 30

    timeframe: str = "4h"


@dataclass
class FlushTrade:
    entry_time: object
    exit_time: object
    entry_index: int
    exit_index: int
    entry_price: float           # fill price, after costs
    exit_price: float            # fill price, after costs
    stop_initial: float
    target_initial: float
    r_multiple: float
    bars_held: int
    exit_reason: str             # "target" | "funding_normalized" | "stop" | "time" | "end_of_data"
    fees_paid: float
    mae_r: float                 # worst excursion against us, in R, over the trade's actual life
    mfe_r: float                 # best excursion in our favor, in R, over the trade's actual life
    entry_funding_pct_rank: float = float("nan")
    # Diagnostic only (does not feed back into the strategy): for STOPPED
    # trades, the best R the price reached in the time_stop_bars bars AFTER
    # the stop, i.e. "did it recover, or keep falling". None for non-stops.
    post_stop_mfe_r: float | None = None
    entry_year: int = None        # filled in by run_flush_validation.py
    regime: str = None            # filled in by run_flush_validation.py (BULL/BEAR/CHOP)
    pair: str = ""                 # filled in by run_flush_validation.py


@dataclass
class FlushBacktestResult:
    params: FlushParams
    trades: list[FlushTrade] = field(default_factory=list)
    spring_loaded_bars: int = 0      # bars where the funding percentile gate was true
    flush_trigger_bars: int = 0      # bars where the candle-shape trigger fired (gate or no gate)
    armed_signals: int = 0           # bars where BOTH conditions held -> an entry was armed


# --------------------------------------------------------------------------- #
# signal computation (pure, vectorized, no lookahead by construction)
# --------------------------------------------------------------------------- #
def compute_flush_trigger(df: pd.DataFrame, params: FlushParams) -> pd.Series:
    """The liquidation-flush candle trigger, evaluated on the COMPLETED bar
    it describes. All three conditions use only that bar's own OHLCV plus a
    trailing average computed from the `trailing_avg_window` bars strictly
    BEFORE it (shift(1)), so the baseline is never contaminated by the
    event itself."""
    tr = ind.true_range(df)
    w = params.trailing_avg_window
    trailing_range_avg = tr.shift(1).rolling(w, min_periods=w).mean()
    trailing_vol_avg = df["volume"].shift(1).rolling(w, min_periods=w).mean()

    range_spike = tr >= params.flush_range_mult * trailing_range_avg
    volume_spike = df["volume"] >= params.flush_volume_mult * trailing_vol_avg

    body = (df["close"] - df["open"]).abs()
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    full_range = (df["high"] - df["low"]).replace(0.0, np.nan)
    close_location = (df["close"] - df["low"]) / full_range

    wick_condition = lower_wick >= params.flush_wick_body_mult * body
    close_condition = close_location >= params.flush_close_location_min

    trigger = range_spike & volume_spike & wick_condition & close_condition
    return trigger.fillna(False)


def compute_spring_loaded(funding_features_aligned: pd.DataFrame, params: FlushParams) -> pd.DataFrame:
    """Returns a DataFrame aligned to the bar index with columns:
      - spring_loaded_now: this bar's aligned funding pct_rank >= threshold
      - spring_loaded_recent: spring_loaded_now was true at any point in the
        trailing `spring_loaded_lookback_bars` bars (INCLUDING this one) --
        a plain backward rolling max, so still strictly no-lookahead.
    """
    pct_rank = funding_features_aligned["pct_rank"]
    now = (pct_rank >= params.funding_pct_threshold).fillna(False)
    recent = now.rolling(params.spring_loaded_lookback_bars, min_periods=1).max().astype(bool)
    return pd.DataFrame({"spring_loaded_now": now, "spring_loaded_recent": recent})


# --------------------------------------------------------------------------- #
# backtest loop
# --------------------------------------------------------------------------- #
def run_flush_backtest(df: pd.DataFrame, funding_features_aligned: pd.DataFrame,
                       params: FlushParams = FlushParams()) -> FlushBacktestResult:
    """Bar-by-bar, one-position-at-a-time, no-lookahead simulation.

    `funding_features_aligned` must already be aligned to df.index (see
    longbot.funding.align_to_bars), with columns funding_rate/pct_rank/
    trailing_median -- i.e. the caller has already enforced the backward
    "as of" join; this function just consumes it.
    """
    ind.validate_ohlcv(df)
    n = len(df)
    res = FlushBacktestResult(params=params)

    atr = ind.atr(df, params.atr_period)
    flush_trigger = compute_flush_trigger(df, params)
    spring = compute_spring_loaded(funding_features_aligned, params)
    trailing_median = funding_features_aligned["trailing_median"]
    pct_rank = funding_features_aligned["pct_rank"]

    res.spring_loaded_bars = int(spring["spring_loaded_now"].sum())
    res.flush_trigger_bars = int(flush_trigger.sum())

    signal = (
        spring["spring_loaded_recent"]
        & flush_trigger
        & atr.notna()
        & pct_rank.notna()
    )
    res.armed_signals = int(signal.sum())

    state = "FLAT"       # FLAT | ARMED | LONG
    armed_index = None
    position = None

    for i in range(n):
        bar = df.iloc[i]

        if state == "LONG":
            assert position is not None
            low, high, open_, close = bar["low"], bar["high"], bar["open"], bar["close"]

            # 1) pessimistic: hard stop first, using a gapped-open-aware fill
            if open_ <= position["stop"]:
                fill = open_ * (1 - params.slippage_pct)
                _close_trade(res, position, fill, df.index[i], i, params, "stop")
                state, position = "FLAT", None
                continue
            if low <= position["stop"]:
                fill = position["stop"] * (1 - params.slippage_pct)
                _close_trade(res, position, fill, df.index[i], i, params, "stop")
                state, position = "FLAT", None
                continue

            # track running MAE/MFE now that the stop has NOT been hit this bar
            position["worst_low"] = min(position["worst_low"], low)
            position["best_high"] = max(position["best_high"], high)

            # 2) price target (favorable, checked before the discretionary funding exit)
            if high >= position["target"]:
                fill = position["target"] * (1 - params.slippage_pct)
                _close_trade(res, position, fill, df.index[i], i, params, "target")
                state, position = "FLAT", None
                continue

            # 3) funding-normalized exit: crowd is no longer aggressively long
            med = trailing_median.iloc[i]
            if pd.notna(med) and pd.notna(pct_rank.iloc[i]) and bar_funding_normalized(
                funding_features_aligned, i, med
            ):
                fill = close * (1 - params.slippage_pct)
                _close_trade(res, position, fill, df.index[i], i, params, "funding_normalized")
                state, position = "FLAT", None
                continue

            # 4) time stop
            if i - position["entry_index"] >= params.time_stop_bars:
                fill = close * (1 - params.slippage_pct)
                _close_trade(res, position, fill, df.index[i], i, params, "time")
                state, position = "FLAT", None
                continue

            continue

        if state == "ARMED" and armed_index == i:
            fill = bar["open"] * (1 + params.slippage_pct)
            fee = fill * params.fee_pct
            atr_i = armed_atr
            stop = fill - params.stop_atr_mult * atr_i
            target = fill + params.target_atr_mult * atr_i
            position = {
                "entry": fill, "entry_time": df.index[i], "entry_index": i,
                "stop": stop, "stop_initial": stop, "target": target,
                "fees": fee, "worst_low": bar["low"], "best_high": bar["high"],
                "entry_funding_pct_rank": armed_pct_rank,
            }
            state = "LONG"
            continue

        if state == "FLAT":
            if signal.iloc[i] and i + 1 < n:
                state = "ARMED"
                armed_index = i + 1
                armed_atr = float(atr.iloc[i])
                armed_pct_rank = float(pct_rank.iloc[i])

    # any still-open position at the end of history: close at the final close
    if state == "LONG" and position is not None:
        last = df.iloc[-1]
        _close_trade(res, position, last["close"] * (1 - params.slippage_pct),
                    df.index[-1], n - 1, params, "end_of_data")

    _annotate_post_stop_diagnostics(df, res.trades, params)
    return res


def bar_funding_normalized(funding_features_aligned: pd.DataFrame, i: int, median: float) -> bool:
    """This bar's funding rate has fallen back to/below its own trailing
    median -- the crowd is no longer aggressively long. Uses only this
    bar's aligned (already-causal) funding_rate value."""
    fr = funding_features_aligned["funding_rate"].iloc[i]
    return bool(pd.notna(fr) and fr <= median)


def _close_trade(res: FlushBacktestResult, position: dict, fill: float, exit_time, exit_index: int,
                 params: FlushParams, reason: str) -> None:
    exit_fee = fill * params.fee_pct
    entry = position["entry"]
    stop_initial = position["stop_initial"]
    r_unit = entry - stop_initial
    # fees are charged in currency (per unit held), not price -- pnl-per-unit
    # is the price move minus the absolute entry+exit fee amounts.
    pnl_per_unit = (fill - entry) - (position["fees"] + exit_fee)
    r_multiple = pnl_per_unit / r_unit if r_unit > 0 else float("nan")

    mae_r = (position["worst_low"] - entry) / r_unit if r_unit > 0 else float("nan")
    mfe_r = (position["best_high"] - entry) / r_unit if r_unit > 0 else float("nan")

    res.trades.append(FlushTrade(
        entry_time=position["entry_time"], exit_time=exit_time,
        entry_index=position["entry_index"], exit_index=exit_index,
        entry_price=entry, exit_price=fill,
        stop_initial=stop_initial, target_initial=position["target"],
        r_multiple=r_multiple, bars_held=exit_index - position["entry_index"],
        exit_reason=reason, fees_paid=position["fees"] + exit_fee,
        mae_r=mae_r, mfe_r=mfe_r,
        entry_funding_pct_rank=position["entry_funding_pct_rank"],
    ))


def _annotate_post_stop_diagnostics(df: pd.DataFrame, trades: list[FlushTrade], params: FlushParams) -> None:
    """DESCRIPTIVE ONLY -- does not affect any entry/exit/stop/threshold.
    For every STOPPED trade, look at the price path for up to
    `time_stop_bars` bars AFTER the stop-exit bar (or to end of data) and
    record the best R the price reached in that window. This is the direct
    empirical read on "did the hard stop correctly cut a real crash, or was
    it too tight for the volatility of a mechanical overshoot".
    """
    n = len(df)
    for t in trades:
        if t.exit_reason != "stop":
            continue
        stop_idx = t.exit_index
        window_end = min(n, stop_idx + 1 + params.time_stop_bars)
        if stop_idx + 1 >= window_end:
            t.post_stop_mfe_r = None
            continue
        future_highs = df["high"].iloc[stop_idx + 1: window_end]
        r_unit = t.entry_price - t.stop_initial
        if r_unit <= 0 or future_highs.empty:
            t.post_stop_mfe_r = None
            continue
        best_high = float(future_highs.max())
        t.post_stop_mfe_r = (best_high - t.entry_price) / r_unit
