"""Honest, bar-by-bar backtest harness (single pair, one position at a time).

This file is deliberately strict, because an over-optimistic backtest is worse
than none — it gives false confidence to risk real money. The rules:

* **No lookahead.** A decision on closed bar ``i`` may use only bars ``<= i``.
  An order created at the close of bar ``i`` can fill no earlier than bar
  ``i+1``.
* **Fills are modelled, not assumed.** A confirmation buy-stop fills only if a
  later bar's price actually trades through it, within a limited validity
  window. Orders that never trigger are counted as *misses* — a signal is not a
  trade.
* **Costs always applied.** Taker fee on both entry and exit, plus slippage
  against us on every fill.
* **Pessimistic intrabar convention.** While long, each bar is first tested for
  a stop-out against the *existing* stop (using the bar's low / a gapped open);
  only if it survives do we ratchet the trailing stop up using that bar's high.
  A single bar can never both protect us and stop us out — we assume the worse
  ordering.
* **Trailing stop ratchets up only, never down** (the Mitigator's logic).

Baselines for honesty: every run is compared against **buy-and-hold** and a
**buy-the-peak** strategy (entering on PEAK_EXTENDED instead of the pullback).
If the pullback strategy can't beat buy-and-hold net of costs, that is the
finding — report it, don't bury it.

Every metric is printed with its assumptions. Read :func:`format_report`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from longbot import indicators as ind
from longbot import scoring_engine as se
from longbot import trade_setter as ts


# --------------------------------------------------------------------------- #
# parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BacktestParams:
    starting_capital: float = 1_000.0
    fee_pct: float = 0.001            # 0.1% taker fee per side (Binance spot default)
    slippage_pct: float = 0.0005      # 0.05% adverse slippage per fill
    risk_pct: float = 0.01            # risk 1% of equity per trade
    stop_atr_mult: float = 2.0        # initial hard stop distance
    trail_atr_mult: float = 3.0       # trailing stop distance (let winners run)
    entry_buffer_atr: float = 0.10    # buy-stop placed this many ATR above local high
    entry_valid_bars: int = 3         # buy-stop is cancelled if unfilled after N bars
    min_notional: float = 10.0        # exchange minimum; trades below are skipped
    timeframe: str = "1h"             # label only, for the report
    # --- node-anchored stop (location layer); off = original ATR stop ---
    use_node_stop: bool = False
    node_buffer_atr: float = 0.25
    stop_atr_floor_mult: float = 1.0
    # --- exit style (the ONLY thing the exit-variant grid changes) ---
    exit_style: str = "TRAIL"         # "TRAIL" | "TIME" | "TARGET"
    time_exit_bars: int = 12          # for exit_style == "TIME"
    target_r: float = 2.0             # for exit_style == "TARGET": take profit at +target_r R


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    stop_initial: float
    pnl: float                 # net of fees and slippage, in quote currency
    r_multiple: float          # pnl in units of initial risk (1R)
    bars_held: int
    exit_reason: str           # "stop" | "trail" | "time" | "target" | "end_of_data"
    fees_paid: float
    tag: str = ""              # optional location label at the signal bar ("AT_NODE"/"IN_VOID")
    peak_price: float = float("nan")   # highest price seen while in the trade (for giveback)

    @property
    def giveback_r(self) -> float:
        """How much of the peak we surrendered by exit, in R (peak -> exit)."""
        risk = self.entry_price - self.stop_initial
        if risk <= 0 or math.isnan(self.peak_price):
            return float("nan")
        return (self.peak_price - self.exit_price) / risk


@dataclass
class BacktestResult:
    params: BacktestParams
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series | None = None
    signals: int = 0           # actionable verdicts seen
    orders_placed: int = 0     # tickets that were feasible and placed
    fills: int = 0             # orders that actually triggered
    misses: int = 0            # orders that expired unfilled
    skipped_infeasible: int = 0  # signals dropped (min notional etc.)
    final_capital: float = 0.0

    # ---- metrics ----
    @property
    def total_return_pct(self) -> float:
        return 100.0 * (self.final_capital / self.params.starting_capital - 1.0)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return float("nan")
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / len(self.trades)

    @property
    def fill_rate(self) -> float:
        if self.orders_placed == 0:
            return float("nan")
        return self.fills / self.orders_placed

    @property
    def miss_rate(self) -> float:
        if self.orders_placed == 0:
            return float("nan")
        return self.misses / self.orders_placed

    @property
    def expectancy_r(self) -> float:
        if not self.trades:
            return float("nan")
        return float(np.mean([t.r_multiple for t in self.trades]))

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = -sum(t.pnl for t in self.trades if t.pnl < 0)
        if gross_loss == 0:
            return float("inf") if gross_win > 0 else float("nan")
        return gross_win / gross_loss

    @property
    def max_drawdown_pct(self) -> float:
        if self.equity_curve is None or self.equity_curve.empty:
            return float("nan")
        running_max = self.equity_curve.cummax()
        dd = (self.equity_curve - running_max) / running_max
        return float(dd.min() * 100.0)

    @property
    def avg_bars_held(self) -> float:
        if not self.trades:
            return float("nan")
        return float(np.mean([t.bars_held for t in self.trades]))


# --------------------------------------------------------------------------- #
# fill helpers (one place so entry/exit slippage is consistent)
# --------------------------------------------------------------------------- #
def _buy_fill_price(entry: float, bar: pd.Series, slippage_pct: float) -> float | None:
    """Price at which a buy-stop at `entry` fills on this bar, or None.

    A gap open above the trigger fills at the (worse) open. Slippage is added
    *against* us (we pay up).
    """
    if bar["open"] >= entry:
        raw = bar["open"]
    elif bar["high"] >= entry:
        raw = entry
    else:
        return None
    return raw * (1.0 + slippage_pct)


def _sell_fill_price(stop: float, bar: pd.Series, slippage_pct: float) -> float | None:
    """Price at which a protective sell-stop at `stop` fills on this bar, or None.

    A gap open below the stop fills at the (worse) open. Slippage subtracted
    against us (we sell lower).
    """
    if bar["open"] <= stop:
        raw = bar["open"]
    elif bar["low"] <= stop:
        raw = stop
    else:
        return None
    return raw * (1.0 - slippage_pct)


# --------------------------------------------------------------------------- #
# core engine
# --------------------------------------------------------------------------- #
def run_backtest(
    df: pd.DataFrame,
    params: BacktestParams = BacktestParams(),
    scoring_params: se.ScoringParams = se.DEFAULT_PARAMS,
    entry_verdicts: tuple[se.Verdict, ...] = (se.Verdict.PULLBACK_IN_UPTREND,),
    regime_df: pd.DataFrame | None = None,
    tag_series: pd.Series | None = None,
    location_features: pd.DataFrame | None = None,
    entry_gate: pd.Series | None = None,
) -> BacktestResult:
    """Simulate the strategy over one pair's OHLCV history.

    ``entry_verdicts`` lets the same engine run the *buy-the-peak* baseline by
    passing ``(Verdict.PEAK_EXTENDED,)``. ``regime_df`` (e.g. BTC) optionally
    gates entries to risk-on bars; pass None to disable the regime gate.

    ``tag_series`` is an optional boolean Series aligned to ``df.index``. When a
    signal places an order at bar ``i``, the resulting trade is tagged
    ``"AT_NODE"`` if ``tag_series.iloc[i]`` is True else ``"IN_VOID"``. This is
    how the PRIMARY location test labels existing trades WITHOUT changing any
    entry or stop — it isolates the location effect.
    """
    ind.validate_ohlcv(df)
    feat = se.compute_features(df, scoring_params, location_features=location_features)

    # Precompute per-bar verdicts once (still no lookahead — each row uses <= i).
    verdicts = [se.classify_setup(feat.iloc[i], scoring_params).verdict for i in range(len(df))]

    # Precompute per-bar regime flag (no lookahead) if a regime series is given.
    regime_on = _regime_flags(df.index, regime_df) if regime_df is not None else None

    # Optional extra AND-gate on entries (e.g. a higher-timeframe/multi-scale
    # gate). Boolean, aligned to df.index; caller is responsible for no-lookahead.
    extra_gate = None
    if entry_gate is not None:
        extra_gate = list(entry_gate.reindex(df.index).fillna(False).astype(bool).values)

    res = BacktestResult(params=params)
    capital = params.starting_capital
    equity_points: list[tuple[pd.Timestamp, float]] = []

    setter_params = ts.TradeSetterParams(
        risk_pct=params.risk_pct,
        stop_atr_mult=params.stop_atr_mult,
        entry_buffer_atr=params.entry_buffer_atr,
        min_notional=params.min_notional,
        use_node_stop=params.use_node_stop,
        node_buffer_atr=params.node_buffer_atr,
        stop_atr_floor_mult=params.stop_atr_floor_mult,
    )

    # state
    state = "FLAT"          # FLAT | PENDING | LONG
    pending: dict | None = None
    position: dict | None = None

    n = len(df)
    for i in range(n):
        bar = df.iloc[i]
        ts_now = df.index[i]

        # ---------------- manage an open position ----------------
        if state == "LONG":
            assert position is not None
            # 1) PESSIMISTIC: test the stop against the EXISTING stop first, so a
            #    single bar can never both save us and stop us out.
            fill = _sell_fill_price(position["stop"], bar, params.slippage_pct)
            if fill is not None:
                capital = _close_position(res, position, fill, ts_now, i, params, capital,
                                          reason="trail" if position["trailing"] else "stop")
                state, position = "FLAT", None
            else:
                # track the peak (for trailing and for giveback reporting)
                position["highest"] = max(position["highest"], bar["high"])
                if params.exit_style == "TIME":
                    # 2a) time-based exit: close at market on the open of entry+N.
                    #     The 2xATR hard stop above still provides disaster cover.
                    if i - position["entry_index"] >= params.time_exit_bars:
                        exit_raw = bar["open"] * (1.0 - params.slippage_pct)
                        capital = _close_position(res, position, exit_raw, ts_now, i, params,
                                                  capital, reason="time")
                        state, position = "FLAT", None
                elif params.exit_style == "TARGET":
                    # 2b) fixed take-profit at +target_r R. The STOP was already
                    #     checked above, so if a bar spans BOTH stop and target we
                    #     have already exited at the stop — the pessimistic rule:
                    #     never credit the favorable target fill on a spanning bar.
                    if bar["high"] >= position["target"]:
                        exit_raw = position["target"] * (1.0 - params.slippage_pct)
                        capital = _close_position(res, position, exit_raw, ts_now, i, params,
                                                  capital, reason="target")
                        state, position = "FLAT", None
                else:
                    # 2c) trailing exit: ratchet the stop UP only.
                    trail = position["highest"] - params.trail_atr_mult * position["atr"]
                    if trail > position["stop"]:
                        position["stop"] = trail
                        position["trailing"] = True

        # ---------------- try to fill a pending order ----------------
        elif state == "PENDING":
            assert pending is not None
            fill = _buy_fill_price(pending["entry"], bar, params.slippage_pct)
            if fill is not None:
                res.fills += 1
                # hard stop measured from the ACTUAL fill (spec: stop from actual
                # entry); node-anchored when the location layer is on.
                stop = ts.compute_stop(fill, pending["atr"], pending.get("support"), setter_params)
                qty = pending["qty"]
                fee = fill * qty * params.fee_pct
                capital -= fee
                target = fill + params.target_r * (fill - stop)   # for TARGET exit
                position = {
                    "entry": fill, "qty": qty, "stop": stop, "stop_initial": stop,
                    "atr": pending["atr"], "highest": bar["high"], "trailing": False,
                    "entry_time": ts_now, "entry_index": i, "fees": fee,
                    "tag": pending.get("tag", ""), "target": target,
                }
                state, pending = "LONG", None
            elif i >= pending["expiry_index"]:
                res.misses += 1            # order expired unfilled — a true "miss"
                state, pending = "FLAT", None

        # ---------------- look for a new signal (only when FLAT) ----------------
        if state == "FLAT":
            v = verdicts[i]
            if v in entry_verdicts:
                res.signals += 1
                gated_out = (regime_on is not None and not regime_on[i]) or \
                            (extra_gate is not None and not extra_gate[i])
                if not gated_out:
                    row = feat.iloc[i]
                    ticket = ts.from_feature_row(
                        "BACKTEST", row, capital,
                        entry_type=ts.EntryType.BUY_STOP, params=setter_params,
                    )
                    if ticket.feasible and i + 1 < n:
                        res.orders_placed += 1
                        support = row.get("vp_support_price")
                        tag = ""
                        if tag_series is not None:
                            tag = "AT_NODE" if bool(tag_series.iloc[i]) else "IN_VOID"
                        pending = {
                            "entry": ticket.entry_price,
                            "qty": ticket.quantity,
                            "atr": float(row["atr"]),
                            "support": float(support) if pd.notna(support) else None,
                            "tag": tag,
                            "expiry_index": i + params.entry_valid_bars,
                        }
                        state = "PENDING"
                    else:
                        res.skipped_infeasible += 1

        # mark-to-market equity (unrealised PnL included while LONG)
        mtm = capital
        if state == "LONG" and position is not None:
            mtm = capital + position["qty"] * (bar["close"] - position["entry"])
        equity_points.append((ts_now, mtm))

    # close any still-open position at the final close (mark-to-realised)
    if state == "LONG" and position is not None:
        last = df.iloc[-1]
        capital = _close_position(res, position, last["close"], df.index[-1], n - 1, params,
                                  capital, reason="end_of_data")

    res.final_capital = capital
    res.equity_curve = pd.Series(dict(equity_points))
    return res


def _close_position(res, position, raw_exit, exit_time, exit_index, params, capital, reason):
    exit_fee = raw_exit * position["qty"] * params.fee_pct
    proceeds = position["qty"] * (raw_exit - position["entry"]) - exit_fee
    capital += proceeds
    total_fees = position["fees"] + exit_fee
    risk_per_unit = position["entry"] - position["stop_initial"]
    r_mult = (proceeds / (risk_per_unit * position["qty"])) if risk_per_unit > 0 else float("nan")
    res.trades.append(Trade(
        entry_time=position["entry_time"], exit_time=exit_time,
        entry_price=position["entry"], exit_price=raw_exit,
        quantity=position["qty"], stop_initial=position["stop_initial"],
        pnl=proceeds, r_multiple=r_mult,
        bars_held=exit_index - position["entry_index"],
        exit_reason=reason, fees_paid=total_fees,
        tag=position.get("tag", ""),
        peak_price=position.get("highest", float("nan")),
    ))
    return capital


def _regime_flags(index: pd.Index, regime_df: pd.DataFrame) -> list[bool]:
    """Per-bar 'is BTC above a rising EMA200' flag, aligned to `index`, no lookahead."""
    close = regime_df["close"]
    ema = ind.ema(close, 200)
    slp = ind.slope(ema, 20)
    on = (close > ema) & (slp > 0)
    on = on.reindex(index, method="ffill").fillna(False)
    return list(on.astype(bool).values)


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
def buy_and_hold_return_pct(df: pd.DataFrame, params: BacktestParams) -> float:
    """Buy at the first usable close, hold to the end, one round-trip of costs."""
    warmup = 250
    if len(df) <= warmup:
        return float("nan")
    entry = df["close"].iloc[warmup] * (1 + params.slippage_pct)
    exit_ = df["close"].iloc[-1] * (1 - params.slippage_pct)
    gross = exit_ / entry
    net = gross * (1 - params.fee_pct) ** 2
    return 100.0 * (net - 1.0)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def format_report(
    res: BacktestResult,
    df: pd.DataFrame,
    peak_baseline: BacktestResult | None = None,
) -> str:
    p = res.params
    bh = buy_and_hold_return_pct(df, p)
    lines = [
        "=" * 66,
        f" BACKTEST REPORT  ({p.timeframe}, {len(df)} bars)",
        "=" * 66,
        " ASSUMPTIONS (read these — results are meaningless without them):",
        f"   starting capital : ${p.starting_capital:,.2f}",
        f"   fee per side     : {p.fee_pct*100:.3f}%   slippage/fill: {p.slippage_pct*100:.3f}%",
        f"   risk per trade   : {p.risk_pct*100:.2f}% of equity",
        f"   stop / trail ATR : {p.stop_atr_mult} / {p.trail_atr_mult}",
        f"   buy-stop valid   : {p.entry_valid_bars} bars (else counted a MISS)",
        f"   min notional     : ${p.min_notional:.2f}",
        "-" * 66,
        " SIGNAL -> ORDER -> FILL funnel (a signal is NOT a trade):",
        f"   actionable signals : {res.signals}",
        f"   orders placed      : {res.orders_placed}"
        f"   (skipped infeasible: {res.skipped_infeasible})",
        f"   fills              : {res.fills}",
        f"   misses (unfilled)  : {res.misses}",
        f"   fill rate          : {_pct(res.fill_rate)}   miss rate: {_pct(res.miss_rate)}",
        "-" * 66,
        " PERFORMANCE (net of fees + slippage):",
        f"   trades closed      : {len(res.trades)}",
        f"   win rate           : {_pct(res.win_rate)}",
        f"   expectancy         : {res.expectancy_r:.3f} R per trade",
        f"   profit factor      : {res.profit_factor:.2f}",
        f"   avg bars held      : {res.avg_bars_held:.1f}",
        f"   max drawdown       : {res.max_drawdown_pct:.2f}%",
        f"   total return       : {res.total_return_pct:+.2f}%   (${res.final_capital:,.2f})",
        "-" * 66,
        " BASELINES (the honesty check):",
        f"   buy & hold         : {bh:+.2f}%",
    ]
    if peak_baseline is not None:
        lines.append(f"   buy-the-peak       : {peak_baseline.total_return_pct:+.2f}%")
    edge = res.total_return_pct - bh
    lines += [
        "-" * 66,
        f" EDGE vs buy & hold   : {edge:+.2f} percentage points",
        ("   -> NO demonstrated edge over buy & hold (net of costs)."
         if edge <= 0 else
         "   -> Beats buy & hold here, BUT one pair/period is NOT proof — "
         "test many pairs & timeframes before trusting it."),
        "=" * 66,
    ]
    return "\n".join(lines)


def _pct(x: float) -> str:
    return "n/a" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x*100:.1f}%"
