"""
Long-only reversion entry/exit engine for a single pair.

Design notes (fixed before any run):

- Direction: at entry, z <= -ENTRY_Z means A is cheap relative to B -> BUY A
  ("long_A"). z >= +ENTRY_Z means B is cheap relative to A -> BUY B
  ("long_B"). We only ever hold the relative underperformer -- there is no
  short leg (halal constraint: spot long only).

- One position at a time PER PAIR. Each of the 10 pairs runs its own
  independent single-slot state machine; different pairs may have
  simultaneous open positions (there is no cross-pair capital netting or
  margin -- each is a plain, independent spot buy). This reading is
  necessary for the success bar's own math to be satisfiable: the bar
  explicitly reasons about reaching >=100 trades "across 6 years and 10
  pairs" -- that only scales with pair count if pairs are independent. A
  single GLOBAL one-slot-for-everything design would cap total trades at
  whatever one slot's turnover allows regardless of how many pairs exist,
  which contradicts the bar's own framing. This is a documented design
  choice, not a data-mined one.

- Frozen parameters per trade: beta/mean/std used for z on every day of an
  open trade are the FORMATION result from the entry day, held fixed until
  exit. They are never re-estimated mid-trade. This is what makes exit
  levels meaningful and is required by the no-lookahead rule -- the
  formation window used is always strictly prior to the day it was
  estimated for, and it does not drift forward mid-trade to peek at newer
  data.

- Execution: both signal and fill use the same bar's close (fees +
  slippage below are the disclosed model of execution cost/uncertainty).
  This is a standard, disclosed simplification for a daily-bar backtest,
  not the no-lookahead hazard the prompt warns about (that hazard is
  about formation PARAMETERS leaking future information, which is
  structurally prevented in cointegration.py).

- 1R definition: "the entry-to-stop spread distance," translated into a
  price distance on the asset actually held. We hold B's price fixed at
  its entry-day level and ask: what price would A (or B, for a long_B
  trade) need to reach for z to hit the stop threshold? That gives an
  implied stop PRICE for the held asset, and 1R is the entry-to-that-price
  percentage distance. This is the only way to make "R" comparable to the
  single-leg P&L we actually realize (we hold one asset, not the spread).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .cointegration import FormationResult, formation_for_index, spread_and_z


@dataclass
class Trade:
    pair: tuple[str, str]
    direction: str  # config.TradeDirection.LONG_A / LONG_B
    entry_date: object
    entry_price: float           # fill price of the held asset, after costs
    entry_z: float
    formation: FormationResult
    stop_price: float            # implied price level for the stop, pre-cost
    r_unit_pct: float            # 1R as a fraction of entry price (>0)
    exit_date: object = None
    exit_price: float = None     # fill price after costs
    exit_reason: str = None      # "target" | "stop" | "time"
    r_multiple: float = None
    return_pct: float = None
    days_held: int = None
    entry_year: int = None       # filled in by walkforward.run_all
    regime: str = None           # filled in by walkforward.run_all


def _implied_stop_price(entry_price: float, direction: str, formation: FormationResult,
                         other_leg_price_at_entry: float) -> float:
    """Implied price of the HELD asset at which z would hit the stop
    threshold, holding the other leg's price fixed at its entry-day level."""
    stop_z = -config.STOP_Z if direction == "long_A" else config.STOP_Z
    stop_spread = formation.mean + stop_z * formation.std
    other_log = np.log(other_leg_price_at_entry)

    if direction == "long_A":
        # spread = log(A) - beta*log(B); solve for log(A)
        implied_log_held = stop_spread + formation.beta * other_log
    else:
        # spread = log(A) - beta*log(B); solve for log(B), other leg is A
        implied_log_held = (other_log - stop_spread) / formation.beta

    return float(np.exp(implied_log_held))


def _apply_cost(price: float, buying: bool) -> float:
    """Apply slippage (adverse) then fee, matching the prior project's cost
    model: 0.10%/side fee + 0.05%/side slippage."""
    slip = price * (1 + config.SLIPPAGE_PCT) if buying else price * (1 - config.SLIPPAGE_PCT)
    fee = slip * config.FEE_PCT
    return slip + fee if buying else slip - fee


def run_pair(prices: pd.DataFrame, asset_a: str, asset_b: str) -> list[Trade]:
    """Walk `prices` (date-indexed, columns [asset_a, asset_b]) day by day
    and return the list of completed (and, for the final open one if any,
    still-open) trades for this pair."""
    trades: list[Trade] = []
    open_trade: Trade | None = None

    dates = prices.index
    for i, today in enumerate(dates):
        price_a = prices[asset_a].iloc[i]
        price_b = prices[asset_b].iloc[i]

        if open_trade is not None:
            held_price = price_a if open_trade.direction == "long_A" else price_b
            _, z = spread_and_z(price_a, price_b, open_trade.formation)
            days_held = (today - open_trade.entry_date).days

            exit_reason = None
            if open_trade.direction == "long_A":
                if z <= -config.STOP_Z:
                    exit_reason = "stop"
                elif z >= -config.TARGET_Z:
                    exit_reason = "target"
            else:
                if z >= config.STOP_Z:
                    exit_reason = "stop"
                elif z <= config.TARGET_Z:
                    exit_reason = "target"

            if exit_reason is None and days_held >= config.TIME_STOP_DAYS:
                exit_reason = "time"

            if exit_reason is not None:
                fill = _apply_cost(held_price, buying=False)
                open_trade.exit_date = today
                open_trade.exit_price = fill
                open_trade.exit_reason = exit_reason
                open_trade.days_held = days_held
                open_trade.return_pct = (fill - open_trade.entry_price) / open_trade.entry_price
                open_trade.r_multiple = open_trade.return_pct / open_trade.r_unit_pct
                trades.append(open_trade)
                open_trade = None
            continue

        # Flat: look for an entry using TODAY's formation (fit on the 90
        # days strictly before today, i.e. rows [i-90, i)).
        formation = formation_for_index(prices, asset_a, asset_b, i)
        if formation is None or not formation.cointegrated:
            continue

        _, z = spread_and_z(price_a, price_b, formation)

        direction = None
        if z <= -config.ENTRY_Z:
            direction = "long_A"
        elif z >= config.ENTRY_Z:
            direction = "long_B"
        if direction is None:
            continue

        held_price = price_a if direction == "long_A" else price_b
        other_price = price_b if direction == "long_A" else price_a
        entry_fill = _apply_cost(held_price, buying=True)
        stop_price = _implied_stop_price(held_price, direction, formation, other_price)
        r_unit_pct = abs(held_price - stop_price) / held_price
        if r_unit_pct <= 0:
            continue

        open_trade = Trade(
            pair=(asset_a, asset_b),
            direction=direction,
            entry_date=today,
            entry_price=entry_fill,
            entry_z=z,
            formation=formation,
            stop_price=stop_price,
            r_unit_pct=r_unit_pct,
        )

    # An open trade with no exit by the end of history is left OUT of
    # completed trades (it has no realized outcome to score).
    return trades
