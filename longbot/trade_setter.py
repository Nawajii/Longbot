"""The Trade Setter — turns a candidate verdict into an exact, placeable ticket.

Given a PULLBACK_IN_UPTREND candidate and the current capital, it computes the
precise numbers a human (phase 1) or the executor (later) would place:

* **Default entry = confirmation buy-stop** just above the pullback's local
  high. It fills only when price *rises through* it — we buy the turn, not the
  falling knife. It may never fill; that "miss" is modelled in the backtest.
* **Alternative entry = dip-buy limit** at the fast EMA — a cheaper fill but a
  higher chance of never filling (or filling right before a deeper drop). We
  compute it but default to the buy-stop.
* **Hard stop = ATR-based**, measured from the *actual* entry, to be placed the
  instant the position opens.
* **Position size = risk-based**: ``size = (capital * risk%) / (entry - stop)``,
  then capped so we never spend more than the available (spot, un-leveraged)
  capital. Respects Binance's minimum order notional and lot/price filters.
* **R-multiple reference levels** (1R/2R/3R) for reasoning about exits.

Pure arithmetic, fully deterministic. No LLM, no network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class EntryType(str, Enum):
    BUY_STOP = "BUY_STOP"      # default: buy on strength through the local high
    DIP_LIMIT = "DIP_LIMIT"    # alternative: buy the dip at the fast EMA


@dataclass(frozen=True)
class TradeSetterParams:
    risk_pct: float = 0.01            # fraction of capital risked per trade (1R)
    stop_atr_mult: float = 2.0        # default hard stop = entry - stop_atr_mult * ATR
    entry_buffer_atr: float = 0.10    # buy-stop sits this many ATR above the local high
    min_notional: float = 10.0        # Binance spot min order value (USDT); flag if below
    max_capital_fraction: float = 1.0 # never deploy more than this fraction of capital
    # --- node-anchored stop (location layer) ---
    use_node_stop: bool = False       # place the stop just beyond the supporting node
    node_buffer_atr: float = 0.25     # stop sits this many ATR below the support node
    stop_atr_floor_mult: float = 1.0  # ATR floor: node stop is never TIGHTER than this


def compute_stop(entry: float, atr: float, support_price: float | None,
                 params: TradeSetterParams) -> float:
    """Where the protective stop goes.

    Default: a pure ATR distance below entry. When ``use_node_stop`` is on and a
    real support level sits below entry, anchor the stop just beyond that node
    (economic meaning: if price decisively leaves the equilibrium zone, the
    thesis is void) — but never tighter than ``stop_atr_floor_mult`` ATR, so a
    node hugging the entry can't produce an absurdly tight, easily-wicked stop.
    """
    if (params.use_node_stop and support_price is not None
            and math.isfinite(support_price) and support_price < entry):
        node_stop = support_price - params.node_buffer_atr * atr
        floor_stop = entry - params.stop_atr_floor_mult * atr
        return min(node_stop, floor_stop)   # lower price = wider stop = respects the ATR floor
    return entry - params.stop_atr_mult * atr


@dataclass(frozen=True)
class SymbolFilters:
    """Exchange rounding rules for one symbol (from Binance exchangeInfo).

    All optional — when omitted, no rounding is applied (fine for backtests).
    """

    price_tick: float | None = None   # PRICE_FILTER tickSize
    qty_step: float | None = None     # LOT_SIZE stepSize
    min_qty: float | None = None      # LOT_SIZE minQty


@dataclass
class OrderTicket:
    symbol: str
    entry_type: EntryType
    entry_price: float
    stop_price: float
    quantity: float
    risk_per_unit: float              # entry - stop, in quote currency per unit
    risk_amount: float                # quantity * risk_per_unit (actual $ at risk)
    notional: float                   # quantity * entry_price (capital deployed)
    r_levels: dict[str, float]        # {"1R","2R","3R"} target prices for reasoning
    alt_entry_price: float            # the dip-limit alternative
    feasible: bool                    # can this actually be placed?
    warnings: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# rounding helpers (respect exchange filters)
# --------------------------------------------------------------------------- #
def _floor_to_step(value: float, step: float | None) -> float:
    if not step:
        return value
    return math.floor(value / step) * step


def _round_to_tick(value: float, tick: float | None, up: bool = False) -> float:
    if not tick:
        return value
    n = value / tick
    n = math.ceil(n) if up else round(n)
    return n * tick


# --------------------------------------------------------------------------- #
# core builder
# --------------------------------------------------------------------------- #
def build_ticket(
    symbol: str,
    *,
    local_high: float,
    ema_fast: float,
    atr: float,
    capital: float,
    support_price: float | None = None,
    entry_type: EntryType = EntryType.BUY_STOP,
    params: TradeSetterParams = TradeSetterParams(),
    filters: SymbolFilters | None = None,
) -> OrderTicket:
    """Build an order ticket from raw geometry.

    Parameters mirror what the scoring engine produces on the candidate bar:
    ``local_high`` (recent swing high), ``ema_fast`` (the fast EMA), and ``atr``
    (the volatility ruler). ``capital`` is available spot quote currency.
    """
    filters = filters or SymbolFilters()
    warnings: list[str] = []

    if atr <= 0 or math.isnan(atr):
        return _infeasible(symbol, entry_type, ["ATR unavailable / non-positive — cannot size a stop"])

    # ---- entry price ----
    buy_stop = local_high + params.entry_buffer_atr * atr
    dip_limit = ema_fast
    if entry_type == EntryType.BUY_STOP:
        entry = _round_to_tick(buy_stop, filters.price_tick, up=True)
        alt = _round_to_tick(dip_limit, filters.price_tick)
    else:
        entry = _round_to_tick(dip_limit, filters.price_tick)
        alt = _round_to_tick(buy_stop, filters.price_tick, up=True)

    # ---- hard stop (measured from the ACTUAL entry; node-anchored if enabled) ----
    stop = _round_to_tick(compute_stop(entry, atr, support_price, params), filters.price_tick)
    risk_per_unit = entry - stop
    if risk_per_unit <= 0:
        return _infeasible(symbol, entry_type, ["stop is at/above entry — geometry invalid"])

    # ---- risk-based size, capped by available capital (spot, no leverage) ----
    risk_budget = capital * params.risk_pct
    qty_by_risk = risk_budget / risk_per_unit
    qty_by_capital = (capital * params.max_capital_fraction) / entry
    quantity = min(qty_by_risk, qty_by_capital)
    if quantity == qty_by_capital < qty_by_risk:
        warnings.append(
            "position capped by available capital (spot, no leverage): "
            "actual risk is below the target risk%"
        )

    quantity = _floor_to_step(quantity, filters.qty_step)

    notional = quantity * entry
    risk_amount = quantity * risk_per_unit

    # ---- feasibility checks (flag, never silently fudge) ----
    feasible = True
    if quantity <= 0:
        feasible = False
        warnings.append("computed quantity is zero after lot-size rounding")
    if filters.min_qty and quantity < filters.min_qty:
        feasible = False
        warnings.append(f"quantity {quantity} below exchange minQty {filters.min_qty}")
    if notional < params.min_notional:
        feasible = False
        warnings.append(
            f"order notional ${notional:.2f} below exchange minimum "
            f"${params.min_notional:.2f} — capital too small for this trade"
        )

    r_levels = {
        "1R": entry + 1 * risk_per_unit,
        "2R": entry + 2 * risk_per_unit,
        "3R": entry + 3 * risk_per_unit,
    }

    return OrderTicket(
        symbol=symbol,
        entry_type=entry_type,
        entry_price=entry,
        stop_price=stop,
        quantity=quantity,
        risk_per_unit=risk_per_unit,
        risk_amount=risk_amount,
        notional=notional,
        r_levels=r_levels,
        alt_entry_price=alt,
        feasible=feasible,
        warnings=warnings,
        meta={
            "atr": atr,
            "local_high": local_high,
            "ema_fast": ema_fast,
            "risk_budget": risk_budget,
            "stop_atr_mult": params.stop_atr_mult,
        },
    )


def _infeasible(symbol: str, entry_type: EntryType, warnings: list[str]) -> OrderTicket:
    return OrderTicket(
        symbol=symbol, entry_type=entry_type,
        entry_price=float("nan"), stop_price=float("nan"), quantity=0.0,
        risk_per_unit=float("nan"), risk_amount=0.0, notional=0.0,
        r_levels={}, alt_entry_price=float("nan"),
        feasible=False, warnings=warnings,
    )


def from_feature_row(
    symbol: str,
    row: pd.Series,
    capital: float,
    entry_type: EntryType = EntryType.BUY_STOP,
    params: TradeSetterParams = TradeSetterParams(),
    filters: SymbolFilters | None = None,
) -> OrderTicket:
    """Convenience: build a ticket from a row of scoring-engine features."""
    support = None
    if params.use_node_stop and "vp_support_price" in row.index:
        val = row["vp_support_price"]
        support = float(val) if pd.notna(val) else None
    return build_ticket(
        symbol,
        local_high=float(row["local_high"]),
        ema_fast=float(row["ema_fast"]),
        atr=float(row["atr"]),
        capital=capital,
        support_price=support,
        entry_type=entry_type,
        params=params,
        filters=filters,
    )
