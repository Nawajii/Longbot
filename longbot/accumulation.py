"""Adaptive DCA (volatility harvesting) — an ACCUMULATION simulator.

This is deliberately NOT a trade backtester. There are no entries/exits,
no stops, no expectancy, nothing is ever sold. It reuses only this repo's
proven CONVENTIONS (no-lookahead discipline, deep Binance fetch + CSV
cache) -- it does not import scoring_engine.py / trade_setter.py /
flush_engine.py, which are all about deciding when to exit a position.
Here there is no position to exit; only a cost basis to build.

THE THESIS (behavioral, not an alpha edge -- see README): retail investors
systematically buy during euphoria and sell during panic. This mechanism
does the opposite on the buy side only -- it buys MORE on red days and
LESS (or nothing) on euphoric green days, and never sells. It is not
timing tops/bottoms; it is a fixed, pre-registered buy-more-on-fear ladder
applied mechanically, every day, forever.

NO-LOOKAHEAD: the buy amount for execution day `t+1` is decided from the
return realized on the COMPLETED bar `t` (``close[t]/close[t-1] - 1``),
then executed at bar `t+1`'s OPEN. Day t+1's own high/low/close are never
read to decide day t+1's own purchase.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LadderParams:
    """The pre-registered ladder. FIXED for the run -- do not sweep these
    numbers to flatter results. Bands are checked top-down; the first
    matching band wins (see `ladder_amount`)."""
    fee_pct: float = 0.001   # 0.10% spot buy fee, applied to every purchase
    # (upper_bound_exclusive_or_None, amount_usd) evaluated top-down; the
    # first row whose r <= threshold (or the catch-all None) applies.
    euphoria_return: float = 0.05     # r > +5% -> skip
    base_band_hi: float = 0.05        # -2% <= r <= +5% -> base
    base_band_lo: float = -0.02
    base_amount: float = 5.0
    tier1_lo: float = -0.05            # -5% < r < -2% -> tier1
    tier1_amount: float = 7.0
    tier2_lo: float = -0.08             # -8% < r <= -5% -> tier2
    tier2_amount: float = 10.0
    tier3_lo: float = -0.12              # -12% < r <= -8% -> tier3
    tier3_amount: float = 15.0
    tier4_amount: float = 25.0            # r <= -12% -> tier4


DEFAULT_LADDER = LadderParams()
FLAT_DAILY_AMOUNT = 5.0   # Flat DCA's constant daily buy, matching the ladder's base band


def ladder_amount(r: float, params: LadderParams = DEFAULT_LADDER) -> float:
    """Pure function: daily return -> USD buy amount, per the pre-registered
    ladder. NaN return (insufficient history) -> $0 (no buy)."""
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return 0.0
    if r > params.euphoria_return:
        return 0.0
    if r >= params.base_band_lo:          # -2% <= r <= +5%
        return params.base_amount
    if r > params.tier1_lo:                # -5% < r < -2%
        return params.tier1_amount
    if r > params.tier2_lo:                 # -8% < r <= -5%
        return params.tier2_amount
    if r > params.tier3_lo:                  # -12% < r <= -8%
        return params.tier3_amount
    return params.tier4_amount                # r <= -12%


@dataclass
class DCABuy:
    signal_date: object          # date of the completed bar the decision was based on (None for flat/lump-sum)
    execution_date: object       # date of the fill (bar whose open was used)
    execution_index: int         # positional index into the price DataFrame
    price: float                 # fill price (bar's open), no slippage modelled
    daily_return: float          # the return that drove the ladder decision (NaN for flat/lump-sum)
    amount_usd: float            # gross USD spent (before fee)
    fee_usd: float
    units: float                 # units bought = (amount_usd - fee_usd) / price


@dataclass
class DCAResult:
    label: str                   # "adaptive" | "flat" | "flat_same_capital" | "lump_sum"
    asset: str
    buys: list[DCABuy] = field(default_factory=list)

    @property
    def total_deployed(self) -> float:
        return float(sum(b.amount_usd for b in self.buys))

    @property
    def total_fees(self) -> float:
        return float(sum(b.fee_usd for b in self.buys))

    @property
    def total_units(self) -> float:
        return float(sum(b.units for b in self.buys))

    @property
    def avg_cost_basis(self) -> float:
        return self.total_deployed / self.total_units if self.total_units > 0 else float("nan")

    def terminal_value(self, final_price: float) -> float:
        """Mark-to-market valuation only -- nothing is ever actually sold,
        so no exit fee is charged here."""
        return self.total_units * final_price


# --------------------------------------------------------------------------- #
# executable-day scaffolding (shared across all three approaches so they are
# compared on an identical opportunity set)
# --------------------------------------------------------------------------- #
def executable_indices(n_bars: int) -> list[int]:
    """Bar indices that can be a BUY EXECUTION day under the no-lookahead
    rule: execution at t+1 needs a signal computed from bar t's return
    (which itself needs bar t-1), so t ranges 1..n-2 and execution index
    ranges 2..n-1."""
    return list(range(2, n_bars))


def _daily_return(close: pd.Series, t: int) -> float:
    if t < 1:
        return float("nan")
    prev = close.iloc[t - 1]
    if prev == 0 or pd.isna(prev) or pd.isna(close.iloc[t]):
        return float("nan")
    return float(close.iloc[t] / prev - 1.0)


# --------------------------------------------------------------------------- #
# simulators
# --------------------------------------------------------------------------- #
def simulate_adaptive_dca(df: pd.DataFrame, asset: str, params: LadderParams = DEFAULT_LADDER) -> DCAResult:
    """For each executable index `i` (>=2), the signal bar is `t = i - 1`;
    its return uses close[t] and close[t-1] -- both strictly BEFORE the
    execution bar `i`. Execution fills at bar i's OPEN."""
    close = df["close"]
    open_ = df["open"]
    result = DCAResult(label="adaptive", asset=asset)

    for i in executable_indices(len(df)):
        t = i - 1
        r = _daily_return(close, t)
        amount = ladder_amount(r, params)
        if amount <= 0:
            continue
        price = float(open_.iloc[i])
        fee = amount * params.fee_pct
        units = (amount - fee) / price
        result.buys.append(DCABuy(
            signal_date=df.index[t], execution_date=df.index[i], execution_index=i,
            price=price, daily_return=r, amount_usd=amount, fee_usd=fee, units=units,
        ))
    return result


def simulate_flat_dca(
    df: pd.DataFrame, asset: str, daily_amount: float = FLAT_DAILY_AMOUNT,
    fee_pct: float = DEFAULT_LADDER.fee_pct, execution_indices: list[int] | None = None,
) -> DCAResult:
    """Buy a constant `daily_amount` on every index in `execution_indices`
    (defaults to the full executable set, i.e. the SAME opportunity set
    adaptive DCA sees -- no information is used, so there is nothing to
    be lookahead-unsafe about, but using the identical day-set keeps the
    comparison to adaptive apples-to-apples)."""
    indices = execution_indices if execution_indices is not None else executable_indices(len(df))
    open_ = df["open"]
    result = DCAResult(label="flat", asset=asset)
    for i in indices:
        price = float(open_.iloc[i])
        fee = daily_amount * fee_pct
        units = (daily_amount - fee) / price
        result.buys.append(DCABuy(
            signal_date=None, execution_date=df.index[i], execution_index=i,
            price=price, daily_return=float("nan"), amount_usd=daily_amount, fee_usd=fee, units=units,
        ))
    return result


def simulate_flat_same_capital(
    df: pd.DataFrame, asset: str, total_capital: float, buy_indices: list[int],
    fee_pct: float = DEFAULT_LADDER.fee_pct,
) -> DCAResult:
    """The same-total-capital control: spread `total_capital` EVENLY across
    exactly the days adaptive DCA actually bought on (`buy_indices` --
    i.e. excluding euphoria-skip days), isolating the effect of VARIABLE
    sizing from the effect of deploying more total capital."""
    if not buy_indices:
        return DCAResult(label="flat_same_capital", asset=asset)
    per_day = total_capital / len(buy_indices)
    open_ = df["open"]
    result = DCAResult(label="flat_same_capital", asset=asset)
    for i in buy_indices:
        price = float(open_.iloc[i])
        fee = per_day * fee_pct
        units = (per_day - fee) / price
        result.buys.append(DCABuy(
            signal_date=None, execution_date=df.index[i], execution_index=i,
            price=price, daily_return=float("nan"), amount_usd=per_day, fee_usd=fee, units=units,
        ))
    return result


def simulate_lump_sum(
    df: pd.DataFrame, asset: str, total_amount: float,
    fee_pct: float = DEFAULT_LADDER.fee_pct, execution_index: int = 0,
) -> DCAResult:
    """Invest the whole amount at once, at `execution_index`'s open
    (default: the very first bar of the period) -- context benchmark, NOT
    the fair fight (see README)."""
    price = float(df["open"].iloc[execution_index])
    fee = total_amount * fee_pct
    units = (total_amount - fee) / price
    result = DCAResult(label="lump_sum", asset=asset)
    result.buys.append(DCABuy(
        signal_date=None, execution_date=df.index[execution_index], execution_index=execution_index,
        price=price, daily_return=float("nan"), amount_usd=total_amount, fee_usd=fee, units=units,
    ))
    return result
