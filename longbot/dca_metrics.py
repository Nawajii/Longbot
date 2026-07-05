"""Accumulation-appropriate metrics for longbot/accumulation.py results.

Deliberately NOT trade metrics (no win rate, no R-multiple, no profit
factor) -- nothing is ever sold, so there is no realized P&L per trade.
Instead: cost basis, money-weighted return (IRR/XIRR), mark-to-market
drawdown of the accumulated stack, and capital-deployed breakdowns by
year/regime.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from longbot.accumulation import DCAResult


def _as_date(d):
    return d.date() if isinstance(d, (pd.Timestamp, datetime)) else d


def xirr(cashflows: list[tuple], low: float = -0.9999, high: float = 10.0,
         tol: float = 1e-8, max_iter: int = 200) -> float:
    """Money-weighted annualized return solved by bisection (no scipy /
    numpy-financial dependency). `cashflows` is a list of (date, amount)
    -- negative for money out (buys), positive for money in (terminal
    mark-to-market valuation). Returns NaN if no sign change is found in
    [low, high] (e.g. too few/degenerate cashflows to solve)."""
    if not cashflows:
        return float("nan")
    dates = [_as_date(d) for d, _ in cashflows]
    t0 = min(dates)

    def npv(rate: float) -> float:
        total = 0.0
        for d, cf in cashflows:
            years = (_as_date(d) - t0).days / 365.0
            total += cf / (1.0 + rate) ** years
        return total

    f_low, f_high = npv(low), npv(high)
    if np.isnan(f_low) or np.isnan(f_high) or f_low * f_high > 0:
        return float("nan")

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        f_mid = npv(mid)
        if abs(f_mid) < tol or (high - low) < 1e-10:
            return mid
        if f_low * f_mid < 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid
    return (low + high) / 2.0


def result_xirr(result: DCAResult, final_date, final_price: float) -> float:
    """Build the cashflow stream for one DCAResult and solve its XIRR.
    The terminal valuation is MARK-TO-MARKET ONLY -- the strategy never
    sells, so no exit fee is applied to it."""
    cashflows = [(b.execution_date, -b.amount_usd) for b in result.buys]
    terminal_value = result.terminal_value(final_price)
    cashflows.append((final_date, terminal_value))
    return xirr(cashflows)


def portfolio_value_series(result: DCAResult, df: pd.DataFrame) -> pd.Series:
    """Daily mark-to-market value of the accumulated stack: cumulative
    units held (as of that bar) times that bar's close. Zero (no units
    yet) before the first buy."""
    units_by_index = np.zeros(len(df))
    for b in result.buys:
        units_by_index[b.execution_index] += b.units
    cum_units = np.cumsum(units_by_index)
    return pd.Series(cum_units * df["close"].to_numpy(), index=df.index)


def max_drawdown_pct(value_series: pd.Series) -> float:
    """Peak-to-trough drawdown of the mark-to-market value curve, in
    percent (negative number). NOTE: this mixes genuine price drawdown
    with the effect of new capital continuously being added -- read it as
    'how far underwater did the CURRENT stack get', not a pure
    already-invested return drawdown (which never-sell DCA doesn't have
    a clean analogue for, since capital keeps arriving)."""
    v = value_series.replace(0.0, np.nan)
    if v.notna().sum() == 0:
        return float("nan")
    peak = v.cummax()
    dd = (v - peak) / peak
    return float(dd.min()) if dd.notna().any() else float("nan")


def deployed_by_bucket(result: DCAResult, bucket_fn) -> dict:
    """Total USD deployed, grouped by whatever `bucket_fn(DCABuy) -> key`
    returns (e.g. a lambda pulling the year, or a regime label lookup)."""
    out: dict = {}
    for b in result.buys:
        k = bucket_fn(b)
        out[k] = out.get(k, 0.0) + b.amount_usd
    return out


def cost_basis_by_bucket(result: DCAResult, bucket_fn) -> dict:
    """Average cost basis (spend / units), grouped the same way."""
    spend: dict = {}
    units: dict = {}
    for b in result.buys:
        k = bucket_fn(b)
        spend[k] = spend.get(k, 0.0) + b.amount_usd
        units[k] = units.get(k, 0.0) + b.units
    return {k: (spend[k] / units[k] if units[k] > 0 else float("nan")) for k in spend}


def year_bucket(b) -> int:
    return b.execution_date.year


def regime_bucket_fn(regime_label_aligned: pd.Series):
    """Build a bucket_fn that looks up a pre-aligned (same index as the
    price DataFrame, forward-filled) BTC regime label for a buy's
    execution date."""
    def _fn(b) -> str:
        return regime_label_aligned.get(b.execution_date, "CHOP")
    return _fn
