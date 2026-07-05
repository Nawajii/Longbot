"""Unit tests for cost basis, XIRR, mark-to-market drawdown, and the
deployed/cost-basis bucket breakdowns."""

import numpy as np
import pandas as pd
import pytest

from longbot.accumulation import simulate_flat_dca, simulate_lump_sum
from longbot.dca_metrics import (
    cost_basis_by_bucket,
    deployed_by_bucket,
    max_drawdown_pct,
    portfolio_value_series,
    regime_bucket_fn,
    result_xirr,
    xirr,
    year_bucket,
)


def _df(closes, start="2021-01-01"):
    idx = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame({"open": closes, "high": closes, "low": closes,
                        "close": closes, "volume": 1000.0}, index=idx)


# --------------------------------------------------------------------------- #
# XIRR
# --------------------------------------------------------------------------- #
def test_xirr_single_cashflow_known_rate():
    # invest $100 today, get back $110 in exactly 1 year -> IRR ~ 10%
    d0 = pd.Timestamp("2021-01-01")
    d1 = pd.Timestamp("2022-01-01")
    rate = xirr([(d0, -100.0), (d1, 110.0)])
    assert rate == pytest.approx(0.10, abs=1e-4)


def test_xirr_self_consistent_on_multi_cashflow_stream():
    """No independently-known closed form for an irregular multi-cashflow
    stream, so verify self-consistency: plugging the solved rate back into
    the NPV formula must give ~0."""
    dates = pd.date_range("2021-01-01", periods=6, freq="90D")
    cashflows = [(dates[0], -50.0), (dates[1], -30.0), (dates[2], -70.0),
                (dates[3], -20.0), (dates[4], -40.0), (dates[5], 260.0)]
    rate = xirr(cashflows)
    assert not np.isnan(rate)

    t0 = dates[0]
    npv = sum(cf / (1 + rate) ** ((d - t0).days / 365.0) for d, cf in cashflows)
    assert abs(npv) < 1e-4


def test_xirr_empty_is_nan():
    assert np.isnan(xirr([]))


def test_result_xirr_uses_terminal_mark_to_market_with_no_exit_fee():
    # a full year between buy and terminal valuation, so the implied
    # annualized rate for a 2x move stays within the solver's bracket
    df = _df([100.0] * 366)
    result = simulate_lump_sum(df, "TEST", total_amount=1000.0)
    # final price double -> terminal value should be exactly units*final_price, no fee subtracted
    rate = result_xirr(result, df.index[-1], final_price=200.0)
    assert not np.isnan(rate)
    assert rate == pytest.approx(1.0, abs=0.05)   # ~doubling over ~1 year -> ~100% IRR
    assert result.terminal_value(200.0) == pytest.approx(result.total_units * 200.0)


# --------------------------------------------------------------------------- #
# portfolio value + drawdown
# --------------------------------------------------------------------------- #
def test_portfolio_value_series_tracks_cumulative_units_times_price():
    df = _df([100.0, 100.0, 50.0, 100.0, 100.0])
    result = simulate_flat_dca(df, "TEST", daily_amount=10.0)  # buys at idx 2,3,4
    values = portfolio_value_series(result, df)
    # before any buy: 0
    assert values.iloc[0] == 0.0
    assert values.iloc[1] == 0.0
    # at idx 2: one buy has landed, value = units_so_far * close[2]
    units_at_2 = next(b.units for b in result.buys if b.execution_index == 2)
    assert values.iloc[2] == pytest.approx(units_at_2 * 50.0)


def test_max_drawdown_detects_a_known_dip():
    # portfolio value ramps up, then price (and value) is cut in half, then recovers
    df = _df([100.0] * 10)
    result = simulate_flat_dca(df, "TEST", daily_amount=10.0)
    values = portfolio_value_series(result, df)
    # manually engineer a 50% dip on the underlying close after accumulation started
    df2 = df.copy()
    df2.loc[df2.index[6], "close"] = 50.0
    values2 = portfolio_value_series(result, df2)
    dd = max_drawdown_pct(values2)
    assert dd < -0.3   # a genuine, large drawdown was detected

def test_max_drawdown_ignores_pre_first_buy_zero_period():
    df = _df([100.0] * 10)
    result = simulate_flat_dca(df, "TEST", daily_amount=10.0, execution_indices=[5, 6, 7])
    values = portfolio_value_series(result, df)
    # the first 5 bars are legitimately zero (no units yet) -- must not be
    # misread as "the portfolio crashed to zero"
    dd = max_drawdown_pct(values)
    assert dd == 0.0 or np.isnan(dd) or dd > -0.01


# --------------------------------------------------------------------------- #
# bucket breakdowns
# --------------------------------------------------------------------------- #
def test_deployed_and_cost_basis_by_year():
    idx = list(pd.date_range("2021-12-30", periods=5, freq="D"))
    df = pd.DataFrame({"open": [100, 100, 200, 200, 200], "high": 0, "low": 0,
                       "close": [100, 100, 200, 200, 200], "volume": 1}, index=idx)
    result = simulate_flat_dca(df, "TEST", daily_amount=10.0, execution_indices=[0, 1, 2, 3, 4])

    deployed = deployed_by_bucket(result, year_bucket)
    assert deployed[2021] == pytest.approx(20.0)   # idx 0,1 -> 2021-12-30/31
    assert deployed[2022] == pytest.approx(30.0)    # idx 2,3,4 -> 2022

    basis = cost_basis_by_bucket(result, year_bucket)
    # cost basis = spend/units, and units are net of the 0.10% fee, so the
    # effective basis is very slightly ABOVE the raw price paid (price /
    # (1 - fee_pct)) -- that's the fee correctly showing up in the basis.
    assert basis[2021] == pytest.approx(100.0 / (1 - 0.001))
    assert basis[2022] == pytest.approx(200.0 / (1 - 0.001))


def test_deployed_by_regime_uses_aligned_label_lookup():
    df = _df([100.0] * 4)
    result = simulate_flat_dca(df, "TEST", daily_amount=5.0, execution_indices=[0, 1, 2, 3])
    labels = pd.Series(["BULL", "BULL", "BEAR", "BEAR"], index=df.index)
    fn = regime_bucket_fn(labels)
    deployed = deployed_by_bucket(result, fn)
    assert deployed["BULL"] == pytest.approx(10.0)
    assert deployed["BEAR"] == pytest.approx(10.0)
