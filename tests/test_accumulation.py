"""Unit tests for the adaptive-DCA ladder, the three simulators, and the
no-lookahead guarantee (signal on bar t's return, execute at bar t+1's
open)."""

import numpy as np
import pandas as pd
import pytest

from longbot.accumulation import (
    DEFAULT_LADDER,
    executable_indices,
    ladder_amount,
    simulate_adaptive_dca,
    simulate_flat_dca,
    simulate_flat_same_capital,
    simulate_lump_sum,
)


def _df(closes, opens=None, start="2021-01-01"):
    n = len(closes)
    idx = pd.date_range(start, periods=n, freq="D")
    opens = opens if opens is not None else closes
    return pd.DataFrame({
        "open": opens, "high": [max(o, c) * 1.001 for o, c in zip(opens, closes)],
        "low": [min(o, c) * 0.999 for o, c in zip(opens, closes)],
        "close": closes, "volume": 1000.0,
    }, index=idx)


# --------------------------------------------------------------------------- #
# ladder boundary tests -- every band edge from the pre-registered table
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("r,expected", [
    (0.06, 0.0),        # > +5% -> skip
    (0.0500001, 0.0),   # just above +5% -> skip
    (0.05, 5.0),        # exactly +5% -> base (inclusive)
    (0.0, 5.0),         # flat day -> base
    (-0.02, 5.0),       # exactly -2% -> base (inclusive)
    (-0.0200001, 7.0),  # just below -2% -> tier1
    (-0.03, 7.0),       # -3% -> tier1
    (-0.05, 10.0),      # exactly -5% -> tier2 (inclusive)
    (-0.0500001, 10.0),  # just below -5% -> tier2
    (-0.06, 10.0),      # -6% -> tier2
    (-0.08, 15.0),      # exactly -8% -> tier3 (inclusive)
    (-0.09, 15.0),      # -9% -> tier3
    (-0.12, 25.0),      # exactly -12% -> tier4 (inclusive)
    (-0.20, 25.0),      # -20% -> tier4
])
def test_ladder_boundaries_match_the_pre_registered_table(r, expected):
    assert ladder_amount(r) == expected


def test_ladder_nan_return_means_no_buy():
    assert ladder_amount(float("nan")) == 0.0


# --------------------------------------------------------------------------- #
# executable index scaffolding
# --------------------------------------------------------------------------- #
def test_executable_indices_range():
    assert executable_indices(10) == list(range(2, 10))


# --------------------------------------------------------------------------- #
# adaptive DCA: execution timing + amount correctness
# --------------------------------------------------------------------------- #
def test_adaptive_entry_uses_prior_bar_return_and_fills_at_next_open():
    # closes: day0=100, day1=100 (flat, r=0 -> base $5 signal), day2 open=105 (execution)
    closes = [100.0, 100.0, 103.0, 103.0]
    opens = [100.0, 100.0, 105.0, 103.0]
    df = _df(closes, opens)
    result = simulate_adaptive_dca(df, "TEST")

    # index 2's signal is bar 1's return = (100/100 - 1) = 0 -> base $5, filled at open[2]=105
    buy_at_2 = next(b for b in result.buys if b.execution_index == 2)
    assert buy_at_2.daily_return == 0.0
    assert buy_at_2.amount_usd == 5.0
    assert buy_at_2.price == 105.0
    assert buy_at_2.signal_date == df.index[1]
    assert buy_at_2.execution_date == df.index[2]


def test_adaptive_dip_triggers_larger_buy_next_open():
    # day1 close crashes -15% vs day0 -> tier4 $25, executed at day2's open
    closes = [100.0, 85.0, 85.0, 85.0]
    opens = [100.0, 100.0, 90.0, 85.0]
    df = _df(closes, opens)
    result = simulate_adaptive_dca(df, "TEST")

    buy_at_2 = next(b for b in result.buys if b.execution_index == 2)
    assert buy_at_2.daily_return == pytest.approx(-0.15)
    assert buy_at_2.amount_usd == 25.0
    assert buy_at_2.price == 90.0


def test_adaptive_euphoria_day_produces_no_buy():
    closes = [100.0, 110.0, 110.0, 110.0]   # day1: +10% -> euphoria, skip
    df = _df(closes)
    result = simulate_adaptive_dca(df, "TEST")
    assert all(b.execution_index != 2 for b in result.buys)


def test_fee_reduces_units_below_naive_amount_over_price():
    closes = [100.0, 100.0, 100.0, 100.0]
    df = _df(closes)
    result = simulate_adaptive_dca(df, "TEST")
    buy = next(b for b in result.buys if b.execution_index == 2)
    assert buy.fee_usd == pytest.approx(buy.amount_usd * DEFAULT_LADDER.fee_pct)
    assert buy.units < buy.amount_usd / buy.price   # fee eats into units bought


# --------------------------------------------------------------------------- #
# no-lookahead: truncation invariance
# --------------------------------------------------------------------------- #
def test_adaptive_dca_truncation_invariance():
    """Every buy decided/executed before a truncation point must be
    identical whether or not later bars exist (or are altered)."""
    rng = np.random.default_rng(11)
    n = 60
    closes = list(100.0 * np.exp(np.cumsum(rng.normal(0, 0.03, n))))
    df = _df(closes)

    cutoff = 40
    full = simulate_adaptive_dca(df, "TEST")
    trunc = simulate_adaptive_dca(df.iloc[:cutoff], "TEST")

    full_before_cutoff = [b for b in full.buys if b.execution_index < cutoff]
    assert len(full_before_cutoff) == len(trunc.buys)
    for bf, bt in zip(full_before_cutoff, trunc.buys):
        assert bf.execution_date == bt.execution_date
        assert bf.amount_usd == bt.amount_usd
        assert bf.price == bt.price
        assert bf.units == bt.units


def test_adaptive_dca_future_replacement_invariance():
    """Replacing the future entirely with different random data must not
    change any already-decided buy."""
    rng = np.random.default_rng(2)
    n_past = 30
    closes_past = list(100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n_past))))

    def build(future_seed):
        frng = np.random.default_rng(future_seed)
        closes_future = list(closes_past[-1] * np.exp(np.cumsum(frng.normal(0, 0.05, 20))))
        return _df(closes_past + closes_future)

    variant_1 = build(101)
    variant_2 = build(202)

    r1 = simulate_adaptive_dca(variant_1, "TEST")
    r2 = simulate_adaptive_dca(variant_2, "TEST")

    r1_past = [b for b in r1.buys if b.execution_index < n_past]
    r2_past = [b for b in r2.buys if b.execution_index < n_past]
    assert len(r1_past) == len(r2_past)
    for b1, b2 in zip(r1_past, r2_past):
        assert b1.amount_usd == b2.amount_usd
        assert b1.price == b2.price


# --------------------------------------------------------------------------- #
# flat DCA / same-capital / lump-sum
# --------------------------------------------------------------------------- #
def test_flat_dca_buys_constant_amount_on_every_executable_day():
    rng = np.random.default_rng(3)
    closes = list(100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, 30))))
    df = _df(closes)
    result = simulate_flat_dca(df, "TEST", daily_amount=5.0)
    assert len(result.buys) == len(executable_indices(len(df)))
    assert all(b.amount_usd == 5.0 for b in result.buys)


def test_flat_same_capital_matches_total_and_spreads_evenly():
    df = _df([100.0] * 10)
    buy_indices = [2, 4, 6, 8]
    result = simulate_flat_same_capital(df, "TEST", total_capital=100.0, buy_indices=buy_indices)
    assert len(result.buys) == 4
    assert all(b.amount_usd == pytest.approx(25.0) for b in result.buys)
    assert result.total_deployed == pytest.approx(100.0)


def test_lump_sum_is_a_single_buy_on_day_one():
    df = _df([100.0, 110.0, 90.0])
    result = simulate_lump_sum(df, "TEST", total_amount=1000.0)
    assert len(result.buys) == 1
    assert result.buys[0].execution_index == 0
    assert result.buys[0].price == 100.0
    assert result.total_deployed == 1000.0
