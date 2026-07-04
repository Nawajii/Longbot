"""End-to-end tests of the flush backtest engine: entry timing (next-bar
open), target/stop/time/funding-normalized exits, R-multiple math, MAE/MFE
tracking, and the full-engine no-lookahead truncation-invariance proof.
"""

import numpy as np
import pandas as pd

from longbot import indicators as ind
from longbot.flush_engine import FlushParams, run_flush_backtest


def _base_df(n=60, price=100.0, start="2021-01-01"):
    idx = pd.date_range(start, periods=n, freq="4h")
    return pd.DataFrame({
        "open": price, "high": price * 1.005, "low": price * 0.995,
        "close": price, "volume": 1000.0,
    }, index=idx)


def _funding_features(n_bars_idx, pct_rank_value=0.5, funding_rate_value=0.0001, median_value=0.0001):
    return pd.DataFrame({
        "funding_rate": funding_rate_value,
        "pct_rank": pct_rank_value,
        "trailing_median": median_value,
    }, index=n_bars_idx)


def _plant_flush(df, i, price=100.0):
    """Overwrite bar i with a hand-built flush/hammer candle."""
    df.loc[df.index[i], ["open", "high", "low", "close", "volume"]] = [
        price, price * 1.005, price * 0.85, price * 0.99, 5000.0,
    ]
    return df


def test_entry_fires_on_bar_after_signal_not_the_signal_bar():
    df = _base_df(60)
    df = _plant_flush(df, 25)
    feat = _funding_features(df.index, pct_rank_value=0.9)  # spring loaded throughout

    result = run_flush_backtest(df, feat, FlushParams())
    assert len(result.trades) >= 1
    t = result.trades[0]
    assert t.entry_time == df.index[26], "entry must be the bar AFTER the flush candle, not the flush bar itself"


def test_target_exit_profitable_and_r_multiple_positive():
    params = FlushParams(target_atr_mult=1.5, stop_atr_mult=2.0, time_stop_bars=30)
    df = _base_df(80)
    df = _plant_flush(df, 25)
    feat = _funding_features(df.index, pct_rank_value=0.9, median_value=0.0)  # never funding-normalizes

    atr26 = float(ind.atr(df, params.atr_period).iloc[25])  # ATR frozen at the SIGNAL bar (25)
    entry_price = df["open"].iloc[26] * (1 + params.slippage_pct)
    target_price = entry_price + params.target_atr_mult * atr26

    # bar 27's high touches the target
    df.loc[df.index[27], "high"] = target_price + 1.0

    result = run_flush_backtest(df, feat, params)
    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.exit_reason == "target"
    assert t.r_multiple > 0


def test_stop_exit_is_a_loss_and_mandatory():
    params = FlushParams()
    df = _base_df(80)
    df = _plant_flush(df, 25)
    feat = _funding_features(df.index, pct_rank_value=0.9, median_value=0.0)

    atr26 = float(ind.atr(df, params.atr_period).iloc[25])
    entry_price = df["open"].iloc[26] * (1 + params.slippage_pct)
    stop_price = entry_price - params.stop_atr_mult * atr26

    # bar 27's low breaches the stop (a real crash continuing down)
    df.loc[df.index[27], "low"] = stop_price - 1.0
    df.loc[df.index[27], "open"] = entry_price  # no gap-through at the open

    result = run_flush_backtest(df, feat, params)
    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.exit_reason == "stop"
    assert t.r_multiple < 0
    assert abs(t.r_multiple) > 0.5   # placed at ~1R


def test_time_stop_after_30_bars_when_nothing_else_triggers():
    params = FlushParams(time_stop_bars=30)
    df = _base_df(120)
    df = _plant_flush(df, 25)
    feat = _funding_features(df.index, pct_rank_value=0.9, median_value=0.0)

    result = run_flush_backtest(df, feat, params)
    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.exit_reason == "time"
    assert t.bars_held == params.time_stop_bars


def test_funding_normalized_exit_fires_when_median_crosses():
    params = FlushParams()
    df = _base_df(80)
    df = _plant_flush(df, 25)
    # funding starts elevated (spring loaded, entry allowed) then normalizes at bar 30
    feat = _funding_features(df.index, pct_rank_value=0.9, funding_rate_value=0.0005, median_value=0.0001)
    feat.loc[df.index[30]:, "funding_rate"] = 0.00001   # crowd de-levers: funding falls below its median

    result = run_flush_backtest(df, feat, params)
    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.exit_reason == "funding_normalized"
    assert t.exit_time == df.index[30]


def test_mae_mfe_tracked_across_trade_life():
    params = FlushParams(time_stop_bars=30)
    df = _base_df(80)
    df = _plant_flush(df, 25)
    feat = _funding_features(df.index, pct_rank_value=0.9, median_value=0.0)

    entry_price = df["open"].iloc[26] * (1 + params.slippage_pct)
    # bar 28 dips (adverse excursion) then bar 29 rallies (favorable excursion), no threshold breached
    df.loc[df.index[28], "low"] = entry_price * 0.99
    df.loc[df.index[29], "high"] = entry_price * 1.01

    result = run_flush_backtest(df, feat, params)
    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.mae_r < 0        # the dip shows up as an adverse excursion
    assert t.mfe_r > 0        # the rally shows up as a favorable excursion


def test_post_stop_diagnostic_is_populated_only_for_stops():
    params = FlushParams()
    df = _base_df(90)
    df = _plant_flush(df, 25)
    feat = _funding_features(df.index, pct_rank_value=0.9, median_value=0.0)

    atr26 = float(ind.atr(df, params.atr_period).iloc[25])
    entry_price = df["open"].iloc[26] * (1 + params.slippage_pct)
    stop_price = entry_price - params.stop_atr_mult * atr26
    df.loc[df.index[27], "low"] = stop_price - 1.0
    df.loc[df.index[27], "open"] = entry_price
    # after the stop, price recovers well above entry
    df.loc[df.index[35], "high"] = entry_price * 1.05

    result = run_flush_backtest(df, feat, params)
    t = result.trades[0]
    assert t.exit_reason == "stop"
    assert t.post_stop_mfe_r is not None
    assert t.post_stop_mfe_r > 0   # the diagnostic should show recovery here


def test_one_position_at_a_time():
    params = FlushParams(time_stop_bars=10)
    df = _base_df(120)
    df = _plant_flush(df, 25)
    df = _plant_flush(df, 30)   # a second flush candle WHILE the first trade should still be open
    feat = _funding_features(df.index, pct_rank_value=0.9, median_value=0.0)

    result = run_flush_backtest(df, feat, params)
    # trades must not overlap in time
    ordered = sorted(result.trades, key=lambda t: t.entry_time)
    for a, b in zip(ordered, ordered[1:]):
        assert a.exit_time <= b.entry_time


def test_full_engine_truncation_invariance():
    """The strongest end-to-end guarantee: replaying the engine on a
    truncated history must not change any trade that completed before the
    truncation point."""
    params = FlushParams(time_stop_bars=15)
    df = _base_df(150)
    df = _plant_flush(df, 25)
    df = _plant_flush(df, 70)
    feat = _funding_features(df.index, pct_rank_value=0.9, median_value=0.0)

    cutoff = 100
    trades_full = run_flush_backtest(df, feat, params).trades
    trades_trunc = run_flush_backtest(df.iloc[:cutoff], feat.iloc[:cutoff], params).trades

    cutoff_time = df.index[cutoff - 1]
    completed_full = [t for t in trades_full if t.exit_time <= cutoff_time]

    assert len(completed_full) == len(trades_trunc)
    for tf, tt in zip(completed_full, trades_trunc):
        assert tf.entry_time == tt.entry_time
        assert tf.entry_price == tt.entry_price
        assert tf.exit_time == tt.exit_time
        assert tf.exit_price == tt.exit_price
        assert tf.exit_reason == tt.exit_reason
        assert tf.r_multiple == tt.r_multiple


def test_fees_and_slippage_reduce_return():
    params = FlushParams()
    df = _base_df(80)
    df = _plant_flush(df, 25)
    feat = _funding_features(df.index, pct_rank_value=0.9, median_value=0.0)

    zero_cost = FlushParams(fee_pct=0.0, slippage_pct=0.0)
    r_with_cost = run_flush_backtest(df, feat, params).trades
    r_no_cost = run_flush_backtest(df, feat, zero_cost).trades
    assert len(r_with_cost) == len(r_no_cost) == 1
    # same exit outcome, but the frictionless run should show a strictly
    # better (or at worst equal) R-multiple once a target/time exit occurs
    if r_with_cost[0].exit_reason != "stop":
        assert r_no_cost[0].r_multiple >= r_with_cost[0].r_multiple
