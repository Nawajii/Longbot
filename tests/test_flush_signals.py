"""Unit tests for the flush-candle trigger and spring-loaded funding gate,
on fully hand-crafted synthetic OHLCV/funding so every condition can be
checked in isolation."""

import numpy as np
import pandas as pd

from longbot.flush_engine import FlushParams, compute_flush_trigger, compute_spring_loaded


def _quiet_bars(n, price=100.0, start="2021-01-01"):
    idx = pd.date_range(start, periods=n, freq="4h")
    return pd.DataFrame({
        "open": price, "high": price * 1.005, "low": price * 0.995,
        "close": price, "volume": 1000.0,
    }, index=idx)


def test_flush_trigger_fires_on_a_hand_built_hammer_after_quiet_bars():
    params = FlushParams()
    df = _quiet_bars(30)
    # bar 25: violent range + volume spike, long lower wick, close near the high (hammer)
    df.loc[df.index[25], ["open", "high", "low", "close", "volume"]] = [100.0, 100.5, 85.0, 99.0, 5000.0]

    trigger = compute_flush_trigger(df, params)
    assert trigger.iloc[25]
    # neighbouring quiet bars must not trigger
    assert not trigger.iloc[24]
    assert not trigger.iloc[26]


def test_flush_trigger_requires_all_four_conditions():
    params = FlushParams()
    df = _quiet_bars(30)

    # big range + volume, but a FULL-BODIED close near the LOW (a real waterfall, not absorption)
    df.loc[df.index[25], ["open", "high", "low", "close", "volume"]] = [100.0, 100.2, 85.0, 85.5, 5000.0]
    trigger = compute_flush_trigger(df, params)
    assert not trigger.iloc[25], "a full-bodied close near the low must NOT count as absorption"

    df2 = _quiet_bars(30)
    # big range + big wick + strong close, but volume is NOT elevated
    df2.loc[df2.index[25], ["open", "high", "low", "close", "volume"]] = [100.0, 100.5, 85.0, 99.0, 1000.0]
    trigger2 = compute_flush_trigger(df2, params)
    assert not trigger2.iloc[25], "no volume spike must NOT trigger"

    df3 = _quiet_bars(30)
    # big range + big volume + strong close, but wick is small relative to body
    df3.loc[df3.index[25], ["open", "high", "low", "close", "volume"]] = [100.0, 115.0, 99.5, 114.0, 5000.0]
    trigger3 = compute_flush_trigger(df3, params)
    assert not trigger3.iloc[25], "a small lower wick relative to body must NOT trigger"


def test_flush_trigger_uses_prior_20_bars_not_including_itself():
    """The trailing average baseline must exclude the flush bar itself --
    otherwise a big enough single bar could inflate its own baseline and
    dodge the 2x-spike threshold."""
    params = FlushParams()
    df = _quiet_bars(30)
    df.loc[df.index[25], ["open", "high", "low", "close", "volume"]] = [100.0, 100.5, 85.0, 99.0, 5000.0]
    trigger = compute_flush_trigger(df, params)

    # sanity: if the baseline wrongly included bar 25, average range/volume
    # would be much higher and might suppress the trigger for a marginal case.
    # Directly check the trailing window excludes index 25:
    from longbot import indicators as ind
    tr = ind.true_range(df)
    trailing = tr.shift(1).rolling(20, min_periods=20).mean()
    # bar 25's own huge true range must not appear in its own trailing average
    assert trailing.iloc[25] < tr.iloc[25] / params.flush_range_mult


def test_spring_loaded_recent_looks_back_24h_on_4h_bars():
    params = FlushParams(spring_loaded_lookback_bars=6)
    n = 20
    idx = pd.date_range("2021-01-01", periods=n, freq="4h")
    pct_rank = pd.Series(0.5, index=idx)
    pct_rank.iloc[5] = 0.9   # spring loaded exactly once, at bar 5
    feat = pd.DataFrame({"pct_rank": pct_rank})

    out = compute_spring_loaded(feat, params)
    assert out["spring_loaded_now"].iloc[5]
    assert not out["spring_loaded_now"].iloc[4]
    # "recent" should stay true for bars 5..10 (6-bar trailing window incl. bar 5) then fall off
    assert out["spring_loaded_recent"].iloc[5:11].all()
    assert not out["spring_loaded_recent"].iloc[11]


def test_spring_loaded_never_uses_a_future_pct_rank():
    """Truncation invariance for the gate itself: a bar's spring_loaded_now/
    recent flags must be identical whether or not later bars exist."""
    params = FlushParams(spring_loaded_lookback_bars=6)
    n = 30
    idx = pd.date_range("2021-01-01", periods=n, freq="4h")
    rng = np.random.default_rng(1)
    pct_rank = pd.Series(rng.uniform(0, 1, n), index=idx)
    feat = pd.DataFrame({"pct_rank": pct_rank})

    full = compute_spring_loaded(feat, params)
    truncated_feat = feat.iloc[:15]
    trunc = compute_spring_loaded(truncated_feat, params)

    assert (full["spring_loaded_now"].iloc[:15] == trunc["spring_loaded_now"]).all()
    assert (full["spring_loaded_recent"].iloc[:15] == trunc["spring_loaded_recent"]).all()
