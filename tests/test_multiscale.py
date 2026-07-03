import numpy as np
import pandas as pd

import backtest as bt
from longbot import multiscale as ms
from run_real_backtest import make_synthetic


def test_daily_gate_no_lookahead():
    """THE critical check: a 4h bar's daily-gate value must not change when future
    bars (including the rest of its own day) are appended. If it does, the gate is
    peeking at the in-progress daily candle — a silent cheat.
    """
    df = make_synthetic(2000, seed=4)
    # pick a 4h bar in the MIDDLE of a day (not the first bar), so the current
    # day's candle is still forming — the hardest case for lookahead.
    i = 1500
    while df.index[i].hour == 0:      # ensure we're mid-day
        i += 1
    full = ms.daily_gate_series(df)
    truncated = ms.daily_gate_series(df.iloc[: i + 1])
    assert bool(full.iloc[i]) == bool(truncated.iloc[i]), (
        f"daily gate at {df.index[i]} changed when future bars were removed — lookahead!"
    )


def test_daily_gate_is_a_pure_filter():
    """AND-ing the daily gate can only REMOVE trades, never add them: the gated
    trade count must be <= the ungated count on identical settings.
    """
    df = make_synthetic(2500, seed=4)
    from longbot import scoring_engine as se
    sp = se.ScoringParams(use_location=True)
    p = bt.BacktestParams(entry_valid_bars=8, use_node_stop=True,
                          exit_style="TRAIL", trail_atr_mult=6.0)
    ungated = bt.run_backtest(df, p, scoring_params=sp)
    gate = ms.daily_gate_series(df)
    gated = bt.run_backtest(df, p, scoring_params=sp, entry_gate=gate)
    assert gated.orders_placed <= ungated.orders_placed


def test_daily_gate_boolean_and_aligned():
    df = make_synthetic(1500, seed=1)
    g = ms.daily_gate_series(df)
    assert g.dtype == bool
    assert g.index.equals(df.index)
    # early bars (daily indicators still warming up) must be False, never NaN
    assert not g.iloc[:50].any()
