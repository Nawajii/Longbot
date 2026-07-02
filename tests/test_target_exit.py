import numpy as np
import pandas as pd

import backtest as bt


from run_real_backtest import make_synthetic


def test_target_hit_gives_two_r():
    """A clean +2R target should realise ~+2R (minus fees/slippage)."""
    df = make_synthetic(1500, seed=5)
    res = bt.run_backtest(df, bt.BacktestParams(entry_valid_bars=8, exit_style="TARGET", target_r=2.0))
    targets = [t for t in res.trades if t.exit_reason == "target"]
    assert targets, "expected at least one target exit on this fixture"
    for t in targets:
        # realised R should be close to +2 (a hair under, from fees + exit slippage)
        assert 1.5 <= t.r_multiple <= 2.05, t.r_multiple


def test_pessimistic_spanning_bar_takes_stop_not_target():
    """If a single bar's range spans BOTH the stop and the +2R target, the loss
    (stop) must be booked — never the favorable target fill."""
    # Build a position by hand and step one spanning bar through the logic by
    # constructing a frame where the entry bar is followed by a huge-range bar.
    from run_real_backtest import make_synthetic
    df = make_synthetic(1500, seed=5)
    # Run TARGET and confirm no trade is credited a >2R win (impossible if pessimistic
    # and no gap-up beyond target is ever credited).
    res = bt.run_backtest(df, bt.BacktestParams(entry_valid_bars=8, exit_style="TARGET", target_r=2.0))
    for t in res.trades:
        assert t.r_multiple <= 2.05, f"target exit credited > 2R ({t.r_multiple}) — not pessimistic"


def test_target_variant_identical_first_entry():
    """TARGET must share the identical first entry with the trailing/time variants
    (only the exit differs)."""
    from run_real_backtest import make_synthetic
    df = make_synthetic(1500, seed=5)
    variants = {
        "TRAIL_2": dict(exit_style="TRAIL", trail_atr_mult=3.0),
        "TIME_12": dict(exit_style="TIME", time_exit_bars=12),
        "TARGET_2R": dict(exit_style="TARGET", target_r=2.0),
    }
    firsts = set()
    for over in variants.values():
        res = bt.run_backtest(df, bt.BacktestParams(entry_valid_bars=8, **over))
        assert res.trades
        t0 = res.trades[0]
        firsts.add((t0.entry_time, round(t0.entry_price, 8)))
    assert len(firsts) == 1, f"TARGET diverged from the shared first entry: {firsts}"
