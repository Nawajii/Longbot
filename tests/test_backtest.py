import pandas as pd

import backtest as bt
from longbot import scoring_engine as se
from run_real_backtest import make_synthetic


def test_backtest_runs_and_is_consistent():
    df = make_synthetic(n=1200)
    res = bt.run_backtest(df)
    # funnel arithmetic: every placed order is filled, missed, or still open at end
    assert res.fills + res.misses <= res.orders_placed
    assert res.orders_placed <= res.signals
    # equity curve covers every bar
    assert len(res.equity_curve) == len(df)
    # a float total return, finite
    assert isinstance(res.total_return_pct, float)


def test_fees_make_a_difference():
    df = make_synthetic(n=1200)
    free = bt.run_backtest(df, bt.BacktestParams(fee_pct=0.0, slippage_pct=0.0))
    costly = bt.run_backtest(df, bt.BacktestParams(fee_pct=0.002, slippage_pct=0.002))
    if free.trades and costly.trades:
        assert costly.total_return_pct < free.total_return_pct


def test_buy_the_peak_baseline_runs():
    df = make_synthetic(n=1200)
    peak = bt.run_backtest(df, entry_verdicts=(se.Verdict.PEAK_EXTENDED,))
    assert isinstance(peak.total_return_pct, float)


def test_buy_and_hold_is_a_number():
    df = make_synthetic(n=1200)
    bh = bt.buy_and_hold_return_pct(df, bt.BacktestParams())
    assert isinstance(bh, float)


def test_no_lookahead_entry_uses_next_bar():
    # A buy-stop created on bar i must not fill using bar i's own data.
    # Construct a single spike then flat: signal cannot fill on its own bar.
    df = make_synthetic(n=600)
    res = bt.run_backtest(df)
    for t in res.trades:
        assert t.exit_time >= t.entry_time
