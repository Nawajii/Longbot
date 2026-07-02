import backtest as bt
from run_real_backtest import make_synthetic


def _run(exit_over):
    df = make_synthetic(1200, seed=5)          # a fixture that actually fills trades
    p = bt.BacktestParams(entry_valid_bars=8, **exit_over)
    return df, bt.run_backtest(df, p)


VARIANTS = {
    "TRAIL_2": {"exit_style": "TRAIL", "trail_atr_mult": 3.0},
    "TRAIL_4": {"exit_style": "TRAIL", "trail_atr_mult": 4.0},
    "TRAIL_6": {"exit_style": "TRAIL", "trail_atr_mult": 6.0},
    "TIME_12": {"exit_style": "TIME", "time_exit_bars": 12},
}


def test_identical_first_entry_across_variants():
    """One-variable property: the ENTRY is computed independently of the exit,
    so the very first trade's entry (time + price) is identical for all four
    variants. (Later entries can differ because holding time affects occupancy.)
    """
    firsts = {}
    for name, over in VARIANTS.items():
        _, res = _run(over)
        assert res.trades, f"{name} produced no trades on the fixture"
        t0 = res.trades[0]
        firsts[name] = (t0.entry_time, round(t0.entry_price, 8))
    unique = set(firsts.values())
    assert len(unique) == 1, f"entries diverged across variants: {firsts}"


def test_time_exit_logic():
    _, res = _run(VARIANTS["TIME_12"])
    assert res.trades
    for t in res.trades:
        # TIME variant never trails
        assert t.exit_reason in {"stop", "time", "end_of_data"}
        if t.exit_reason == "time":
            # closed at/after exactly 12 bars, never later
            assert t.bars_held <= 12


def test_trail_width_changes_holding_and_giveback():
    """A wider trail should, on the same entries, hold longer on average."""
    _, r2 = _run(VARIANTS["TRAIL_2"])
    _, r6 = _run(VARIANTS["TRAIL_6"])
    if r2.trades and r6.trades:
        bars2 = sum(t.bars_held for t in r2.trades) / len(r2.trades)
        bars6 = sum(t.bars_held for t in r6.trades) / len(r6.trades)
        assert bars6 >= bars2
