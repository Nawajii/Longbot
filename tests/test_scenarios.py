from dataclasses import dataclass

import numpy as np

import run_scenarios as rs
from run_real_backtest import make_synthetic


@dataclass
class FakeTrade:
    pnl: float
    r_multiple: float


def test_agg_math():
    trades = [FakeTrade(10, 2.0), FakeTrade(-5, -1.0), FakeTrade(20, 4.0), FakeTrade(-5, -1.0)]
    a = rs._agg(trades)
    assert a["n"] == 4
    assert a["win_rate"] == 0.5
    assert abs(a["expectancy_r"] - 1.0) < 1e-9  # (2-1+4-1)/4


def test_agg_empty():
    a = rs._agg([])
    assert a["n"] == 0
    assert np.isnan(a["win_rate"])


def test_sweep_end_to_end_with_synthetic(monkeypatch):
    # No network: every fetch returns synthetic data so we exercise the whole
    # sweep + aggregation + summary formatting path.
    monkeypatch.setattr(rs, "fetch_binance_klines_range",
                        lambda *a, **k: make_synthetic(500))
    scenarios = {"BULL": ("2023-11-01", "2024-03-01"), "CHOP": ("2023-05-01", "2023-09-01")}
    records = rs.run_sweep(["BTCUSDT", "ETHUSDT"], "4h", scenarios, entry_valid_bars=8)
    assert len(records) == 4  # 2 pairs x 2 scenarios
    assert all(r.skipped is None for r in records)
    out = rs.summarize(records, scenarios)
    assert "COMBINED SUMMARY" in out
    assert "PER-SCENARIO BREAKDOWN" in out
    assert "BULL" in out and "CHOP" in out


def test_sweep_skips_insufficient_data(monkeypatch):
    monkeypatch.setattr(rs, "fetch_binance_klines_range",
                        lambda *a, **k: make_synthetic(50))  # below MIN_USABLE_BARS
    scenarios = {"BEAR": ("2022-04-01", "2022-08-01")}
    records = rs.run_sweep(["BTCUSDT"], "4h", scenarios)
    assert all(r.skipped is not None for r in records)
    # summary must not crash on an all-skipped sweep
    out = rs.summarize(records, scenarios)
    assert "TOTAL trades : 0" in out
