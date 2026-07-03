from dataclasses import dataclass

import numpy as np
import pandas as pd

import run_validation as V


@dataclass
class FakeTrade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    r_multiple: float
    pnl: float


def _t(year, r):
    ts = pd.Timestamp(f"{year}-06-01")
    return FakeTrade(ts, ts + pd.Timedelta(hours=4), r, r)


def test_fold_bucketing():
    assert V._fold_of(pd.Timestamp("2020-03-01")) == "IS"     # design era
    assert V._fold_of(pd.Timestamp("2021-12-31")) == "IS"
    assert V._fold_of(pd.Timestamp("2022-01-01")) == "2022"
    assert V._fold_of(pd.Timestamp("2024-07-01")) == "2024"
    assert V._fold_of(pd.Timestamp("2025-01-01")) == "2025+"
    assert V._fold_of(pd.Timestamp("2026-07-01")) == "2025+"


def test_dd_r_is_negative_on_a_drawdown():
    # cumulative R: +1, -3 -> -2, +1 -> -1 ; peak was +1, trough -2 => dd -3
    trades = [_t(2022, 1.0), _t(2023, -3.0), _t(2024, 1.0)]
    # exit_time order follows year here
    assert abs(V._dd_r(trades) - (-3.0)) < 1e-9


def test_rung2_concentration_flags_outlier_dependence():
    # 30 tiny losers + 5 huge winners: removing the top 5 must flip it negative
    trades = [_t(2021, -0.2) for _ in range(30)] + [_t(2021, 20.0) for _ in range(5)]
    R = {"trades": {"TRAIL_6": trades}}
    _lines, ok, _notes = V.rung2(R, "TRAIL_6")
    assert ok is False            # outlier-dependent AND all winners from mania


def test_rung2_passes_when_broad_and_recent():
    # many modestly-positive trades across years, winners not all mania
    trades = ([_t(2022, 0.4) for _ in range(40)]
              + [_t(2023, 0.4) for _ in range(40)]
              + [_t(2024, 3.0), _t(2024, 3.2), _t(2025, 3.1), _t(2023, 3.3), _t(2022, 3.4)])
    R = {"trades": {"TRAIL_6": trades}}
    _lines, ok, _notes = V.rung2(R, "TRAIL_6")
    assert ok is True


def test_regime_split_uses_asof():
    idx = pd.date_range("2022-01-01", periods=100, freq="4h")
    label = pd.Series(["BEAR"] * 100, index=idx)
    tr = FakeTrade(idx[50], idx[51], 1.0, 1.0)
    split = V._regime_split([tr], label)
    assert len(split["BEAR"]) == 1 and len(split["BULL"]) == 0
