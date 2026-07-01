from dataclasses import dataclass

import numpy as np

import run_location_test as X
from run_real_backtest import make_synthetic


@dataclass
class FakeTrade:
    pnl: float
    r_multiple: float
    tag: str = ""


def test_expectancy_math():
    n, wr, e = X._expectancy([FakeTrade(1, 2.0), FakeTrade(-1, -1.0)])
    assert n == 2 and wr == 0.5 and abs(e - 0.5) < 1e-9


def test_expectancy_empty():
    n, wr, e = X._expectancy([])
    assert n == 0 and np.isnan(wr) and np.isnan(e)


def test_experiment_runs_and_reports(monkeypatch):
    seeds = iter(range(1, 999))
    monkeypatch.setattr(X, "_fetch_window",
                        lambda *a, **k: make_synthetic(700, seed=next(seeds)))
    scen = {"BULL": ("2023-11-01", "2024-03-01"), "CHOP": ("2023-05-01", "2023-09-01")}
    R = X.run_experiment(["P1", "P2"], "4h", scen, 1000.0, 8, "BTCUSDT")
    assert R["executed"] == 4          # 2 pairs x 2 scenarios
    out = X.format_experiment(R)
    assert "PRIMARY TEST" in out
    assert "SECONDARY TEST" in out
    assert "VERDICT" in out
    # both AT_NODE and IN_VOID buckets exist as keys even if empty
    assert "AT_NODE" in R["tagged"]
