"""
Entry/exit state-machine tests on fully controlled synthetic data: the
first 90 days establish a real (fitted, not hand-waved) formation window,
then subsequent days are engineered using that exact formation's beta/mean/
std to hit precise z thresholds -- proving the target/stop/time-stop logic
and R-multiple sign/math are correct.
"""

import numpy as np
import pandas as pd

from pairs_pipeline import config
from pairs_pipeline.cointegration import fit_formation_window
from pairs_pipeline.engine import run_pair


def _formation_days(n=90, seed=42):
    rng = np.random.default_rng(seed)
    log_b = np.cumsum(rng.normal(0, 0.02, n))
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.85 * noise[i - 1] + rng.normal(0, 0.01)
    log_a = 1.0 * log_b + 0.2 + noise
    return log_a, log_b


def _price_for_z(z, formation, log_b_fixed):
    log_a = formation.mean + z * formation.std + formation.beta * log_b_fixed
    return float(np.exp(log_a))


def _build_frame(extra_days_z, formation_seed=42):
    """90-day real formation + one row per z value in extra_days_z, all
    with B held fixed at its last formation-day price."""
    log_a90, log_b90 = _formation_days(seed=formation_seed)
    formation = fit_formation_window(
        pd.Series(log_a90), pd.Series(log_b90),
    )
    assert formation.cointegrated, "test fixture must produce a cointegrated formation window"

    b_fixed_log = log_b90[-1]
    b_fixed_price = float(np.exp(b_fixed_log))

    dates = pd.date_range("2021-01-01", periods=90 + len(extra_days_z), freq="D").date
    a_prices = list(np.exp(log_a90))
    b_prices = list(np.exp(log_b90))
    for z in extra_days_z:
        a_prices.append(_price_for_z(z, formation, b_fixed_log))
        b_prices.append(b_fixed_price)

    prices = pd.DataFrame({"A": a_prices, "B": b_prices}, index=dates)
    return prices, formation


def test_entry_then_target_exit_is_profitable():
    # day 90: entry at z=-2.2 (breaches -2.0); day 91: z=-0.20 (breaches -0.25 target)
    prices, formation = _build_frame(extra_days_z=[-2.2, -0.20])
    trades = run_pair(prices, "A", "B")

    assert len(trades) == 1
    t = trades[0]
    assert t.direction == "long_A"
    assert t.entry_date == prices.index[90]
    assert t.exit_date == prices.index[91]
    assert t.exit_reason == "target"
    assert t.r_multiple > 0


def test_entry_then_stop_exit_is_a_loss():
    # day 90: entry at z=-2.2; day 91: z=-3.6 (breaches -3.5 stop)
    prices, formation = _build_frame(extra_days_z=[-2.2, -3.6])
    trades = run_pair(prices, "A", "B")

    assert len(trades) == 1
    t = trades[0]
    assert t.direction == "long_A"
    assert t.exit_reason == "stop"
    assert t.r_multiple < 0
    # by definition the stop is placed at ~1R away
    assert abs(t.r_multiple) > 0.5


def test_entry_then_time_stop_after_30_days():
    # day 90: entry at z=-2.2. Days 91..120 (30 more days): z=-1.0, never
    # touching target (-0.25) or stop (-3.5) -> must exit on day 120 as "time".
    extra = [-2.2] + [-1.0] * 30
    prices, formation = _build_frame(extra_days_z=extra)
    trades = run_pair(prices, "A", "B")

    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "time"
    assert t.days_held == config.TIME_STOP_DAYS
    assert t.exit_date == prices.index[90 + 30]


def test_long_b_direction_on_positive_z_breach():
    log_a90, log_b90 = _formation_days(seed=42)
    formation = fit_formation_window(pd.Series(log_a90), pd.Series(log_b90))
    assert formation.cointegrated

    b_fixed_log = log_b90[-1]
    # For direction long_B we need to move B's price to hit z, holding A fixed.
    a_fixed_log = log_a90[-1]
    a_fixed_price = float(np.exp(a_fixed_log))

    def price_b_for_z(z):
        # spread = log(A) - beta*log(B) => log(B) = (log(A) - spread) / beta
        spread = formation.mean + z * formation.std
        log_b = (a_fixed_log - spread) / formation.beta
        return float(np.exp(log_b))

    dates = pd.date_range("2021-01-01", periods=92, freq="D").date
    a_prices = list(np.exp(log_a90)) + [a_fixed_price, a_fixed_price]
    b_prices = list(np.exp(log_b90)) + [price_b_for_z(2.2), price_b_for_z(0.20)]
    prices = pd.DataFrame({"A": a_prices, "B": b_prices}, index=dates)

    trades = run_pair(prices, "A", "B")
    assert len(trades) == 1
    t = trades[0]
    assert t.direction == "long_B"
    assert t.exit_reason == "target"
    assert t.r_multiple > 0


def test_fees_and_slippage_reduce_return_vs_frictionless():
    from pairs_pipeline.engine import _apply_cost
    price = 100.0
    buy_fill = _apply_cost(price, buying=True)
    sell_fill = _apply_cost(price, buying=False)
    assert buy_fill > price
    assert sell_fill < price
    # round trip at unchanged mid price should be a small net loss from costs
    round_trip_return = (sell_fill - buy_fill) / buy_fill
    assert round_trip_return < 0
    # Fee is applied on top of the already-slipped price, so the exact
    # round-trip cost has a small second-order term beyond the linear
    # approximation below -- assert it's close, not bit-exact.
    expected_cost = 2 * (config.FEE_PCT + config.SLIPPAGE_PCT)
    assert abs(round_trip_return + expected_cost) < 1e-4
