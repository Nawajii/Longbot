import numpy as np
import pandas as pd

from longbot import volume_profile as vp
from run_real_backtest import make_synthetic


def _flat_candles(price, vol, n):
    return (np.full(n, price - 0.5), np.full(n, price + 0.5), np.full(n, vol))


def test_poc_finds_the_busiest_price():
    # 100 candles trading around 50 (heavy), a few around 80 (light)
    lows = np.concatenate([np.full(100, 49.5), np.full(5, 79.5)])
    highs = np.concatenate([np.full(100, 50.5), np.full(5, 80.5)])
    vols = np.concatenate([np.full(100, 1000.0), np.full(5, 10.0)])
    prof = vp.compute_profile(lows, highs, vols)
    assert abs(prof.poc_price - 50.0) < 2.0          # POC sits at the busy price
    assert prof.poc_price < prof.vah                 # POC inside its value area


def test_value_area_holds_about_70pct():
    rng = np.random.default_rng(0)
    center = rng.normal(100, 2, 3000)                # volume concentrated near 100
    lows, highs, vols = center - 0.25, center + 0.25, np.ones(3000)
    prof = vp.compute_profile(lows, highs, vols, n_bins=50)
    in_va_mask = (prof.bin_centers >= prof.val) & (prof.bin_centers <= prof.vah)
    va_vol = prof.volume[in_va_mask].sum()
    frac = va_vol / prof.total_volume
    assert 0.60 <= frac <= 0.80                       # ~70%, within discretization slack


def test_hvn_and_lvn_identified():
    # two volume humps (HVNs) separated by a valley (LVN)
    lows = np.concatenate([np.full(200, 9.5), np.full(200, 19.5)])
    highs = np.concatenate([np.full(200, 10.5), np.full(200, 20.5)])
    vols = np.concatenate([np.full(200, 500.0), np.full(200, 500.0)])
    prof = vp.compute_profile(lows, highs, vols)
    assert len(prof.hvn_prices) >= 1
    # there should be a node near 10 and near 20
    assert any(abs(h - 10) < 1.5 for h in prof.hvn_prices)
    assert any(abs(h - 20) < 1.5 for h in prof.hvn_prices)


def test_location_at_node_vs_void():
    prof = vp.compute_profile(*_flat_candles(100.0, 1000.0, 120))
    at = vp.location(100.0, prof)                     # right on the busy price
    assert at.at_node and not at.in_void
    far = vp.location(130.0, prof)                    # nowhere near the node
    assert far.in_void and not far.at_node


def test_degenerate_window_does_not_crash():
    prof = vp.compute_profile(np.full(10, 5.0), np.full(10, 5.0), np.zeros(10))
    loc = vp.location(5.0, prof)
    assert np.isfinite(loc.support_price)


def test_no_lookahead_profile_is_stable():
    """THE critical check: bar i's location features must not change when future
    bars are appended. If they do, the profile is peeking ahead — a silent cheat.
    """
    df = make_synthetic(400)
    i = 250
    full = vp.compute_location_features(df)
    truncated = vp.compute_location_features(df.iloc[: i + 1])   # only bars <= i
    for col in vp.LOCATION_COLUMNS:
        a = full[col].iloc[i]
        b = truncated[col].iloc[i]
        if isinstance(a, (bool, np.bool_)):
            assert a == b, col
        else:
            assert (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9, col
