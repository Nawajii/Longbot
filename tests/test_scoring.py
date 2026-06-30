import pandas as pd

from longbot import scoring_engine as se


def make_row(**overrides):
    """A confirmed-uptrend, healthy-pullback row by default. Override to break it."""
    base = dict(
        close=110.0, ema_fast=108.0, ema_mid=105.0, ema_slow=100.0,
        ema_slow_slope=0.5, multiweek_return=0.20, adx=30.0,
        plus_di=30.0, minus_di=10.0,
        rsi=50.0, stoch_k=50.0, willr=-50.0, cci=0.0,
        atr=2.0, local_high=115.0, pullback_pct=0.04,
        stretch_atr=1.0, dist_fast_atr=0.5,
        rel_vol=1.2, obv_slope=5.0, aroon_up=80.0, aroon_osc=60.0,
        bb_pct_b=0.5, bb_bandwidth=0.1,
    )
    base.update(overrides)
    return pd.Series(base)


def test_healthy_pullback_is_actionable():
    score = se.classify_setup(make_row())
    assert score.verdict == se.Verdict.PULLBACK_IN_UPTREND
    assert score.is_actionable
    assert score.momentum_state in (se.MomentumState.COOLED, se.MomentumState.NEUTRAL)


def test_exhausted_momentum_is_peak_not_buy():
    # strong trend BUT overbought oscillators + stretched price = the trap
    score = se.classify_setup(make_row(rsi=82.0, stoch_k=90.0, willr=-10.0, stretch_atr=3.0))
    assert score.verdict == se.Verdict.PEAK_EXTENDED
    assert score.momentum_state == se.MomentumState.EXHAUSTED
    assert not score.is_actionable


def test_no_uptrend_is_skipped():
    score = se.classify_setup(make_row(close=95.0, adx=12.0, ema_slow_slope=-0.2))
    assert score.verdict == se.Verdict.NO_UPTREND


def test_deep_pullback_oversold_is_risk():
    score = se.classify_setup(make_row(rsi=25.0))
    assert score.verdict == se.Verdict.DEEP_PULLBACK_RISK


def test_stretched_but_not_exhausted_is_still_peak():
    # price far above fast EMA even with calm oscillators => skip, don't chase
    score = se.classify_setup(make_row(stretch_atr=3.5))
    assert score.verdict == se.Verdict.PEAK_EXTENDED


def test_thin_volume_blocks_actionable():
    score = se.classify_setup(make_row(rel_vol=0.3, obv_slope=-1.0))
    assert score.verdict != se.Verdict.PULLBACK_IN_UPTREND


def test_no_data_when_features_missing():
    row = make_row(adx=float("nan"))
    assert se.classify_setup(row).verdict == se.Verdict.NO_DATA
