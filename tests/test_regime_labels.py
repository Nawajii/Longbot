import numpy as np
import pandas as pd

import run_deep_grid as D


def _seg(start_price, per_bar_ret, n, start_ts):
    idx = pd.date_range(start_ts, periods=n, freq="4h")
    price = start_price * np.exp(np.cumsum(np.full(n, per_bar_ret)))
    return pd.DataFrame({"open": price, "high": price * 1.001, "low": price * 0.999,
                         "close": price, "volume": np.ones(n)}, index=idx)


def test_regime_labels_separate_bull_bear_chop():
    """A clear up -> down -> flat series must yield BULL, BEAR AND CHOP bars —
    none near zero. Uses small macro params so the trend line warms up fast.
    """
    up = _seg(100, +0.004, 400, "2020-01-01")
    down = _seg(up["close"].iloc[-1], -0.004, 400, up.index[-1] + pd.Timedelta(hours=4))
    flat = _seg(down["close"].iloc[-1], 0.0, 400, down.index[-1] + pd.Timedelta(hours=4))
    btc = pd.concat([up, down, flat])

    labels = D.btc_regime_labels(btc, ema_bars=60, slope_bars=20)
    counts = {r: int((labels == r).sum()) for r in D.REGIMES}
    # all three regimes represented, none collapsed to zero
    assert counts["BULL"] > 0 and counts["BEAR"] > 0 and counts["CHOP"] > 0, counts
    # the up segment is predominantly BULL, the down segment predominantly BEAR
    up_lab = labels.iloc[100:400]
    down_lab = labels.iloc[500:800]
    assert (up_lab == "BULL").mean() > 0.5
    assert (down_lab == "BEAR").mean() > 0.5


def test_regime_label_is_independent_of_the_entry_gate():
    """The macro label must NOT be the same as the entry gate (EMA200 on the
    trading timeframe) — otherwise every gate-permitted trade is tautologically
    BULL. Prove the macro label produces BEAR bars where the *gate* (EMA200) is
    briefly bullish, i.e. during a downtrend rally.
    """
    from longbot import indicators as ind

    down = _seg(100, -0.003, 900, "2021-01-01")
    # inject a multi-day counter-trend rally in the middle (a bear rally)
    rally = np.concatenate([np.zeros(450), np.full(40, 0.02), np.zeros(410)])
    close = pd.Series(down["close"].to_numpy() * np.exp(np.cumsum(rally)), index=down.index)
    btc = pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999,
                        "close": close, "volume": np.ones(len(close))}, index=down.index)

    macro = D.btc_regime_labels(btc, ema_bars=200, slope_bars=40)
    gate_on = (btc["close"] > ind.ema(btc["close"], 200)) & (ind.slope(ind.ema(btc["close"], 200), 20) > 0)

    # during the rally the fine gate opens, but the macro regime is still BEAR
    rally_window = slice(455, 495)
    assert gate_on.iloc[rally_window].any(), "fixture should open the gate during the rally"
    assert (macro.iloc[rally_window] == "BEAR").any(), "macro label should still read BEAR in a bear rally"
