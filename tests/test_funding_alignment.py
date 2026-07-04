"""No-lookahead proofs for the funding-rate alignment layer -- the other
half (besides the flush-candle trigger) of what makes this strategy sound.
A 4h bar must never see a funding print timestamped after it, and a
percentile/median computed for a given print must never depend on a LATER
print.
"""

import numpy as np
import pandas as pd

from longbot.funding import (
    align_to_bars,
    build_funding_features,
    funding_percentile_rank,
    funding_trailing_median,
)


def _funding_df(times, rates):
    return pd.DataFrame({"funding_rate": rates}, index=pd.DatetimeIndex(times, name="funding_time"))


def test_align_backward_asof_never_uses_a_later_print():
    times = pd.date_range("2021-01-01", periods=5, freq="8h")
    rates = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
    funding_df = _funding_df(times, rates)

    # bars: exactly on print 1 (idx1), 1 minute before print 2, 1 minute after print 2
    bar_index = pd.DatetimeIndex([
        times[1],
        times[2] - pd.Timedelta(minutes=1),
        times[2] + pd.Timedelta(minutes=1),
        times[0] - pd.Timedelta(hours=1),   # before ANY print
    ])
    aligned = align_to_bars(funding_df, bar_index)

    assert aligned["funding_rate"].iloc[0] == rates[1]           # exactly at print 1 -> print 1
    assert aligned["funding_rate"].iloc[1] == rates[1]           # just before print 2 -> still print 1
    assert aligned["funding_rate"].iloc[2] == rates[2]           # just after print 2 -> print 2
    assert np.isnan(aligned["funding_rate"].iloc[3])             # before any print -> NaN


def test_align_truncation_invariance():
    """Changing/removing LATER funding prints must not change the aligned
    value for an EARLIER bar."""
    times = pd.date_range("2021-01-01", periods=10, freq="8h")
    rates = list(np.linspace(0.0001, 0.001, 10))
    full = _funding_df(times, rates)
    truncated = full.iloc[:5]

    bar_index = pd.DatetimeIndex([times[3] + pd.Timedelta(hours=1)])
    aligned_full = align_to_bars(full, bar_index)
    aligned_trunc = align_to_bars(truncated, bar_index)

    assert aligned_full["funding_rate"].iloc[0] == aligned_trunc["funding_rate"].iloc[0]


def test_percentile_rank_is_causal_truncation_invariant():
    """The percentile rank AT print i must not change if prints after i are
    altered or removed -- it may only ever look backward."""
    rng = np.random.default_rng(3)
    n = 400
    rates = rng.normal(0.0001, 0.0003, n)
    window = 90

    full = pd.Series(rates)
    pct_full = funding_percentile_rank(full, window)

    i = 200
    truncated = pd.Series(rates[: i + 1])
    pct_trunc = funding_percentile_rank(truncated, window)

    assert pct_full.iloc[i] == pct_trunc.iloc[i]

    # even replacing the future with entirely different random data:
    variant = pd.Series(np.concatenate([rates[: i + 1], rng.normal(0, 1, 50)]))
    pct_variant = funding_percentile_rank(variant, window)
    assert pct_full.iloc[i] == pct_variant.iloc[i]


def test_percentile_rank_formula_on_a_known_array():
    # last 10 values 1..10 in a window of 10: value 10 (the last) should rank at 1.0 (100th pct)
    s = pd.Series(np.arange(1, 11, dtype=float))
    pct = funding_percentile_rank(s, window=10)
    assert pct.iloc[9] == 1.0   # 10 is <= itself and every other value in the window

    # a window where the last value is the smallest -> lowest rank (1/window)
    s2 = pd.Series([5, 4, 3, 2, 1], dtype=float)
    pct2 = funding_percentile_rank(s2, window=5)
    assert pct2.iloc[4] == 1.0 / 5.0


def test_trailing_median_is_causal():
    rng = np.random.default_rng(9)
    rates = rng.normal(0, 1, 300)
    window = 90
    full = pd.Series(rates)
    med_full = funding_trailing_median(full, window)

    i = 150
    truncated = pd.Series(rates[: i + 1])
    med_trunc = funding_trailing_median(truncated, window)
    assert med_full.iloc[i] == med_trunc.iloc[i]


def test_build_funding_features_columns_and_alignment_roundtrip():
    times = pd.date_range("2021-01-01", periods=300, freq="8h")
    rng = np.random.default_rng(5)
    rates = rng.normal(0.0001, 0.0004, 300)
    funding_df = _funding_df(times, rates)

    feat = build_funding_features(funding_df, window=90)
    assert list(feat.columns) == ["funding_rate", "pct_rank", "trailing_median"]
    assert feat["pct_rank"].iloc[:89].isna().all()
    assert feat["pct_rank"].iloc[89:].notna().all()

    bar_index = pd.date_range("2021-01-01", periods=600, freq="4h")
    aligned = align_to_bars(feat, bar_index)
    assert list(aligned.columns) == ["funding_rate", "pct_rank", "trailing_median"]
    assert len(aligned) == len(bar_index)


def test_empty_funding_df_yields_all_nan():
    empty = pd.DataFrame(columns=["funding_rate"]).rename_axis("funding_time")
    bar_index = pd.date_range("2021-01-01", periods=10, freq="4h")
    aligned = align_to_bars(empty, bar_index)
    assert aligned["funding_rate"].isna().all()


def test_align_handles_mismatched_datetime_resolutions():
    """Regression test: a real run hit merge_asof's "incompatible merge keys
    dtype('<M8[us]') and dtype('<M8[ms]')" error because the funding index
    (parsed from a fresh fetch, ms-resolution) and the bar index (parsed
    from a CSV round-trip, us-resolution) ended up at different datetime64
    resolutions despite representing the same instants. align_to_bars must
    normalize both merge keys internally and merge cleanly regardless of
    which resolution each side arrives in.
    """
    times = pd.date_range("2021-01-01", periods=5, freq="8h")
    rates = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]

    # funding index at millisecond resolution (mirrors fetch_funding_rate_range,
    # which builds funding_time via pd.to_datetime(..., unit="ms"))
    ms_index = pd.DatetimeIndex(times, name="funding_time").astype("datetime64[ms]")
    funding_df = pd.DataFrame({"funding_rate": rates}, index=ms_index)
    assert funding_df.index.dtype == np.dtype("datetime64[ms]")

    # bar index at microsecond resolution (mirrors a CSV round-trip's string inference)
    bar_index = pd.DatetimeIndex([
        times[1],
        times[2] - pd.Timedelta(minutes=1),
        times[2] + pd.Timedelta(minutes=1),
    ]).astype("datetime64[us]")
    assert bar_index.dtype == np.dtype("datetime64[us]")

    aligned = align_to_bars(funding_df, bar_index)  # must not raise MergeError

    assert aligned["funding_rate"].iloc[0] == rates[1]
    assert aligned["funding_rate"].iloc[1] == rates[1]
    assert aligned["funding_rate"].iloc[2] == rates[2]
    # the returned frame's index must still be the exact index the caller passed in
    assert aligned.index.equals(bar_index)
