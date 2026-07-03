"""
The most important tests in this repo. They prove, mechanically, that
nothing computed for a trading decision on day i can ever depend on price
data at or after day i -- the classic stat-arb backtest killer the prompt
calls out by name.

Strategy: truncation invariance. If a function has no lookahead, then
computing it on a dataset truncated right after the day in question must
give the EXACT same result as computing it on the full dataset (or on a
dataset where we've replaced the future with different random data
entirely). If truncating/altering the future ever changes a past decision,
that's lookahead, full stop.
"""

import numpy as np
import pandas as pd

from pairs_pipeline.cointegration import formation_for_index, fit_formation_window
from pairs_pipeline.engine import run_pair


def test_formation_result_unaffected_by_future_data(cointegrated_prices):
    """formation_for_index(prices, a, b, i) must be identical regardless of
    what rows exist after i."""
    full = cointegrated_prices
    i = 250  # comfortably past the 90-bar formation minimum, comfortably before the end

    result_full = formation_for_index(full, "A", "B", i)

    truncated = full.iloc[: i + 1]  # everything up to and including day i
    result_truncated = formation_for_index(truncated, "A", "B", i)

    assert result_full.beta == result_truncated.beta
    assert result_full.mean == result_truncated.mean
    assert result_full.std == result_truncated.std
    assert result_full.adf_pvalue == result_truncated.adf_pvalue


def test_formation_result_unaffected_by_replacing_the_future():
    """Stronger version: keep the past identical, replace the future with
    completely different random data, and confirm the formation result at
    day i (computed from data strictly before i) does not move at all."""
    rng = np.random.default_rng(1)
    n_past = 200
    n_future_variant = 50

    dates_past = pd.date_range("2020-01-01", periods=n_past, freq="D").date
    log_b_past = np.cumsum(rng.normal(0, 0.02, n_past))
    log_a_past = 1.1 * log_b_past + rng.normal(0, 0.01, n_past)

    def build(future_seed):
        frng = np.random.default_rng(future_seed)
        dates_future = pd.date_range("2020-07-19", periods=n_future_variant, freq="D").date
        log_b_future = log_b_past[-1] + np.cumsum(frng.normal(0, 0.02, n_future_variant))
        log_a_future = log_a_past[-1] + np.cumsum(frng.normal(0, 0.02, n_future_variant))
        prices = pd.DataFrame({
            "A": np.exp(np.concatenate([log_a_past, log_a_future])),
            "B": np.exp(np.concatenate([log_b_past, log_b_future])),
        }, index=list(dates_past) + list(dates_future))
        return prices

    variant_1 = build(future_seed=101)
    variant_2 = build(future_seed=202)  # totally different future

    i = n_past - 1  # decision day sits right at the boundary; formation window is strictly before it
    r1 = formation_for_index(variant_1, "A", "B", i)
    r2 = formation_for_index(variant_2, "A", "B", i)

    assert r1.beta == r2.beta
    assert r1.mean == r2.mean
    assert r1.std == r2.std
    assert r1.adf_pvalue == r2.adf_pvalue


def test_formation_window_excludes_decision_day_itself(cointegrated_prices):
    """Changing ONLY the price on decision day i (leaving every prior day
    untouched) must not change the formation result for day i."""
    prices = cointegrated_prices.copy()
    i = 300

    result_before = formation_for_index(prices, "A", "B", i)

    mutated = prices.copy()
    mutated.iloc[i, mutated.columns.get_loc("A")] *= 5.0  # wildly change day i's price only
    result_after = formation_for_index(mutated, "A", "B", i)

    assert result_before.beta == result_after.beta
    assert result_before.mean == result_after.mean
    assert result_before.std == result_after.std
    assert result_before.adf_pvalue == result_after.adf_pvalue


def test_full_engine_truncation_invariance(cointegrated_prices):
    """The strongest end-to-end guarantee: replay the full engine on a
    truncated history vs. the full history, and require every trade that
    completes before the truncation point to be byte-identical between the
    two runs. If a later run's future data ever changed an earlier
    decision, this would fail."""
    full_prices = cointegrated_prices
    cutoff = 300

    truncated_prices = full_prices.iloc[:cutoff]

    trades_full = run_pair(full_prices, "A", "B")
    trades_truncated = run_pair(truncated_prices, "A", "B")

    cutoff_date = full_prices.index[cutoff - 1]
    trades_full_before_cutoff = [t for t in trades_full if t.exit_date is not None and t.exit_date <= cutoff_date]

    assert len(trades_full_before_cutoff) == len(trades_truncated)
    for t_full, t_trunc in zip(trades_full_before_cutoff, trades_truncated):
        assert t_full.direction == t_trunc.direction
        assert t_full.entry_date == t_trunc.entry_date
        assert t_full.entry_price == t_trunc.entry_price
        assert t_full.exit_date == t_trunc.exit_date
        assert t_full.exit_price == t_trunc.exit_price
        assert t_full.exit_reason == t_trunc.exit_reason
        assert t_full.r_multiple == t_trunc.r_multiple
