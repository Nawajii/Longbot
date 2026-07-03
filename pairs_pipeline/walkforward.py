"""
Walk-forward OOS runner.

Because the formation window rolls forward continuously (a fresh trailing
90-day fit every day, per config.ROLL_CADENCE_DAYS) and is never fit on
data that includes or follows the day being traded, EVERY trading decision
in this codebase is out-of-sample by construction -- there is no separate
train/test split to perform. What "per calendar-year OOS fold" means here
is: slice the (already fully OOS) trade list by the calendar year of trade
entry, so 2022 can be inspected in isolation as the critical bear-market
fold the strategy's whole thesis depends on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import config, data_fetch, regime
from .cointegration import rolling_formation
from .engine import Trade, run_pair


@dataclass
class PairResult:
    pair: tuple[str, str]
    skipped: bool = False
    skip_reason: str = ""
    trades: list[Trade] = field(default_factory=list)
    n_formation_tested: int = 0
    n_formation_cointegrated: int = 0
    history_start: object = None
    history_end: object = None
    n_bars: int = 0


MIN_HISTORY_BARS = config.FORMATION_WINDOW_DAYS + 30


def compute_coint_frequency(prices: pd.DataFrame, asset_a: str, asset_b: str) -> tuple[int, int]:
    """Independent full pass over every day's formation window (regardless
    of trading activity) to report how often the pair was cointegrated.
    Kept separate from engine.run_pair, which only fits a formation when
    flat -- that path would under/over-sample relative to a plain calendar
    frequency."""
    tested = 0
    coint = 0
    for _, formation in rolling_formation(prices, asset_a, asset_b):
        if formation is None:
            continue
        tested += 1
        if formation.cointegrated:
            coint += 1
    return tested, coint


def run_all(pairs: list[tuple[str, str]] | None = None) -> dict[tuple[str, str], PairResult]:
    pairs = pairs or config.PAIR_UNIVERSE
    btc_close = data_fetch.get_symbol_series(config.REGIME_SYMBOL)
    regime_series = regime.compute_btc_regime(btc_close)

    results: dict[tuple[str, str], PairResult] = {}
    for asset_a, asset_b in pairs:
        prices = data_fetch.get_pair_prices(asset_a, asset_b)
        if len(prices) < MIN_HISTORY_BARS:
            results[(asset_a, asset_b)] = PairResult(
                pair=(asset_a, asset_b),
                skipped=True,
                skip_reason=(
                    f"only {len(prices)} overlapping daily bars "
                    f"(need >= {MIN_HISTORY_BARS} = {config.FORMATION_WINDOW_DAYS}d formation "
                    f"+ 30d minimum trading history)"
                ),
            )
            continue

        trades = run_pair(prices, asset_a, asset_b)
        for t in trades:
            t.entry_year = t.entry_date.year
            t.regime = regime_series.get(t.entry_date, "unknown")

        tested, coint = compute_coint_frequency(prices, asset_a, asset_b)

        results[(asset_a, asset_b)] = PairResult(
            pair=(asset_a, asset_b),
            trades=trades,
            n_formation_tested=tested,
            n_formation_cointegrated=coint,
            history_start=prices.index[0],
            history_end=prices.index[-1],
            n_bars=len(prices),
        )

    return results
