"""Layer 1 — the Pair Picker: a funnel that ELIMINATES, then ranks.

Run on closed candles only, once per candle close. Five stages, in order:

  1. Regime gate (global on/off): only hunt longs when the whole market is
     risk-on — BTC above a *rising* EMA200 on the chosen timeframe. Otherwise
     do nothing at all.
  2. Liquidity floor (hard disqualifiers): drop pairs below a min 24h quote
     volume, above a max spread, leveraged tokens, stablecoin pairs, and very
     recently listed coins. A filter, not a tiebreaker.
  3. Confirmed uptrend: price > EMA50 > EMA200, EMA200 sloping up, positive
     multi-week return (the scoring engine's Trend layer).
  4. Pullback / not overbought: cooled momentum, near the fast EMA, pulled back
     off the local high but holding above EMA50. This stage filters OUT the
     most extended pairs (the PEAK_EXTENDED trap).
  5. Rank survivors by trend quality over a multi-day lookback (never by
     short-term % move) and take the best one.

The picker is pure, deterministic code and does NOT fetch data — callers pass
in the OHLCV frames and per-pair metadata. (Keeping I/O out of the brain makes
it trivial to backtest and impossible for an LLM to creep into selection.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import indicators as ind
from . import scoring_engine as se


@dataclass(frozen=True)
class PickerParams:
    min_quote_volume_24h: float = 10_000_000.0   # USDT traded in last 24h
    max_spread_pct: float = 0.005                # 0.5% max bid/ask spread
    min_history_bars: int = 250                  # enough to warm up EMA200 etc.
    # regime gate (on BTC):
    regime_ema: int = 200
    regime_slope_lookback: int = 20


@dataclass(frozen=True)
class PairMeta:
    """Per-pair market metadata used by the liquidity floor (stage 2)."""

    quote_volume_24h: float
    spread_pct: float
    is_leveraged_token: bool = False   # UP/DOWN/BULL/BEAR etc.
    is_stablecoin_pair: bool = False   # e.g. USDC/USDT
    bars_available: int = 10_000       # proxy for listing age


@dataclass
class PairData:
    symbol: str
    df: pd.DataFrame          # OHLCV, closed candles only, ascending
    meta: PairMeta


@dataclass
class Candidate:
    symbol: str
    score: se.SetupScore
    feature_row: pd.Series


@dataclass
class PickResult:
    regime_on: bool
    chosen: Candidate | None
    candidates: list[Candidate]                 # all actionable survivors, ranked best-first
    funnel: dict[str, int] = field(default_factory=dict)   # stage -> count remaining
    dropped: dict[str, str] = field(default_factory=dict)  # symbol -> reason it was eliminated


# --------------------------------------------------------------------------- #
# Stage 1 — regime gate
# --------------------------------------------------------------------------- #
def regime_ok(btc_df: pd.DataFrame, params: PickerParams = PickerParams()) -> tuple[bool, str]:
    """Is the whole market risk-on? BTC above a *rising* EMA200.

    Returns (ok, human_reason). When this is False the bot does nothing — this
    is the single biggest protection against buying pullbacks that are actually
    the start of a market-wide rollover.
    """
    if len(btc_df) < params.min_history_bars:
        return False, "not enough BTC history to judge regime"
    close = btc_df["close"]
    ema = ind.ema(close, params.regime_ema)
    slope = ind.slope(ema, params.regime_slope_lookback)
    last_close = close.iloc[-1]
    last_ema = ema.iloc[-1]
    last_slope = slope.iloc[-1]
    if pd.isna(last_ema) or pd.isna(last_slope):
        return False, "regime EMA still warming up"
    above = last_close > last_ema
    rising = last_slope > 0
    if above and rising:
        return True, "risk-on: BTC above a rising EMA200"
    why = []
    if not above:
        why.append("BTC below EMA200")
    if not rising:
        why.append("EMA200 not rising")
    return False, "risk-off: " + ", ".join(why)


# --------------------------------------------------------------------------- #
# Stage 2 — liquidity floor
# --------------------------------------------------------------------------- #
def passes_liquidity(meta: PairMeta, params: PickerParams = PickerParams()) -> tuple[bool, str]:
    if meta.is_leveraged_token:
        return False, "leveraged token (forbidden: not spot)"
    if meta.is_stablecoin_pair:
        return False, "stablecoin pair (no trend to ride)"
    if meta.bars_available < params.min_history_bars:
        return False, f"too little history ({meta.bars_available} bars) — recently listed"
    if meta.quote_volume_24h < params.min_quote_volume_24h:
        return False, (
            f"24h volume ${meta.quote_volume_24h:,.0f} < "
            f"${params.min_quote_volume_24h:,.0f} floor"
        )
    if meta.spread_pct > params.max_spread_pct:
        return False, f"spread {meta.spread_pct*100:.2f}% > {params.max_spread_pct*100:.2f}% max"
    return True, "liquidity ok"


# --------------------------------------------------------------------------- #
# The funnel
# --------------------------------------------------------------------------- #
def pick_pair(
    pairs: list[PairData],
    btc_df: pd.DataFrame,
    scoring_params: se.ScoringParams = se.DEFAULT_PARAMS,
    picker_params: PickerParams = PickerParams(),
) -> PickResult:
    """Run the full 5-stage funnel and return the single best candidate (or None)."""
    dropped: dict[str, str] = {}
    funnel: dict[str, int] = {"input": len(pairs)}

    # Stage 1 — regime gate (global kill switch for the whole scan)
    on, regime_reason = regime_ok(btc_df, picker_params)
    if not on:
        return PickResult(
            regime_on=False, chosen=None, candidates=[],
            funnel={**funnel, "after_regime": 0},
            dropped={"__regime__": regime_reason},
        )

    # Stage 2 — liquidity floor
    survivors: list[PairData] = []
    for pd_ in pairs:
        ok, reason = passes_liquidity(pd_.meta, picker_params)
        if ok:
            survivors.append(pd_)
        else:
            dropped[pd_.symbol] = f"liquidity: {reason}"
    funnel["after_liquidity"] = len(survivors)

    # Stages 3 & 4 — confirmed uptrend + pullback (scoring engine on each pair)
    candidates: list[Candidate] = []
    after_trend = 0
    for pd_ in survivors:
        if len(pd_.df) < picker_params.min_history_bars:
            dropped[pd_.symbol] = "insufficient candles after liquidity stage"
            continue
        feat = se.compute_features(pd_.df, scoring_params)
        row = feat.iloc[-1]
        score = se.classify_setup(row, scoring_params)

        if not score.trend_ok:
            dropped[pd_.symbol] = f"no confirmed uptrend ({score.verdict.value})"
            continue
        after_trend += 1

        if score.verdict != se.Verdict.PULLBACK_IN_UPTREND:
            dropped[pd_.symbol] = f"trend ok but not a clean pullback ({score.verdict.value})"
            continue
        candidates.append(Candidate(symbol=pd_.symbol, score=score, feature_row=row))

    funnel["after_trend"] = after_trend
    funnel["after_pullback"] = len(candidates)

    # Stage 5 — rank by trend quality (NOT by recent % move) and take the best
    candidates.sort(key=lambda c: c.score.trend_quality, reverse=True)
    chosen = candidates[0] if candidates else None

    return PickResult(
        regime_on=True,
        chosen=chosen,
        candidates=candidates,
        funnel=funnel,
        dropped=dropped,
    )
