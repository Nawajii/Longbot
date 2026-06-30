"""The scoring engine — the "brain" of the pullback hunter.

It evaluates four **independent** layers on closed candles and combines them
into a single verdict via :func:`classify_setup`:

    1. Trend       — is this a confirmed, healthy uptrend? (direction + strength)
    2. Momentum    — is short-term momentum EXHAUSTED, COOLED, or neutral?
    3. Volatility  — how stretched is price, how deep is the pullback, what's ATR?
    4. Volume      — does independent volume confirm the move?

The single most important design rule (carried over from the POC spec):

    **Trend and Momentum are used in OPPOSITE directions.**

    * Strong trend  + EXHAUSTED momentum + stretched price  -> PEAK_EXTENDED (skip; the trap)
    * Strong trend  + COOLED momentum + shallow pullback     -> PULLBACK_IN_UPTREND (the candidate)

Momentum is therefore returned as an *exhaustion state*, never summed into a
"more bullish = higher score" number. Buying because RSI is high is exactly the
mistake this engine exists to avoid.

Everything here is deterministic. No LLM is involved in producing a verdict.
Every threshold in :class:`ScoringParams` is a *tunable hypothesis*, not a fact —
the backtest decides which values (if any) carry an edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from . import indicators as ind


# --------------------------------------------------------------------------- #
# Verdicts and states
# --------------------------------------------------------------------------- #
class Verdict(str, Enum):
    """The output classification for a single bar / pair."""

    PULLBACK_IN_UPTREND = "PULLBACK_IN_UPTREND"  # the candidate — actionable
    PEAK_EXTENDED = "PEAK_EXTENDED"              # strong trend but exhausted/stretched — skip
    NO_UPTREND = "NO_UPTREND"                    # trend not confirmed — skip
    DEEP_PULLBACK_RISK = "DEEP_PULLBACK_RISK"    # pullback too deep / oversold — possible trend break, skip
    NEUTRAL_WAIT = "NEUTRAL_WAIT"                # in an uptrend but no clean setup yet — wait
    NO_DATA = "NO_DATA"                          # not enough history to judge


class MomentumState(str, Enum):
    """Momentum is an exhaustion *state*, never a bullish magnitude."""

    EXHAUSTED = "EXHAUSTED"    # overbought on multiple oscillators — danger near highs
    NEUTRAL = "NEUTRAL"        # neither hot nor cold
    COOLED = "COOLED"          # pulled back into a healthy buy-the-dip band
    OVERSOLD = "OVERSOLD"      # very weak — in an uptrend this warns of a possible break


ACTIONABLE = Verdict.PULLBACK_IN_UPTREND


# --------------------------------------------------------------------------- #
# Tunable parameters (HYPOTHESES, not facts — the backtest validates these)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScoringParams:
    # --- moving averages ---
    ema_fast: int = 20
    ema_mid: int = 50
    ema_slow: int = 200
    ema_slow_slope_lookback: int = 20      # bars used to measure EMA200 slope

    # --- trend confirmation ---
    adx_period: int = 14
    adx_min_trend: float = 20.0            # below this = no real trend
    trend_return_lookback: int = 120       # bars for "positive multi-week return"

    # --- momentum / exhaustion ---
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_cooled_low: float = 40.0           # the "cooled" buy-the-dip band ...
    rsi_cooled_high: float = 60.0          # ... is [cooled_low, cooled_high]
    rsi_oversold: float = 30.0
    stoch_overbought: float = 80.0
    willr_overbought: float = -20.0
    cci_overbought: float = 100.0
    exhaustion_votes: int = 2              # how many oscillators must agree => EXHAUSTED

    # --- volatility / stretch / pullback geometry ---
    atr_period: int = 14
    bb_period: int = 20
    local_high_lookback: int = 20          # window defining the recent local high
    stretch_max_atr: float = 2.0           # price > this many ATR above fast EMA = stretched
    near_fast_max_atr: float = 1.5         # "near the fast EMA" means within this many ATR
    pullback_min_pct: float = 0.01         # must be at least this far off the local high
    pullback_max_pct: float = 0.12         # ... but not deeper than this (else trend at risk)

    # --- volume confirmation ---
    rel_vol_period: int = 20
    rel_vol_min: float = 0.8               # volume not bone-dry relative to its average
    require_obv_confirm: bool = True       # OBV must be rising over the trend window

    # --- aroon (trend youth, used in ranking) ---
    aroon_period: int = 25


DEFAULT_PARAMS = ScoringParams()


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class SetupScore:
    verdict: Verdict
    momentum_state: MomentumState
    trend_ok: bool
    pullback_ok: bool
    volume_ok: bool
    trend_quality: float          # 0..1, used ONLY for ranking survivors
    reasons: list[str] = field(default_factory=list)
    # raw numbers, handy for logging / the Coroner / debugging
    features: dict[str, float] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return self.verdict == ACTIONABLE


# --------------------------------------------------------------------------- #
# Feature computation (vectorised once over the whole frame; no lookahead)
# --------------------------------------------------------------------------- #
def compute_features(df: pd.DataFrame, params: ScoringParams = DEFAULT_PARAMS) -> pd.DataFrame:
    """Compute every indicator the engine needs, aligned to ``df``'s index.

    Computed once over the full series for backtest efficiency. Each indicator
    only uses trailing data, so reading row ``i`` of the result uses no
    information from bars after ``i`` — *provided the caller reads row i only
    after bar i has closed* (the backtest enforces this).
    """
    ind.validate_ohlcv(df)
    close = df["close"]

    feat = pd.DataFrame(index=df.index)
    feat["close"] = close
    feat["ema_fast"] = ind.ema(close, params.ema_fast)
    feat["ema_mid"] = ind.ema(close, params.ema_mid)
    feat["ema_slow"] = ind.ema(close, params.ema_slow)
    feat["ema_slow_slope"] = ind.slope(feat["ema_slow"], params.ema_slow_slope_lookback)

    feat["rsi"] = ind.rsi(close, params.rsi_period)
    feat["atr"] = ind.atr(df, params.atr_period)

    adx_df = ind.adx(df, params.adx_period)
    feat["adx"] = adx_df["adx"]
    feat["plus_di"] = adx_df["plus_di"]
    feat["minus_di"] = adx_df["minus_di"]

    stoch = ind.stochastic(df)
    feat["stoch_k"] = stoch["k"]
    feat["willr"] = ind.williams_r(df)
    feat["cci"] = ind.cci(df)

    bb = ind.bollinger(close, params.bb_period)
    feat["bb_pct_b"] = bb["pct_b"]
    feat["bb_bandwidth"] = bb["bandwidth"]

    feat["obv"] = ind.obv(close, df["volume"])
    feat["obv_slope"] = ind.slope(feat["obv"], params.trend_return_lookback)
    feat["rel_vol"] = ind.relative_volume(df["volume"], params.rel_vol_period)

    aroon = ind.aroon(df, params.aroon_period)
    feat["aroon_up"] = aroon["up"]
    feat["aroon_osc"] = aroon["oscillator"]

    # derived geometry
    feat["local_high"] = df["high"].rolling(
        params.local_high_lookback, min_periods=params.local_high_lookback
    ).max()
    feat["pullback_pct"] = (feat["local_high"] - close) / feat["local_high"]
    feat["multiweek_return"] = close / close.shift(params.trend_return_lookback) - 1.0
    # "stretch" and "distance to fast EMA" measured in ATR units (volatility-normalised)
    feat["stretch_atr"] = (close - feat["ema_fast"]) / feat["atr"]
    feat["dist_fast_atr"] = (close - feat["ema_fast"]).abs() / feat["atr"]

    return feat


# --------------------------------------------------------------------------- #
# The four layers (each judges one row of features independently)
# --------------------------------------------------------------------------- #
def _trend_layer(row: pd.Series, p: ScoringParams) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    ok = True
    if not (row["close"] > row["ema_mid"] > row["ema_slow"]):
        ok = False
        reasons.append("price not stacked above EMA50>EMA200")
    if not (row["ema_slow_slope"] > 0):
        ok = False
        reasons.append("EMA200 not sloping up")
    if not (row["multiweek_return"] > 0):
        ok = False
        reasons.append("multi-week return not positive")
    if not (row["adx"] >= p.adx_min_trend):
        ok = False
        reasons.append(f"ADX {row['adx']:.1f} < {p.adx_min_trend} (weak trend)")
    if ok:
        reasons.append("confirmed uptrend: price>EMA50>EMA200, EMA200 up, ADX ok")
    return ok, reasons


def _momentum_layer(row: pd.Series, p: ScoringParams) -> tuple[MomentumState, list[str]]:
    """Return an EXHAUSTION STATE, deliberately NOT a bullish magnitude."""
    votes = 0
    if row["rsi"] >= p.rsi_overbought:
        votes += 1
    if row["stoch_k"] >= p.stoch_overbought:
        votes += 1
    if row["willr"] >= p.willr_overbought:
        votes += 1
    if row["cci"] >= p.cci_overbought:
        votes += 1

    if votes >= p.exhaustion_votes:
        return MomentumState.EXHAUSTED, [f"momentum EXHAUSTED ({votes} oscillators overbought)"]
    if row["rsi"] <= p.rsi_oversold:
        return MomentumState.OVERSOLD, [f"momentum OVERSOLD (RSI {row['rsi']:.1f})"]
    if p.rsi_cooled_low <= row["rsi"] <= p.rsi_cooled_high:
        return MomentumState.COOLED, [f"momentum COOLED (RSI {row['rsi']:.1f} in buy-dip band)"]
    return MomentumState.NEUTRAL, [f"momentum NEUTRAL (RSI {row['rsi']:.1f})"]


def _volatility_pullback_layer(row: pd.Series, p: ScoringParams) -> tuple[bool, bool, list[str]]:
    """Return (pullback_ok, is_stretched, reasons)."""
    reasons: list[str] = []
    is_stretched = bool(row["stretch_atr"] > p.stretch_max_atr)
    if is_stretched:
        reasons.append(
            f"price stretched {row['stretch_atr']:.1f} ATR above fast EMA (>{p.stretch_max_atr})"
        )

    pulled_off_high = p.pullback_min_pct <= row["pullback_pct"] <= p.pullback_max_pct
    above_mid = row["close"] > row["ema_mid"]
    near_fast = row["dist_fast_atr"] <= p.near_fast_max_atr

    pullback_ok = bool(pulled_off_high and above_mid and near_fast and not is_stretched)
    if pullback_ok:
        reasons.append(
            f"healthy pullback: {row['pullback_pct']*100:.1f}% off high, near fast EMA, above EMA50"
        )
    else:
        if not pulled_off_high:
            reasons.append(
                f"pullback {row['pullback_pct']*100:.1f}% outside "
                f"[{p.pullback_min_pct*100:.0f}%,{p.pullback_max_pct*100:.0f}%] band"
            )
        if not near_fast:
            reasons.append(f"price {row['dist_fast_atr']:.1f} ATR from fast EMA (not near)")
        if not above_mid:
            reasons.append("price below EMA50 (pullback too deep)")
    return pullback_ok, is_stretched, reasons


def _volume_layer(row: pd.Series, p: ScoringParams) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    vol_ok = row["rel_vol"] >= p.rel_vol_min
    if not vol_ok:
        reasons.append(f"relative volume {row['rel_vol']:.2f} < {p.rel_vol_min} (too thin)")
    if p.require_obv_confirm:
        obv_ok = row["obv_slope"] > 0
        if not obv_ok:
            reasons.append("OBV not rising (volume not confirming trend)")
        vol_ok = vol_ok and obv_ok
    if vol_ok:
        reasons.append("volume confirms (rel-vol ok, OBV rising)")
    return bool(vol_ok), reasons


def _trend_quality(row: pd.Series, p: ScoringParams) -> float:
    """A 0..1 trend-quality score used ONLY to rank survivors, never to trigger.

    Blends trend strength (ADX), trend youth (Aroon), EMA200 up-slope and the
    multi-day return. Deliberately ignores short-term % move so we don't chase
    whatever pumped hardest this hour.
    """
    parts: list[float] = []
    # ADX 20..50 -> 0..1
    parts.append(float(np.clip((row["adx"] - p.adx_min_trend) / 30.0, 0.0, 1.0)))
    # Aroon up 50..100 -> 0..1
    parts.append(float(np.clip((row["aroon_up"] - 50.0) / 50.0, 0.0, 1.0)))
    # EMA200 slope normalised by price (per-bar drift) 0..0.2%/bar -> 0..1
    norm_slope = row["ema_slow_slope"] / row["close"] if row["close"] else 0.0
    parts.append(float(np.clip(norm_slope / 0.002, 0.0, 1.0)))
    # multi-week return 0..40% -> 0..1
    parts.append(float(np.clip(row["multiweek_return"] / 0.40, 0.0, 1.0)))
    return float(np.mean(parts))


# --------------------------------------------------------------------------- #
# The classifier
# --------------------------------------------------------------------------- #
def classify_setup(row: pd.Series, params: ScoringParams = DEFAULT_PARAMS) -> SetupScore:
    """Combine the four layers into a single verdict for one bar.

    This is where the "opposite directions" rule lives: a confirmed uptrend
    whose momentum is EXHAUSTED (or whose price is stretched) is the PEAK trap
    we skip; a confirmed uptrend that has COOLED and pulled back shallowly, with
    volume confirming, is the candidate.
    """
    # Not enough history? Any key feature NaN => we simply cannot judge.
    key = ["ema_slow", "ema_slow_slope", "adx", "rsi", "atr", "multiweek_return", "local_high"]
    if any(pd.isna(row.get(k, np.nan)) for k in key):
        return SetupScore(
            verdict=Verdict.NO_DATA,
            momentum_state=MomentumState.NEUTRAL,
            trend_ok=False, pullback_ok=False, volume_ok=False,
            trend_quality=0.0,
            reasons=["insufficient history (indicators still warming up)"],
        )

    trend_ok, trend_reasons = _trend_layer(row, params)
    momentum_state, mom_reasons = _momentum_layer(row, params)
    pullback_ok, is_stretched, vol_reasons = _volatility_pullback_layer(row, params)
    volume_ok, volm_reasons = _volume_layer(row, params)
    quality = _trend_quality(row, params)

    reasons = trend_reasons + mom_reasons + vol_reasons + volm_reasons

    # ---- decision tree (order matters) ----
    if not trend_ok:
        verdict = Verdict.NO_UPTREND
    elif momentum_state == MomentumState.EXHAUSTED or is_stretched:
        # confirmed trend BUT momentum hot / price stretched == the peak trap
        verdict = Verdict.PEAK_EXTENDED
    elif momentum_state == MomentumState.OVERSOLD:
        # in an uptrend but momentum collapsed — treat as possible trend break
        verdict = Verdict.DEEP_PULLBACK_RISK
    elif pullback_ok and volume_ok and momentum_state in (MomentumState.COOLED, MomentumState.NEUTRAL):
        verdict = Verdict.PULLBACK_IN_UPTREND
    else:
        verdict = Verdict.NEUTRAL_WAIT

    features = {
        k: (float(row[k]) if pd.notna(row[k]) else float("nan"))
        for k in row.index
        if np.isscalar(row[k]) or isinstance(row[k], (int, float, np.floating))
    }

    return SetupScore(
        verdict=verdict,
        momentum_state=momentum_state,
        trend_ok=trend_ok,
        pullback_ok=pullback_ok,
        volume_ok=volume_ok,
        trend_quality=quality,
        reasons=reasons,
        features=features,
    )


def score_latest(df: pd.DataFrame, params: ScoringParams = DEFAULT_PARAMS) -> SetupScore:
    """Convenience: compute features and classify the most recent *closed* bar.

    NOTE: pass a DataFrame whose last row is a CLOSED candle. Never pass a
    forming candle — that is lookahead in live trading and a bug in backtests.
    """
    feat = compute_features(df, params)
    return classify_setup(feat.iloc[-1], params)
