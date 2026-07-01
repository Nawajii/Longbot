"""Volume Profile — the "location" layer.

The pullback engine knows *direction* (trend), *timing* (momentum) and *risk*
(ATR stops), but it was blind to *location*: it would buy a pullback without
checking whether real support sat beneath it. Volume Profile supplies that
missing gear by mapping where trading volume actually concentrated — a
volume-at-price histogram whose peaks are equilibrium zones (support/resistance)
and whose valleys are voids price slides through.

APPROXIMATION (state it plainly, it matters):
    True volume-at-price needs tick data. We only have OHLCV candles, so we
    APPROXIMATE each candle's volume-at-price by spreading its volume evenly
    across the price bins its high–low range spans (proportional to how much of
    the bin the candle's range covers). This is the standard OHLCV approximation
    and it is only that — an approximation. It cannot see intrabar structure.

NO-LOOKAHEAD (the sacred rule):
    The profile at bar `i` is built ONLY from bars in the trailing window ending
    at `i`. It never uses a single future bar. There is a unit test that proves
    bar `i`'s profile is byte-for-byte unchanged when later bars are appended.

Fixed design parameters (set ONCE — deliberately NOT swept; tuning these until
the backtest looks good would be overfitting):
    VP_LOOKBACK     = 120 bars
    N_BINS          = 50
    VALUE_AREA_PCT  = 0.70   (standard 70% value area)
    NEAR_NODE_PCT   = 0.015  (price is "at" a node within 1.5% of it)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---- fixed, conventional parameters (do NOT sweep) ----
VP_LOOKBACK = 120
N_BINS = 50
VALUE_AREA_PCT = 0.70
NEAR_NODE_PCT = 0.015


@dataclass
class VolumeProfile:
    bin_edges: np.ndarray          # length N_BINS+1
    bin_centers: np.ndarray        # length N_BINS
    volume: np.ndarray             # length N_BINS, volume-at-price
    total_volume: float
    poc_price: float               # Point of Control (highest-volume bin center)
    vah: float                     # Value Area High
    val: float                     # Value Area Low
    hvn_prices: list[float] = field(default_factory=list)  # High-Volume Nodes (support/res)
    lvn_prices: list[float] = field(default_factory=list)  # Low-Volume Nodes (voids)


@dataclass
class Location:
    """Where a given price sits relative to a profile."""

    nearest_hvn: float
    dist_hvn_pct: float            # distance to nearest HVN as fraction of price
    in_value_area: bool            # VAL <= price <= VAH
    at_node: bool                  # price is at/near real support (HVN or VAL)
    in_void: bool                  # the complement: price hangs in a low-volume void
    support_price: float           # nearest supporting level at/below price (for stops)


# --------------------------------------------------------------------------- #
# core profile computation (single window)
# --------------------------------------------------------------------------- #
def compute_profile(
    lows: np.ndarray,
    highs: np.ndarray,
    volumes: np.ndarray,
    n_bins: int = N_BINS,
    value_area_pct: float = VALUE_AREA_PCT,
) -> VolumeProfile:
    """Build a volume-at-price profile from one window of candles.

    Volume is distributed across price bins in proportion to how much of each
    bin the candle's [low, high] range overlaps (the OHLCV approximation).
    """
    lo = float(np.min(lows))
    hi = float(np.max(highs))
    total_volume = float(np.sum(volumes))

    # degenerate windows: flat price or no volume -> single-node profile
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo or total_volume <= 0:
        price = lo if np.isfinite(lo) else 0.0
        vol = np.zeros(n_bins)
        vol[0] = total_volume
        return VolumeProfile(
            bin_edges=np.array([price, price]), bin_centers=np.array([price]),
            volume=np.array([total_volume]), total_volume=total_volume,
            poc_price=price, vah=price, val=price,
            hvn_prices=[price], lvn_prices=[],
        )

    edges = np.linspace(lo, hi, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    lower_e = edges[:-1]
    upper_e = edges[1:]

    vol = np.zeros(n_bins)
    for l, h, v in zip(lows, highs, volumes):
        if v <= 0:
            continue
        span = h - l
        if span <= 0:
            # candle with no range: dump its volume in the bin containing `l`
            idx = min(int(np.searchsorted(edges, l, side="right") - 1), n_bins - 1)
            idx = max(idx, 0)
            vol[idx] += v
            continue
        # overlap of [l, h] with each bin [lower_e, upper_e]
        overlap = np.clip(np.minimum(h, upper_e) - np.maximum(l, lower_e), 0.0, None)
        vol += v * overlap / span

    poc_idx = int(np.argmax(vol))
    poc_price = float(centers[poc_idx])

    val, vah = _value_area(vol, edges, poc_idx, total_volume, value_area_pct)
    hvn_prices, lvn_prices = _nodes(vol, centers)

    return VolumeProfile(
        bin_edges=edges, bin_centers=centers, volume=vol, total_volume=total_volume,
        poc_price=poc_price, vah=vah, val=val,
        hvn_prices=hvn_prices, lvn_prices=lvn_prices,
    )


def _value_area(vol, edges, poc_idx, total_volume, value_area_pct):
    """Expand out from the POC bin until `value_area_pct` of volume is enclosed.

    Standard approach: at each step add whichever adjacent side holds more volume.
    """
    n = len(vol)
    target = value_area_pct * total_volume
    included = {poc_idx}
    cum = vol[poc_idx]
    lo_i, hi_i = poc_idx - 1, poc_idx + 1
    while cum < target and (lo_i >= 0 or hi_i < n):
        left = vol[lo_i] if lo_i >= 0 else -1.0
        right = vol[hi_i] if hi_i < n else -1.0
        if right >= left:
            included.add(hi_i)
            cum += max(right, 0.0)
            hi_i += 1
        else:
            included.add(lo_i)
            cum += max(left, 0.0)
            lo_i -= 1
    val = float(edges[min(included)])
    vah = float(edges[max(included) + 1])
    return val, vah


def _nodes(vol, centers):
    """High/Low-Volume Nodes via contiguous runs relative to the mean bin volume.

    A High-Volume Node is a run of bins ABOVE the mean (a volume hump); we take
    the peak of each run. A Low-Volume Node is a run BELOW the mean (a void); we
    take its trough. Using runs (not strict local maxima) makes detection robust
    to the flat plateaus the even-volume OHLCV approximation can produce.
    """
    mean_v = float(np.mean(vol))
    n = len(vol)

    def _runs(mask, pick):
        out, j = [], 0
        while j < n:
            if mask[j]:
                k = j
                while k + 1 < n and mask[k + 1]:
                    k += 1
                local = int(pick(vol[j : k + 1])) + j
                out.append(float(centers[local]))
                j = k + 1
            else:
                j += 1
        return out

    hvn = _runs(vol > mean_v, np.argmax)
    lvn = _runs(vol < mean_v, np.argmin)
    if not hvn:                        # always expose at least the POC as a node
        hvn.append(float(centers[int(np.argmax(vol))]))
    return hvn, lvn


# --------------------------------------------------------------------------- #
# location of a price relative to a profile
# --------------------------------------------------------------------------- #
def location(price: float, profile: VolumeProfile, near_pct: float = NEAR_NODE_PCT) -> Location:
    """Classify where `price` sits: at real support (AT_NODE) or in a void (IN_VOID)."""
    hvns = np.array(profile.hvn_prices) if profile.hvn_prices else np.array([profile.poc_price])
    dists = np.abs(hvns - price) / price
    k = int(np.argmin(dists))
    nearest_hvn = float(hvns[k])
    dist_hvn_pct = float(dists[k])

    in_va = profile.val <= price <= profile.vah
    near_hvn = dist_hvn_pct <= near_pct
    near_val = abs(price - profile.val) / price <= near_pct

    # "at support" = sitting on a high-volume node, or resting on the value-area low
    at_node = bool(near_hvn or near_val)
    in_void = not at_node

    # nearest supporting level at/below price -> where a node-anchored stop goes
    candidates = list(profile.hvn_prices) + [profile.val, profile.poc_price]
    below = [c for c in candidates if c <= price]
    support_price = float(max(below)) if below else float(min(candidates))

    return Location(
        nearest_hvn=nearest_hvn, dist_hvn_pct=dist_hvn_pct,
        in_value_area=in_va, at_node=at_node, in_void=in_void,
        support_price=support_price,
    )


# --------------------------------------------------------------------------- #
# rolling features over a whole OHLCV frame (no-lookahead)
# --------------------------------------------------------------------------- #
LOCATION_COLUMNS = [
    "vp_poc", "vp_vah", "vp_val",
    "vp_nearest_hvn", "vp_dist_hvn_pct",
    "vp_in_value_area", "vp_at_node", "vp_in_void", "vp_support_price",
]


def compute_location_features(
    df: pd.DataFrame,
    lookback: int = VP_LOOKBACK,
    n_bins: int = N_BINS,
    value_area_pct: float = VALUE_AREA_PCT,
    near_pct: float = NEAR_NODE_PCT,
) -> pd.DataFrame:
    """Per-bar location features, using ONLY the trailing `lookback` bars at each i.

    Bars before the window is full are NaN/False (unknown location -> not tradable).
    """
    lows = df["low"].to_numpy()
    highs = df["high"].to_numpy()
    vols = df["volume"].to_numpy()
    closes = df["close"].to_numpy()
    n = len(df)

    out = {c: np.full(n, np.nan) for c in LOCATION_COLUMNS}
    for c in ("vp_in_value_area", "vp_at_node", "vp_in_void"):
        out[c] = np.zeros(n, dtype=bool)

    for i in range(lookback - 1, n):
        s = i - lookback + 1
        prof = compute_profile(lows[s : i + 1], highs[s : i + 1], vols[s : i + 1],
                               n_bins=n_bins, value_area_pct=value_area_pct)
        loc = location(closes[i], prof, near_pct=near_pct)
        out["vp_poc"][i] = prof.poc_price
        out["vp_vah"][i] = prof.vah
        out["vp_val"][i] = prof.val
        out["vp_nearest_hvn"][i] = loc.nearest_hvn
        out["vp_dist_hvn_pct"][i] = loc.dist_hvn_pct
        out["vp_in_value_area"][i] = loc.in_value_area
        out["vp_at_node"][i] = loc.at_node
        out["vp_in_void"][i] = loc.in_void
        out["vp_support_price"][i] = loc.support_price

    return pd.DataFrame(out, index=df.index)
