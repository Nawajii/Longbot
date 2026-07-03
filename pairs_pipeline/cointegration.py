"""
Formation-window cointegration + spread engine.

This is the module the whole project's correctness depends on. The rule
that must never be violated: everything computed here for a decision made
on trading day t uses ONLY price data with date < t (the 90 days strictly
before t). Nothing in this module ever looks at a date >= the day being
decided. tests/test_no_lookahead.py asserts this by construction: feeding
the same formation window with different/absent future data must produce
byte-identical output.

Engle-Granger procedure, applied per formation window:
  1. OLS: log(A) = alpha + beta * log(B) + resid   (over the formation window)
  2. ADF test on `resid` -> p-value. Pair is "tradeable" for the following
     window iff p < 0.05.
  3. Because of the OLS normal equations, mean(resid) == 0 over the
     formation window by construction, so mean(log(A) - beta*log(B)) over
     the same window equals `alpha` exactly. That lets us define the raw
     trading spread exactly as the prompt specifies -- S = log(A) - beta*log(B)
     -- while still centering it correctly: the formation-window mean of S
     is alpha, and its std is the OLS residual std. z = (S - alpha) / std.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from . import config


@dataclass(frozen=True)
class FormationResult:
    formation_start: object
    formation_end: object  # exclusive -- last date used is formation_end - 1 bar
    beta: float
    mean: float   # == OLS alpha, by construction (see module docstring)
    std: float
    adf_pvalue: float
    n_obs: int

    @property
    def cointegrated(self) -> bool:
        return self.adf_pvalue < config.ADF_P_THRESHOLD and self.std > 0


def fit_formation_window(log_a: pd.Series, log_b: pd.Series) -> FormationResult:
    """Fit beta/alpha/std/ADF p-value on the given window. The caller is
    responsible for ensuring log_a/log_b contain ONLY in-formation-window
    data -- this function has no notion of "today" and will happily use
    every row it's given, which is exactly why the no-lookahead guarantee
    lives in how the caller slices the data (see rolling_formation below
    and engine.py), not in this function.
    """
    if len(log_a) != len(log_b):
        raise ValueError("log_a and log_b must be aligned/same length")
    n = len(log_a)
    if n < config.FORMATION_WINDOW_DAYS:
        raise ValueError(f"formation window needs >= {config.FORMATION_WINDOW_DAYS} obs, got {n}")

    X = sm.add_constant(log_b.values)
    ols = sm.OLS(log_a.values, X).fit()
    alpha, beta = ols.params[0], ols.params[1]
    resid = ols.resid

    std = float(np.std(resid, ddof=1))
    if std == 0.0 or not np.isfinite(std):
        adf_p = 1.0
    else:
        try:
            adf_p = float(adfuller(resid, autolag="AIC")[1])
        except Exception:
            adf_p = 1.0

    return FormationResult(
        formation_start=log_a.index[0],
        formation_end=log_a.index[-1],
        beta=float(beta),
        mean=float(alpha),
        std=std,
        adf_pvalue=adf_p,
        n_obs=n,
    )


def formation_for_index(prices: pd.DataFrame, asset_a: str, asset_b: str, i: int) -> FormationResult | None:
    """Formation result usable for a decision at row i, fit on the
    FORMATION_WINDOW_DAYS rows strictly before i (rows [i-w, i)). Returns
    None if there isn't yet enough prior history. This is the single
    choke point every no-lookahead guarantee in this codebase rests on:
    row i itself, and every row after it, is never read here."""
    w = config.FORMATION_WINDOW_DAYS
    if i < w:
        return None
    log_a = np.log(prices[asset_a])
    log_b = np.log(prices[asset_b])
    window_a = log_a.iloc[i - w:i]
    window_b = log_b.iloc[i - w:i]
    return fit_formation_window(window_a, window_b)


def rolling_formation(prices: pd.DataFrame, asset_a: str, asset_b: str):
    """Generator yielding (decision_date, FormationResult | None) for every
    date in `prices` from the point enough history exists onward.

    decision_date is the date the formation result becomes usable for a
    trading decision -- i.e. it is computed from the FORMATION_WINDOW_DAYS
    rows strictly before decision_date. decision_date itself is never
    included in the fit.

    Kept mainly for tests/inspection; engine.py calls formation_for_index
    directly (and only when flat) to avoid fitting a formation window on
    days when no entry decision is even possible.
    """
    dates = prices.index
    for i in range(len(dates)):
        yield dates[i], formation_for_index(prices, asset_a, asset_b, i)


def spread_and_z(price_a: float, price_b: float, formation: FormationResult) -> tuple[float, float]:
    """Apply a FROZEN formation result to a (possibly later) price pair.
    Used both for the entry-day decision and for every subsequent day of
    an open trade (with the entry day's formation result held fixed)."""
    spread = np.log(price_a) - formation.beta * np.log(price_b)
    if formation.std == 0:
        return spread, 0.0
    z = (spread - formation.mean) / formation.std
    return float(spread), float(z)
