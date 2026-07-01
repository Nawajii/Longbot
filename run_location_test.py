"""The Volume Profile "location" experiment — pre-registered, falsifiable.

HYPOTHESIS (registered BEFORE seeing the result):
    Pullback entries that occur INTO a high-volume node (a real equilibrium /
    support zone) have higher expectancy than pullback entries that occur in a
    low-volume void — and adding this location filter turns the strategy's
    aggregate expectancy positive where the blind version was negative.

Two tests, on the SAME 20-pairs x 3-windows x 4h sweep as run_scenarios.py:

  PRIMARY (does location discriminate?):
      Take every pullback the OLD blind engine would trade, and TAG each by its
      location at the signal bar — AT_NODE (into an HVN / value area) vs IN_VOID.
      Nothing about entries or stops changes; we just label existing trades.
      Report expectancy / win rate / count for each subset. If AT_NODE clearly
      beats IN_VOID, location matters. (This isolates location from the stop
      change — the cleanest single-variable test.)

  SECONDARY (does the full arrangement work?):
      Run the complete new gearbox (location gate ON + node-anchored stops) and
      compare its aggregate + per-scenario expectancy head-to-head against the
      old blind engine (location OFF), on the identical sweep.

Honesty rules: fixed VP config (no sweeping), no-lookahead, fees + slippage as
in the harness. Cells with <30 trades are flagged NOT CONCLUSIVE. "Beating buy
& hold in a down window" is NOT credited as edge (it usually means sitting in
cash). Run where Binance is reachable:  python run_location_test.py
"""

from __future__ import annotations

import argparse

import numpy as np

import backtest as bt
from longbot import scoring_engine as se
from longbot import volume_profile as vp
from run_scenarios import DEFAULT_PAIRS, DEFAULT_SCENARIOS, MIN_USABLE_BARS, _fetch_window

MIN_CONCLUSIVE = 30


def _expectancy(trades) -> tuple[int, float, float]:
    """(n, win_rate, expectancy_R) for a list of trades."""
    if not trades:
        return 0, float("nan"), float("nan")
    rs = [t.r_multiple for t in trades if not np.isnan(t.r_multiple)]
    wins = sum(1 for t in trades if t.pnl > 0)
    exp = float(np.mean(rs)) if rs else float("nan")
    return len(trades), wins / len(trades), exp


def _flag(n: int) -> str:
    return "" if n >= MIN_CONCLUSIVE else "  (<30: NOT CONCLUSIVE)"


def run_experiment(pairs, timeframe, scenarios, capital, entry_valid_bars, regime_symbol):
    blind_bt = bt.BacktestParams(
        starting_capital=capital, timeframe=timeframe, entry_valid_bars=entry_valid_bars)
    located_bt = bt.BacktestParams(
        starting_capital=capital, timeframe=timeframe, entry_valid_bars=entry_valid_bars,
        use_node_stop=True)
    located_sp = se.ScoringParams(use_location=True)

    # collectors
    tagged = {"AT_NODE": [], "IN_VOID": []}                 # primary
    tagged_by_scen = {s: {"AT_NODE": [], "IN_VOID": []} for s in scenarios}
    blind_trades, located_trades = [], []                   # secondary (aggregate)
    blind_by_scen = {s: [] for s in scenarios}
    located_by_scen = {s: [] for s in scenarios}
    blind_orders = blind_fills = located_orders = located_fills = 0
    executed = skipped = 0

    for scen, (start, end) in scenarios.items():
        try:
            regime_df = _fetch_window(regime_symbol, timeframe, start, end)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {scen}: no regime data ({exc}); running ungated.")
            regime_df = None

        for pair in pairs:
            try:
                df = _fetch_window(pair, timeframe, start, end)
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                print(f"[skip] {scen} {pair}: fetch failed ({exc})")
                continue
            if len(df) < MIN_USABLE_BARS:
                skipped += 1
                print(f"[skip] {scen} {pair}: insufficient data ({len(df)} bars)")
                continue
            executed += 1

            # compute the location profile ONCE and reuse for tags + located engine
            loc = vp.compute_location_features(df)
            tags = loc["vp_at_node"]

            # PRIMARY: old blind engine, trades merely labelled by location
            old = bt.run_backtest(df, blind_bt, regime_df=regime_df, tag_series=tags)
            for t in old.trades:
                tagged.setdefault(t.tag, []).append(t)
                tagged_by_scen[scen].setdefault(t.tag, []).append(t)
            blind_trades += old.trades
            blind_by_scen[scen] += old.trades
            blind_orders += old.orders_placed
            blind_fills += old.fills

            # SECONDARY: full new gearbox (location gate + node stops)
            new = bt.run_backtest(df, located_bt, scoring_params=located_sp,
                                  regime_df=regime_df, location_features=loc)
            located_trades += new.trades
            located_by_scen[scen] += new.trades
            located_orders += new.orders_placed
            located_fills += new.fills

            print(f"[run] {scen:<4} {pair:<9} blind={len(old.trades):>2} trades  "
                  f"located={len(new.trades):>2} trades  ({len(df)} bars)")

    return {
        "tagged": tagged, "tagged_by_scen": tagged_by_scen,
        "blind_trades": blind_trades, "located_trades": located_trades,
        "blind_by_scen": blind_by_scen, "located_by_scen": located_by_scen,
        "blind_fill": (blind_fills, blind_orders),
        "located_fill": (located_fills, located_orders),
        "executed": executed, "skipped": skipped, "scenarios": scenarios,
    }


def format_experiment(R: dict) -> str:
    scen = R["scenarios"]
    L = ["=" * 78,
         " VOLUME-PROFILE LOCATION EXPERIMENT — pre-registered hypothesis:",
         "   Pullbacks INTO a high-volume node have higher expectancy than pullbacks",
         "   in a low-volume void; adding the location filter turns aggregate",
         "   expectancy positive where the blind engine was negative.",
         f"   (fixed VP config: lookback={vp.VP_LOOKBACK}, bins={vp.N_BINS}, "
         f"VA={vp.VALUE_AREA_PCT:.0%}, near={vp.NEAR_NODE_PCT:.1%}; no sweeping)",
         "=" * 78,
         f" runs executed: {R['executed']}   skipped (insufficient data): {R['skipped']}",
         "-" * 78,
         " PRIMARY TEST — does location discriminate? (same trades, just labelled)",
         f"   {'subset':<10}{'trades':>8}{'win rate':>11}{'expectancy':>14}"]

    at = R["tagged"].get("AT_NODE", [])
    void = R["tagged"].get("IN_VOID", [])
    for name, tr in (("AT_NODE", at), ("IN_VOID", void)):
        n, wr, e = _expectancy(tr)
        L.append(f"   {name:<10}{n:>8}{_pct(wr):>11}{_r(e):>14}{_flag(n)}")
    _, _, e_at = _expectancy(at)
    _, _, e_void = _expectancy(void)
    gap = e_at - e_void if (not np.isnan(e_at) and not np.isnan(e_void)) else float("nan")
    L.append(f"   gap (AT_NODE - IN_VOID): {_r(gap)} R  "
             f"{'-> location discriminates' if gap > 0 else '-> no discrimination'}")
    L.append("   per-scenario (AT_NODE vs IN_VOID expectancy):")
    for s in scen:
        a = _expectancy(R['tagged_by_scen'][s].get('AT_NODE', []))
        v = _expectancy(R['tagged_by_scen'][s].get('IN_VOID', []))
        L.append(f"     {s:<5} AT_NODE {_r(a[2])} (n={a[0]}){_flag(a[0])}   "
                 f"IN_VOID {_r(v[2])} (n={v[0]}){_flag(v[0])}")

    L += ["-" * 78,
          " SECONDARY TEST — full new gearbox vs old blind engine (head-to-head)",
          f"   {'engine':<20}{'trades':>8}{'win rate':>11}{'expectancy':>14}{'fill rate':>12}"]
    bn, bwr, be = _expectancy(R["blind_trades"])
    ln, lwr, le = _expectancy(R["located_trades"])
    L.append(f"   {'OLD (blind)':<20}{bn:>8}{_pct(bwr):>11}{_r(be):>14}"
             f"{_pct(_ratio(R['blind_fill'])):>12}{_flag(bn)}")
    L.append(f"   {'NEW (located+node)':<20}{ln:>8}{_pct(lwr):>11}{_r(le):>14}"
             f"{_pct(_ratio(R['located_fill'])):>12}{_flag(ln)}")
    L.append("   per-scenario expectancy (blind -> located):")
    for s in scen:
        b = _expectancy(R['blind_by_scen'][s])
        l = _expectancy(R['located_by_scen'][s])
        L.append(f"     {s:<5} blind {_r(b[2])} (n={b[0]})  ->  "
                 f"located {_r(l[2])} (n={l[0]}){_flag(min(b[0], l[0]))}")

    # verdicts
    L += ["-" * 78, " VERDICT:"]
    prim = ("SUPPORTED: AT_NODE expectancy exceeds IN_VOID."
            if gap > 0 else "NOT SUPPORTED: location did not separate winners from losers.")
    L.append(f"   Primary (location discriminates?): {prim}")
    turned_positive = (not np.isnan(be) and not np.isnan(le) and be <= 0 < le)
    sec = ("SUPPORTED: located engine expectancy is positive where blind was not."
           if turned_positive else
           "NOT SUPPORTED: the full arrangement did not turn aggregate expectancy positive.")
    L.append(f"   Secondary (full arrangement works?): {sec}")
    bull_n, _, bull_e = _expectancy(R['located_by_scen'].get('BULL', []))
    L.append(f"   Decisive cell — located BULL expectancy: {_r(bull_e)} R "
             f"(n={bull_n}){_flag(bull_n)}")
    L += ["   NOTE: expectancy that is positive only in BULL, but flat/negative in",
          "   CHOP and BEAR, is the 'no real edge, just long a bull market' pattern.",
          "   A genuine edge survives CHOP and limits damage in BEAR, net of costs.",
          "=" * 78]
    return "\n".join(L)


def _pct(x):
    return "n/a" if (x is None or np.isnan(x)) else f"{x*100:.1f}%"


def _r(x):
    return "n/a" if (x is None or np.isnan(x)) else f"{x:+.3f}"


def _ratio(fill_orders):
    f, o = fill_orders
    return f / o if o else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="*", default=DEFAULT_PAIRS)
    ap.add_argument("--timeframe", default="4h")
    ap.add_argument("--capital", type=float, default=1000.0)
    ap.add_argument("--entry-valid-bars", type=int, default=3)
    ap.add_argument("--regime-symbol", default="BTCUSDT")
    args = ap.parse_args()

    print(__doc__.split("Honesty rules")[0])
    R = run_experiment(args.pairs, args.timeframe, DEFAULT_SCENARIOS,
                       args.capital, args.entry_valid_bars, args.regime_symbol)
    print("\n" + format_experiment(R))


if __name__ == "__main__":
    main()
