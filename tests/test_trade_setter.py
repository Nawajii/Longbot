import math

from longbot import trade_setter as ts


def test_risk_based_sizing_and_levels():
    t = ts.build_ticket(
        "BTCUSDT", local_high=100.0, ema_fast=98.0, atr=2.0, capital=1000.0,
    )
    # entry = local_high + 0.1*ATR = 100.2 ; stop = entry - 2*ATR = 96.2
    assert math.isclose(t.entry_price, 100.2, rel_tol=1e-9)
    assert math.isclose(t.stop_price, 96.2, rel_tol=1e-9)
    assert math.isclose(t.risk_per_unit, 4.0, rel_tol=1e-9)
    # qty = (1000 * 1%) / 4 = 2.5 ; risk_amount = 10
    assert math.isclose(t.quantity, 2.5, rel_tol=1e-9)
    assert math.isclose(t.risk_amount, 10.0, rel_tol=1e-6)
    assert math.isclose(t.r_levels["1R"], 104.2, rel_tol=1e-9)
    assert math.isclose(t.r_levels["2R"], 108.2, rel_tol=1e-9)
    assert t.feasible


def test_capital_cap_limits_size():
    # huge risk budget but tiny capital must cap notional at <= capital
    p = ts.TradeSetterParams(risk_pct=1.0)  # absurd 100% risk
    t = ts.build_ticket("X", local_high=100.0, ema_fast=99.0, atr=2.0, capital=500.0, params=p)
    assert t.notional <= 500.0 + 1e-6
    assert any("capped by available capital" in w for w in t.warnings)


def test_min_notional_flagged_infeasible():
    t = ts.build_ticket("X", local_high=100.0, ema_fast=99.0, atr=2.0, capital=20.0)
    assert not t.feasible
    assert any("below exchange minimum" in w for w in t.warnings)


def test_lot_and_tick_rounding():
    f = ts.SymbolFilters(price_tick=0.1, qty_step=0.01, min_qty=0.001)
    t = ts.build_ticket("X", local_high=100.03, ema_fast=98.0, atr=2.0,
                        capital=5000.0, filters=f)
    # entry rounded UP to tick so the buy-stop stays above the high
    assert round(t.entry_price / 0.1) * 0.1 == t.entry_price
    # qty floored to step
    assert math.isclose(t.quantity, round(t.quantity / 0.01) * 0.01, abs_tol=1e-9)


def test_invalid_atr_is_infeasible():
    t = ts.build_ticket("X", local_high=100.0, ema_fast=99.0, atr=0.0, capital=1000.0)
    assert not t.feasible
