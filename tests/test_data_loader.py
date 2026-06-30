import pandas as pd

from run_real_backtest import load_csv, make_synthetic


def test_load_headerless_binance_dump(tmp_path):
    # data.binance.vision monthly klines: headerless, 12 cols, ms-epoch open_time
    rows, t = [], 1609459200000  # 2021-01-01 00:00 UTC
    for i in range(50):
        rows.append([t, 100 + i, 101 + i, 99 + i, 100.5 + i, 1000, t + 3599999, 0, 0, 0, 0, 0])
        t += 3600000
    p = tmp_path / "dump.csv"
    pd.DataFrame(rows).to_csv(p, index=False, header=False)
    df = load_csv(str(p))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert str(df.index[0]) == "2021-01-01 00:00:00"
    assert len(df) == 50


def test_load_headered_csv_roundtrip(tmp_path):
    # our own --save-csv output (datetime index, named OHLCV columns)
    src = make_synthetic(120)
    p = tmp_path / "saved.csv"
    src.to_csv(p)
    df = load_csv(str(p))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == len(src)
    assert df.index.is_monotonic_increasing
