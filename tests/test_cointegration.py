import numpy as np

from pairs_pipeline import config
from pairs_pipeline.cointegration import fit_formation_window, spread_and_z


def test_cointegrated_pair_usually_passes_adf(cointegrated_prices):
    log_a = np.log(cointegrated_prices["A"])
    log_b = np.log(cointegrated_prices["B"])
    result = fit_formation_window(log_a.iloc[:200], log_b.iloc[:200])
    assert result.adf_pvalue < config.ADF_P_THRESHOLD
    assert result.cointegrated


def test_independent_walks_usually_fail_adf(independent_prices):
    log_a = np.log(independent_prices["A"])
    log_b = np.log(independent_prices["B"])
    result = fit_formation_window(log_a.iloc[:200], log_b.iloc[:200])
    # Not a hard guarantee for any single random seed, but with this fixed
    # seed two unrelated random walks should not spuriously cointegrate.
    assert result.adf_pvalue >= config.ADF_P_THRESHOLD
    assert not result.cointegrated


def test_formation_window_too_short_raises():
    import pandas as pd
    short_a = pd.Series(np.log(np.arange(10, 20)), index=pd.date_range("2020-01-01", periods=10))
    short_b = pd.Series(np.log(np.arange(20, 30)), index=pd.date_range("2020-01-01", periods=10))
    try:
        fit_formation_window(short_a, short_b)
        assert False, "expected ValueError for undersized formation window"
    except ValueError:
        pass


def test_spread_z_formula_matches_formation_mean_and_std():
    from pairs_pipeline.cointegration import FormationResult
    formation = FormationResult(
        formation_start=None, formation_end=None,
        beta=2.0, mean=1.0, std=0.5, adf_pvalue=0.01, n_obs=90,
    )
    # spread = log(A) - beta*log(B); pick A,B so spread == mean exactly -> z == 0
    price_b = 10.0
    price_a = np.exp(formation.mean + formation.beta * np.log(price_b))
    spread, z = spread_and_z(price_a, price_b, formation)
    assert abs(z) < 1e-9

    # spread one std above mean -> z == 1
    price_a_plus = np.exp(formation.mean + formation.std + formation.beta * np.log(price_b))
    _, z_plus = spread_and_z(price_a_plus, price_b, formation)
    assert abs(z_plus - 1.0) < 1e-9
