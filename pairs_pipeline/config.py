"""
Pre-registered configuration for the long-only cointegration pairs strategy.

Every constant in this file is a parameter that was fixed BEFORE any
out-of-sample result was looked at. Do not tune these against OOS output.
If the strategy fails the bar in report.py, the correct response is to
retire it, not to edit this file and re-run.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Timeframe
# ---------------------------------------------------------------------------
# Daily bars, not 4h. Justification (fixed before any run):
#   1. The formation window is specified as "90 days" -- on a daily bar this
#      is exactly 90 observations with no rounding/ambiguity.
#   2. Engle-Granger/ADF cointegration tests are biased by the strong
#      intraday autocorrelation and microstructure noise present in 4h
#      crypto bars; daily closes are the standard resolution used in the
#      academic and practitioner pairs-trading literature.
#   3. It cuts data volume by ~6x for the same wall-clock history, which
#      matters when fetching 14 symbols back to 2019 from a rate-limited
#      public API.
INTERVAL = "1d"
BINANCE_INTERVAL = "1d"

# ---------------------------------------------------------------------------
# Pair universe (FIXED -- economic kinship, not data-mined).
# Binance ticker for each leg is resolved in data_fetch.SYMBOL_ALIASES.
# ---------------------------------------------------------------------------
PAIR_UNIVERSE: list[tuple[str, str]] = [
    ("ETH", "BTC"),     # the two majors
    ("SOL", "AVAX"),    # competing high-throughput L1s
    ("AVAX", "DOT"),    # L1 smart-contract peers
    ("ADA", "DOT"),     # L1 peers, similar era
    ("NEAR", "SOL"),    # L1 peers
    ("LTC", "BCH"),     # BTC-derived payment coins
    ("UNI", "AAVE"),    # DeFi blue chips
    ("LINK", "ETH"),    # ETH-beta infrastructure
    ("MATIC", "ETH"),   # ETH-scaling, ETH-beta (ticker resolved to MATIC/POL splice)
    ("ATOM", "DOT"),    # interoperability L1s
]

# ---------------------------------------------------------------------------
# Cointegration / spread engine (formation window)
# ---------------------------------------------------------------------------
FORMATION_WINDOW_DAYS = 90
ADF_P_THRESHOLD = 0.05

# Roll cadence: the prompt fixes the formation window length (90d) but does
# not fix how often it is re-estimated. We re-estimate DAILY (a fresh
# trailing 90-day formation window is computed every day a pair is flat),
# which is the standard convention in the pairs-trading literature and the
# most conservative reading of "no lookahead": each day's entry decision
# uses only the 90 days strictly before that day. Once a trade is opened,
# its beta/mean/std are FROZEN for the life of that trade (see engine.py) --
# they are not re-estimated mid-trade. This choice was fixed before any
# backtest was run.
ROLL_CADENCE_DAYS = 1

# ---------------------------------------------------------------------------
# Entry / exit thresholds (FIXED -- no sweeping)
# ---------------------------------------------------------------------------
ENTRY_Z = 2.0          # |z| >= 2.0 to open
TARGET_Z = 0.25         # |z| <= 0.25 to take profit (relationship normalised)
STOP_Z = 3.5            # |z| >= 3.5 => cointegration assumed broken, exit at loss
TIME_STOP_DAYS = 30      # exit flat if neither target nor stop hit

# ---------------------------------------------------------------------------
# Costs (reused from the prior single-asset project's design)
# ---------------------------------------------------------------------------
FEE_PCT = 0.0010         # 0.10% per side
SLIPPAGE_PCT = 0.0005    # 0.05% per side

# ---------------------------------------------------------------------------
# Regime bucketing (reused design: BTC 200-EMA)
# ---------------------------------------------------------------------------
REGIME_SYMBOL = "BTC"
REGIME_EMA_SPAN = 200

# ---------------------------------------------------------------------------
# Data range
# ---------------------------------------------------------------------------
# Requested start predates every symbol's actual Binance listing date; the
# fetcher simply receives data starting from each symbol's real listing.
DATA_START = "2017-01-01"

# ---------------------------------------------------------------------------
# Walk-forward OOS folds (calendar year of trade entry)
# ---------------------------------------------------------------------------
CRITICAL_BEAR_FOLD = 2022
FOLD_YEARS_OF_INTEREST = [2022, 2023, 2024, 2025]

# ---------------------------------------------------------------------------
# Binding success bar (fixed BEFORE results; see report.py::evaluate_bar)
# ---------------------------------------------------------------------------
MIN_AGGREGATE_EXPECTANCY_R = 0.15
MIN_OOS_TRADES = 100
MIN_BEAR_FOLD_SAMPLE = 30
MIN_BEAR_FOLD_EXPECTANCY_R = 0.0
CONCENTRATION_TOP_N_REMOVED = 5
MIN_EXPECTANCY_AFTER_REMOVAL_R = 0.05


@dataclass(frozen=True)
class TradeDirection:
    LONG_A = "long_A"
    LONG_B = "long_B"
