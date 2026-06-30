"""Central configuration via environment variables / a .env file.

Secrets (exchange keys, the Gemini/Telegram tokens) are read from the
environment and are **never** committed. See .env.example for the shape.

Security rules that this config assumes (your job to enforce on the exchange):
  * the API key has TRADE enabled and WITHDRAWAL DISABLED;
  * the key is IP-whitelisted to the host that runs the bot;
  * keys live only in the environment / a secrets store, never in git.

Nothing here affects the backtest — backtesting uses only free public klines
and needs no keys.
"""

from __future__ import annotations

try:
    from pydantic import Field
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError as exc:  # pragma: no cover - clearer message than a raw import error
    raise ImportError(
        "pydantic and pydantic-settings are required for live/paper config. "
        "Install with: pip install pydantic pydantic-settings"
    ) from exc


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="LONGBOT_", extra="ignore")

    # ---- exchange (NO withdrawal permission, IP-whitelisted) ----
    binance_api_key: str = Field(default="", description="trade-only key, withdrawal disabled")
    binance_api_secret: str = Field(default="")
    use_testnet: bool = Field(default=True, description="point ccxt at testnet.binance.vision")

    # ---- trading universe / timeframe ----
    quote_currency: str = "USDT"
    timeframe: str = Field(default="1h", description="candle timeframe; keep it a parameter")
    capital: float = Field(default=1000.0, description="spot quote capital the bot may use")

    # ---- risk governor (the dumb hard kill-switch) ----
    max_daily_loss_pct: float = 0.05         # halt if down 5% on the day
    max_drawdown_pct: float = 0.20           # halt if equity 20% below peak
    max_consecutive_losses: int = 4          # halt after N losers in a row

    # ---- human-in-the-loop ----
    autotrade: bool = Field(default=False, description="False = suggest only, human places trades")

    # ---- notifications (Telegram) ----
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # ---- Coroner (post-trade LLM analysis, reports to human only) ----
    gemini_api_key: str = ""
    coroner_model: str = "gemini-1.5-flash"
    # optional deeper-reasoning fallback (e.g. sentiment-quality veto, later)
    anthropic_api_key: str = ""

    # ---- storage ----
    db_path: str = "longbot.db"
    log_path: str = "longbot.log"


def load_settings() -> "Settings":
    return Settings()
