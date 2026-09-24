"""Central configuration for the data layer, loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class ConfigError(RuntimeError):
    """Raised when a required setting is missing."""


@dataclass(frozen=True)
class Settings:
    alpaca_api_key_id: str | None
    alpaca_api_secret_key: str | None
    alpaca_data_base_url: str
    alpaca_trading_base_url: str
    fred_api_key: str | None
    finnhub_api_key: str | None
    edgar_user_agent: str | None
    cache_dir: Path

    def require(self, field: str) -> str:
        value = getattr(self, field)
        if not value:
            raise ConfigError(
                f"Missing required setting '{field}'. Set it in your .env file "
                "(see .env.example)."
            )
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        alpaca_api_key_id=os.getenv("ALPACA_API_KEY_ID"),
        alpaca_api_secret_key=os.getenv("ALPACA_API_SECRET_KEY"),
        alpaca_data_base_url=os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets"),
        alpaca_trading_base_url=os.getenv(
            "ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets"
        ),
        fred_api_key=os.getenv("FRED_API_KEY"),
        finnhub_api_key=os.getenv("FINNHUB_API_KEY"),
        edgar_user_agent=os.getenv("SEC_EDGAR_USER_AGENT"),
        cache_dir=Path(os.getenv("DATA_LAYER_CACHE_DIR", ".cache")),
    )
