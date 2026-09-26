"""Finnhub company news client (free plan).

Free plan reportedly rejects US stock candle requests (spec section 5), so
this client is intentionally news-only - use AlpacaClient for prices, not
this. Sentiment is a paid Finnhub feature, so the bias/sentiment agents are
expected to score sentiment themselves from these raw headlines.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import requests

from .base_client import BaseClient
from .cache import DiskCache
from .config import Settings, get_settings

BASE_URL = "https://finnhub.io/api/v1"


class FinnhubClient(BaseClient):
    def __init__(
        self,
        settings: Settings | None = None,
        session: requests.Session | None = None,
        cache: DiskCache | None = None,
    ):
        super().__init__(session=session, cache=cache)
        self._settings = settings or get_settings()

    def get_company_news(
        self, symbol: str, from_date: date, to_date: date, cache_ttl: float = 900.0
    ) -> list[dict[str, Any]]:
        """News items for `symbol` between from_date and to_date.

        Each item is tagged with published_at/age_hours so the staleness
        check (spec section 6: "the staleness check needs a timestamp on
        every news item") doesn't need to re-derive it downstream.
        """
        params = {
            "symbol": symbol,
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
            "token": self._settings.require("finnhub_api_key"),
        }
        items = self._get_json(f"{BASE_URL}/company-news", params=params, cache_ttl=cache_ttl)
        now = datetime.now(timezone.utc)
        for item in items:
            published = datetime.fromtimestamp(item["datetime"], tz=timezone.utc)
            item["published_at"] = published.isoformat()
            item["age_hours"] = (now - published).total_seconds() / 3600.0
        return items

    def get_earnings_calendar(
        self, symbol: str, from_date: date, to_date: date, cache_ttl: float = 6 * 3600.0
    ) -> list[dict[str, Any]]:
        """Earnings reports for `symbol` between the two dates: date, hour
        (bmo/amc/dmh), fiscal quarter/year, EPS and revenue estimates, and
        the actuals once reported. On the free plan (checked live Sept 26,
        2026) it lists reports about a month back and months ahead."""
        params = {
            "symbol": symbol,
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
            "token": self._settings.require("finnhub_api_key"),
        }
        data = self._get_json(f"{BASE_URL}/calendar/earnings", params=params, cache_ttl=cache_ttl)
        return list(data.get("earningsCalendar") or [])
