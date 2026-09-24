"""FRED (Federal Reserve Economic Data) macro series client.

Free API key, ~120 req/min, 800k+ series (spec section 5). Macro series are
slow-changing, so cache aggressively by default.
"""
from __future__ import annotations

from typing import Any

import requests

from .base_client import BaseClient
from .cache import DiskCache
from .config import Settings, get_settings

BASE_URL = "https://api.stlouisfed.org/fred"


class FredClient(BaseClient):
    def __init__(
        self,
        settings: Settings | None = None,
        session: requests.Session | None = None,
        cache: DiskCache | None = None,
    ):
        super().__init__(session=session, cache=cache)
        self._settings = settings or get_settings()

    def get_series_observations(
        self,
        series_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
        cache_ttl: float = 6 * 3600.0,
    ) -> list[dict[str, Any]]:
        """Observations for a series, e.g. series_id="FEDFUNDS".

        start_date/end_date are "YYYY-MM-DD" strings, per the FRED API.
        """
        params: dict[str, Any] = {
            "series_id": series_id,
            "api_key": self._settings.require("fred_api_key"),
            "file_type": "json",
        }
        if start_date:
            params["observation_start"] = start_date
        if end_date:
            params["observation_end"] = end_date

        data = self._get_json(
            f"{BASE_URL}/series/observations", params=params, cache_ttl=cache_ttl
        )
        return data.get("observations", [])
