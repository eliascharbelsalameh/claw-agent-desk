"""SEC EDGAR filings + fundamentals client (free, no API key).

Fair-access rules require a descriptive User-Agent identifying the
requester and cap traffic at roughly 10 requests/second (spec section 5).
This client enforces a minimum interval between requests to stay under
that, and refuses to run without SEC_EDGAR_USER_AGENT set - don't hardcode
a default contact here, since the challenge entry may end up public
(spec section 2).
"""
from __future__ import annotations

import threading
import time
from typing import Any

import requests

from .base_client import BaseClient
from .cache import DiskCache
from .config import Settings, get_settings

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# SEC's fair access guidance targets ~10 req/s; stay comfortably under it.
_MIN_REQUEST_INTERVAL = 0.12


class _RateLimiter:
    def __init__(self, min_interval: float):
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            remaining = self._min_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
            self._last_call = time.monotonic()


class EdgarClient(BaseClient):
    def __init__(
        self,
        settings: Settings | None = None,
        session: requests.Session | None = None,
        cache: DiskCache | None = None,
    ):
        super().__init__(session=session, cache=cache)
        self._settings = settings or get_settings()
        self._rate_limiter = _RateLimiter(_MIN_REQUEST_INTERVAL)
        self._ticker_to_cik: dict[str, str] | None = None

    def _headers(self) -> dict[str, str]:
        return {"User-Agent": self._settings.require("edgar_user_agent")}

    def _get_json(self, url: str, *, cache_ttl: float | None = None, **kwargs: Any) -> Any:
        self._rate_limiter.wait()
        return super()._get_json(url, headers=self._headers(), cache_ttl=cache_ttl, **kwargs)

    def get_cik(self, ticker: str) -> str:
        if self._ticker_to_cik is None:
            data = self._get_json(TICKERS_URL, cache_ttl=7 * 24 * 3600.0)
            self._ticker_to_cik = {
                row["ticker"].upper(): str(row["cik_str"]).zfill(10) for row in data.values()
            }
        cik = self._ticker_to_cik.get(ticker.upper())
        if cik is None:
            raise KeyError(f"no CIK found for ticker {ticker!r}")
        return cik

    def get_company_submissions(self, ticker: str, cache_ttl: float = 3600.0) -> dict[str, Any]:
        """Recent filings list for a ticker (10-K/10-Q/8-K, dates, etc.)."""
        cik = self.get_cik(ticker)
        return self._get_json(SUBMISSIONS_URL.format(cik=cik), cache_ttl=cache_ttl)

    def get_company_facts(self, ticker: str, cache_ttl: float = 24 * 3600.0) -> dict[str, Any]:
        """XBRL-tagged fundamentals (all reported facts) for a ticker."""
        cik = self.get_cik(ticker)
        return self._get_json(COMPANY_FACTS_URL.format(cik=cik), cache_ttl=cache_ttl)
