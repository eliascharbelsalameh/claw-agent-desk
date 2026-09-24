"""Alpaca Market Data (IEX feed, free Basic plan) + paper trading account.

Free-plan caveat (spec section 6): the Basic plan only carries the IEX
feed, which was ~4% of overall US equity volume in Q2 2026. Absolute
volume numbers are not comparable to consolidated tape volume - always
reason about *relative* volume against a recent rolling baseline instead
of trusting the absolute number (see get_relative_volume below).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from .base_client import BaseClient
from .cache import DiskCache
from .config import Settings, get_settings
from .http_utils import request_with_retry


def _bucket_start(ts: datetime, hours: int) -> datetime:
    ts = ts.astimezone(timezone.utc)
    bucket_hour = (ts.hour // hours) * hours
    return ts.replace(hour=bucket_hour, minute=0, second=0, microsecond=0)


def aggregate_bars(bars: list[dict[str, Any]], hours: int = 4) -> list[dict[str, Any]]:
    """Aggregate ascending 1-hour OHLCV bars into `hours`-hour buckets.

    Fallback path for get_bars_4h when Alpaca's native multi-hour timeframe
    isn't available on the free plan (unverified at spec time - see spec
    section 5).
    """
    buckets: dict[datetime, dict[str, Any]] = {}
    order: list[datetime] = []
    for bar in bars:
        ts = datetime.fromisoformat(bar["t"].replace("Z", "+00:00"))
        bucket = _bucket_start(ts, hours)
        if bucket not in buckets:
            buckets[bucket] = {
                "t": bucket.isoformat().replace("+00:00", "Z"),
                "o": bar["o"],
                "h": bar["h"],
                "l": bar["l"],
                "c": bar["c"],
                "v": bar["v"],
            }
            order.append(bucket)
        else:
            agg = buckets[bucket]
            agg["h"] = max(agg["h"], bar["h"])
            agg["l"] = min(agg["l"], bar["l"])
            agg["c"] = bar["c"]
            agg["v"] += bar["v"]
    return [buckets[b] for b in order]


class AlpacaClient(BaseClient):
    def __init__(
        self,
        settings: Settings | None = None,
        session: requests.Session | None = None,
        cache: DiskCache | None = None,
    ):
        super().__init__(session=session, cache=cache)
        self._settings = settings or get_settings()

    def _auth_headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self._settings.require("alpaca_api_key_id"),
            "APCA-API-SECRET-KEY": self._settings.require("alpaca_api_secret_key"),
        }

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Hour",
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 10000,
        feed: str = "iex",
        cache_ttl: float | None = 300.0,
        max_pages: int = 50,
    ) -> list[dict[str, Any]]:
        url = f"{self._settings.alpaca_data_base_url}/v2/stocks/{symbol}/bars"
        params: dict[str, Any] = {"timeframe": timeframe, "limit": limit, "feed": feed}
        if start is not None:
            params["start"] = start.astimezone(timezone.utc).isoformat()
        if end is not None:
            params["end"] = end.astimezone(timezone.utc).isoformat()

        bars: list[dict[str, Any]] = []
        page_token: str | None = None
        for _ in range(max_pages):
            page_params = dict(params)
            if page_token:
                page_params["page_token"] = page_token
            data = self._get_json(
                url, params=page_params, headers=self._auth_headers(), cache_ttl=cache_ttl
            )
            bars.extend(data.get("bars") or [])
            page_token = data.get("next_page_token")
            if not page_token:
                break
        return bars

    def get_bars_4h(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        prefer_native: bool = True,
    ) -> list[dict[str, Any]]:
        """4-hour bars for the day-trading agent's global-view timeframe
        (spec section 3, step 5). Tries Alpaca's native "4Hour" timeframe
        first; falls back to aggregating 1-hour bars if the API rejects it.
        """
        if prefer_native:
            try:
                return self.get_bars(symbol, timeframe="4Hour", start=start, end=end)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status not in (400, 404, 422):
                    raise
        hourly = self.get_bars(symbol, timeframe="1Hour", start=start, end=end)
        return aggregate_bars(hourly, hours=4)

    def get_relative_volume(self, symbol: str, lookback_days: int = 20) -> dict[str, Any]:
        """Latest daily IEX volume vs the average of the prior `lookback_days`.

        Spec section 6: absolute IEX volume understates true market volume,
        so agents should reason about relative volume against a recent
        rolling baseline (weeks, not years) rather than the absolute figure.
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days + 5)
        daily_bars = self.get_bars(
            symbol, timeframe="1Day", start=start, end=end, cache_ttl=3600.0
        )
        if len(daily_bars) < 2:
            raise ValueError(f"not enough daily bars for {symbol} to compute relative volume")

        window = daily_bars[-(lookback_days + 1):]
        *history, latest = window
        if not history:
            raise ValueError(f"not enough history for {symbol} to compute relative volume")
        baseline = sum(bar["v"] for bar in history) / len(history)
        latest_volume = latest["v"]
        return {
            "symbol": symbol,
            "date": latest["t"],
            "latest_volume": latest_volume,
            "baseline_avg_volume": baseline,
            "relative_volume": (latest_volume / baseline) if baseline else float("inf"),
            "feed": "iex",
        }

    # --- Paper trading account (simulated execution, spec section 5) ---

    def get_account(self) -> dict[str, Any]:
        url = f"{self._settings.alpaca_trading_base_url}/v2/account"
        return self._get_json(url, headers=self._auth_headers())

    def list_positions(self) -> list[dict[str, Any]]:
        url = f"{self._settings.alpaca_trading_base_url}/v2/positions"
        return self._get_json(url, headers=self._auth_headers())

    def submit_market_order(
        self, symbol: str, qty: float, side: str, time_in_force: str = "day"
    ) -> dict[str, Any]:
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        url = f"{self._settings.alpaca_trading_base_url}/v2/orders"
        response = request_with_retry(
            self._session,
            "POST",
            url,
            headers=self._auth_headers(),
            json={
                "symbol": symbol,
                "qty": str(qty),
                "side": side,
                "type": "market",
                "time_in_force": time_in_force,
            },
        )
        response.raise_for_status()
        return response.json()
