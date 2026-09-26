"""Alpaca Market Data (IEX feed, free Basic plan) + paper trading account.

Free-plan caveat (spec section 6): the Basic plan only carries the IEX
feed, which was ~4% of overall US equity volume in Q2 2026. Absolute
volume numbers are not comparable to consolidated tape volume - always
reason about *relative* volume against a recent rolling baseline instead
of trusting the absolute number (see get_relative_volume below).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
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


def session_in_progress(daily_bar: dict[str, Any], clock: dict[str, Any]) -> bool:
    """Whether `daily_bar` is today's session and that session hasn't closed.

    Daily bars are stamped at midnight US/Eastern (e.g. 2026-09-25T04:00:00Z),
    so their date prefix is the session's date; the clock's timestamp carries
    the Eastern offset, so its date prefix is today's exchange date. Today's
    session is still running (or not yet started) exactly when the next close
    is today; after the close, next_close moves to the next trading day.
    """
    today = str(clock.get("timestamp", ""))[:10]
    if not today:
        return False
    return daily_bar["t"][:10] == today and str(clock.get("next_close", ""))[:10] == today


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
        adjustment: str = "split",
    ) -> list[dict[str, Any]]:
        """Bars in ascending time order. Split-adjusted by default: Alpaca
        serves raw prices otherwise, and a year of NFLX bars across its
        10-for-1 split put the stock 94% below its "52-week high" (live,
        Sept 26, 2026). Split adjustment scales volume too, and leaves bars
        after the last split unchanged, so the latest price is the real one."""
        url = f"{self._settings.alpaca_data_base_url}/v2/stocks/{symbol}/bars"
        params: dict[str, Any] = {
            "timeframe": timeframe, "limit": limit, "feed": feed, "adjustment": adjustment,
        }
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

    def get_relative_volume(
        self, symbol: str, lookback_days: int = 20, clock: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Latest completed session's daily IEX volume vs the average of the
        `lookback_days` sessions before it.

        Spec section 6: absolute IEX volume understates true market volume,
        so agents should reason about relative volume against a recent
        rolling baseline (weeks, not years) rather than the absolute figure.

        While a session is still running, its daily bar holds only the volume
        traded so far; comparing that with full-day averages made relative
        volume look falsely low all morning. So a still-running session is
        left out of the comparison and reported separately
        (`in_progress_volume`). `clock` is get_clock()'s result; it is
        fetched when not given, and if the clock can't be read the latest
        bar is used as before, with `session_in_progress` None (unknown).
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days + 5)
        daily_bars = self.get_bars(
            symbol, timeframe="1Day", start=start, end=end, cache_ttl=3600.0
        )
        if clock is None:
            try:
                clock = self.get_clock()
            except Exception:  # noqa: BLE001 - relative volume still works, flagged as unknown
                clock = None
        in_progress = None
        if daily_bars and clock is not None and session_in_progress(daily_bars[-1], clock):
            in_progress = daily_bars[-1]
            daily_bars = daily_bars[:-1]
        if len(daily_bars) < 2:
            raise ValueError(f"not enough daily bars for {symbol} to compute relative volume")

        window = daily_bars[-(lookback_days + 1):]
        *history, latest = window
        if not history:
            raise ValueError(f"not enough history for {symbol} to compute relative volume")
        baseline = sum(bar["v"] for bar in history) / len(history)
        latest_volume = latest["v"]
        result = {
            "symbol": symbol,
            "date": latest["t"],
            "latest_volume": latest_volume,
            "baseline_avg_volume": baseline,
            "relative_volume": (latest_volume / baseline) if baseline else float("inf"),
            "feed": "iex",
            "session_in_progress": None if clock is None else in_progress is not None,
        }
        if in_progress is not None:
            result["in_progress_date"] = in_progress["t"]
            result["in_progress_volume"] = in_progress["v"]
        return result

    def get_clock(self) -> dict[str, Any]:
        """Alpaca's market clock: timestamp, is_open, next_open, next_close.
        Timestamps carry the exchange's own (US/Eastern) offset, and the clock
        accounts for holidays and early closes. Never cached."""
        url = f"{self._settings.alpaca_trading_base_url}/v2/clock"
        return self._get_json(url, headers=self._auth_headers())

    def get_calendar(self, start: date, end: date, cache_ttl: float = 24 * 3600.0) -> list[dict[str, Any]]:
        """Trading sessions between two dates: [{"date", "open", "close"}],
        holidays excluded and early closes included. The portfolio counts
        holding periods in these sessions, not calendar days."""
        url = f"{self._settings.alpaca_trading_base_url}/v2/calendar"
        params = {"start": start.isoformat(), "end": end.isoformat()}
        return self._get_json(url, params=params, headers=self._auth_headers(), cache_ttl=cache_ttl)

    # --- Paper trading account (simulated execution, spec section 5) ---

    def get_account(self) -> dict[str, Any]:
        url = f"{self._settings.alpaca_trading_base_url}/v2/account"
        return self._get_json(url, headers=self._auth_headers())

    def list_positions(self) -> list[dict[str, Any]]:
        url = f"{self._settings.alpaca_trading_base_url}/v2/positions"
        return self._get_json(url, headers=self._auth_headers())

    def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        time_in_force: str = "day",
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Market order on the paper account. This POST goes through the
        same retry as every request, so a connection dropped after Alpaca
        accepted the order would be sent again: pass a `client_order_id`
        derived from the decision, and Alpaca rejects the repeat (HTTP 422)
        instead of placing a second order - see get_order_by_client_id."""
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        url = f"{self._settings.alpaca_trading_base_url}/v2/orders"
        body = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": "market",
            "time_in_force": time_in_force,
        }
        if client_order_id:
            body["client_order_id"] = client_order_id
        response = request_with_retry(self._session, "POST", url, headers=self._auth_headers(), json=body)
        response.raise_for_status()
        return response.json()

    def get_order_by_client_id(self, client_order_id: str) -> dict[str, Any]:
        url = f"{self._settings.alpaca_trading_base_url}/v2/orders:by_client_order_id"
        return self._get_json(url, params={"client_order_id": client_order_id}, headers=self._auth_headers())
