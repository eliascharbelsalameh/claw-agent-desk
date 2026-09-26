import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import requests

from data_layer.alpaca_client import AlpacaClient, aggregate_bars, session_in_progress
from data_layer.config import Settings


def _bar(hour, o, h, l, c, v, day=24):
    ts = datetime(2026, 9, day, hour, 0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    return {"t": ts, "o": o, "h": h, "l": l, "c": c, "v": v}


def _settings(**overrides):
    base = dict(
        alpaca_api_key_id="k",
        alpaca_api_secret_key="s",
        alpaca_data_base_url="https://data.alpaca.markets",
        alpaca_trading_base_url="https://paper-api.alpaca.markets",
        fred_api_key=None,
        finnhub_api_key=None,
        edgar_user_agent=None,
        nvidia_api_key=None,
        nvidia_base_url="",
        cache_dir=Path("/tmp/does-not-matter"),
    )
    base.update(overrides)
    return Settings(**base)


def test_aggregate_bars_groups_by_4_hours():
    bars = [
        _bar(0, 10, 12, 9, 11, 100),
        _bar(1, 11, 13, 10, 12, 100),
        _bar(2, 12, 14, 11, 13, 100),
        _bar(3, 13, 15, 12, 14, 100),
        _bar(4, 14, 16, 13, 15, 50),
    ]
    result = aggregate_bars(bars, hours=4)
    assert len(result) == 2
    first = result[0]
    assert first["o"] == 10
    assert first["h"] == 15
    assert first["l"] == 9
    assert first["c"] == 14
    assert first["v"] == 400
    assert result[1]["v"] == 50


def test_get_bars_4h_falls_back_when_native_unsupported(monkeypatch):
    client = AlpacaClient(settings=_settings())

    def fake_get_bars(symbol, timeframe="1Hour", **kwargs):
        if timeframe == "4Hour":
            response = requests.Response()
            response.status_code = 422
            raise requests.HTTPError(response=response)
        return [_bar(0, 1, 1, 1, 1, 1), _bar(1, 1, 1, 1, 1, 1)]

    monkeypatch.setattr(client, "get_bars", fake_get_bars)
    result = client.get_bars_4h("AAPL")
    assert len(result) == 1
    assert result[0]["v"] == 2


def test_get_bars_4h_uses_native_when_available(monkeypatch):
    client = AlpacaClient(settings=_settings())
    native_bars = [_bar(0, 1, 1, 1, 1, 1)]

    def fake_get_bars(symbol, timeframe="1Hour", **kwargs):
        assert timeframe == "4Hour"
        return native_bars

    monkeypatch.setattr(client, "get_bars", fake_get_bars)
    result = client.get_bars_4h("AAPL")
    assert result is native_bars


def test_get_bars_4h_reraises_unexpected_errors(monkeypatch):
    client = AlpacaClient(settings=_settings())

    def fake_get_bars(symbol, timeframe="1Hour", **kwargs):
        response = requests.Response()
        response.status_code = 500
        raise requests.HTTPError(response=response)

    monkeypatch.setattr(client, "get_bars", fake_get_bars)
    with pytest.raises(requests.HTTPError):
        client.get_bars_4h("AAPL")


CLOSED_CLOCK = {"timestamp": "2026-09-21T18:00:00-04:00", "is_open": False,
                "next_open": "2026-09-22T09:30:00-04:00", "next_close": "2026-09-22T16:00:00-04:00"}


def test_get_relative_volume(monkeypatch):
    client = AlpacaClient(settings=_settings())
    bars = [_bar(4, 1, 1, 1, 1, 100, day=d) for d in range(1, 21)] + [
        _bar(4, 1, 1, 1, 1, 400, day=21)
    ]

    monkeypatch.setattr(client, "get_bars", lambda *a, **k: bars)
    result = client.get_relative_volume("AAPL", lookback_days=20, clock=CLOSED_CLOCK)
    assert result["relative_volume"] == pytest.approx(4.0)
    assert result["baseline_avg_volume"] == pytest.approx(100.0)
    assert result["feed"] == "iex"
    assert result["session_in_progress"] is False
    assert "in_progress_volume" not in result


def test_relative_volume_leaves_out_a_session_still_running(monkeypatch):
    client = AlpacaClient(settings=_settings())
    # 20 full days at 100, then day 21 at 300, then today's (day 22) partial 30
    bars = ([_bar(4, 1, 1, 1, 1, 100, day=d) for d in range(1, 21)]
            + [_bar(4, 1, 1, 1, 1, 300, day=21), _bar(4, 1, 1, 1, 1, 30, day=22)])
    monkeypatch.setattr(client, "get_bars", lambda *a, **k: bars)
    open_clock = {"timestamp": "2026-09-22T10:45:00-04:00", "is_open": True,
                  "next_open": "2026-09-23T09:30:00-04:00", "next_close": "2026-09-22T16:00:00-04:00"}

    result = client.get_relative_volume("AAPL", lookback_days=20, clock=open_clock)

    assert result["relative_volume"] == pytest.approx(3.0)  # day 21 vs days 1-20, not 30 vs 100
    assert result["date"].startswith("2026-09-21")
    assert result["session_in_progress"] is True
    assert result["in_progress_volume"] == 30 and result["in_progress_date"].startswith("2026-09-22")


def test_premarket_bar_counts_as_in_progress_but_yesterday_after_close_does_not():
    today_bar = _bar(4, 1, 1, 1, 1, 5, day=22)
    premarket = {"timestamp": "2026-09-22T08:00:00-04:00", "is_open": False,
                 "next_close": "2026-09-22T16:00:00-04:00"}
    after_close = {"timestamp": "2026-09-22T17:00:00-04:00", "is_open": False,
                   "next_close": "2026-09-23T16:00:00-04:00"}
    assert session_in_progress(today_bar, premarket) is True
    assert session_in_progress(today_bar, after_close) is False
    assert session_in_progress(_bar(4, 1, 1, 1, 1, 5, day=21), premarket) is False
    assert session_in_progress(today_bar, {}) is False


def test_relative_volume_without_a_clock_uses_latest_bar_and_says_unknown(monkeypatch):
    client = AlpacaClient(settings=_settings())
    bars = [_bar(4, 1, 1, 1, 1, 100, day=d) for d in range(1, 22)]
    monkeypatch.setattr(client, "get_bars", lambda *a, **k: bars)

    def no_clock():
        raise requests.ConnectionError("down")

    monkeypatch.setattr(client, "get_clock", no_clock)
    result = client.get_relative_volume("AAPL", lookback_days=20)
    assert result["session_in_progress"] is None and result["relative_volume"] == pytest.approx(1.0)


class _PagedSession:
    """Two pages of bars, then no next_page_token."""

    def __init__(self):
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append(kwargs["params"])
        page = len(self.calls)
        response = requests.Response()
        response.status_code = 200
        body = {"bars": [_bar(page, 1, 1, 1, 1, 1)], "next_page_token": "p2" if page == 1 else None}
        response._content = json.dumps(body).encode()
        return response


def test_get_bars_asks_for_split_adjusted_bars_and_follows_pages():
    session = _PagedSession()
    client = AlpacaClient(settings=_settings(), session=session)
    bars = client.get_bars("NFLX", timeframe="1Day", start=datetime(2025, 9, 1, tzinfo=timezone.utc))
    assert len(bars) == 2
    assert session.calls[0]["adjustment"] == "split" and session.calls[0]["feed"] == "iex"
    assert "page_token" not in session.calls[0] and session.calls[1]["page_token"] == "p2"
