from datetime import datetime, timezone
from pathlib import Path

import pytest
import requests

from data_layer.alpaca_client import AlpacaClient, aggregate_bars
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


def test_get_relative_volume(monkeypatch):
    client = AlpacaClient(settings=_settings())
    bars = [_bar(0, 1, 1, 1, 1, 100, day=d) for d in range(1, 21)] + [
        _bar(0, 1, 1, 1, 1, 400, day=21)
    ]

    monkeypatch.setattr(client, "get_bars", lambda *a, **k: bars)
    result = client.get_relative_volume("AAPL", lookback_days=20)
    assert result["relative_volume"] == pytest.approx(4.0)
    assert result["baseline_avg_volume"] == pytest.approx(100.0)
    assert result["feed"] == "iex"
