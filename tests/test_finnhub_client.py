from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from data_layer.config import Settings
from data_layer.finnhub_client import FinnhubClient


class _FakeResponse:
    def __init__(self, json_data):
        self._json = json_data
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


class _FakeSession:
    def __init__(self, json_data):
        self._json_data = json_data
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return _FakeResponse(self._json_data)


def _settings():
    return Settings(
        alpaca_api_key_id=None,
        alpaca_api_secret_key=None,
        alpaca_data_base_url="",
        alpaca_trading_base_url="",
        fred_api_key=None,
        finnhub_api_key="finnhub-key",
        edgar_user_agent=None,
        cache_dir=Path("/tmp/x"),
    )


def test_get_company_news_adds_age():
    now_ts = int(datetime.now(timezone.utc).timestamp()) - 3600
    session = _FakeSession([{"headline": "x", "datetime": now_ts, "source": "s"}])
    client = FinnhubClient(settings=_settings(), session=session)
    news = client.get_company_news("AAPL", date(2026, 9, 1), date(2026, 9, 24))
    assert news[0]["age_hours"] == pytest.approx(1.0, abs=0.05)
    assert "published_at" in news[0]
    _, _, kwargs = session.calls[0]
    assert kwargs["params"]["token"] == "finnhub-key"
