from pathlib import Path

import pytest

from data_layer.config import ConfigError, Settings
from data_layer.edgar_client import EdgarClient


class _FakeResponse:
    def __init__(self, json_data):
        self._json = json_data
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


class _FakeSession:
    def __init__(self, responses):
        self._responses = responses
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return _FakeResponse(self._responses[url])


def _settings(**overrides):
    base = dict(
        alpaca_api_key_id=None,
        alpaca_api_secret_key=None,
        alpaca_data_base_url="",
        alpaca_trading_base_url="",
        fred_api_key=None,
        finnhub_api_key=None,
        edgar_user_agent="TestApp test@example.com",
        cache_dir=Path("/tmp/x"),
    )
    base.update(overrides)
    return Settings(**base)


TICKERS = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}


def test_get_cik_and_facts(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    session = _FakeSession(
        {
            "https://www.sec.gov/files/company_tickers.json": TICKERS,
            "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json": {"facts": {}},
        }
    )
    client = EdgarClient(settings=_settings(), session=session)
    assert client.get_cik("aapl") == "0000320193"
    facts = client.get_company_facts("aapl")
    assert facts == {"facts": {}}

    for _, _, kwargs in session.calls:
        assert kwargs["headers"]["User-Agent"] == "TestApp test@example.com"


def test_unknown_ticker_raises_keyerror(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    session = _FakeSession({"https://www.sec.gov/files/company_tickers.json": TICKERS})
    client = EdgarClient(settings=_settings(), session=session)
    with pytest.raises(KeyError):
        client.get_cik("NOPE")


def test_missing_user_agent_raises(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    session = _FakeSession({"https://www.sec.gov/files/company_tickers.json": TICKERS})
    client = EdgarClient(settings=_settings(edgar_user_agent=None), session=session)
    with pytest.raises(ConfigError):
        client.get_cik("AAPL")
