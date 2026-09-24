from pathlib import Path

import pytest

from data_layer.config import ConfigError, Settings
from data_layer.fred_client import FredClient


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


def _settings(**overrides):
    base = dict(
        alpaca_api_key_id=None,
        alpaca_api_secret_key=None,
        alpaca_data_base_url="",
        alpaca_trading_base_url="",
        fred_api_key="fred-key",
        finnhub_api_key=None,
        edgar_user_agent=None,
        cache_dir=Path("/tmp/x"),
    )
    base.update(overrides)
    return Settings(**base)


def test_get_series_observations():
    session = _FakeSession({"observations": [{"date": "2026-01-01", "value": "5.0"}]})
    client = FredClient(settings=_settings(), session=session)
    obs = client.get_series_observations("FEDFUNDS")
    assert obs == [{"date": "2026-01-01", "value": "5.0"}]
    _, _, kwargs = session.calls[0]
    assert kwargs["params"]["series_id"] == "FEDFUNDS"
    assert kwargs["params"]["api_key"] == "fred-key"


def test_missing_api_key_raises():
    session = _FakeSession({})
    client = FredClient(settings=_settings(fred_api_key=None), session=session)
    with pytest.raises(ConfigError):
        client.get_series_observations("FEDFUNDS")
