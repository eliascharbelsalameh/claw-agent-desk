from pathlib import Path

import pytest

from data_layer.config import ConfigError, Settings
from data_layer.llm_client import LlmClient


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


def _settings(nvidia_api_key="nvidia-key"):
    return Settings(
        alpaca_api_key_id=None,
        alpaca_api_secret_key=None,
        alpaca_data_base_url="",
        alpaca_trading_base_url="",
        fred_api_key=None,
        finnhub_api_key=None,
        edgar_user_agent=None,
        nvidia_api_key=nvidia_api_key,
        nvidia_base_url="https://integrate.api.nvidia.com/v1",
        cache_dir=Path("/tmp/x"),
    )


def _completion_payload(text="hello"):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def test_chat_completion_posts_model_and_messages():
    session = _FakeSession(_completion_payload())
    client = LlmClient(settings=_settings(), session=session)
    messages = [{"role": "user", "content": "hi"}]

    data = client.chat_completion("some/model", messages)

    assert data == _completion_payload()
    method, url, kwargs = session.calls[0]
    assert method == "POST"
    assert url == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert kwargs["json"]["model"] == "some/model"
    assert kwargs["json"]["messages"] == messages
    assert kwargs["headers"]["Authorization"] == "Bearer nvidia-key"


def test_chat_completion_requires_api_key():
    session = _FakeSession(_completion_payload())
    client = LlmClient(settings=_settings(nvidia_api_key=None), session=session)

    with pytest.raises(ConfigError):
        client.chat_completion("some/model", [{"role": "user", "content": "hi"}])


def test_complete_text_extracts_message_content():
    session = _FakeSession(_completion_payload("the answer"))
    client = LlmClient(settings=_settings(), session=session)

    text = client.complete_text("some/model", [{"role": "user", "content": "hi"}])

    assert text == "the answer"


def test_rate_limit_paces_calls_per_model(monkeypatch):
    session = _FakeSession(_completion_payload())
    client = LlmClient(settings=_settings(), session=session, rpm_limits={"m": 2})

    fake_now = [0.0]
    sleeps: list[float] = []

    def fake_monotonic():
        return fake_now[0]

    def fake_sleep(seconds):
        sleeps.append(seconds)
        fake_now[0] += seconds

    monkeypatch.setattr("data_layer.llm_client.time.monotonic", fake_monotonic)
    monkeypatch.setattr("data_layer.llm_client.time.sleep", fake_sleep)

    client.chat_completion("m", [{"role": "user", "content": "1"}])
    client.chat_completion("m", [{"role": "user", "content": "2"}])
    assert sleeps == []

    client.chat_completion("m", [{"role": "user", "content": "3"}])
    assert sleeps == [60.0]


def test_rate_limit_is_independent_per_model(monkeypatch):
    session = _FakeSession(_completion_payload())
    client = LlmClient(settings=_settings(), session=session, rpm_limits={"m1": 1, "m2": 1})

    monkeypatch.setattr("data_layer.llm_client.time.monotonic", lambda: 0.0)
    monkeypatch.setattr(
        "data_layer.llm_client.time.sleep", lambda s: pytest.fail("should not sleep")
    )

    client.chat_completion("m1", [{"role": "user", "content": "1"}])
    client.chat_completion("m2", [{"role": "user", "content": "1"}])
