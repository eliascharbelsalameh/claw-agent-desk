import json
from pathlib import Path

import pytest
import requests

from data_layer.config import ConfigError, Settings
from data_layer.llm_client import LlmClient


def _sse(chunks, fail_after=None):
    lines = []
    for i, chunk in enumerate(chunks):
        if fail_after is not None and i == fail_after:
            lines.append(requests.exceptions.ChunkedEncodingError("Response ended prematurely"))
        lines.append(f"data: {json.dumps(chunk)}".encode())
        lines.append(b"")
    lines.append(b"data: [DONE]")
    return lines


def _stream_chunks(text="hello", reasoning=None):
    chunks = [{"id": "c1", "model": "some/model", "choices": [{"delta": {"role": "assistant"}}]}]
    if reasoning:
        chunks.append({"choices": [{"delta": {"reasoning_content": reasoning}}]})
    for piece in (text[: len(text) // 2], text[len(text) // 2:]):
        chunks.append({"choices": [{"delta": {"content": piece}}]})
    chunks.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    chunks.append({"choices": [], "usage": {"total_tokens": 7}})
    return chunks


class _FakeResponse:
    def __init__(self, json_data=None, lines=None):
        self._json = json_data
        self._lines = lines
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._json

    def iter_lines(self):
        for line in self._lines:
            if isinstance(line, Exception):
                raise line
            yield line


class _FakeSession:
    """Non-streaming requests get json_data; streaming ones pop the next
    SSE line list from `streams` (json_data's text if none given)."""

    def __init__(self, json_data, streams=None):
        self._json_data = json_data
        self._streams = list(streams or [])
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if kwargs.get("stream"):
            lines = self._streams.pop(0) if self._streams else _sse(
                _stream_chunks(self._json_data["choices"][0]["message"]["content"])
            )
            return _FakeResponse(lines=lines)
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

    data = client.chat_completion("some/model", messages, stream=False)

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


def test_streams_by_default_and_reassembles_response():
    session = _FakeSession(
        _completion_payload(), streams=[_sse(_stream_chunks("the answer", reasoning="hmm"))]
    )
    client = LlmClient(settings=_settings(), session=session)

    data = client.chat_completion("some/model", [{"role": "user", "content": "hi"}])

    _, _, kwargs = session.calls[0]
    assert kwargs["stream"] is True
    assert kwargs["json"]["stream"] is True
    assert kwargs["json"]["stream_options"] == {"include_usage": True}
    choice = data["choices"][0]
    assert choice["message"]["content"] == "the answer"
    assert choice["message"]["reasoning_content"] == "hmm"
    assert choice["finish_reason"] == "stop"
    assert data["usage"] == {"total_tokens": 7}
    assert data["id"] == "c1"


def test_non_streaming_omits_stream_fields():
    session = _FakeSession(_completion_payload())
    client = LlmClient(settings=_settings(), session=session)

    client.chat_completion("some/model", [{"role": "user", "content": "hi"}], stream=False)

    _, _, kwargs = session.calls[0]
    assert "stream" not in kwargs["json"] and kwargs["stream"] is False


def test_dropped_stream_is_retried(monkeypatch):
    monkeypatch.setattr("data_layer.llm_client.time.sleep", lambda s: None)
    session = _FakeSession(
        _completion_payload(),
        streams=[_sse(_stream_chunks("partial"), fail_after=2), _sse(_stream_chunks("full"))],
    )
    client = LlmClient(settings=_settings(), session=session)

    text = client.complete_text("some/model", [{"role": "user", "content": "hi"}])

    assert text == "full"
    assert len(session.calls) == 2


def test_dropped_stream_gives_up_after_retry_budget(monkeypatch):
    monkeypatch.setattr("data_layer.llm_client.time.sleep", lambda s: None)
    broken = [_sse(_stream_chunks("x"), fail_after=1) for _ in range(3)]
    session = _FakeSession(_completion_payload(), streams=broken)
    client = LlmClient(settings=_settings(), session=session)

    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        client.chat_completion("some/model", [{"role": "user", "content": "hi"}])
    assert len(session.calls) == 3  # first try + STREAM_RETRIES


def test_stream_without_finish_reason_is_retried_as_a_drop(monkeypatch):
    monkeypatch.setattr("data_layer.llm_client.time.sleep", lambda s: None)
    truncated = _sse([{"id": "c1", "choices": [{"delta": {"role": "assistant"}}]}])  # then [DONE]
    session = _FakeSession(_completion_payload(), streams=[truncated, _sse(_stream_chunks("ok"))])
    client = LlmClient(settings=_settings(), session=session)

    assert client.complete_text("some/model", [{"role": "user", "content": "hi"}]) == "ok"
    assert len(session.calls) == 2
