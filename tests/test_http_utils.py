import json
import time

import pytest
import requests

from data_layer.http_utils import RateLimitExceeded, redact, request_with_retry


def test_redact_hides_keys_in_urls_and_leaves_the_rest():
    text = ("HTTPSConnectionPool(host='api.stlouisfed.org', port=443): Max retries exceeded with url: "
            "/fred/series/observations?series_id=DGS10&api_key=abc123&file_type=json (Caused by x) "
            "| 403 for url: https://finnhub.io/api/v1/company-news?symbol=AAPL&token=tok456")
    out = redact(text)
    assert "abc123" not in out and "tok456" not in out
    assert "series_id=DGS10&api_key=***&file_type=json (Caused by x)" in out
    assert out.endswith("symbol=AAPL&token=***")
    # inside a JSON line the escaped quote after the value survives
    assert json.loads(redact(json.dumps({"e": 'url "x?token=tok456"'}))) == {"e": 'url "x?token=***"'}


class _FakeResponse:
    def __init__(self, status_code, headers=None):
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def request(self, method, url, **kwargs):
        self.calls += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_succeeds_first_try():
    session = _FakeSession([_FakeResponse(200)])
    resp = request_with_retry(session, "GET", "http://x", max_retries=3, backoff_base=0.01)
    assert resp.status_code == 200
    assert session.calls == 1


def test_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    session = _FakeSession([_FakeResponse(429), _FakeResponse(429), _FakeResponse(200)])
    resp = request_with_retry(session, "GET", "http://x", max_retries=3, backoff_base=0.01)
    assert resp.status_code == 200
    assert session.calls == 3


def test_honors_retry_after(monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    session = _FakeSession(
        [_FakeResponse(429, headers={"Retry-After": "2"}), _FakeResponse(200)]
    )
    request_with_retry(session, "GET", "http://x", max_retries=3, backoff_base=0.01)
    assert slept[0] >= 2


def test_raises_after_max_retries(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    session = _FakeSession([_FakeResponse(429), _FakeResponse(429), _FakeResponse(429)])
    with pytest.raises(RateLimitExceeded):
        request_with_retry(session, "GET", "http://x", max_retries=2, backoff_base=0.01)


def test_non_retryable_status_returned_immediately():
    session = _FakeSession([_FakeResponse(404)])
    resp = request_with_retry(session, "GET", "http://x", max_retries=3, backoff_base=0.01)
    assert resp.status_code == 404
    assert session.calls == 1


def test_retries_dropped_connection(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    session = _FakeSession([requests.ConnectionError("remote closed"), _FakeResponse(200)])
    resp = request_with_retry(session, "GET", "http://x", max_retries=3, backoff_base=0.01)
    assert resp.status_code == 200
    assert session.calls == 2


def test_dropped_connection_reraises_after_budget(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    session = _FakeSession([requests.ConnectionError("remote closed")] * 3)
    with pytest.raises(requests.ConnectionError):
        request_with_retry(session, "GET", "http://x", max_retries=2, backoff_base=0.01)
    assert session.calls == 3


def test_timeout_is_not_retried(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    session = _FakeSession([requests.ReadTimeout("slow"), _FakeResponse(200)])
    with pytest.raises(requests.ReadTimeout):
        request_with_retry(session, "GET", "http://x", max_retries=3, backoff_base=0.01)
    assert session.calls == 1
