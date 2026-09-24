import time

import pytest
import requests

from data_layer.http_utils import RateLimitExceeded, request_with_retry


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
        return self._responses.pop(0)


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
