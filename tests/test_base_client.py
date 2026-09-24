import requests

from data_layer.base_client import BaseClient
from data_layer.cache import DiskCache


class _FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        return self._json


class _FakeSession:
    def __init__(self, json_data):
        self._json_data = json_data
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return _FakeResponse(self._json_data)


def test_get_json_without_cache():
    session = _FakeSession({"ok": True})
    client = BaseClient(session=session)
    data = client._get_json("http://x")
    assert data == {"ok": True}
    assert len(session.calls) == 1


def test_get_json_uses_cache(tmp_path):
    session = _FakeSession({"ok": True})
    cache = DiskCache(tmp_path)
    client = BaseClient(session=session, cache=cache)
    client._get_json("http://x", cache_ttl=60)
    client._get_json("http://x", cache_ttl=60)
    assert len(session.calls) == 1


def test_get_json_without_cache_ttl_never_caches(tmp_path):
    session = _FakeSession({"ok": True})
    cache = DiskCache(tmp_path)
    client = BaseClient(session=session, cache=cache)
    client._get_json("http://x")
    client._get_json("http://x")
    assert len(session.calls) == 2
