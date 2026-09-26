import time

from data_layer.cache import DiskCache, cache_key


def test_roundtrip(tmp_path):
    cache = DiskCache(tmp_path)
    cache.set("k", {"a": 1}, ttl_seconds=60)
    assert cache.get("k") == {"a": 1}


def test_expiry(tmp_path):
    cache = DiskCache(tmp_path)
    cache.set("k", {"a": 1}, ttl_seconds=0.01)
    time.sleep(0.05)
    assert cache.get("k") is None


def test_missing_key(tmp_path):
    cache = DiskCache(tmp_path)
    assert cache.get("nope") is None


def test_cache_key_is_stable():
    assert cache_key("a", 1, None) == cache_key("a", 1, None)
    assert cache_key("a", 1) != cache_key("a", 2)


def test_a_failed_write_is_swallowed_and_leaves_no_temp_file(tmp_path, monkeypatch):
    import data_layer.cache as cache_module

    cache = DiskCache(tmp_path)

    def locked(*args, **kwargs):
        raise PermissionError("the file is being used by another process")

    monkeypatch.setattr(cache_module.os, "replace", locked)
    cache.set("k", {"v": 1}, ttl_seconds=60)  # must not raise
    assert cache.get("k") is None
    assert list(tmp_path.glob("*.tmp")) == []
