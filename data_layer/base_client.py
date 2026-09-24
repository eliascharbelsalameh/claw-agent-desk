"""Shared request/cache plumbing used by every data-source client."""
from __future__ import annotations

from typing import Any

import requests

from .cache import DiskCache, cache_key
from .http_utils import request_with_retry


class BaseClient:
    def __init__(
        self,
        session: requests.Session | None = None,
        cache: DiskCache | None = None,
    ):
        self._session = session or requests.Session()
        self._cache = cache

    def _get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        cache_ttl: float | None = None,
        **kwargs: Any,
    ) -> Any:
        key = cache_key(url, params) if cache_ttl else None
        if key is not None and self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                return cached

        response = request_with_retry(
            self._session, "GET", url, params=params, headers=headers, **kwargs
        )
        response.raise_for_status()
        data = response.json()

        if key is not None and self._cache is not None:
            self._cache.set(key, data, cache_ttl)
        return data
