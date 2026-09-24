"""Small dependency-free disk cache with per-entry TTL.

Slow-changing data (EDGAR filings, FRED series, the ticker->CIK map) should
be cached aggressively (spec section 5: "Cache everything that changes
slowly"); fast-changing data (news, intraday bars) should use a short TTL
or none at all. Callers decide the TTL per call.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


def cache_key(*parts: Any) -> str:
    return "|".join(str(p) for p in parts)


class DiskCache:
    def __init__(self, cache_dir: Path | str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if payload.get("expires_at", 0) < time.time():
            path.unlink(missing_ok=True)
            return None
        return payload["value"]

    def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        path = self._path(key)
        payload = {"expires_at": time.time() + ttl_seconds, "value": value}
        path.write_text(json.dumps(payload), encoding="utf-8")
