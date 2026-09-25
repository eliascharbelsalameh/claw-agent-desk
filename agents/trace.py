"""Timestamped JSONL trail of every agent's input and output.

Spec section 3: "Every agent's input and output is logged with a
timestamp, so the demo video can show the back-and-forth and not just a
final buy or hold." One JSON object per line, append-only, so a multi-day
run can be tailed live and replayed afterwards without a database.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TraceLogger:
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def log(self, agent: str, event: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "agent": agent,
            "event": event,
            **payload,
        }
        line = json.dumps(record, default=str, ensure_ascii=False)
        with self._lock, self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return record
