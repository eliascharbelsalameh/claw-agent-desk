"""Timestamped JSONL trail of every agent's input and output.

Spec section 3: "Every agent's input and output is logged with a
timestamp, so the demo video can show the back-and-forth and not just a
final buy or hold." One JSON object per line, append-only, so a multi-day
run can be tailed live and replayed afterwards without a database.
"""
from __future__ import annotations

import json
import threading
from collections import Counter
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


def read_trace(path: str | Path) -> list[dict[str, Any]]:
    events = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a line cut short by an interrupted run
    return events


def summarize_trace(events: list[dict[str, Any]], since: str | None = None) -> dict[str, Any]:
    """LLM calls, tokens, failures and fallbacks, optionally only events at
    or after the ISO timestamp `since` (a trace file holds a whole day)."""
    if since is not None:
        events = [e for e in events if e.get("ts", "") >= since]
    responses = [e for e in events if e.get("event") == "llm_response"]
    usage = [e.get("usage") or {} for e in responses]
    return {
        "llm_calls": len(responses),
        "llm_errors": sum(e.get("event") == "llm_error" for e in events),
        "fallbacks": sum(e.get("event") == "fallback" for e in events),
        "prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usage),
        "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usage),
        "total_tokens": sum(u.get("total_tokens") or 0 for u in usage),
        "calls_by_model": dict(Counter(e.get("model", "?") for e in responses)),
    }


def failures_by_hour(events: list[dict[str, Any]]) -> dict[int, dict[str, int]]:
    """LLM calls that answered vs failed, per UTC hour of the day. Build's
    reliability varies by the hour (Sept 25, 2026: 14 of 17 failures came
    between 21:00 and 24:00 UTC), so the scheduler logs this every cycle."""
    hours: dict[int, dict[str, int]] = {}
    for e in events:
        kind = e.get("event")
        if kind not in ("llm_response", "llm_error") or len(e.get("ts", "")) < 13:
            continue
        bucket = hours.setdefault(int(e["ts"][11:13]), {"ok": 0, "failed": 0})
        bucket["ok" if kind == "llm_response" else "failed"] += 1
    return dict(sorted(hours.items()))
