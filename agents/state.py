"""What the desk remembers between cycles: one JSON file.

The scheduler (agents/scheduler.py) loads this at every step and saves it
after every change, so a restart - or a crash mid-cycle - loses at most the
step in progress:
- `pending`: stocks deferred by a Build failure, with everything the desk
  had already produced, resumed from the failed step on the next pass of
  the same trading session (pipeline.DeskPipeline.resume_symbol);
- `outages`: when each model's current outage began (pipeline.ModelOutages),
  which decides when an analyst may fall back to a backup;
- `positions`: the paper positions the desk opened itself, with the
  horizon each one was bought for (the broker stays the source of truth for
  quantities; positions the desk didn't open are never touched);
- `decisions` and `cycles`: a rolling log of final decisions and of what
  every pass did, including LLM failures per hour.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

STATE_VERSION = 1
# Rolling logs: enough for the multi-day demo without the file growing forever.
MAX_DECISIONS = 500
MAX_CYCLES = 200


@dataclass
class DeskState:
    outages: dict[str, str] = field(default_factory=dict)
    pending: dict[str, dict[str, Any]] = field(default_factory=dict)
    positions: dict[str, dict[str, Any]] = field(default_factory=dict)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    cycles: list[dict[str, Any]] = field(default_factory=list)
    # The trading session whose decision cycle already ran (YYYY-MM-DD).
    last_decision_session: str | None = None
    version: int = STATE_VERSION

    @classmethod
    def load(cls, path: str | Path) -> DeskState:
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != STATE_VERSION:
            raise ValueError(f"{path} has state version {data.get('version')}, expected {STATE_VERSION}")
        return cls(**data)

    def save(self, path: str | Path) -> None:
        """Write atomically (temp file + rename), so a crash mid-write
        leaves the previous state intact rather than a truncated file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.decisions = self.decisions[-MAX_DECISIONS:]
        self.cycles = self.cycles[-MAX_CYCLES:]
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=1, default=str), encoding="utf-8")
        os.replace(tmp, path)
