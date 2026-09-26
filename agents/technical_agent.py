"""Technical expert agent - step 5 of the pipeline (spec section 3).

"Classic, book-based day-trading methods with multiple timeframes: start on
the 4-hour chart for the global view, then drop to smaller charts for entry
timing." It runs last, on an agreed buy that passed the bias gate, and only
times the entry: "enter" (buy at the next open) or "wait" (not today - the
desk re-examines the stock at its next decision cycle). It never revisits
whether the stock is worth buying, and it can't create a buy.

The prompt asks for a specific, chart-based reason to wait: the pick has
already survived two analysts, the critic and the bias agents, and a timing
agent that waited on general uncertainty would be one more brake on a desk
that already rarely buys (Sept 26, 2026).

Like the other gates: an unreachable model defers the stock (retried next
pass), an unusable answer fails the gate (no entry that day).
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from data_layer.llm_client import DEFAULT_MODEL_HEALTH, ModelHealth, role_models

from .analyst_agent import EVIDENCE_STATUS_NOTE, HORIZON, check_evidence
from .llm_json import request_json, unreachable
from .macro_agent import IEX_VOLUME_NOTE, StockContext
from .trace import TraceLogger

AGENT_NAME = "technical"
TECHNICAL_MAX_TOKENS = 8192
TECHNICAL_TEMPERATURE = 0.0
# Bars shown per timeframe (most recent): ~3 weeks of 4H, ~1 week of 1H.
MAX_BARS = {"bars_4h": 30, "bars_1h": 35}

ENTER, WAIT = "enter", "wait"
PASSED, WAITING, FAILED, DEFERRED = "passed", "waiting", "failed", "deferred"

SYSTEM_PROMPT = f"""You are the technical analyst on an equity research desk. The desk has already decided to buy this US stock for a long-only portfolio of real shares over {HORIZON}: two analysts agreed and the bias checks passed. You do not revisit that decision. Your job is the entry, using classic multiple-timeframe chart reading: the 4-hour chart for the prevailing trend and the key levels, then the 1-hour chart for the timing.

Decide:
- "enter": buy at the next market open.
- "wait": do not buy today; the desk looks at the stock again at its next daily decision. Wait only for a specific reason you can point to on the charts (for example, price pressing into a resistance level right after a sharp run, or a breakdown under way on the 1-hour chart), not for general uncertainty.

Rules:
- Use only the bars and source facts given. Name levels as prices taken from the bars.
- {IEX_VOLUME_NOTE}
- {EVIDENCE_STATUS_NOTE}

Reply with ONLY a JSON object, no prose before or after:
{{
  "timing": "enter" or "wait",
  "trend_4h": "up" or "down" or "sideways",
  "support": a price or null,
  "resistance": a price or null,
  "reason": "1-3 sentences",
  "evidence": [{{"fact": "path, e.g. chart.bars_1h[3].c or technicals.rsi_14", "value": "the value at that path, copied exactly", "why": "why it matters"}}]
}}"""


def _compact(bars: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return [{k: b.get(k) for k in ("t", "o", "h", "l", "c", "v")} for b in bars[-limit:]]


def chart(ctx: StockContext) -> dict[str, Any]:
    """The bars the technical agent reads, oldest first, trimmed."""
    return {name: _compact(getattr(ctx, name), limit) for name, limit in MAX_BARS.items()}


@dataclass
class TechnicalResult:
    symbol: str
    model: str
    generated_at: str
    outcome: str = FAILED  # PASSED (enter), WAITING, FAILED or DEFERRED
    timing: str | None = None
    trend_4h: str | None = None
    support: float | None = None
    resistance: float | None = None
    reason: str | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    evidence_check: dict[str, int] = field(default_factory=dict)
    json_repairs: list[str] = field(default_factory=list)
    fallbacks: list[dict[str, Any]] = field(default_factory=list)
    primary_model: str | None = None
    error: str | None = None
    call_failed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def gate(self) -> dict[str, Any]:
        """The entry for Decision.gates."""
        reason = self.reason if self.error is None else self.error
        return {"gate": "technical", "passed": self.outcome == PASSED, "reason": f"{self.timing or self.outcome}: {reason}"}


def _price(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError("'support' and 'resistance' must be prices or null") from None


def validate_timing(obj: dict[str, Any]) -> dict[str, Any]:
    timing = str(obj.get("timing", "")).strip().lower()
    if timing not in (ENTER, WAIT):
        raise ValueError(f"'timing' must be 'enter' or 'wait', got {timing!r}")
    trend = str(obj.get("trend_4h", "")).strip().lower()
    if trend not in ("up", "down", "sideways"):
        raise ValueError("'trend_4h' must be 'up', 'down' or 'sideways'")
    reason = obj.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("'reason' must be a non-empty string")
    evidence = obj.get("evidence") or []
    if not isinstance(evidence, list):
        raise ValueError("'evidence' must be a list")
    return {"timing": timing, "trend_4h": trend, "support": _price(obj.get("support")),
            "resistance": _price(obj.get("resistance")), "reason": reason.strip(),
            "evidence": [dict(e) for e in evidence if isinstance(e, dict) and e.get("fact") and "value" in e]}


class TechnicalAgent:
    def __init__(
        self,
        llm: Any,
        *,
        trace: TraceLogger | None = None,
        model: str | None = None,
        models: Sequence[str] | None = None,
        health: ModelHealth | None = None,
        max_tokens: int = TECHNICAL_MAX_TOKENS,
        temperature: float = TECHNICAL_TEMPERATURE,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self._llm = llm
        self.models = list(models) if models else [model] if model else role_models(AGENT_NAME)
        self.model = self.models[0]
        self._health = health if health is not None else DEFAULT_MODEL_HEALTH
        self._trace = trace
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._now = now

    def _log(self, event: str, payload: dict[str, Any]) -> None:
        if self._trace is not None:
            self._trace.log(AGENT_NAME, event, payload)

    def time_entry(self, ctx: StockContext) -> TechnicalResult:
        result = TechnicalResult(symbol=ctx.symbol, model=self.model, primary_model=self.model,
                                 generated_at=self._now().isoformat())
        bars = chart(ctx)
        if not bars["bars_4h"] and not bars["bars_1h"]:
            result.error = "no 4-hour or 1-hour bars to read"
            self._log("timing_failed", {"symbol": ctx.symbol, "error": result.error})
            return result
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"{ctx.to_prompt()}\n\n=== CHART (IEX bars, oldest first; cite as chart.bars_4h[i] / chart.bars_1h[i]) ===\n"
                f"{json.dumps(bars, ensure_ascii=False, default=str)}")},
        ]
        base = {"symbol": ctx.symbol}
        self._log("llm_request", {**base, "models": self.models, "messages": messages})
        reply = request_json(
            self._llm, self.models, messages,
            validate=validate_timing, log=self._log, base=base,
            max_tokens=self._max_tokens, temperature=self._temperature,
            invalid_event="timing_invalid", repaired_event="timing_repaired", health=self._health,
        )
        result.model = reply.model or result.model
        result.fallbacks = reply.fallbacks
        if not reply.ok:
            result.error = reply.error if reply.call_failed else f"unparseable timing: {reply.error}"
            result.call_failed = unreachable(reply)
            result.outcome = DEFERRED if result.call_failed else FAILED
            self._log("timing_failed", {**base, "model": result.model, "error": result.error,
                                        "call_failed": result.call_failed})
            return result
        for key, value in reply.value.items():
            setattr(result, key, value)
        result.json_repairs = reply.repairs
        result.outcome = PASSED if result.timing == ENTER else WAITING
        if result.evidence:
            result.evidence_check = check_evidence(result.evidence, {**ctx.facts(), "chart": bars})
        self._log("timing", result.to_dict())
        return result
