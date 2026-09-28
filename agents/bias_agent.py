"""Bias/sentiment agents - step 4 of the pipeline (spec section 3).

"Two bias/sentiment agents (redundancy). Only run when the analysts align.
They check for emotional or news-driven skew, trends, and stale news that
may already be priced in." They run on agreed buys only - the one outcome
that opens a position - after the critic loop, and each returns a
BiasCheck: its own score of the news sentiment (Finnhub's sentiment is a
paid feature, spec section 5), three named checks, and pass or flag.

Like the critic, a bias agent never recommends anything: it says whether
the buy case rests on skewed inputs. The gate (bias_gate) vetoes a buy only
when BOTH agents flag it, mirroring the strict two-analyst match - no single
model's weakness decides a veto (spec section 1). A single flag is kept in
the decision's record. An agent that can't be reached defers the stock
(retried like any Build failure); an unusable answer fails the gate,
because a check that couldn't be done is not a pass.

Decided Sept 26, 2026, so a buy never waits all session on one model:
- bias_1 goes first, and its pass clears the gate at once: without its
  flag no veto is possible, so bias_2 is asked only when bias_1 flags (or
  can't be reached - then bias_2's check is kept for the retry).
- Once bias_2 has been unreachable on a stock for BIAS_SOLO_AFTER - every
  attempt for that stock failed since - bias_1's flag vetoes the buy alone.
  In the Sept 26 dry run four agreed buys waited all afternoon on bias_2,
  whose only eligible model (mistral-nemotron) failed 61% of its calls that
  day. Not the other way round: an unreachable bias_1 still defers.

Runtime independence: each bias agent excludes the models the analysts
actually used, and bias_2 also excludes the model bias_1 ran on.
"""
from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from data_layer.llm_client import DEFAULT_MODEL_HEALTH, ModelHealth, role_models

from .analyst_agent import EVIDENCE_STATUS_NOTE, HORIZON, AnalystVerdict, check_evidence
from .llm_json import request_json, unreachable
from .macro_agent import StockContext
from .trace import TraceLogger

BIAS_ROLES = ("bias_1", "bias_2")
# Output tokens per check, by role. Decided Sept 28, 2026: bias_2's model
# (nemotron-3.5-lightning) reasons at length - on LLY it spent all 8,192
# tokens reasoning, twice, never reached its JSON, and the gate failed.
BIAS_MAX_TOKENS = {"bias_1": 8192, "bias_2": 16384}
BIAS_TEMPERATURE = 0.0
CHECKS = ("news_driven", "stale_news", "trend_chasing")

PASSED, VETOED, FAILED, DEFERRED = "passed", "vetoed", "failed", "deferred"

# The agent whose pass clears the gate and whose flag vetoes alone once
# the other has been unreachable on a stock this long (module docstring).
LEAD_ROLE = "bias_1"
BIAS_SOLO_AFTER = timedelta(hours=2)

SYSTEM_PROMPT = f"""You are a bias and sentiment checker on an equity research desk. Two analysts, on different models, independently read the same context packet for one US stock and both recommend buying it for a long-only portfolio of real shares over {HORIZON}; a critic has already challenged their reasoning. Your job is narrower: check whether this buy rests on skewed inputs rather than on the evidence. You do not decide whether to buy and you give no recommendation.

Check:
- news_sentiment: score the tone of the company news in the packet yourself, from -1 (very negative) to +1 (very positive). The desk has no other sentiment source.
- news_driven: does the buy case lean on headlines (unverified third-party reporting) more than on the source facts?
- stale_news: does the buy case treat news older than 72 hours as new, when the price may already reflect it (compare the news dates with the price moves)?
- trend_chasing: does the buy case extrapolate a recent run-up or excitement that the other source facts do not support?

Rules:
- The "Source facts" block is authoritative. Your background knowledge may be outdated: never cite figures or events from memory. When you rely on a fact, give its path and the value at that path, copied exactly.
- {EVIDENCE_STATUS_NOTE}
- Flag only a skew that matters to the buy case. A case with sound support in the source facts passes even when some of its news is positive. Missing data the desk never provides (listed under limitations) is not a skew.

Reply with ONLY a JSON object, no prose before or after:
{{
  "news_sentiment": number from -1 to 1,
  "checks": {{
    "news_driven": {{"skewed": true or false, "why": "one or two sentences"}},
    "stale_news": {{"skewed": true or false, "why": "one or two sentences", "items": [news indices]}},
    "trend_chasing": {{"skewed": true or false, "why": "one or two sentences"}}
  }},
  "verdict": "pass" or "flag",
  "reason": "1-3 sentences",
  "evidence": [{{"fact": "path into the source facts", "value": "the value at that path, copied exactly", "why": "why it matters"}}]
}}"""


@dataclass
class BiasCheck:
    symbol: str
    role: str
    model: str
    generated_at: str
    verdict: str | None = None  # "pass" or "flag"
    news_sentiment: float | None = None
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)
    reason: str | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    evidence_check: dict[str, int] = field(default_factory=dict)
    json_repairs: list[str] = field(default_factory=list)
    fallbacks: list[dict[str, Any]] = field(default_factory=list)
    primary_model: str | None = None
    error: str | None = None
    call_failed: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def flagged(self) -> bool:
        return self.verdict == "flag"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_bias_check(obj: dict[str, Any]) -> dict[str, Any]:
    """Normalized fields, or ValueError naming the first problem."""
    verdict = str(obj.get("verdict", "")).strip().lower()
    if verdict not in ("pass", "flag"):
        raise ValueError(f"'verdict' must be 'pass' or 'flag', got {verdict!r}")
    try:
        sentiment = float(obj.get("news_sentiment"))
    except (TypeError, ValueError):
        raise ValueError("'news_sentiment' must be a number from -1 to 1") from None
    if not -1 <= sentiment <= 1:
        raise ValueError("'news_sentiment' must be a number from -1 to 1")
    checks = obj.get("checks")
    if not isinstance(checks, dict):
        raise ValueError(f"'checks' must be an object with {', '.join(CHECKS)}")
    normalized = {}
    for name in CHECKS:
        check = checks.get(name)
        if not isinstance(check, dict) or not isinstance(check.get("skewed"), bool):
            raise ValueError(f"'checks.{name}' needs a true/false 'skewed' and a 'why'")
        normalized[name] = {"skewed": check["skewed"], "why": str(check.get("why") or "").strip()}
        if name == "stale_news" and isinstance(check.get("items"), list):
            normalized[name]["items"] = [i for i in check["items"] if isinstance(i, int)]
    reason = obj.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("'reason' must be a non-empty string")
    evidence = obj.get("evidence") or []
    if not isinstance(evidence, list):
        raise ValueError("'evidence' must be a list")
    kept = [dict(e) for e in evidence if isinstance(e, dict) and e.get("fact") and "value" in e]
    return {"verdict": verdict, "news_sentiment": round(sentiment, 3), "checks": normalized,
            "reason": reason.strip(), "evidence": kept}


class BiasAgent:
    def __init__(
        self,
        llm: Any,
        role: str = "bias_1",
        *,
        trace: TraceLogger | None = None,
        model: str | None = None,
        models: Sequence[str] | None = None,
        health: ModelHealth | None = None,
        max_tokens: int | None = None,
        temperature: float = BIAS_TEMPERATURE,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self._llm = llm
        self.role = role
        self.models = list(models) if models else [model] if model else role_models(role)
        self.model = self.models[0]
        self._health = health if health is not None else DEFAULT_MODEL_HEALTH
        self._trace = trace
        self._max_tokens = max_tokens or BIAS_MAX_TOKENS.get(role, BIAS_MAX_TOKENS["bias_1"])
        self._temperature = temperature
        self._now = now

    def _log(self, event: str, payload: dict[str, Any]) -> None:
        if self._trace is not None:
            self._trace.log(self.role, event, payload)

    def review(
        self, ctx: StockContext, verdicts: dict[str, AnalystVerdict], *, exclude: Collection[str] = ()
    ) -> BiasCheck:
        check = BiasCheck(symbol=ctx.symbol, role=self.role, model=self.model, primary_model=self.model,
                          generated_at=self._now().isoformat())
        briefs = {role: v.brief() for role, v in sorted(verdicts.items())}
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"{ctx.to_prompt()}\n\n=== THE AGREED BUY CASE (both analysts' final verdicts) ===\n"
                f"{json.dumps(briefs, indent=1, ensure_ascii=False, default=str)}")},
        ]
        excluded = sorted({v.model for v in verdicts.values()} | set(exclude))
        base = {"symbol": ctx.symbol}
        self._log("llm_request", {**base, "models": self.models, "exclude": excluded, "messages": messages})
        reply = request_json(
            self._llm, self.models, messages,
            validate=validate_bias_check, log=self._log, base=base,
            max_tokens=self._max_tokens, temperature=self._temperature,
            invalid_event="bias_check_invalid", repaired_event="bias_check_repaired",
            health=self._health, exclude=excluded,
        )
        check.model = reply.model or check.model
        check.fallbacks = reply.fallbacks
        if not reply.ok:
            check.error = reply.error if reply.call_failed else f"unparseable bias check: {reply.error}"
            check.call_failed = unreachable(reply)
            self._log("bias_check_failed", {**base, "model": check.model, "error": check.error,
                                            "call_failed": check.call_failed})
            return check
        for key, value in reply.value.items():
            setattr(check, key, value)
        check.json_repairs = reply.repairs
        if check.evidence:
            check.evidence_check = check_evidence(check.evidence, ctx.facts())
        self._log("bias_check", check.to_dict())
        return check


@dataclass
class BiasGateResult:
    symbol: str
    outcome: str  # PASSED, VETOED, FAILED or DEFERRED
    reason: str
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Role -> when this stock's first attempt to reach it failed (UTC, ISO),
    # kept while every retry fails: what BIAS_SOLO_AFTER is measured from.
    unreachable_since: dict[str, str] = field(default_factory=dict)
    # LEAD_ROLE's flag vetoed alone (the other agent unreachable for BIAS_SOLO_AFTER).
    solo: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def gate(self) -> dict[str, Any]:
        """The entry for Decision.gates."""
        return {"gate": "bias", "passed": self.outcome == PASSED, "reason": self.reason}


def _duration(delta: timedelta) -> str:
    minutes = int(delta.total_seconds() // 60)
    return f"{minutes // 60}h {minutes % 60:02d}m"


def bias_gate(
    ctx: StockContext,
    verdicts: dict[str, AnalystVerdict],
    agents: dict[str, BiasAgent],
    *,
    previous: BiasGateResult | None = None,
    trace: TraceLogger | None = None,
    solo_after: timedelta | None = BIAS_SOLO_AFTER,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> BiasGateResult:
    """Run the bias agents in role order (each after the previous, so it
    can exclude the model that one used), stopping at a LEAD_ROLE pass.
    With `previous` - a deferred result - checks that already came back
    are kept and only the others are asked again. solo_after=None never
    lets LEAD_ROLE's flag veto alone."""
    kept = {role: BiasCheck(**d) for role, d in (previous.checks if previous else {}).items()
            if d.get("error") is None}
    checks: dict[str, BiasCheck] = {}
    for role in sorted(agents):
        lead = checks.get(LEAD_ROLE)
        if role in kept:
            checks[role] = kept[role]
        elif lead is None or not lead.ok or lead.flagged:
            used = {c.model for c in (*kept.values(), *checks.values()) if c.ok}
            checks[role] = agents[role].review(ctx, verdicts, exclude=used)

    records = {role: c.to_dict() for role, c in checks.items()}
    unusable = [r for r, c in checks.items() if not c.ok and not c.call_failed]
    unreached = [r for r, c in checks.items() if not c.ok and c.call_failed]
    flags = [r for r, c in checks.items() if c.ok and c.flagged]
    at = now()
    earlier = previous.unreachable_since if previous else {}
    since = {r: earlier.get(r) or at.isoformat() for r in unreached}
    lead = checks.get(LEAD_ROLE)
    cleared = lead is not None and lead.ok and not lead.flagged
    # LEAD_ROLE's flag stands alone only when every other agent is
    # unreachable, counted from the latest of their first failures.
    alone = (solo_after is not None and lead is not None and lead.ok and lead.flagged
             and bool(unreached) and len(unreached) == len(checks) - 1)
    start = max((datetime.fromisoformat(s) for s in since.values()), default=at)
    solo = alone and at - start >= solo_after
    absent = ", ".join(unreached)
    if unusable:
        outcome, reason = FAILED, f"bias check unusable from {', '.join(unusable)}; a check that couldn't be done is not a pass"
    elif cleared:
        outcome = PASSED
        reason = f"passed: {LEAD_ROLE} passes it, which clears the gate (a veto needs both bias agents)"
        if flags:  # a check kept from an attempt where LEAD_ROLE couldn't be reached
            reason += f"; only {flags[0]} flags it ({checks[flags[0]].reason})"
    elif solo:
        outcome = VETOED
        reason = (f"{absent} could not be reached for {_duration(at - start)} (since {start:%H:%M} UTC), "
                  f"so {LEAD_ROLE} decides alone and flags the buy: {lead.reason}")
    elif unreached:
        outcome, reason = DEFERRED, f"could not reach {absent}: a Build failure, retried next pass"
        if alone:
            reason += (f"; {LEAD_ROLE}'s flag vetoes the buy alone if {absent} is still unreachable "
                       f"at {start + solo_after:%H:%M} UTC")
    elif len(flags) == len(checks):
        outcome = VETOED
        reason = "both bias agents flag the buy: " + " | ".join(f"{r}: {checks[r].reason}" for r in flags)
    elif flags:
        outcome = PASSED
        reason = (f"passed: only {flags[0]} flags it ({checks[flags[0]].reason}); "
                  "a veto needs both bias agents")
    else:
        outcome, reason = PASSED, "passed: neither bias agent flags the buy"
    result = BiasGateResult(symbol=ctx.symbol, outcome=outcome, reason=reason, checks=records,
                            unreachable_since=since, solo=solo)
    if trace is not None:
        trace.log("bias_gate", "decision", {"symbol": ctx.symbol, "outcome": outcome, "reason": reason,
                                             "sentiment": {r: c.news_sentiment for r, c in checks.items()},
                                             "flags": flags, "unreachable_since": since, "solo": solo})
    return result
