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
model's weakness decides (spec section 1), neither a pass nor a veto. A
single flag is kept in the decision's record. An agent that can't be
reached defers the stock (retried like any Build failure); an unusable
answer fails the gate, because a check that couldn't be done is not a pass.

Runtime independence: each bias agent excludes the models the analysts
actually used, and bias_2 also excludes the model bias_1 ran on.
"""
from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from data_layer.llm_client import DEFAULT_MODEL_HEALTH, ModelHealth, role_models

from .analyst_agent import EVIDENCE_STATUS_NOTE, HORIZON, AnalystVerdict, check_evidence
from .llm_json import request_json, unreachable
from .macro_agent import StockContext
from .trace import TraceLogger

BIAS_ROLES = ("bias_1", "bias_2")
BIAS_MAX_TOKENS = 8192
BIAS_TEMPERATURE = 0.0
CHECKS = ("news_driven", "stale_news", "trend_chasing")

PASSED, VETOED, FAILED, DEFERRED = "passed", "vetoed", "failed", "deferred"

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
        max_tokens: int = BIAS_MAX_TOKENS,
        temperature: float = BIAS_TEMPERATURE,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self._llm = llm
        self.role = role
        self.models = list(models) if models else [model] if model else role_models(role)
        self.model = self.models[0]
        self._health = health if health is not None else DEFAULT_MODEL_HEALTH
        self._trace = trace
        self._max_tokens = max_tokens
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def gate(self) -> dict[str, Any]:
        """The entry for Decision.gates."""
        return {"gate": "bias", "passed": self.outcome == PASSED, "reason": self.reason}


def bias_gate(
    ctx: StockContext,
    verdicts: dict[str, AnalystVerdict],
    agents: dict[str, BiasAgent],
    *,
    previous: BiasGateResult | None = None,
    trace: TraceLogger | None = None,
) -> BiasGateResult:
    """Run the bias agents in role order (each after the previous, so it
    can exclude the model that one used). With `previous` - a deferred
    result - checks that already came back are kept and only the others
    are asked again."""
    kept = {role: BiasCheck(**d) for role, d in (previous.checks if previous else {}).items()
            if d.get("error") is None}
    checks: dict[str, BiasCheck] = {}
    for role in sorted(agents):
        if role in kept:
            checks[role] = kept[role]
            continue
        used = {c.model for c in (*kept.values(), *checks.values()) if c.ok}
        checks[role] = agents[role].review(ctx, verdicts, exclude=used)

    records = {role: c.to_dict() for role, c in checks.items()}
    unusable = [r for r, c in checks.items() if not c.ok and not c.call_failed]
    unreached = [r for r, c in checks.items() if not c.ok and c.call_failed]
    flags = [r for r, c in checks.items() if c.ok and c.flagged]
    if unusable:
        outcome, reason = FAILED, f"bias check unusable from {', '.join(unusable)}; a check that couldn't be done is not a pass"
    elif unreached:
        outcome, reason = DEFERRED, f"could not reach {', '.join(unreached)}: a Build failure, retried next pass"
    elif len(flags) == len(checks):
        outcome = VETOED
        reason = "both bias agents flag the buy: " + " | ".join(f"{r}: {checks[r].reason}" for r in flags)
    elif flags:
        outcome = PASSED
        reason = (f"passed: only {flags[0]} flags it ({checks[flags[0]].reason}); "
                  "a veto needs both bias agents")
    else:
        outcome, reason = PASSED, "passed: neither bias agent flags the buy"
    result = BiasGateResult(symbol=ctx.symbol, outcome=outcome, reason=reason, checks=records)
    if trace is not None:
        trace.log("bias_gate", "decision", {"symbol": ctx.symbol, "outcome": outcome, "reason": reason,
                                             "sentiment": {r: c.news_sentiment for r, c in checks.items()},
                                             "flags": flags})
    return result
