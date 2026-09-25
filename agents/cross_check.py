"""Analyst cross-check - the gate after step 2 of the pipeline (spec section 3).

Two analysts on different models judge the same context independently;
this decides what happens to the stock from their two verdicts. It is a
deterministic rule, not an LLM call, so the same pair of verdicts always
gets the same outcome and the demo can show exactly why.

Rules (decided Sept 25, 2026 - strict matching, no confidence gate):
- identical recommendations           -> "agree" (carries that recommendation)
- buy vs hold                          -> "critic" (re-votes in the critic loop;
                                          aborts there if still unmatched)
- buy vs avoid                         -> "abort" (a real contradiction)
- hold vs avoid                        -> "abort" (not covered by the decisions;
                                          strict matching applies. Either way
                                          no new position results)
- either verdict failed                -> "abort" (a failed analyst never counts
                                          as agreement)
- both analysts on the same model      -> "abort" (the check would not be
                                          independent - spec section 4)

Confidence is reported for the trace but never used: the two models report
it on different scales (consistency test, Sept 25, 2026).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .analyst_agent import AnalystVerdict
from .trace import TraceLogger

AGENT_NAME = "cross_check"

AGREE = "agree"
CRITIC = "critic"
ABORT = "abort"


@dataclass
class CrossCheckResult:
    symbol: str
    outcome: str
    recommendation: str | None
    reason: str
    verdicts: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _summary(v: AnalystVerdict) -> dict[str, Any]:
    return {
        "model": v.model,
        "ok": v.ok,
        "error": v.error,
        "recommendation": v.recommendation,
        "confidence": v.confidence,
        "evidence_check": v.evidence_check,
        "thesis": v.thesis,
    }


def _decide(a: AnalystVerdict, b: AnalystVerdict) -> tuple[str, str | None, str]:
    failed = [v.role for v in (a, b) if not v.ok]
    if failed:
        return ABORT, None, f"verdict failed for {', '.join(failed)}; a failed analyst never counts as agreement"
    if a.model == b.model:
        return ABORT, None, f"both analysts ran on {a.model}; the cross-check would not be independent"
    if a.recommendation == b.recommendation:
        return AGREE, a.recommendation, f"both analysts recommend {a.recommendation}"
    pair = {a.recommendation, b.recommendation}
    split = f"{a.role} says {a.recommendation}, {b.role} says {b.recommendation}"
    if pair == {"buy", "hold"}:
        return CRITIC, None, f"{split}: buy vs hold goes to the critic loop for re-votes"
    return ABORT, None, f"{split}: contradictory recommendations"


def cross_check(
    a: AnalystVerdict, b: AnalystVerdict, trace: TraceLogger | None = None
) -> CrossCheckResult:
    if a.symbol != b.symbol:
        raise ValueError(f"verdicts are for different symbols: {a.symbol} vs {b.symbol}")
    if a.role == b.role:
        raise ValueError(f"both verdicts come from the same role: {a.role}")
    outcome, recommendation, reason = _decide(a, b)
    result = CrossCheckResult(
        symbol=a.symbol,
        outcome=outcome,
        recommendation=recommendation,
        reason=reason,
        verdicts={a.role: _summary(a), b.role: _summary(b)},
    )
    if trace is not None:
        trace.log(AGENT_NAME, "decision", result.to_dict())
    return result
