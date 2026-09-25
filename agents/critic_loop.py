"""Critic loop - challenge, re-vote, re-check (spec section 3, step 3).

Runs after the cross-check for two kinds of stock:
- "split": the analysts said buy vs hold. Decided Sept 25, 2026: instead
  of aborting at once, the split goes to re-votes, and aborts if it is
  still unmatched afterwards.
- "agreed_buy": both analysts said buy. Spec section 3 has agents
  challenge each other's reasoning "before anything is decided", and a
  buy is the only outcome that opens a position, so an agreed buy is
  challenged once before it moves on. Agreed hold/avoid pass straight
  through (nothing to act on). CHALLENGE_AGREED_BUYS switches this off.

Each round: the critic reviews both current verdicts, each analyst
re-votes seeing its own verdict, the other's, and the challenges, and the
revised pair goes through the same cross_check as the first verdicts:
- agree  -> done, with the agreed recommendation (may differ from before);
- critic -> another round, up to MAX_ROUNDS, then abort;
- abort  -> abort (a contradiction, or a failed re-vote).
A failed critic call also aborts: an unchallenged buy is exactly what this
step exists to prevent, so a failed step never counts as a pass.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .analyst_agent import AnalystAgent, AnalystVerdict
from .critic_agent import CriticAgent
from .cross_check import ABORT, AGREE, CRITIC, CrossCheckResult, cross_check
from .macro_agent import StockContext
from .trace import TraceLogger

AGENT_NAME = "critic_loop"

# Each round costs one critic call plus one re-vote per analyst (3 calls).
MAX_ROUNDS = 2
CHALLENGE_AGREED_BUYS = True

SPLIT = "split"
AGREED_BUY = "agreed_buy"


@dataclass
class CriticLoopResult:
    symbol: str
    trigger: str
    outcome: str  # AGREE or ABORT
    recommendation: str | None
    reason: str
    rounds: list[dict[str, Any]] = field(default_factory=list)
    final_verdicts: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def needs_critic(result: CrossCheckResult, challenge_agreed_buys: bool = CHALLENGE_AGREED_BUYS) -> str | None:
    """The loop's trigger for this cross-check outcome, or None to skip it."""
    if result.outcome == CRITIC:
        return SPLIT
    if challenge_agreed_buys and result.outcome == AGREE and result.recommendation == "buy":
        return AGREED_BUY
    return None


def _verdict_summary(v: AnalystVerdict) -> dict[str, Any]:
    return {
        "model": v.model,
        "ok": v.ok,
        "error": v.error,
        "previous_recommendation": v.previous_recommendation,
        "recommendation": v.recommendation,
        "changed": v.changed,
        "confidence": v.confidence,
        "challenges_accepted": v.challenges_accepted,
        "challenges_rejected": v.challenges_rejected,
        "response_to_critique": v.response_to_critique,
        "evidence_check": v.evidence_check,
    }


def run_critic_loop(
    ctx: StockContext,
    verdicts: dict[str, AnalystVerdict],
    analysts: dict[str, AnalystAgent],
    critic: CriticAgent,
    trigger: str,
    *,
    trace: TraceLogger | None = None,
    max_rounds: int = MAX_ROUNDS,
) -> CriticLoopResult:
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    roles = tuple(sorted(verdicts))
    if len(roles) != 2 or set(roles) != set(analysts):
        raise ValueError(f"need exactly two analysts with matching verdicts, got {roles} / {sorted(analysts)}")

    def log(event: str, payload: dict[str, Any]) -> None:
        if trace is not None:
            trace.log(AGENT_NAME, event, {"symbol": ctx.symbol, **payload})

    def finish(outcome: str, recommendation: str | None, reason: str) -> CriticLoopResult:
        result = CriticLoopResult(
            symbol=ctx.symbol,
            trigger=trigger,
            outcome=outcome,
            recommendation=recommendation,
            reason=reason,
            rounds=rounds,
            final_verdicts={role: _verdict_summary(current[role]) for role in roles},
        )
        log("final", {k: v for k, v in result.to_dict().items() if k not in ("symbol", "rounds")})
        return result

    current = dict(verdicts)
    rounds: list[dict[str, Any]] = []
    log("start", {"trigger": trigger, "max_rounds": max_rounds,
                  "recommendations": {r: current[r].recommendation for r in roles}})

    for n in range(1, max_rounds + 1):
        critique = critic.review(ctx, current, n)
        if not critique.ok:
            rounds.append({"round": n, "critique": critique.to_dict()})
            return finish(ABORT, None, f"critic failed in round {n}: {critique.error}")

        revised: dict[str, AnalystVerdict] = {}
        for role in roles:
            other = roles[1] if role == roles[0] else roles[0]
            revised[role] = analysts[role].revise(
                ctx,
                current[role],
                current[other],
                assessment=critique.assessment,
                challenges_to_me=critique.for_role(role),
                challenges_to_other=critique.for_role(other),
                review_round=n,
            )
        check = cross_check(revised[roles[0]], revised[roles[1]], trace=trace)
        current = revised
        round_record = {
            "round": n,
            "critique": critique.to_dict(),
            "verdicts": {role: _verdict_summary(revised[role]) for role in roles},
            "cross_check": check.to_dict(),
        }
        rounds.append(round_record)
        log("round", {k: v for k, v in round_record.items() if k != "critique"}
            | {"challenges": {r: len(critique.for_role(r)) for r in roles}})

        if check.outcome == AGREE:
            return finish(AGREE, check.recommendation, f"after round {n}: {check.reason}")
        if check.outcome == ABORT:
            return finish(ABORT, None, f"after round {n}: {check.reason}")

    return finish(ABORT, None, f"still split after {max_rounds} rounds: {check.reason}")
