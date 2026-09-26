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
- abort  -> abort (a contradiction, or a re-vote that came back unusable).
A critic that answers unusably also aborts: an unchallenged buy is exactly
what this step exists to prevent, so a failed step never counts as a pass.

Decided Sept 26, 2026: when a model can't be reached at all (a Build
failure), the loop is *deferred* instead - not a decision, and never a
pass. The result keeps what the round already produced (the critique, any
re-vote that came back) in `resume`, and run_critic_loop(resume_from=...)
picks up at that round next cycle, repeating only the calls that failed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .analyst_agent import AnalystAgent, AnalystVerdict
from .critic_agent import Critique, CriticAgent
from .cross_check import ABORT, AGREE, CRITIC, DEFERRED, CrossCheckResult, cross_check
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
    outcome: str  # AGREE, ABORT or DEFERRED
    recommendation: str | None
    reason: str
    rounds: list[dict[str, Any]] = field(default_factory=list)
    final_verdicts: dict[str, dict[str, Any]] = field(default_factory=dict)
    # The full verdicts the loop ended on (AnalystVerdict.to_dict()), for
    # the stages after it: the bias agents review the final buy case.
    verdicts: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Only when DEFERRED: the round to pick up, the full verdicts going into
    # it, and what of it already succeeded ("critique", and "revised" for
    # re-votes that came back), so a retry repeats only the failed calls.
    resume: dict[str, Any] | None = None

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
    resume_from: CriticLoopResult | None = None,
) -> CriticLoopResult:
    """Run the loop from round 1, or - with `resume_from`, a deferred result
    of an earlier call - from the round where that one stopped (`verdicts`
    and `trigger` then come from the deferred result)."""
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    if resume_from is not None and (resume_from.outcome != DEFERRED or not resume_from.resume):
        raise ValueError("only a deferred critic loop can be resumed")
    roles = tuple(sorted(resume_from.resume["verdicts"] if resume_from is not None else verdicts))
    if len(roles) != 2 or set(roles) != set(analysts):
        raise ValueError(f"need exactly two analysts with matching verdicts, got {roles} / {sorted(analysts)}")

    def log(event: str, payload: dict[str, Any]) -> None:
        if trace is not None:
            trace.log(AGENT_NAME, event, {"symbol": ctx.symbol, **payload})

    def result(outcome: str, recommendation: str | None, reason: str,
               resume: dict[str, Any] | None = None) -> CriticLoopResult:
        return CriticLoopResult(
            symbol=ctx.symbol,
            trigger=trigger,
            outcome=outcome,
            recommendation=recommendation,
            reason=reason,
            rounds=rounds,
            final_verdicts={role: _verdict_summary(current[role]) for role in roles},
            verdicts={role: current[role].to_dict() for role in roles},
            resume=resume,
        )

    def finish(outcome: str, recommendation: str | None, reason: str) -> CriticLoopResult:
        done = result(outcome, recommendation, reason)
        log("final", {k: v for k, v in done.to_dict().items()
                      if k not in ("symbol", "rounds", "resume", "verdicts")})
        return done

    def defer(n: int, critique: Critique | None, revised: dict[str, AnalystVerdict], reason: str) -> CriticLoopResult:
        state = {
            "round": n,
            "verdicts": {role: current[role].to_dict() for role in roles},
            "critique": critique.to_dict() if critique is not None else None,
            "revised": {role: v.to_dict() for role, v in revised.items()},
        }
        log("deferred", {"trigger": trigger, "round": n, "reason": reason,
                         "kept": {"critique": critique is not None, "revised": sorted(revised)}})
        return result(DEFERRED, None, reason, resume=state)

    if resume_from is None:
        start = 1
        current = dict(verdicts)
        rounds: list[dict[str, Any]] = []
        pending_critique: Critique | None = None
        pending_revised: dict[str, AnalystVerdict] = {}
        log("start", {"trigger": trigger, "max_rounds": max_rounds,
                      "recommendations": {r: current[r].recommendation for r in roles}})
    else:
        state = resume_from.resume
        trigger = resume_from.trigger
        start = state["round"]
        current = {role: AnalystVerdict(**d) for role, d in state["verdicts"].items()}
        rounds = list(resume_from.rounds)
        pending_critique = Critique(**state["critique"]) if state.get("critique") else None
        pending_revised = {role: AnalystVerdict(**d) for role, d in state.get("revised", {}).items()}
        log("resume", {"trigger": trigger, "round": start,
                       "kept": {"critique": pending_critique is not None, "revised": sorted(pending_revised)}})

    check: CrossCheckResult | None = None
    for n in range(start, max_rounds + 1):
        reuse = n == start
        critique = pending_critique if reuse and pending_critique is not None else critic.review(ctx, current, n)
        if not critique.ok:
            if critique.call_failed:
                return defer(n, None, {}, f"critic could not be reached in round {n}: {critique.error}")
            rounds.append({"round": n, "critique": critique.to_dict()})
            return finish(ABORT, None, f"critic failed in round {n}: {critique.error}")

        revised: dict[str, AnalystVerdict] = dict(pending_revised) if reuse else {}
        for role in roles:
            if role in revised:
                continue
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
        unusable = [r for r in roles if not revised[r].ok and not revised[r].call_failed]
        unreached = [r for r in roles if not revised[r].ok and revised[r].call_failed]
        if unreached and not unusable:
            models = ", ".join(f"{r} ({revised[r].model})" for r in unreached)
            return defer(n, critique, {r: v for r, v in revised.items() if v.ok},
                         f"could not reach {models} for the round {n} re-vote: a Build failure, "
                         "retried next cycle")

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

    reason = check.reason if check is not None else "no round left to run"
    return finish(ABORT, None, f"still split after {max_rounds} rounds: {reason}")
