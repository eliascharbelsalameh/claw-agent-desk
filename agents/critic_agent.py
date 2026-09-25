"""Critic agent - step 3 of the pipeline (spec section 3).

"Agents challenge each other's reasoning over several rounds before
anything is decided." The critic reads the context packet and both
analysts' verdicts and challenges specific points of their reasoning:
claims the source facts don't support, misread or mis-cited evidence,
ignored facts or data gaps, and arguments one analyst raised that the
other never answered.

The critic never gives a recommendation of its own and never says which
analyst is right: the decision stays with the two independent analysts,
who re-vote after reading the challenges (critic_loop.py). A critic that
voted would turn the desk into a majority vote and bypass the strict
two-analyst match decided on Sept 25, 2026.

Challenges that cite a fact path go through the same deterministic check
as the analysts' evidence, so a critic that misquotes the data is flagged
the same way an analyst would be.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from data_layer.llm_client import DEFAULT_MODEL_HEALTH, ModelHealth, role_models

from .analyst_agent import EVIDENCE_STATUS_NOTE, HORIZON, AnalystVerdict, check_evidence
from .llm_json import request_json
from .macro_agent import StockContext
from .trace import TraceLogger

AGENT_NAME = "critic"

# Most important first; extras are dropped (and counted) rather than
# failing the reply, so a verbose critic doesn't cost a correction turn.
MAX_CHALLENGES_PER_ANALYST = 3
CRITIC_MAX_TOKENS = 8192
CRITIC_TEMPERATURE = 0.0

SYSTEM_PROMPT = f"""You are the critic on an equity research desk. Two analysts, running on different models, independently analysed the same context packet for one US stock and gave verdicts on whether it belongs in a long-only portfolio of real shares over {HORIZON}. Your job is to test their reasoning, not to decide: you never give a recommendation of your own and never say which analyst is right. After your review, each analyst reconsiders and votes again.

Challenge only what matters:
- claims the source facts do not support, or that contradict them;
- evidence that is misread or cited with a wrong value (anything whose "status" is not "matches_source" deserves scrutiny);
- relevant facts, data_gaps or limitations an analyst ignored;
- drivers or risks raised by one analyst that the other did not address.

Rules:
- Apply the same standard to both analysts. If an analysis holds up, give it no challenges rather than inventing weak ones.
- At most {MAX_CHALLENGES_PER_ANALYST} challenges per analyst, most important first.
- The "Source facts" block is authoritative. Your background knowledge may be outdated: never cite figures or events from memory. When a challenge rests on a fact, give its path and the value at that path, copied exactly.
- {EVIDENCE_STATUS_NOTE}

Reply with ONLY a JSON object, no prose before or after:
{{
  "assessment": "2-4 sentences: where the two analyses agree, where they diverge, and what the divergence rests on",
  "challenges": [
    {{"to": "analyst_1" or "analyst_2", "point": "the specific claim or omission being challenged", "why": "why it is unsupported, wrong or incomplete", "fact": "optional path into the source facts", "value": "the value at that path, if fact is given"}}
  ]
}}"""


@dataclass
class Critique:
    symbol: str
    model: str
    review_round: int
    generated_at: str
    assessment: str | None = None
    challenges: list[dict[str, Any]] = field(default_factory=list)
    # Status counts for challenges that cite a fact (same check as evidence).
    evidence_check: dict[str, int] = field(default_factory=dict)
    dropped_challenges: int = 0
    json_repairs: list[str] = field(default_factory=list)
    # Models tried before `model` answered, and why each failed.
    fallbacks: list[dict[str, Any]] = field(default_factory=list)
    # The critic's configured primary; `model` differs when a backup answered
    # (including when the primary was excluded or skipped as cooling).
    primary_model: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def for_role(self, role: str) -> list[dict[str, Any]]:
        return [c for c in self.challenges if c["to"] == role]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_critique(obj: dict[str, Any], roles: tuple[str, ...]) -> dict[str, Any]:
    """Normalized critique fields, or ValueError naming the first problem."""
    assessment = obj.get("assessment")
    if not isinstance(assessment, str) or not assessment.strip():
        raise ValueError("'assessment' must be a non-empty string")
    challenges = obj.get("challenges")
    if not isinstance(challenges, list):
        raise ValueError("'challenges' must be a list (empty if nothing needs challenging)")

    kept: list[dict[str, Any]] = []
    per_role = {role: 0 for role in roles}
    dropped = 0
    for item in challenges:
        if not isinstance(item, dict):
            raise ValueError("each challenge must be an object")
        to = str(item.get("to", "")).strip()
        if to not in roles:
            raise ValueError(f"each challenge's 'to' must be one of {roles}, got {to!r}")
        point, why = item.get("point"), item.get("why")
        if not (isinstance(point, str) and point.strip() and isinstance(why, str) and why.strip()):
            raise ValueError("each challenge needs non-empty 'point' and 'why' strings")
        if per_role[to] >= MAX_CHALLENGES_PER_ANALYST:
            dropped += 1
            continue
        per_role[to] += 1
        challenge = {"to": to, "point": point.strip(), "why": why.strip()}
        fact = item.get("fact")
        if isinstance(fact, str) and fact.strip() and "value" in item and item["value"] not in (None, ""):
            challenge["fact"] = fact.strip()
            challenge["value"] = item["value"]
        kept.append(challenge)
    return {"assessment": assessment.strip(), "challenges": kept, "dropped_challenges": dropped}


class CriticAgent:
    def __init__(
        self,
        llm: Any,
        *,
        trace: TraceLogger | None = None,
        model: str | None = None,
        models: Sequence[str] | None = None,
        health: ModelHealth | None = None,
        max_tokens: int = CRITIC_MAX_TOKENS,
        temperature: float = CRITIC_TEMPERATURE,
        extra_params: dict[str, Any] | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        """`models` is the ordered candidate list (primary, then backups);
        by default the critic's list from AGENT_MODELS/AGENT_MODEL_BACKUPS.
        Passing `model` pins a single model with no backups."""
        self._llm = llm
        self.models = list(models) if models else [model] if model else role_models(AGENT_NAME)
        self.model = self.models[0]
        self._health = health if health is not None else DEFAULT_MODEL_HEALTH
        self._trace = trace
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._extra_params = extra_params or {}
        self._now = now

    def _log(self, event: str, payload: dict[str, Any]) -> None:
        if self._trace is not None:
            self._trace.log(AGENT_NAME, event, payload)

    def review(
        self, ctx: StockContext, verdicts: dict[str, AnalystVerdict], review_round: int
    ) -> Critique:
        roles = tuple(sorted(verdicts))
        critique = Critique(
            symbol=ctx.symbol,
            model=self.model,
            primary_model=self.model,
            review_round=review_round,
            generated_at=self._now().isoformat(),
        )
        briefs = {role: verdicts[role].brief() for role in roles}
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{ctx.to_prompt()}\n\n=== VERDICTS TO REVIEW (round {review_round}) ===\n"
                    f"{json.dumps(briefs, indent=1, ensure_ascii=False, default=str)}"
                ),
            },
        ]
        # A critic running on an analyst's model would be reviewing its own
        # reasoning, so the models the analysts actually used are off limits.
        exclude = sorted({v.model for v in verdicts.values()})
        base = {"symbol": ctx.symbol, "review_round": review_round}
        self._log("llm_request", {**base, "models": self.models, "exclude": exclude, "messages": messages})
        reply = request_json(
            self._llm,
            self.models,
            messages,
            validate=lambda obj: validate_critique(obj, roles),
            log=self._log,
            base=base,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            extra_params=self._extra_params,
            invalid_event="critique_invalid",
            repaired_event="critique_repaired",
            health=self._health,
            exclude=exclude,
        )
        critique.model = reply.model or critique.model
        critique.fallbacks = reply.fallbacks
        if not reply.ok:
            critique.error = reply.error if reply.call_failed else f"unparseable critique: {reply.error}"
            return critique

        critique.assessment = reply.value["assessment"]
        critique.challenges = reply.value["challenges"]
        critique.dropped_challenges = reply.value["dropped_challenges"]
        critique.json_repairs = reply.repairs
        cited = [c for c in critique.challenges if "fact" in c]
        if cited:
            critique.evidence_check = check_evidence(cited, ctx.facts())
        self._log("critique", critique.to_dict())
        return critique
