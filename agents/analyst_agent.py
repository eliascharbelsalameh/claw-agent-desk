"""Analyst agents - step 2 of the pipeline (spec section 3).

Two analysts do the same task on different models (AGENT_MODELS
"analyst_1" / "analyst_2", from different labs) and are compared
afterwards: if they contradict each other the stock is not picked. This
module is the single analyst; one AnalystAgent per role.

Each analyst reads the macro agent's StockContext and returns a structured
AnalystVerdict. Structure is what makes the later cross-check mechanical
(compare `recommendation`, not prose), and it allows a deterministic
grounding check: every piece of evidence must cite a fact path in the
context plus the value found there, and check_evidence() resolves each
path and compares. An analyst that quotes a number not in the data is
flagged in the verdict (and the trace), not silently believed.

In the critic loop (critic_loop.py) an analyst also revises its verdict:
it sees its own previous verdict, the other analyst's, and the critic's
challenges, and re-votes (`revise`).

A verdict that can't be parsed after one correction turn comes back with
`error` set. Downstream must treat that as "no pick" - an analyst that
failed to answer must never count as agreement.
"""
from __future__ import annotations

import json
import re
from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from data_layer.llm_client import DEFAULT_MODEL_HEALTH, ModelHealth, role_models

from .llm_json import (  # noqa: F401 - parsing helpers re-exported for existing callers
    PARSE_ATTEMPTS,
    extract_json_object,
    repair_json_quotes,
    request_json,
    unreachable,
)
from .macro_agent import IEX_VOLUME_NOTE, StockContext
from .trace import TraceLogger

RECOMMENDATIONS = ("buy", "hold", "avoid")

# A forward projection: is holding the shares over this window worthwhile,
# judged only from data available now. Decided Sept 25, 2026: 2-5 trading
# days, so the ~2-day paper run (spec section 7) can actually test the
# calls on camera; a multi-week call would be unverifiable in the demo.
# Intraday entry timing stays with the technical agent. Only the window is
# stated - what to weigh at this range is left to the analysts, unsteered.
HORIZON = "the next 2 to 5 trading days"

# Reasoning models (analyst_2's Nemotron) spend part of this budget on
# hidden reasoning before the JSON; observed completions stay under ~3k.
ANALYST_MAX_TOKENS = 8192

# At 0.2, 4 of 6 stock/analyst pairs flipped between buy and hold across 5
# repeats on identical input. At 0 (Sept 25, 2026), gemma-4-31b-it returned
# the identical verdict and confidence every time; nemotron-3-super still
# split 2-1 on two of three stocks (reasoning models on Build are not
# deterministic even at 0), so disagreements remain for the critic loop.
ANALYST_TEMPERATURE = 0.0

# Relative tolerance when comparing a cited number with the fact it names:
# models round (0.6073944 -> 0.61), and that is not a hallucination.
EVIDENCE_REL_TOL = 0.01

# Shown to every agent that sees check_evidence statuses (the critic, and
# analysts re-voting on each other's verdicts).
EVIDENCE_STATUS_NOTE = (
    'Each evidence item\'s "status" comes from an automatic check of the quote against the '
    'source facts. "matches_source" only means the quoted value is what the source facts say; '
    "it does not make the underlying claim true. A news headline or summary with that status is "
    "still unverified third-party reporting."
)

# The three recommendations take one symmetric bar (decided Sept 26, 2026).
# Until then the prompt told each analyst that a disagreement drops the
# stock, "so only recommend buy when the evidence genuinely supports it",
# and defined hold as "no strong case either way" - a one-sided brake on
# top of the desk's own (strict matching, the critic challenging every
# buy). Across the Sept 25-26 tests muse-glimmer never said buy (0 of 23)
# and 38 of 43 holds cited the missing valuation data. Caution now lives in
# the desk's structure, where the trace shows it, not inside each analyst.
SYSTEM_PROMPT = f"""You are an equity analyst on a research desk. You receive a context packet for one US stock and decide whether it belongs in a long-only portfolio over {HORIZON}. A second analyst, on a different model, analyses the same packet independently.

Rules:
- Use only the context packet. Your background knowledge of this company may be outdated: where it conflicts with the packet, the packet wins, and never cite prices, figures or events from memory.
- The "Source facts" block is authoritative. The briefing is a convenience summary written by another model; if they disagree, trust the source facts.
- {IEX_VOLUME_NOTE}
- News older than 72 hours may already be priced in. Headlines are not verified facts.
- Two different lists describe what is missing. "data_gaps" lists sources that failed to load for this run (often empty). "limitations" lists what this desk never provides (always present). Both matter.
- Do not claim growth, valuation, margins or trends unless the source facts contain the numbers that show them (e.g. fundamentals yoy_pct_change). Do not treat anything listed in limitations as known or "implied".
- "data_concerns" is your own assessment, not a copy of data_gaps: list each missing, stale or unreliable piece of information (from data_gaps, from limitations, or anything else you noticed) that actually affected this recommendation or its confidence, and how. An empty data_gaps does not mean there are no data concerns.
- Weigh the drivers against the risks and judge where the balance of evidence leans for someone holding the shares over the horizon: "buy" if it leans toward a gain, "avoid" if it leans toward a loss, "hold" if it is balanced or too thin to lean either way. Buy and avoid take the same bar.
- The portfolio holds real shares only, long-only: no short selling, options or other derivatives, leverage, or negotiated deals. "buy" means buying the shares outright; "avoid" means not holding them, never shorting.

Reply with ONLY a JSON object, no prose before or after, with exactly these keys:
{{
  "recommendation": "buy" | "hold" | "avoid",
  "confidence": number from 0 to 1,
  "thesis": "2-4 sentences",
  "drivers": ["reasons supporting the recommendation"],
  "risks": ["what could make it wrong"],
  "evidence": [{{"fact": "path into the source facts, e.g. price.change_20d_pct or macro.DGS10.latest or fundamentals.revenue.value or news[2].headline", "value": "the value at that path, copied exactly", "why": "why it matters"}}],
  "data_concerns": ["each missing, stale or unreliable piece of information that affected this call, and how"]
}}
Give at least 3 evidence items, each pointing at a real path in the source facts."""

# Appended to the context packet when an analyst re-votes in the critic loop.
REVISION_PROMPT = """=== REVIEW ROUND {review_round} ===
Before any decision, your verdict was reviewed together with that of a second analyst (a different model, same context packet), and a critic challenged the reasoning of both. You are {role}.

Your previous verdict:
{own}

The other analyst's verdict:
{other}

The critic's assessment:
{assessment}

The critic's challenges to you:
{mine}

The critic's challenges to the other analyst:
{theirs}

{status_note}

The critic is another model and can be wrong. Check each challenge addressed to you against the source facts: accept it only if the facts support it, and reject it, saying why, if they don't or if it doesn't matter for the recommendation. Change your recommendation only if the challenges you accept expose a real error, a fact you ignored, or an argument you cannot answer from the source facts. Do not change it just to agree with the other analyst, and do not keep it just to stay consistent.

The same rules and the same JSON format apply, with one extra key, "response_to_critique": a list with one entry per challenge addressed to you, each {{"point": "the challenge's point", "accept": true or false, "reason": "one or two sentences"}}. Use an empty list if there were no challenges to you."""


@dataclass
class AnalystVerdict:
    symbol: str
    role: str
    model: str
    generated_at: str
    recommendation: str | None = None
    confidence: float | None = None
    thesis: str | None = None
    drivers: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    data_concerns: list[str] = field(default_factory=list)
    evidence_check: dict[str, int] = field(default_factory=dict)
    # Quote fixes applied to the reply's JSON before it parsed (empty = none).
    json_repairs: list[str] = field(default_factory=list)
    # 0 for the independent first verdict; n after critic round n.
    review_round: int = 0
    previous_recommendation: str | None = None
    # One {point, accept, reason} per challenge the analyst answered; accept
    # is None when the model didn't say (e.g. it replied in free text).
    response_to_critique: list[dict[str, Any]] = field(default_factory=list)
    # Models tried before `model` answered, and why each failed.
    fallbacks: list[dict[str, Any]] = field(default_factory=list)
    # The role's configured primary. `model` differs from it when a backup
    # answered - including when the primary was skipped without a call
    # because it failed minutes earlier (so `fallbacks` is empty).
    primary_model: str | None = None
    error: str | None = None
    # With `error` set: True when no model could be reached (a Build
    # failure - the pipeline defers the stock and retries next cycle), False
    # when a model answered but unusably (final: the stock is aborted).
    call_failed: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def used_backup(self) -> bool:
        return self.primary_model is not None and self.model != self.primary_model

    @property
    def changed(self) -> bool:
        """Whether a revised verdict changed its recommendation."""
        return self.previous_recommendation is not None and self.recommendation != self.previous_recommendation

    @property
    def challenges_accepted(self) -> int:
        return sum(1 for r in self.response_to_critique if r.get("accept") is True)

    @property
    def challenges_rejected(self) -> int:
        return sum(1 for r in self.response_to_critique if r.get("accept") is False)

    def brief(self) -> dict[str, Any]:
        """What the other agents see of this verdict: the argument and its
        checked evidence, but not the model (it would invite brand bias)
        or the confidence (the models report it on different scales)."""
        brief: dict[str, Any] = {
            "recommendation": self.recommendation,
            "thesis": self.thesis,
            "drivers": self.drivers,
            "risks": self.risks,
            "evidence": [
                {k: e[k] for k in ("fact", "value", "why", "status", "found_at", "actual") if k in e}
                for e in self.evidence
            ],
            "data_concerns": self.data_concerns,
        }
        if self.response_to_critique:
            brief["response_to_critique"] = self.response_to_critique
        return brief

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- parsing / validation (no LLM) ---


def _str_list(obj: dict[str, Any], key: str, required: bool) -> list[str]:
    value = obj.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"'{key}' must be a list of strings")
    if required and not value:
        raise ValueError(f"'{key}' must not be empty")
    return value


def validate_verdict(obj: dict[str, Any]) -> dict[str, Any]:
    """Normalized verdict fields, or ValueError naming the first problem."""
    recommendation = str(obj.get("recommendation", "")).strip().lower()
    if recommendation not in RECOMMENDATIONS:
        raise ValueError(f"'recommendation' must be one of {RECOMMENDATIONS}, got {recommendation!r}")

    try:
        confidence = float(obj.get("confidence"))
    except (TypeError, ValueError):
        raise ValueError("'confidence' must be a number from 0 to 1") from None
    if 1 < confidence <= 100:  # a percentage instead of a fraction
        confidence /= 100
    if not 0 <= confidence <= 1:
        raise ValueError("'confidence' must be a number from 0 to 1")

    thesis = obj.get("thesis")
    if not isinstance(thesis, str) or not thesis.strip():
        raise ValueError("'thesis' must be a non-empty string")

    evidence = obj.get("evidence")
    if not isinstance(evidence, list) or len(evidence) < 3:
        raise ValueError("'evidence' must be a list of at least 3 items")
    for item in evidence:
        if not isinstance(item, dict) or not item.get("fact") or "value" not in item:
            raise ValueError("each evidence item needs 'fact' and 'value'")

    return {
        "recommendation": recommendation,
        "confidence": round(confidence, 3),
        "thesis": thesis.strip(),
        "drivers": _str_list(obj, "drivers", required=True),
        "risks": _str_list(obj, "risks", required=True),
        "evidence": [dict(item) for item in evidence],
        "data_concerns": _str_list(obj, "data_concerns", required=False),
    }


def _as_text(value: Any) -> str | None:
    """Free-text field that models sometimes return as a list or object."""
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list):
        return " ".join(str(v).strip() for v in value if str(v).strip()) or None
    return json.dumps(value, ensure_ascii=False)


_ACCEPT_WORDS = {"true", "yes", "accept", "accepted"}
_REJECT_WORDS = {"false", "no", "reject", "rejected"}


def parse_critique_responses(value: Any) -> list[dict[str, Any]]:
    """Normalize an analyst's response_to_critique into {point, accept, reason}
    entries. Lenient on purpose: a re-vote is never failed over this field.
    Free text or a {point: reason} mapping (both seen live) is kept, with
    accept=None since the model didn't say."""
    if value is None:
        return []
    if isinstance(value, dict):
        value = [{"point": k, "reason": v} for k, v in value.items()]
    if not isinstance(value, list):
        text = _as_text(value)
        return [{"point": None, "accept": None, "reason": text}] if text else []
    responses = []
    for item in value:
        if not isinstance(item, dict):
            text = _as_text(item)
            if text:
                responses.append({"point": None, "accept": None, "reason": text})
            continue
        accept = item.get("accept")
        if isinstance(accept, str):
            word = accept.strip().lower()
            accept = True if word in _ACCEPT_WORDS else False if word in _REJECT_WORDS else None
        responses.append({
            "point": _as_text(item.get("point")),
            "accept": accept if isinstance(accept, bool) else None,
            "reason": _as_text(item.get("reason") or item.get("response")),
        })
    return responses


# --- grounding check (no LLM) ---

_PATH_TOKEN_RE = re.compile(r"([^.\[\]]+)|\[(\d+)\]")
_UNRESOLVED = object()


def resolve_fact_path(facts: Any, path: str) -> Any:
    """Look up "news[2].headline" / "macro.DGS10.latest" in the facts
    dict; returns _UNRESOLVED if any step doesn't exist."""
    node = facts
    for key, index in _PATH_TOKEN_RE.findall(path.strip()):
        if index:
            if not isinstance(node, list) or int(index) >= len(node):
                return _UNRESOLVED
            node = node[int(index)]
        else:
            if not isinstance(node, dict) or key not in node:
                return _UNRESOLVED
            node = node[key]
    return node


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").rstrip("%").lstrip("$")
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def values_match(cited: Any, actual: Any) -> bool:
    cited_num, actual_num = _as_number(cited), _as_number(actual)
    if cited_num is not None and actual_num is not None:
        scale = max(abs(actual_num), 1e-9)
        return abs(cited_num - actual_num) / scale <= EVIDENCE_REL_TOL
    # Compare words only: models rewrite punctuation when quoting (live, an
    # em dash came back as two non-breaking hyphens).
    cited_str = _words(cited)
    actual_str = _words(actual)
    if not cited_str or not actual_str:
        return False
    return cited_str in actual_str or actual_str in cited_str


_NON_WORD_RE = re.compile(r"[\W_]+")


def _words(value: Any) -> str:
    return _NON_WORD_RE.sub(" ", str(value).casefold()).strip()


_FIRST_INDEX_RE = re.compile(r"\[(\d+)\]")


def _find_at_other_index(facts: dict[str, Any], path: str, cited: Any) -> str | None:
    """For "news[13].headline", the path where the cited value actually is
    (e.g. "news[4].headline"), if anywhere. Live, an analyst quoted a real
    headline under the wrong index; that is sloppy, not fabricated, and the
    critic should be able to tell the two apart."""
    match = _FIRST_INDEX_RE.search(path)
    if match is None:
        return None
    container = resolve_fact_path(facts, path[: match.start()])
    if not isinstance(container, list):
        return None
    for i in range(len(container)):
        candidate = f"{path[: match.start()]}[{i}]{path[match.end():]}"
        actual = resolve_fact_path(facts, candidate)
        if actual is not _UNRESOLVED and values_match(cited, actual):
            return candidate
    return None


def check_evidence(evidence: list[dict[str, Any]], facts: dict[str, Any]) -> dict[str, int]:
    """Annotate each item with a status and return counts per status:
    matches_source (the quoted value is what the facts say at that path),
    wrong_index (real value, wrong list position - `found_at` says where),
    mismatch (the value at that path differs - `actual` says what it is),
    or unknown_path (the path doesn't exist in the facts).

    The check confirms quotes, not truth. It was called "verified" until
    Sept 25, 2026, when a critic read a news item's "verified" status as
    "confirmed fact" and pushed an analyst toward buy on it (see
    EVIDENCE_STATUS_NOTE)."""
    counts = {"matches_source": 0, "wrong_index": 0, "mismatch": 0, "unknown_path": 0}
    for item in evidence:
        path = str(item["fact"])
        actual = resolve_fact_path(facts, path)
        if actual is not _UNRESOLVED and values_match(item["value"], actual):
            item["status"] = "matches_source"
        elif (found_at := _find_at_other_index(facts, path, item["value"])) is not None:
            item["status"] = "wrong_index"
            item["found_at"] = found_at
        elif actual is _UNRESOLVED:
            item["status"] = "unknown_path"
        else:
            item["status"] = "mismatch"
            item["actual"] = actual
        counts[item["status"]] += 1
    return counts


def _pretty(value: Any) -> str:
    return json.dumps(value, indent=1, ensure_ascii=False, default=str)


class AnalystAgent:
    def __init__(
        self,
        llm: Any,
        role: str = "analyst_1",
        *,
        trace: TraceLogger | None = None,
        model: str | None = None,
        models: Sequence[str] | None = None,
        health: ModelHealth | None = None,
        max_tokens: int = ANALYST_MAX_TOKENS,
        temperature: float = ANALYST_TEMPERATURE,
        extra_params: dict[str, Any] | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        """`models` is the ordered candidate list (primary, then backups);
        by default the role's list from AGENT_MODELS/AGENT_MODEL_BACKUPS.
        Passing `model` pins a single model with no backups."""
        self._llm = llm
        self.role = role
        self.models = list(models) if models else [model] if model else role_models(role)
        self.model = self.models[0]
        self._health = health if health is not None else DEFAULT_MODEL_HEALTH
        self._trace = trace
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._extra_params = extra_params or {}
        self._now = now

    def _log(self, event: str, payload: dict[str, Any]) -> None:
        if self._trace is not None:
            self._trace.log(self.role, event, payload)

    def _new_verdict(self, ctx: StockContext, **fields: Any) -> AnalystVerdict:
        return AnalystVerdict(
            symbol=ctx.symbol,
            role=self.role,
            model=self.model,
            primary_model=self.model,
            generated_at=self._now().isoformat(),
            **fields,
        )

    def _complete(
        self,
        ctx: StockContext,
        verdict: AnalystVerdict,
        messages: list[dict[str, str]],
        models: Sequence[str],
        exclude: Collection[str],
    ) -> dict[str, Any] | None:
        """Ask the candidate models, in order, for the verdict JSON and fill
        `verdict` in place (including the model that actually answered).
        Returns the raw parsed object, or None with `verdict.error` set."""
        base = {"symbol": ctx.symbol, "review_round": verdict.review_round}
        self._log("llm_request", {**base, "models": list(models), "exclude": sorted(exclude),
                                  "messages": messages})
        reply = request_json(
            self._llm,
            models,
            messages,
            validate=validate_verdict,
            log=self._log,
            base=base,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            extra_params=self._extra_params,
            invalid_event="verdict_invalid",
            repaired_event="verdict_repaired",
            health=self._health,
            exclude=exclude,
        )
        verdict.model = reply.model or verdict.model
        verdict.fallbacks = reply.fallbacks
        if not reply.ok:
            verdict.error = reply.error if reply.call_failed else f"unparseable verdict: {reply.error}"
            verdict.call_failed = unreachable(reply)
            self._log("verdict_failed", {"symbol": ctx.symbol, "review_round": verdict.review_round,
                                         "model": verdict.model, "error": verdict.error,
                                         "call_failed": verdict.call_failed})
            return None
        for key, value in reply.value.items():
            setattr(verdict, key, value)
        verdict.json_repairs = reply.repairs
        verdict.evidence_check = check_evidence(verdict.evidence, ctx.facts())
        return reply.raw

    def analyze(
        self, ctx: StockContext, *, exclude: Collection[str] = (), allow_backup: bool = False
    ) -> AnalystVerdict:
        """Independent first verdict. `exclude` names models this analyst
        must not use - the model the other analyst already ran on.

        Only the role's primary answers unless `allow_backup`. Decided Sept
        26, 2026: when the primary can't be reached the stock is deferred
        and retried next cycle, rather than decided by a backup that judges
        differently (muse-glimmer, analyst_1's backup, said buy 0 times in
        23 verdicts). The pipeline allows backups only once a primary has
        been down for a long stretch (pipeline.ANALYST_BACKUP_AFTER)."""
        verdict = self._new_verdict(ctx)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ctx.to_prompt()},
        ]
        models = self.models if allow_backup else self.models[:1]
        if self._complete(ctx, verdict, messages, models, set(exclude)) is not None:
            self._log("verdict", verdict.to_dict())
        return verdict

    def revise(
        self,
        ctx: StockContext,
        own: AnalystVerdict,
        other: AnalystVerdict,
        *,
        assessment: str,
        challenges_to_me: list[dict[str, Any]],
        challenges_to_other: list[dict[str, Any]],
        review_round: int,
    ) -> AnalystVerdict:
        """Re-vote after a critic round: same packet, plus both previous
        verdicts and the critic's challenges. Returns a new verdict (the
        previous one is left untouched) with previous_recommendation set."""
        if not (own.ok and other.ok):
            raise ValueError("revise needs two successful verdicts")
        verdict = self._new_verdict(
            ctx, review_round=review_round, previous_recommendation=own.recommendation
        )

        def listed(challenges: list[dict[str, Any]]) -> str:
            shown = [{k: v for k, v in c.items() if k != "to"} for c in challenges]
            return _pretty(shown) if shown else "(none)"

        review = REVISION_PROMPT.format(
            review_round=review_round,
            role=self.role,
            own=_pretty(own.brief()),
            other=_pretty(other.brief()),
            assessment=assessment,
            mine=listed(challenges_to_me),
            theirs=listed(challenges_to_other),
            status_note=EVIDENCE_STATUS_NOTE,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{ctx.to_prompt()}\n\n{review}"},
        ]
        # Only the model that wrote the verdict revises it: a re-vote from
        # another model would be a different analyst. If it can't be reached
        # the critic loop is deferred and this re-vote retried next cycle.
        raw = self._complete(ctx, verdict, messages, [own.model], {other.model})
        if raw is not None:
            verdict.response_to_critique = parse_critique_responses(raw.get("response_to_critique"))
            self._log("verdict", verdict.to_dict())
        return verdict
