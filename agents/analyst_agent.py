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

A verdict that can't be parsed after one correction turn comes back with
`error` set. Downstream must treat that as "no pick" - an analyst that
failed to answer must never count as agreement.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from data_layer.llm_client import AGENT_MODELS

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
PARSE_ATTEMPTS = 2

# At 0.2, 4 of 6 stock/analyst pairs flipped between buy and hold across 5
# repeats on identical input. At 0 (Sept 25, 2026), gemma-4-31b-it returned
# the identical verdict and confidence every time; nemotron-3-super still
# split 2-1 on two of three stocks (reasoning models on Build are not
# deterministic even at 0), so disagreements remain for the critic loop.
ANALYST_TEMPERATURE = 0.0

# Relative tolerance when comparing a cited number with the fact it names:
# models round (0.6073944 -> 0.61), and that is not a hallucination.
EVIDENCE_REL_TOL = 0.01

SYSTEM_PROMPT = f"""You are an equity analyst on a research desk. You receive a context packet for one US stock and decide whether it belongs in a long-only portfolio over {HORIZON}. A second analyst, on a different model, analyses the same packet independently; if you disagree, the stock is not picked, so only recommend "buy" when the evidence genuinely supports it.

Rules:
- Use only the context packet. Your background knowledge of this company may be outdated: where it conflicts with the packet, the packet wins, and never cite prices, figures or events from memory.
- The "Source facts" block is authoritative. The briefing is a convenience summary written by another model; if they disagree, trust the source facts.
- {IEX_VOLUME_NOTE}
- News older than 72 hours may already be priced in. Headlines are not verified facts.
- Two different lists describe what is missing. "data_gaps" lists sources that failed to load for this run (often empty). "limitations" lists what this desk never provides (always present). Both matter.
- Do not claim growth, valuation, margins or trends unless the source facts contain the numbers that show them (e.g. fundamentals yoy_pct_change). Do not treat anything listed in limitations as known or "implied".
- "data_concerns" is your own assessment, not a copy of data_gaps: list each missing, stale or unreliable piece of information (from data_gaps, from limitations, or anything else you noticed) that actually affected this recommendation or its confidence, and how. An empty data_gaps does not mean there are no data concerns.
- "hold" means no strong case either way; "avoid" means the risks outweigh the case for buying.
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
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- parsing / validation (no LLM) ---

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json_object(text: str, repairs: list[str] | None = None) -> dict[str, Any]:
    """The outermost {...} in a reply, tolerating code fences and stray
    prose around it. Raises ValueError if there is no parseable object.

    If the JSON doesn't parse, repair_json_quotes gets one try; when that
    produces valid JSON, a description of each fix is appended to `repairs`
    (if given) so the caller can log exactly what was changed.
    """
    text = _FENCE_RE.sub("", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in reply")
    body = text[start : end + 1]
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as exc:
        repaired, fixes = repair_json_quotes(body)
        try:
            obj = json.loads(repaired) if fixes else None
        except json.JSONDecodeError:
            obj = None
        if obj is None:
            raise ValueError(f"invalid JSON: {exc}") from exc
        if repairs is not None:
            repairs.extend(fixes)
    if not isinstance(obj, dict):
        raise ValueError("reply JSON is not an object")
    return obj


# `"key": value` on one line, as models pretty-print their JSON.
_KEY_VALUE_RE = re.compile(r'^(\s*"[^"\\]+"\s*:\s*)(.*?)\s*$')
# A value that is already valid JSON syntax at its start.
_JSON_VALUE_START_RE = re.compile(r'^(["\[{]|-?\d|true\b|false\b|null\b)')


def _has_closing_quote(line: str) -> bool:
    """For a line starting with '"', whether an unescaped closing quote follows."""
    escaped = False
    for ch in line[1:]:
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            return True
    return False


def repair_json_quotes(body: str) -> tuple[str, list[str]]:
    """Add the quotes a model left off string values, and nothing else.

    nemotron-3-super repeatedly emitted exactly these shapes (5 of 38
    replies on Sept 25, 2026, and its correction turn reproduced them
    character for character):
      "why": Indicates strong growth.        <- both quotes missing
      "why": Indicates strong growth."       <- opening quote missing
      "Continued heavy AI capex could ...    <- list item never closed
    Only whole lines are touched and no word of the content changes; the
    caller re-parses and re-validates, so a wrong guess fails as before.
    """
    lines = body.split("\n")
    fixes: list[str] = []
    for i, line in enumerate(lines):
        match = _KEY_VALUE_RE.match(line)
        if match:
            prefix, value = match.groups()
            if not value or _JSON_VALUE_START_RE.match(value):
                continue
            comma = value.endswith(",")
            inner = value[:-1].rstrip() if comma else value
            if inner.endswith('"') and not inner.endswith('\\"'):
                inner = inner[:-1]
            lines[i] = prefix + json.dumps(inner, ensure_ascii=False) + ("," if comma else "")
            fixes.append(f"line {i + 1}: quoted the unquoted value of {prefix.strip().rstrip(':').strip()}")
            continue
        stripped = line.strip()
        if stripped.startswith('"') and not _has_closing_quote(stripped):
            following = next((l.strip() for l in lines[i + 1 :] if l.strip()), "")
            lines[i] = line.rstrip() + ('"' if following.startswith(("]", "}")) else '",')
            fixes.append(f"line {i + 1}: closed an unterminated string")
    return "\n".join(lines), fixes


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
    verified, wrong_index (real value, wrong list position - `found_at`
    says where), mismatch (the value at that path differs - `actual` says
    what it is), or unknown_path (the path doesn't exist in the facts)."""
    counts = {"verified": 0, "wrong_index": 0, "mismatch": 0, "unknown_path": 0}
    for item in evidence:
        path = str(item["fact"])
        actual = resolve_fact_path(facts, path)
        if actual is not _UNRESOLVED and values_match(item["value"], actual):
            item["status"] = "verified"
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


class AnalystAgent:
    def __init__(
        self,
        llm: Any,
        role: str = "analyst_1",
        *,
        trace: TraceLogger | None = None,
        model: str | None = None,
        max_tokens: int = ANALYST_MAX_TOKENS,
        temperature: float = ANALYST_TEMPERATURE,
        extra_params: dict[str, Any] | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self._llm = llm
        self.role = role
        self.model = model or AGENT_MODELS[role]
        self._trace = trace
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._extra_params = extra_params or {}
        self._now = now

    def _log(self, event: str, payload: dict[str, Any]) -> None:
        if self._trace is not None:
            self._trace.log(self.role, event, payload)

    def analyze(self, ctx: StockContext) -> AnalystVerdict:
        verdict = AnalystVerdict(
            symbol=ctx.symbol,
            role=self.role,
            model=self.model,
            generated_at=self._now().isoformat(),
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ctx.to_prompt()},
        ]
        self._log("llm_request", {"symbol": ctx.symbol, "model": self.model, "messages": messages})

        for attempt in range(1, PARSE_ATTEMPTS + 1):
            base = {"symbol": ctx.symbol, "model": self.model, "attempt": attempt}
            try:
                response = self._llm.chat_completion(
                    self.model,
                    messages,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    **self._extra_params,
                )
                choice = response["choices"][0]
                content = choice["message"].get("content") or ""
                finish_reason = choice.get("finish_reason")
            except Exception as exc:  # noqa: BLE001 - a failed analyst means "no pick"
                verdict.error = f"{type(exc).__name__}: {exc}"
                self._log("llm_error", {**base, "error": repr(exc)})
                return verdict

            self._log(
                "llm_response",
                {
                    **base,
                    "finish_reason": finish_reason,
                    "content": content,
                    "reasoning_content": choice["message"].get("reasoning_content"),
                    "usage": response.get("usage"),
                },
            )
            repairs: list[str] = []
            try:
                if not content.strip():
                    raise ValueError(f"empty reply (finish_reason={finish_reason})")
                fields = validate_verdict(extract_json_object(content, repairs))
            except ValueError as exc:
                verdict.error = f"unparseable verdict: {exc}"
                self._log("verdict_invalid", {**base, "problem": str(exc)})
                # One correction turn: show the model its reply and the problem.
                messages = messages + [
                    {"role": "assistant", "content": content[-4000:]},
                    {
                        "role": "user",
                        "content": (
                            f"That reply could not be used: {exc}. Reply again with ONLY "
                            "the JSON object in the required format."
                        ),
                    },
                ]
                continue

            if repairs:
                self._log("verdict_repaired", {**base, "repairs": repairs})
            for key, value in fields.items():
                setattr(verdict, key, value)
            verdict.json_repairs = repairs
            verdict.evidence_check = check_evidence(verdict.evidence, ctx.facts())
            verdict.error = None
            self._log("verdict", verdict.to_dict())
            return verdict

        return verdict
