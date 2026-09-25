"""Shared "ask a model for one JSON object" loop for the desk's agents.

Every agent that needs structured output from a Build model goes through
the same steps: call, log the raw reply, parse it (repairing the quote
slips nemotron-3-super is known for), validate it, and on failure give the
model exactly one correction turn. Keeping this in one place means the
analysts, the critic, and the agents still to come all fail, retry, and
log the same way.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

PARSE_ATTEMPTS = 2

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


@dataclass
class JsonReply:
    """Outcome of request_json. `value` is whatever `validate` returned;
    `raw` is the parsed object it was built from. On failure both are None
    and `error` says why; `call_failed` separates "the call itself failed"
    (connection, HTTP) from "the model answered but unusably"."""

    value: Any = None
    raw: dict[str, Any] | None = None
    repairs: list[str] = field(default_factory=list)
    error: str | None = None
    call_failed: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None


def request_json(
    llm: Any,
    model: str,
    messages: list[dict[str, str]],
    *,
    validate: Callable[[dict[str, Any]], Any],
    log: Callable[[str, dict[str, Any]], None],
    base: dict[str, Any],
    max_tokens: int,
    temperature: float,
    extra_params: dict[str, Any] | None = None,
    attempts: int = PARSE_ATTEMPTS,
    invalid_event: str = "reply_invalid",
    repaired_event: str = "reply_repaired",
) -> JsonReply:
    """Ask `model` for one JSON object that passes `validate`.

    `validate` takes the parsed object and returns the normalized result or
    raises ValueError naming the problem; that message is what the model
    sees in its correction turn. Every reply and failure is sent to `log`
    with `base` merged in, so each agent's trace reads the same way.
    """
    messages = list(messages)
    error: str | None = None
    for attempt in range(1, attempts + 1):
        info = {**base, "attempt": attempt}
        try:
            response = llm.chat_completion(
                model,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **(extra_params or {}),
            )
            choice = response["choices"][0]
            content = choice["message"].get("content") or ""
            finish_reason = choice.get("finish_reason")
        except Exception as exc:  # noqa: BLE001 - callers turn this into a failed result
            log("llm_error", {**info, "error": repr(exc)})
            return JsonReply(error=f"{type(exc).__name__}: {exc}", call_failed=True)

        log(
            "llm_response",
            {
                **info,
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
            raw = extract_json_object(content, repairs)
            value = validate(raw)
        except ValueError as exc:
            error = str(exc)
            log(invalid_event, {**info, "problem": error})
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
            log(repaired_event, {**info, "repairs": repairs})
        return JsonReply(value=value, raw=raw, repairs=repairs)
    return JsonReply(error=error)
