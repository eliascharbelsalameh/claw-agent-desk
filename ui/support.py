"""Plain-Python helpers for the Streamlit app, kept out of the app script so
they can be tested without Streamlit: symbol parsing, credential presence,
loading Windows user-level env vars, and summarizing a trace file."""
from __future__ import annotations

import os
import re
from typing import Any, Callable, Iterable

CREDENTIAL_NAMES = (
    "ALPACA_API_KEY_ID",
    "ALPACA_API_SECRET_KEY",
    "FRED_API_KEY",
    "FINNHUB_API_KEY",
    "SEC_EDGAR_USER_AGENT",
    "NVIDIA_API_KEY",
)

# Plain US tickers, plus class shares like BRK.B.
_SYMBOL_RE = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")

REC_COLORS = {"buy": "green", "hold": "blue", "avoid": "red"}
OUTCOME_COLORS = {"agree": "green", "critic": "orange", "abort": "red", "deferred": "violet"}


# Trace reading lives with the logger (the scheduler uses it too).
from agents.trace import read_trace, summarize_trace  # noqa: E402,F401


def parse_symbols(text: str) -> tuple[list[str], list[str]]:
    """Comma/space separated tickers -> (valid unique symbols in order, rejected tokens)."""
    valid, rejected = [], []
    for token in re.split(r"[\s,;]+", text.upper()):
        if not token:
            continue
        if _SYMBOL_RE.match(token):
            if token not in valid:
                valid.append(token)
        else:
            rejected.append(token)
    return valid, rejected


def credential_status(names: Iterable[str] = CREDENTIAL_NAMES) -> dict[str, bool]:
    """Which credentials are present in this process - never their values."""
    return {name: bool(os.environ.get(name)) for name in names}


def _read_windows_user_env(name: str) -> str | None:
    import winreg  # Windows only

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            return winreg.QueryValueEx(key, name)[0]
    except FileNotFoundError:
        return None


def load_user_env(
    names: Iterable[str] = CREDENTIAL_NAMES,
    reader: Callable[[str], str | None] | None = None,
) -> list[str]:
    """Fill missing credentials from the Windows user environment.

    The credentials live as user-level env vars (see CLAUDE.md), and a
    process started from a shell that predates `setx` doesn't see them.
    Only missing names are read, values are never returned or printed, and
    this is a no-op off Windows or when CLAW_DESK_NO_REGISTRY is set (tests).
    Returns the names that were filled in.
    """
    if reader is None:
        if os.name != "nt" or os.environ.get("CLAW_DESK_NO_REGISTRY"):
            return []
        reader = _read_windows_user_env
    filled = []
    for name in names:
        if os.environ.get(name):
            continue
        value = reader(name)
        if value:
            os.environ[name] = value
            filled.append(name)
    return filled


def event_row(event: dict[str, Any]) -> dict[str, Any]:
    """One trace event flattened for a table: who, what, and a short gist."""
    gist = ""
    kind = event.get("event")
    if kind == "llm_response":
        usage = event.get("usage") or {}
        gist = f"{usage.get('total_tokens', '?')} tokens, finish={event.get('finish_reason')}"
    elif kind in ("llm_error", "fallback"):
        gist = str(event.get("error") or event.get("reason") or "")[:160]
        if kind == "fallback":
            gist = f"{event.get('from_model')} -> {event.get('to_model')}: {gist}"
    elif kind == "verdict":
        gist = f"{event.get('recommendation')} {event.get('confidence')}"
    elif kind == "decision":
        gist = f"{event.get('outcome')} {event.get('recommendation') or ''} - {event.get('reason', '')}"[:160]
    elif kind == "final":
        gist = f"{event.get('outcome')} {event.get('recommendation') or ''} - {event.get('reason', '')}"[:160]
    elif kind == "critique":
        gist = f"{len(event.get('challenges') or [])} challenges"
    elif kind in ("verdict_invalid", "critique_invalid", "briefing_rejected"):
        gist = str(event.get("problem") or event.get("problems") or "")[:160]
    return {
        "time": str(event.get("ts", ""))[11:19],
        "symbol": event.get("symbol", ""),
        "agent": event.get("agent", ""),
        "event": kind,
        "model": event.get("model") or event.get("from_model") or "",
        "detail": gist,
    }
