import json

from ui.support import (
    event_row,
    load_user_env,
    parse_symbols,
    read_trace,
    summarize_trace,
)


def test_parse_symbols_dedupes_keeps_order_and_rejects_junk():
    assert parse_symbols("aapl, MSFT msft;nvda") == (["AAPL", "MSFT", "NVDA"], [])
    assert parse_symbols("BRK.B, TOOLONGX, 1234") == (["BRK.B"], ["TOOLONGX", "1234"])
    assert parse_symbols("  ") == ([], [])


def test_load_user_env_fills_only_missing_names_and_reports_which(monkeypatch):
    monkeypatch.setenv("A_SET", "already")
    monkeypatch.delenv("B_MISSING", raising=False)
    monkeypatch.delenv("C_ABSENT", raising=False)
    seen = []

    def reader(name):
        seen.append(name)
        return {"B_MISSING": "from-registry"}.get(name)

    filled = load_user_env(["A_SET", "B_MISSING", "C_ABSENT"], reader=reader)
    assert filled == ["B_MISSING"]
    assert seen == ["B_MISSING", "C_ABSENT"]  # a set var is never read
    import os
    assert os.environ["A_SET"] == "already" and os.environ["B_MISSING"] == "from-registry"


def test_load_user_env_is_off_when_disabled(monkeypatch):
    monkeypatch.setenv("CLAW_DESK_NO_REGISTRY", "1")
    monkeypatch.delenv("B_MISSING", raising=False)
    assert load_user_env(["B_MISSING"]) == []


def test_read_and_summarize_trace(tmp_path):
    path = tmp_path / "t.jsonl"
    lines = [
        {"ts": "2026-09-25T10:00:00", "agent": "analyst_1", "event": "llm_response", "model": "old",
         "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}},
        {"ts": "2026-09-25T12:00:00", "agent": "analyst_1", "event": "llm_response", "model": "m",
         "usage": {"prompt_tokens": 70, "completion_tokens": 30, "total_tokens": 100}},
        {"ts": "2026-09-25T12:01:00", "agent": "critic", "event": "llm_error", "model": "c", "error": "x"},
        {"ts": "2026-09-25T12:02:00", "agent": "critic", "event": "fallback", "from_model": "c", "to_model": "d",
         "reason": "ConnectionError"},
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines) + '\n{"cut sho', encoding="utf-8")
    events = read_trace(path)
    assert len(events) == 4  # the truncated last line is skipped
    s = summarize_trace(events, since="2026-09-25T11:00:00")
    assert (s["llm_calls"], s["total_tokens"], s["llm_errors"], s["fallbacks"]) == (1, 100, 1, 1)
    assert s["calls_by_model"] == {"m": 1}
    assert event_row(events[3])["detail"].startswith("c -> d: ConnectionError")
    assert event_row(events[1])["detail"] == "100 tokens, finish=None"
