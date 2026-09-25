import json
from datetime import datetime, timezone

import pytest

from agents.analyst_agent import (
    _UNRESOLVED,
    PARSE_ATTEMPTS,
    AnalystAgent,
    check_evidence,
    extract_json_object,
    repair_json_quotes,
    resolve_fact_path,
    validate_verdict,
    values_match,
)
from agents.macro_agent import StockContext
from agents.trace import TraceLogger

NOW = datetime(2026, 9, 25, 15, 0, tzinfo=timezone.utc)


def _ctx():
    return StockContext(
        symbol="AAPL",
        generated_at=NOW.isoformat(),
        macro={"DGS10": {"label": "10y", "latest": 5.11, "date": "2026-09-23"}},
        price={"last_close": 335.88, "change_20d_pct": 7.15},
        relative_volume={"relative_volume": 0.6073944227327381, "feed": "iex"},
        fundamentals={"revenue": {"value": 109417000000, "period_end": "2026-06-27"}},
        news=[{"headline": "Apple stock is giving new CEO John Ternus plenty to smile about"}],
        briefing="MACRO: ...",
    )


def _verdict_json(**overrides):
    verdict = {
        "recommendation": "buy",
        "confidence": 0.7,
        "thesis": "Momentum and earnings support the case.",
        "drivers": ["20d momentum", "strong quarter"],
        "risks": ["yields rising"],
        "evidence": [
            {"fact": "price.change_20d_pct", "value": 7.15, "why": "momentum"},
            {"fact": "fundamentals.revenue.value", "value": "109,417,000,000", "why": "scale"},
            {"fact": "relative_volume.relative_volume", "value": 0.61, "why": "quiet volume"},
        ],
        "data_concerns": [],
    }
    verdict.update(overrides)
    return json.dumps(verdict)


class FakeLlm:
    def __init__(self, contents, fail=False):
        self.contents = list(contents)
        self.fail = fail
        self.calls = []

    def chat_completion(self, model, messages, **kwargs):
        self.calls.append((model, messages, kwargs))
        if self.fail:
            raise ConnectionError("build dropped")
        return {
            "choices": [{
                "message": {"content": self.contents.pop(0), "reasoning_content": "thinking"},
                "finish_reason": "stop",
            }],
            "usage": {"total_tokens": 99},
        }


def _analyst(llm, trace=None):
    return AnalystAgent(llm, "analyst_1", trace=trace, model="test/model", now=lambda: NOW)


# --- parsing ---

def test_extract_json_object_tolerates_fences_and_prose():
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json_object('Here you go: {"a": {"b": 2}} hope it helps') == {"a": {"b": 2}}
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json_object("I think it's a buy")
    with pytest.raises(ValueError, match="invalid JSON"):
        extract_json_object('{"a": 1,,}')


def test_validate_verdict_normalizes():
    fields = validate_verdict(json.loads(_verdict_json(recommendation=" BUY ", confidence=70)))
    assert fields["recommendation"] == "buy"
    assert fields["confidence"] == 0.7


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"recommendation": "strong buy"}, "recommendation"),
        ({"confidence": "high"}, "confidence"),
        ({"confidence": 150}, "confidence"),
        ({"thesis": ""}, "thesis"),
        ({"drivers": []}, "drivers"),
        ({"risks": "none"}, "risks"),
        ({"evidence": [{"fact": "price.last_close", "value": 1}]}, "at least 3"),
        ({"evidence": [{"fact": "a"}, {"fact": "b"}, {"fact": "c"}]}, "'fact' and 'value'"),
    ],
)
def test_validate_verdict_rejects(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_verdict(json.loads(_verdict_json(**overrides)))


# --- grounding ---

def test_resolve_fact_path():
    facts = _ctx().facts()
    assert resolve_fact_path(facts, "macro.DGS10.latest") == 5.11
    assert resolve_fact_path(facts, "news[0].headline").startswith("Apple stock")
    assert resolve_fact_path(facts, "fundamentals.eps.value") is _UNRESOLVED
    assert resolve_fact_path(facts, "news[5].headline") is _UNRESOLVED
    assert resolve_fact_path(facts, "price[0]") is _UNRESOLVED
    assert resolve_fact_path(facts, "data_gaps") == []  # present-but-empty is not missing


def test_values_match():
    assert values_match(0.61, 0.6073944227327381)  # rounding is fine
    assert values_match("109,417,000,000", 109417000000)
    assert values_match("7.15%", 7.15)
    assert not values_match(8.0, 7.15)
    assert values_match("Ternus plenty to smile", "Apple stock is giving new CEO John Ternus plenty to smile about")
    assert not values_match("", "anything")
    assert not values_match("--", "anything")  # punctuation-only is not a quote
    # models rewrite punctuation when quoting
    assert values_match(
        "Musk Teases 3 Nvidia Chip Waves \u2011\u2011 Google Books",
        "SPCX Climbs As Musk Teases 3 Nvidia Chip Waves \u2014 Google Books SpaceX Ride",
    )


def test_check_evidence_flags_mismatch_and_unknown_paths():
    evidence = [
        {"fact": "price.last_close", "value": 335.88},
        {"fact": "price.change_20d_pct", "value": 12.0},
        {"fact": "fundamentals.free_cash_flow.value", "value": 1},
    ]
    counts = check_evidence(evidence, _ctx().facts())
    assert counts == {"verified": 1, "wrong_index": 0, "mismatch": 1, "unknown_path": 1}
    assert evidence[1]["status"] == "mismatch" and evidence[1]["actual"] == 7.15
    assert evidence[2]["status"] == "unknown_path"


def test_check_evidence_distinguishes_wrong_index_from_fabrication():
    ctx = _ctx()
    ctx.news = [{"headline": "Synack expands pen testing"}, {"headline": "Goldman doubles down on MSFT"}]
    evidence = [
        {"fact": "news[0].headline", "value": "Goldman doubles down on MSFT"},  # real, wrong slot
        {"fact": "news[7].headline", "value": "Goldman doubles down on MSFT"},  # out of range, real
        {"fact": "news[0].headline", "value": "Apple buys the moon"},           # fabricated
    ]
    counts = check_evidence(evidence, ctx.facts())
    assert counts == {"verified": 0, "wrong_index": 2, "mismatch": 1, "unknown_path": 0}
    assert evidence[0]["found_at"] == "news[1].headline"
    assert evidence[1]["found_at"] == "news[1].headline"
    assert evidence[2]["actual"] == "Synack expands pen testing"


# --- agent ---

def test_analyze_returns_checked_verdict_and_logs(tmp_path):
    trace = TraceLogger(tmp_path / "t.jsonl")
    llm = FakeLlm([_verdict_json()])
    verdict = _analyst(llm, trace).analyze(_ctx())

    assert verdict.ok
    assert verdict.recommendation == "buy" and verdict.confidence == 0.7
    assert verdict.evidence_check == {
        "verified": 3, "wrong_index": 0, "mismatch": 0, "unknown_path": 0,
    }
    assert verdict.role == "analyst_1" and verdict.model == "test/model"

    model, messages, kwargs = llm.calls[0]
    assert "Source facts (authoritative" in messages[1]["content"]
    assert "IEX" in messages[0]["content"] and "only recommend \"buy\"" in messages[0]["content"]
    assert "Do not claim growth, valuation" in messages[0]["content"]
    assert "An empty data_gaps does not mean there are no data concerns" in messages[0]["content"]
    assert '"limitations"' in messages[1]["content"]
    assert kwargs["max_tokens"] >= 4096

    records = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]
    assert [r["event"] for r in records] == ["llm_request", "llm_response", "verdict"]
    assert all(r["agent"] == "analyst_1" for r in records)
    assert records[1]["reasoning_content"] == "thinking"
    assert records[2]["recommendation"] == "buy"


def test_invalid_reply_gets_one_correction_turn():
    llm = FakeLlm(["I'd say buy, it's great", _verdict_json(recommendation="hold")])
    verdict = _analyst(llm).analyze(_ctx())

    assert verdict.ok and verdict.recommendation == "hold"
    retry_messages = llm.calls[1][1]
    assert retry_messages[-2] == {"role": "assistant", "content": "I'd say buy, it's great"}
    assert "could not be used: no JSON object" in retry_messages[-1]["content"]


def test_persistently_invalid_reply_is_a_failed_verdict():
    llm = FakeLlm(['{"recommendation": "moon"}'] * PARSE_ATTEMPTS)
    verdict = _analyst(llm).analyze(_ctx())

    assert not verdict.ok
    assert verdict.recommendation is None
    assert "recommendation" in verdict.error
    assert len(llm.calls) == PARSE_ATTEMPTS


def test_llm_failure_is_a_failed_verdict_not_an_exception():
    verdict = _analyst(FakeLlm([], fail=True)).analyze(_ctx())
    assert not verdict.ok
    assert verdict.error.startswith("ConnectionError")


# --- JSON quote repair (shapes seen live from nemotron-3-super) ---

BROKEN_REPLY = """{
  "recommendation": "buy",
  "confidence": 0.78,
  "thesis": "Growth is strong.",
  "drivers": ["momentum"],
  "risks": [
    "Continued heavy AI capex could pressure cash flow, as Burry’s alarm notes.
    "Rising yields."
  ],
  "evidence": [
    {
      "fact": "price.change_20d_pct",
      "value": 7.15,
      "why": Indicates strong momentum, per the "20d" window.
    },
    {
      "fact": "fundamentals.revenue.value",
      "value": "109,417,000,000",
      "why": Shows scale."
    },
    {
      "fact": "relative_volume.relative_volume",
      "value": 0.61,
      "why": Quiet volume,
      "note": "extra"
    }
  ],
  "data_concerns": []
}"""


def test_repair_fixes_the_three_live_shapes_without_changing_words():
    repairs = []
    obj = extract_json_object(BROKEN_REPLY, repairs)
    fields = validate_verdict(obj)

    assert fields["risks"] == [
        "Continued heavy AI capex could pressure cash flow, as Burry’s alarm notes.",
        "Rising yields.",
    ]
    whys = [e["why"] for e in fields["evidence"]]
    assert whys == ['Indicates strong momentum, per the "20d" window.', "Shows scale.", "Quiet volume"]
    assert fields["evidence"][2]["note"] == "extra"  # the comma separator was kept
    assert len(repairs) == 4
    assert repairs[0] == "line 7: closed an unterminated string"
    assert 'quoted the unquoted value of "why"' in repairs[1]


def test_valid_json_is_never_touched():
    repairs = []
    extract_json_object(_verdict_json(), repairs)
    assert repairs == []
    assert repair_json_quotes('{\n  "a": 1,\n  "b": "x"\n}')[1] == []


def test_unrepairable_json_still_fails():
    with pytest.raises(ValueError, match="invalid JSON"):
        extract_json_object('{\n  "a": [1, 2,,\n}')


def test_repaired_reply_is_accepted_and_logged(tmp_path):
    trace = TraceLogger(tmp_path / "t.jsonl")
    llm = FakeLlm([BROKEN_REPLY])
    verdict = _analyst(llm, trace).analyze(_ctx())

    assert verdict.ok and verdict.recommendation == "buy"
    assert len(llm.calls) == 1  # no correction turn needed
    assert len(verdict.json_repairs) == 4
    events = [json.loads(l)["event"] for l in trace.path.read_text(encoding="utf-8").splitlines()]
    assert events == ["llm_request", "llm_response", "verdict_repaired", "verdict"]
