import json

import pytest

from agents.analyst_agent import AnalystVerdict
from agents.cross_check import ABORT, AGREE, CRITIC, cross_check
from agents.trace import TraceLogger


def _v(role, rec, model=None, confidence=0.7, error=None, symbol="AAPL"):
    return AnalystVerdict(
        symbol=symbol,
        role=role,
        model=model or f"lab-{role}/model",
        generated_at="2026-09-25T12:00:00+00:00",
        recommendation=None if error else rec,
        confidence=None if error else confidence,
        thesis=None if error else "t",
        evidence_check={} if error else {"matches_source": 3},
        error=error,
    )


@pytest.mark.parametrize(
    "rec_1, rec_2, outcome, recommendation",
    [
        ("buy", "buy", AGREE, "buy"),
        ("hold", "hold", AGREE, "hold"),
        ("avoid", "avoid", AGREE, "avoid"),
        ("buy", "hold", CRITIC, None),
        ("hold", "buy", CRITIC, None),
        ("buy", "avoid", ABORT, None),
        ("avoid", "buy", ABORT, None),
        ("hold", "avoid", ABORT, None),
        ("avoid", "hold", ABORT, None),
    ],
)
def test_decision_table(rec_1, rec_2, outcome, recommendation):
    result = cross_check(_v("analyst_1", rec_1), _v("analyst_2", rec_2))
    assert result.outcome == outcome
    assert result.recommendation == recommendation


def test_confidence_never_changes_the_outcome():
    low = cross_check(_v("analyst_1", "buy", confidence=0.05), _v("analyst_2", "buy", confidence=0.1))
    high = cross_check(_v("analyst_1", "buy", confidence=0.99), _v("analyst_2", "buy", confidence=0.95))
    assert low.outcome == high.outcome == AGREE


@pytest.mark.parametrize("failed_role", ["analyst_1", "analyst_2"])
def test_failed_verdict_aborts_even_if_the_other_says_buy(failed_role):
    a = _v("analyst_1", "buy", error="ConnectionError" if failed_role == "analyst_1" else None)
    b = _v("analyst_2", "buy", error="ConnectionError" if failed_role == "analyst_2" else None)
    result = cross_check(a, b)
    assert result.outcome == ABORT and result.recommendation is None
    assert failed_role in result.reason


def test_both_failed_aborts():
    result = cross_check(_v("analyst_1", None, error="x"), _v("analyst_2", None, error="y"))
    assert result.outcome == ABORT
    assert "analyst_1, analyst_2" in result.reason


def test_same_model_aborts_as_not_independent():
    result = cross_check(_v("analyst_1", "buy", model="same/m"), _v("analyst_2", "buy", model="same/m"))
    assert result.outcome == ABORT
    assert "not be independent" in result.reason


def test_rejects_mismatched_symbols_or_roles():
    with pytest.raises(ValueError, match="different symbols"):
        cross_check(_v("analyst_1", "buy"), _v("analyst_2", "buy", symbol="MSFT"))
    with pytest.raises(ValueError, match="same role"):
        cross_check(_v("analyst_1", "buy"), _v("analyst_1", "buy", model="other/m"))


def test_result_carries_both_verdicts_and_is_logged(tmp_path):
    trace = TraceLogger(tmp_path / "t.jsonl")
    result = cross_check(_v("analyst_1", "buy", confidence=0.8), _v("analyst_2", "hold"), trace=trace)

    assert set(result.verdicts) == {"analyst_1", "analyst_2"}
    assert result.verdicts["analyst_1"]["confidence"] == 0.8
    assert "analyst_1 says buy, analyst_2 says hold" in result.reason

    record = json.loads(trace.path.read_text(encoding="utf-8").splitlines()[0])
    assert record["agent"] == "cross_check" and record["event"] == "decision"
    assert record["outcome"] == CRITIC and record["symbol"] == "AAPL"
