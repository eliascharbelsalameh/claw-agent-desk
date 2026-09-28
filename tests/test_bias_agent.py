"""Bias/sentiment agents and the bias gate (spec section 3, step 4)."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from agents.analyst_agent import AnalystVerdict
from agents.bias_agent import (
    BIAS_SOLO_AFTER,
    DEFERRED,
    FAILED,
    PASSED,
    SYSTEM_PROMPT,
    VETOED,
    BiasAgent,
    BiasCheck,
    bias_gate,
    validate_bias_check,
)
from agents.macro_agent import StockContext
from agents.pipeline import DeskPipeline
from agents.portfolio import Decision
from data_layer.llm_client import ModelHealth
from tests.fakes import FakeAnalyst, FakeBias, FakeCritic, FakeMacro
from tests.test_backups import DEAD, RoutedLlm


def _ctx():
    return StockContext(symbol="MSFT", generated_at="t", macro={}, price={"last_close": 516.16},
                        news=[{"index": 0, "headline": "Microsoft wins a contract", "age_hours": 90.0}])


def _verdicts():
    return {role: AnalystVerdict(symbol="MSFT", role=role, model=model, generated_at="t", recommendation="buy",
                                 confidence=0.7, thesis="t", drivers=["d"], risks=["r"])
            for role, model in (("analyst_1", "a1/m"), ("analyst_2", "a2/m"))}


def _reply(verdict="pass", sentiment=0.3, stale=False):
    return json.dumps({
        "news_sentiment": sentiment,
        "checks": {"news_driven": {"skewed": verdict == "flag", "why": "w"},
                   "stale_news": {"skewed": stale, "why": "w", "items": [0] if stale else []},
                   "trend_chasing": {"skewed": False, "why": "w"}},
        "verdict": verdict, "reason": f"{verdict} because",
        "evidence": [{"fact": "price.last_close", "value": 516.16, "why": "price"}],
    })


def test_validate_bias_check_normalizes_and_rejects():
    out = validate_bias_check(json.loads(_reply(verdict="FLAG", stale=True)))
    assert out["verdict"] == "flag" and out["checks"]["stale_news"] == {"skewed": True, "why": "w", "items": [0]}
    for bad, message in (({"verdict": "maybe"}, "verdict"),
                         ({**json.loads(_reply()), "news_sentiment": 3}, "news_sentiment"),
                         ({**json.loads(_reply()), "checks": {"news_driven": {"skewed": "yes"}}}, "checks"),
                         ({**json.loads(_reply()), "reason": ""}, "reason")):
        with pytest.raises(ValueError, match=message):
            validate_bias_check(bad)


def test_prompt_scores_sentiment_itself_and_gives_no_recommendation():
    assert "score the tone of the company news in the packet yourself" in SYSTEM_PROMPT
    assert "you give no recommendation" in SYSTEM_PROMPT
    assert "does not make the underlying claim true" in SYSTEM_PROMPT  # EVIDENCE_STATUS_NOTE


def test_review_checks_evidence_and_never_uses_an_analysts_model():
    llm = RoutedLlm({"b2/m": _reply()})
    agent = BiasAgent(llm, "bias_1", models=["a1/m", "b2/m"], health=ModelHealth())
    check = agent.review(_ctx(), _verdicts())
    assert check.ok and check.model == "b2/m" and check.verdict == "pass"
    assert check.evidence_check["matches_source"] == 1
    assert [m for m, _ in llm.calls] == ["b2/m"]


def test_bias_2_gets_the_larger_token_budget():
    # Sept 28, 2026: lightning spent all 8,192 tokens reasoning on LLY, twice
    for role, budget in (("bias_1", 8192), ("bias_2", 16384)):
        llm = RoutedLlm({"b/m": _reply()})
        BiasAgent(llm, role, models=["b/m"], health=ModelHealth()).review(_ctx(), _verdicts())
        assert llm.calls[0][1]["max_tokens"] == budget


def test_review_failures_say_whether_to_retry():
    down = BiasAgent(RoutedLlm({"b/m": DEAD}), "bias_1", models=["b/m"], health=ModelHealth()).review(
        _ctx(), _verdicts())
    assert not down.ok and down.call_failed
    garbled = BiasAgent(RoutedLlm({"b/m": "no json"}), "bias_1", models=["b/m"], health=ModelHealth()).review(
        _ctx(), _verdicts())
    assert not garbled.ok and not garbled.call_failed


class ScriptedBias:
    def __init__(self, role, results):
        self.role, self.results, self.excludes = role, list(results), []

    def review(self, ctx, verdicts, *, exclude=()):
        self.excludes.append(set(exclude))
        kind = self.results.pop(0)
        check = BiasCheck(symbol=ctx.symbol, role=self.role, model=f"{self.role}/m", generated_at="t")
        if kind == "DOWN":
            check.error, check.call_failed = "ConnectionError", True
        elif kind == "GARBLED":
            check.error = "unparseable bias check: no JSON"
        else:
            check.verdict, check.news_sentiment, check.reason = kind, 0.2, f"{self.role} says {kind}"
        return check


T0 = datetime(2026, 9, 28, 12, 30, tzinfo=timezone.utc)


def _gate(r1, r2, previous=None, agents=None, at=T0, solo_after=BIAS_SOLO_AFTER):
    agents = agents or {"bias_1": ScriptedBias("bias_1", r1), "bias_2": ScriptedBias("bias_2", r2)}
    result = bias_gate(_ctx(), _verdicts(), agents, previous=previous, solo_after=solo_after, now=lambda: at)
    return result, agents


@pytest.mark.parametrize("r1, r2, outcome", [
    ("pass", "pass", PASSED),
    ("pass", "DOWN", PASSED),     # a bias_1 pass clears the gate: bias_2 isn't asked
    ("flag", "pass", PASSED),     # one flag is recorded, but a veto needs both
    ("flag", "flag", VETOED),
    ("flag", "DOWN", DEFERRED),
    ("flag", "GARBLED", FAILED),  # a check that couldn't be done is not a pass
    ("GARBLED", "pass", FAILED),
    ("DOWN", "pass", DEFERRED),
    ("GARBLED", "DOWN", FAILED),  # unusable is final, retrying the other can't help
])
def test_gate_rules(r1, r2, outcome):
    result, _ = _gate([r1], [r2])
    assert result.outcome == outcome
    assert result.gate["passed"] is (outcome == PASSED)


def test_single_flag_is_kept_in_the_reason():
    result, _ = _gate(["flag"], ["pass"])
    assert "only bias_1 flags it (bias_1 says flag)" in result.reason


def test_bias_2_excludes_the_model_bias_1_used():
    _, agents = _gate(["flag"], ["pass"])
    assert agents["bias_2"].excludes == [{"bias_1/m"}]


def test_a_bias_1_pass_clears_the_gate_without_asking_bias_2():
    result, agents = _gate(["pass"], ["DOWN"])
    assert result.outcome == PASSED and not result.solo and result.unreachable_since == {}
    assert result.reason == "passed: bias_1 passes it, which clears the gate (a veto needs both bias agents)"
    assert set(result.checks) == {"bias_1"} and agents["bias_2"].excludes == []


def test_a_late_bias_1_pass_keeps_bias_2s_earlier_flag_in_the_record():
    first, agents = _gate(["DOWN", "pass"], ["flag"])  # bias_2 is asked while bias_1 is down
    assert first.outcome == DEFERRED
    second, _ = _gate(None, None, previous=first, agents=agents)
    assert second.outcome == PASSED and set(second.checks) == {"bias_1", "bias_2"}
    assert second.reason.endswith("; only bias_2 flags it (bias_2 says flag)")
    assert agents["bias_2"].excludes == [set()]  # asked once only


def test_deferred_gate_resumes_keeping_the_check_that_came_back():
    first, agents = _gate(["DOWN", "flag"], ["flag"])
    assert first.outcome == DEFERRED
    second, _ = _gate(None, None, previous=first, agents=agents)
    assert second.outcome == VETOED
    assert len(agents["bias_2"].results) == 0 and agents["bias_2"].excludes == [set()]  # asked once only


# --- a bias_1 flag vetoes alone after a long bias_2 outage (decided Sept 26, 2026) ---

def test_bias_1_flag_vetoes_alone_once_bias_2_has_been_unreachable_long_enough():
    first, agents = _gate(["flag"], ["DOWN", "DOWN", "DOWN"])
    assert first.outcome == DEFERRED and not first.solo
    assert first.unreachable_since == {"bias_2": T0.isoformat()}
    assert first.reason.endswith("; bias_1's flag vetoes the buy alone if bias_2 is still unreachable at 14:30 UTC")

    # a retry just short of BIAS_SOLO_AFTER still waits, and the wait keeps its start
    almost = T0 + BIAS_SOLO_AFTER - timedelta(minutes=1)
    second, _ = _gate(None, None, previous=first, agents=agents, at=almost)
    assert second.outcome == DEFERRED and second.unreachable_since == first.unreachable_since

    third, _ = _gate(None, None, previous=second, agents=agents, at=T0 + BIAS_SOLO_AFTER)
    assert third.outcome == VETOED and third.solo and third.gate["passed"] is False
    assert third.reason == ("bias_2 could not be reached for 2h 00m (since 12:30 UTC), so bias_1 decides alone "
                            "and flags the buy: bias_1 says flag")
    assert agents["bias_1"].excludes == [set()]  # asked once: its check was kept
    assert agents["bias_2"].results == []        # asked on every pass


def test_alone_a_single_flag_vetoes_which_it_never_does_with_both_agents():
    both, _ = _gate(["flag"], ["pass"])
    assert both.outcome == PASSED
    first, agents = _gate(["flag"], ["DOWN", "DOWN"])
    alone, _ = _gate(None, None, previous=first, agents=agents, at=T0 + timedelta(hours=3))
    assert alone.outcome == VETOED and "flags the buy: bias_1 says flag" in alone.reason


def test_bias_2_answering_late_brings_back_the_two_agent_rule():
    first, agents = _gate(["flag"], ["DOWN", "pass"])
    late, _ = _gate(None, None, previous=first, agents=agents, at=T0 + timedelta(hours=3))
    assert late.outcome == PASSED and not late.solo and late.unreachable_since == {}
    assert "a veto needs both bias agents" in late.reason


def test_bias_2_never_decides_alone():
    first, agents = _gate(["DOWN", "DOWN"], ["flag"])
    assert first.outcome == DEFERRED and "alone" not in first.reason
    later, _ = _gate(None, None, previous=first, agents=agents, at=T0 + timedelta(hours=5))
    assert later.outcome == DEFERRED and not later.solo
    assert later.unreachable_since == {"bias_1": T0.isoformat()}


def test_solo_rule_off_keeps_waiting():
    first, agents = _gate(["flag"], ["DOWN", "DOWN"], solo_after=None)
    later, _ = _gate(None, None, previous=first, agents=agents, at=T0 + timedelta(hours=5), solo_after=None)
    assert later.outcome == DEFERRED and not later.solo and "alone" not in later.reason


def test_pipeline_runs_the_gate_only_on_agreed_buys_and_decisions_carry_it():
    FakeBias.flag = {"MSFT"}
    try:
        desk = DeskPipeline(FakeMacro(), analysts={r: FakeAnalyst(None, r) for r in ("analyst_1", "analyst_2")},
                            critic=FakeCritic(None),
                            bias_agents={r: FakeBias(None, r) for r in ("bias_1", "bias_2")})
        runs = {r.symbol: r for r in desk.run(["MSFT", "NVDA"])}
    finally:
        FakeBias.flag = set()
    assert runs["NVDA"].bias is None  # agreed hold: nothing to check
    assert runs["MSFT"].bias.outcome == VETOED
    decision = Decision.from_run(runs["MSFT"], "2026-09-28", "t")
    assert decision.agreed == "buy" and decision.blocked
    assert decision.gates[0]["gate"] == "bias" and decision.gates[0]["passed"] is False
