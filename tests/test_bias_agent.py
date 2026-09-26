"""Bias/sentiment agents and the bias gate (spec section 3, step 4)."""
import json

import pytest

from agents.analyst_agent import AnalystVerdict
from agents.bias_agent import (
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


def _gate(r1, r2, previous=None, agents=None):
    agents = agents or {"bias_1": ScriptedBias("bias_1", r1), "bias_2": ScriptedBias("bias_2", r2)}
    return bias_gate(_ctx(), _verdicts(), agents, previous=previous), agents


@pytest.mark.parametrize("r1, r2, outcome", [
    ("pass", "pass", PASSED),
    ("flag", "pass", PASSED),     # one flag is recorded, but a veto needs both
    ("flag", "flag", VETOED),
    ("GARBLED", "pass", FAILED),  # a check that couldn't be done is not a pass
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
    _, agents = _gate(["pass"], ["pass"])
    assert agents["bias_2"].excludes == [{"bias_1/m"}]


def test_deferred_gate_resumes_keeping_the_check_that_came_back():
    first, agents = _gate(["DOWN", "flag"], ["flag"])
    assert first.outcome == DEFERRED
    second, _ = _gate(None, None, previous=first, agents=agents)
    assert second.outcome == VETOED
    assert len(agents["bias_2"].results) == 0 and agents["bias_2"].excludes == [set()]  # asked once only


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
