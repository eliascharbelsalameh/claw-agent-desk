import json
from datetime import datetime, timezone

import pytest

from agents.analyst_agent import AnalystAgent, AnalystVerdict, parse_critique_responses
from agents.critic_agent import MAX_CHALLENGES_PER_ANALYST, CriticAgent, Critique, validate_critique
from agents.critic_loop import (
    AGREED_BUY,
    MAX_ROUNDS,
    SPLIT,
    CriticLoopResult,
    needs_critic,
    run_critic_loop,
)
from agents.cross_check import ABORT, AGREE, CRITIC, DEFERRED, CrossCheckResult
from agents.macro_agent import StockContext
from agents.trace import TraceLogger

NOW = datetime(2026, 9, 25, 15, 0, tzinfo=timezone.utc)
ROLES = ("analyst_1", "analyst_2")


def _ctx():
    return StockContext(
        symbol="AAPL",
        generated_at=NOW.isoformat(),
        macro={"DGS10": {"label": "10y", "latest": 5.11, "date": "2026-09-23"}},
        price={"last_close": 335.88, "change_20d_pct": 7.15},
        relative_volume={"relative_volume": 0.61, "feed": "iex"},
        news=[{"index": 0, "headline": "Apple stock is giving new CEO John Ternus plenty to smile about"}],
    )


def _verdict(role, rec, *, error=None, model=None, review_round=0, previous=None, call_failed=False):
    return AnalystVerdict(
        symbol="AAPL",
        role=role,
        model=model or f"lab-{role}/model",
        generated_at=NOW.isoformat(),
        recommendation=None if error else rec,
        confidence=None if error else 0.7,
        thesis=None if error else f"{role} thinks {rec}",
        drivers=[] if error else ["momentum"],
        risks=[] if error else ["yields"],
        evidence=[] if error else [{"fact": "price.change_20d_pct", "value": 7.15, "why": "m", "status": "matches_source"}],
        review_round=review_round,
        previous_recommendation=previous,
        error=error,
        call_failed=call_failed,
    )


def _completion(content):
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": {"total_tokens": 1}}


class ScriptedLlm:
    def __init__(self, contents, fail=False):
        self.contents = list(contents)
        self.fail = fail
        self.calls = []

    def chat_completion(self, model, messages, **kwargs):
        self.calls.append((model, messages, kwargs))
        if self.fail:
            raise ConnectionError("build dropped")
        return _completion(self.contents.pop(0))


def _critique_json(challenges=None, assessment="They diverge on momentum."):
    return json.dumps({"assessment": assessment, "challenges": challenges if challenges is not None else [
        {"to": "analyst_1", "point": "Ignores low volume", "why": "Relative volume is 0.61",
         "fact": "relative_volume.relative_volume", "value": 0.61},
        {"to": "analyst_2", "point": "Understates momentum", "why": "20d change is 7.15%",
         "fact": "price.change_20d_pct", "value": 9.99},
    ]})


# --- critic agent ---

def test_validate_critique_caps_per_analyst_and_keeps_facts_only_with_values():
    many = [{"to": "analyst_1", "point": f"p{i}", "why": "w"} for i in range(MAX_CHALLENGES_PER_ANALYST + 2)]
    many.append({"to": "analyst_2", "point": "p", "why": "w", "fact": "price.last_close"})  # no value
    out = validate_critique({"assessment": "a", "challenges": many}, ROLES)
    assert len([c for c in out["challenges"] if c["to"] == "analyst_1"]) == MAX_CHALLENGES_PER_ANALYST
    assert out["dropped_challenges"] == 2
    assert "fact" not in out["challenges"][-1]


@pytest.mark.parametrize(
    "obj, message",
    [
        ({"challenges": []}, "assessment"),
        ({"assessment": "a", "challenges": "none"}, "challenges"),
        ({"assessment": "a", "challenges": [{"to": "analyst_3", "point": "p", "why": "w"}]}, "'to'"),
        ({"assessment": "a", "challenges": [{"to": "analyst_1", "point": "", "why": "w"}]}, "point"),
    ],
)
def test_validate_critique_rejects(obj, message):
    with pytest.raises(ValueError, match=message):
        validate_critique(obj, ROLES)


def test_empty_challenges_are_valid():
    assert validate_critique({"assessment": "Both hold up.", "challenges": []}, ROLES)["challenges"] == []


def test_review_checks_cited_facts_and_hides_model_and_confidence(tmp_path):
    trace = TraceLogger(tmp_path / "t.jsonl")
    llm = ScriptedLlm([_critique_json()])
    critic = CriticAgent(llm, trace=trace, model="critic/model", now=lambda: NOW)
    verdicts = {"analyst_1": _verdict("analyst_1", "buy"), "analyst_2": _verdict("analyst_2", "hold")}

    critique = critic.review(_ctx(), verdicts, review_round=1)

    assert critique.ok and critique.review_round == 1
    assert critique.evidence_check == {"matches_source": 1, "wrong_index": 0, "mismatch": 1, "unknown_path": 0}
    assert critique.for_role("analyst_2")[0]["status"] == "mismatch"
    prompt = llm.calls[0][1][1]["content"]
    assert "VERDICTS TO REVIEW (round 1)" in prompt and "analyst_1 thinks buy" in prompt
    assert "lab-analyst_1/model" not in prompt and "confidence" not in prompt.split("VERDICTS TO REVIEW")[1]
    system = llm.calls[0][1][0]["content"]
    assert "never give a recommendation of your own" in system
    assert "Apply the same standard to both analysts" in system
    events = [json.loads(l)["event"] for l in trace.path.read_text(encoding="utf-8").splitlines()]
    assert events == ["llm_request", "llm_response", "critique"]


def test_review_failure_is_a_failed_critique_not_an_exception():
    critic = CriticAgent(ScriptedLlm([], fail=True), model="critic/model")
    critique = critic.review(_ctx(), {"analyst_1": _verdict("analyst_1", "buy"),
                                      "analyst_2": _verdict("analyst_2", "hold")}, 1)
    assert not critique.ok and critique.error.startswith("ConnectionError")


# --- analyst re-vote ---

def _verdict_reply(rec, response=None):
    if response is None:
        response = [{"point": "Ignores low volume", "accept": True, "reason": "Fair; it lowers my confidence."}]
    return json.dumps({
        "recommendation": rec, "confidence": 0.6, "thesis": "Revised.", "drivers": ["d"], "risks": ["r"],
        "evidence": [{"fact": "price.change_20d_pct", "value": 7.15, "why": "w"},
                     {"fact": "price.last_close", "value": 335.88, "why": "w"},
                     {"fact": "macro.DGS10.latest", "value": 5.11, "why": "w"}],
        "data_concerns": [], "response_to_critique": response,
    })


def test_revise_shows_both_verdicts_and_challenges_and_records_the_flip():
    llm = ScriptedLlm([_verdict_reply("hold")])
    analyst = AnalystAgent(llm, "analyst_1", model="a1/model", now=lambda: NOW)
    own, other = _verdict("analyst_1", "buy"), _verdict("analyst_2", "hold")
    challenge = {"to": "analyst_1", "point": "Ignores low volume", "why": "0.61x baseline"}

    revised = analyst.revise(_ctx(), own, other, assessment="They diverge.", challenges_to_me=[challenge],
                             challenges_to_other=[], review_round=1)

    assert revised.ok and revised.recommendation == "hold"
    assert revised.previous_recommendation == "buy" and revised.changed
    assert revised.review_round == 1
    assert revised.response_to_critique == [
        {"point": "Ignores low volume", "accept": True, "reason": "Fair; it lowers my confidence."}
    ]
    assert revised.challenges_accepted == 1 and revised.challenges_rejected == 0
    assert own.recommendation == "buy"  # the previous verdict is untouched
    prompt = llm.calls[0][1][1]["content"]
    assert "REVIEW ROUND 1" in prompt and "You are analyst_1" in prompt
    assert "analyst_2 thinks hold" in prompt and "Ignores low volume" in prompt
    assert "Do not change it just to agree with the other analyst" in prompt
    assert "The critic is another model and can be wrong" in prompt
    assert "does not make the underlying claim true" in prompt
    assert '"to"' not in prompt.split("challenges to you:")[1].split("challenges to the other")[0]
    assert "a1/model" not in prompt and "lab-analyst_2/model" not in prompt


def test_revise_keeping_the_recommendation_is_not_a_change():
    analyst = AnalystAgent(ScriptedLlm([_verdict_reply("buy", response=["No.", "Still buy."])]), "analyst_1")
    revised = analyst.revise(_ctx(), _verdict("analyst_1", "buy"), _verdict("analyst_2", "hold"),
                             assessment="a", challenges_to_me=[], challenges_to_other=[], review_round=1)
    assert not revised.changed
    assert [r["reason"] for r in revised.response_to_critique] == ["No.", "Still buy."]
    assert revised.challenges_accepted == revised.challenges_rejected == 0


def test_revise_refuses_failed_inputs():
    analyst = AnalystAgent(ScriptedLlm([]), "analyst_1")
    with pytest.raises(ValueError):
        analyst.revise(_ctx(), _verdict("analyst_1", None, error="x"), _verdict("analyst_2", "buy"),
                       assessment="a", challenges_to_me=[], challenges_to_other=[], review_round=1)


# --- loop ---

class FakeCritic:
    """Fails in `fail_round`: unusably by default, or unreachable (a Build
    failure) with unreachable=True."""

    def __init__(self, fail_round=None, unreachable=False):
        self.fail_round = fail_round
        self.unreachable = unreachable
        self.rounds = []

    def review(self, ctx, verdicts, review_round):
        self.rounds.append((review_round, {r: v.recommendation for r, v in verdicts.items()}))
        c = Critique(symbol=ctx.symbol, model="critic/model", review_round=review_round,
                     generated_at=NOW.isoformat())
        if review_round == self.fail_round:
            c.error = "ConnectionError: dropped" if self.unreachable else "unparseable critique: no JSON"
            c.call_failed = self.unreachable
            return c
        c.assessment = f"round {review_round}"
        c.challenges = [{"to": "analyst_1", "point": "p1", "why": "w"}, {"to": "analyst_2", "point": "p2", "why": "w"}]
        return c


class FakeAnalyst:
    """Re-votes from a script: one recommendation per round, or 'FAIL' (an
    unusable answer) or 'DOWN' (its model unreachable)."""

    def __init__(self, role, script):
        self.role = role
        self.script = list(script)
        self.seen = []

    def revise(self, ctx, own, other, *, assessment, challenges_to_me, challenges_to_other, review_round):
        self.seen.append((review_round, own.recommendation, other.recommendation, len(challenges_to_me)))
        rec = self.script.pop(0)
        if rec == "FAIL":
            return _verdict(self.role, None, error="unparseable verdict: no JSON", review_round=review_round,
                            previous=own.recommendation)
        if rec == "DOWN":
            return _verdict(self.role, None, error="ConnectionError: dropped", review_round=review_round,
                            previous=own.recommendation, call_failed=True)
        return _verdict(self.role, rec, review_round=review_round, previous=own.recommendation)


def _run(first, script_1, script_2, *, critic=None, trigger=SPLIT, trace=None, max_rounds=MAX_ROUNDS):
    analysts = {"analyst_1": FakeAnalyst("analyst_1", script_1), "analyst_2": FakeAnalyst("analyst_2", script_2)}
    verdicts = {"analyst_1": _verdict("analyst_1", first[0]), "analyst_2": _verdict("analyst_2", first[1])}
    critic = critic or FakeCritic()
    result = run_critic_loop(_ctx(), verdicts, analysts, critic, trigger, trace=trace, max_rounds=max_rounds)
    return result, analysts, critic


def test_needs_critic():
    def cc(outcome, rec=None):
        return CrossCheckResult(symbol="AAPL", outcome=outcome, recommendation=rec, reason="r")
    assert needs_critic(cc(CRITIC)) == SPLIT
    assert needs_critic(cc(AGREE, "buy")) == AGREED_BUY
    assert needs_critic(cc(AGREE, "buy"), challenge_agreed_buys=False) is None
    assert needs_critic(cc(AGREE, "hold")) is None
    assert needs_critic(cc(AGREE, "avoid")) is None
    assert needs_critic(cc(ABORT)) is None


def test_split_resolved_in_first_round(tmp_path):
    trace = TraceLogger(tmp_path / "t.jsonl")
    result, analysts, _ = _run(("buy", "hold"), ["hold"], ["hold"], trace=trace)
    assert result.outcome == AGREE and result.recommendation == "hold"
    assert len(result.rounds) == 1
    assert result.final_verdicts["analyst_1"]["changed"] is True
    assert analysts["analyst_1"].seen == [(1, "buy", "hold", 1)]
    events = [(json.loads(l)["agent"], json.loads(l)["event"]) for l in trace.path.read_text(encoding="utf-8").splitlines()]
    assert events == [("critic_loop", "start"), ("cross_check", "decision"),
                      ("critic_loop", "round"), ("critic_loop", "final")]


def test_split_that_persists_aborts_after_max_rounds():
    result, _, critic = _run(("buy", "hold"), ["buy", "buy"], ["hold", "hold"])
    assert result.outcome == ABORT and result.recommendation is None
    assert "still split after 2 rounds" in result.reason
    assert [r for r, _ in critic.rounds] == [1, 2]
    # round 2 reviews the round-1 revisions, not the originals
    assert critic.rounds[1][1] == {"analyst_1": "buy", "analyst_2": "hold"}


def test_split_resolved_in_second_round():
    result, _, _ = _run(("buy", "hold"), ["buy", "buy"], ["hold", "buy"])
    assert result.outcome == AGREE and result.recommendation == "buy"
    assert len(result.rounds) == 2 and "after round 2" in result.reason


def test_split_that_becomes_a_contradiction_aborts():
    result, _, _ = _run(("buy", "hold"), ["buy"], ["avoid"])
    assert result.outcome == ABORT and "contradictory" in result.reason


def test_agreed_buy_that_survives_the_challenge_passes_after_one_round():
    result, _, critic = _run(("buy", "buy"), ["buy"], ["buy"], trigger=AGREED_BUY)
    assert result.outcome == AGREE and result.recommendation == "buy"
    assert result.trigger == AGREED_BUY and len(critic.rounds) == 1


def test_agreed_buy_both_downgraded_becomes_agreed_hold():
    result, _, _ = _run(("buy", "buy"), ["hold"], ["hold"], trigger=AGREED_BUY)
    assert result.outcome == AGREE and result.recommendation == "hold"


def test_critic_failure_aborts():
    result, analysts, _ = _run(("buy", "buy"), [], [], critic=FakeCritic(fail_round=1), trigger=AGREED_BUY)
    assert result.outcome == ABORT and "critic failed in round 1" in result.reason
    assert analysts["analyst_1"].seen == []  # no re-votes without a critique
    assert result.resume is None


def test_unreachable_critic_defers_then_resumes_at_the_same_round(tmp_path):
    trace = TraceLogger(tmp_path / "t.jsonl")
    result, analysts, _ = _run(("buy", "hold"), ["hold"], ["hold"],
                               critic=FakeCritic(fail_round=1, unreachable=True), trace=trace)
    assert result.outcome == DEFERRED and result.recommendation is None
    assert "critic could not be reached in round 1" in result.reason
    assert result.resume["round"] == 1 and result.resume["critique"] is None
    assert analysts["analyst_1"].seen == []

    # next cycle: from the serialized result, as the scheduler stores it
    stored = CriticLoopResult(**json.loads(json.dumps(result.to_dict())))
    critic = FakeCritic()
    resumed = run_critic_loop(_ctx(), {}, analysts, critic, "ignored", trace=trace, resume_from=stored)
    assert resumed.outcome == AGREE and resumed.recommendation == "hold" and resumed.trigger == SPLIT
    assert [r for r, _ in critic.rounds] == [1]
    assert critic.rounds[0][1] == {"analyst_1": "buy", "analyst_2": "hold"}
    events = [json.loads(l)["event"] for l in trace.path.read_text(encoding="utf-8").splitlines()
              if json.loads(l)["agent"] == "critic_loop"]
    assert events == ["start", "deferred", "resume", "round", "final"]


def test_unreachable_revote_defers_keeping_the_critique_and_the_other_revote():
    result, analysts, critic = _run(("buy", "hold"), ["DOWN", "hold"], ["hold"])
    assert result.outcome == DEFERRED and "could not reach analyst_1" in result.reason
    assert result.resume["critique"]["assessment"] == "round 1"
    assert set(result.resume["revised"]) == {"analyst_2"}
    assert result.rounds == []  # the round isn't complete yet

    class NoCritic:
        def review(self, *a, **k):
            raise AssertionError("the round-1 critique is reused, not asked for again")

    resumed = run_critic_loop(_ctx(), {}, analysts, NoCritic(), SPLIT, resume_from=result)
    assert resumed.outcome == AGREE and resumed.recommendation == "hold"
    assert len(analysts["analyst_1"].seen) == 2 and len(analysts["analyst_2"].seen) == 1
    assert resumed.rounds[0]["critique"]["assessment"] == "round 1"


def test_an_unusable_revote_aborts_even_if_the_other_is_unreachable():
    result, _, _ = _run(("buy", "hold"), ["FAIL"], ["DOWN"])
    assert result.outcome == ABORT and "verdict failed for analyst_1" in result.reason


def test_only_a_deferred_loop_can_be_resumed():
    done, analysts, critic = _run(("buy", "hold"), ["hold"], ["hold"])
    with pytest.raises(ValueError, match="deferred"):
        run_critic_loop(_ctx(), {}, analysts, critic, SPLIT, resume_from=done)


def test_failed_revote_aborts():
    result, _, _ = _run(("buy", "hold"), ["FAIL"], ["buy"])
    assert result.outcome == ABORT and "verdict failed for analyst_1" in result.reason


def test_rejects_bad_inputs():
    with pytest.raises(ValueError, match="max_rounds"):
        _run(("buy", "hold"), [], [], max_rounds=0)


def test_parse_critique_responses_handles_live_shapes():
    structured = parse_critique_responses([
        {"point": "a", "accept": True, "reason": "fair"},
        {"point": "b", "accept": "Rejected", "reason": "facts say otherwise"},
        {"point": "c", "accept": "maybe", "response": "unclear"},
    ])
    assert [r["accept"] for r in structured] == [True, False, None]
    assert structured[2]["reason"] == "unclear"
    # gemma answered with a {challenge point: reply} mapping on Sept 25
    mapping = parse_critique_responses({"Quality metrics claim": "I accept this."})
    assert mapping == [{"point": "Quality metrics claim", "accept": None, "reason": "I accept this."}]
    assert parse_critique_responses("I accept all three.") == [
        {"point": None, "accept": None, "reason": "I accept all three."}
    ]
    assert parse_critique_responses(None) == [] and parse_critique_responses([]) == []


def test_critic_prompt_explains_status_is_not_truth():
    llm = ScriptedLlm([_critique_json()])
    CriticAgent(llm, model="critic/model").review(
        _ctx(), {"analyst_1": _verdict("analyst_1", "buy"), "analyst_2": _verdict("analyst_2", "hold")}, 1
    )
    system = llm.calls[0][1][0]["content"]
    assert "matches_source" in system and "does not make the underlying claim true" in system
    assert '"verified"' not in system
