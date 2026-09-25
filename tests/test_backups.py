"""Per-role backup models: health ordering, failover, independence exclusions."""
import json

import pytest

from agents.analyst_agent import AnalystAgent, AnalystVerdict
from agents.critic_agent import CriticAgent
from agents.llm_json import BACKUP_CONNECT_RETRIES, request_json
from agents.macro_agent import MacroContextAgent, StockContext
from agents.trace import TraceLogger
from data_layer.llm_client import AGENT_MODEL_BACKUPS, AGENT_MODELS, ModelHealth, role_models


def _completion(content):
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": {"total_tokens": 1}}


class RoutedLlm:
    """Per-model behaviour: a string reply, an exception to raise, or a list of those in call order."""

    def __init__(self, behaviour):
        self.behaviour = {m: (list(b) if isinstance(b, list) else b) for m, b in behaviour.items()}
        self.calls = []

    def chat_completion(self, model, messages, **kwargs):
        self.calls.append((model, kwargs))
        b = self.behaviour[model]
        item = b.pop(0) if isinstance(b, list) else b
        if isinstance(item, Exception):
            raise item
        return _completion(item)


VERDICT = json.dumps({
    "recommendation": "buy", "confidence": 0.7, "thesis": "t", "drivers": ["d"], "risks": ["r"],
    "evidence": [{"fact": "price.last_close", "value": 1, "why": "w"}] * 3, "data_concerns": [],
})
DEAD = ConnectionError("Remote end closed connection without response")


def _ctx():
    return StockContext(symbol="AAPL", generated_at="t", macro={}, price={"last_close": 1})


# --- role config ---

def test_role_models_primary_first_without_repeats():
    for role in AGENT_MODELS:
        models = role_models(role)
        assert models[0] == AGENT_MODELS[role]
        assert len(models) == len(set(models))
        assert models[1:] == [m for m in AGENT_MODEL_BACKUPS.get(role, []) if m != AGENT_MODELS[role]]


def test_analyst_candidate_lists_never_share_a_model():
    assert not set(role_models("analyst_1")) & set(role_models("analyst_2"))


# --- health ---

def test_model_health_moves_failed_models_last_until_cooldown_or_success():
    now = [0.0]
    health = ModelHealth(cooldown_seconds=100, clock=lambda: now[0])
    health.record_failure("a")
    assert health.order(["a", "b", "c"]) == ["b", "c", "a"]
    assert health.order(["a"]) == ["a"]  # never skipped outright
    now[0] = 101
    assert health.order(["a", "b"]) == ["a", "b"]  # cooled down
    health.record_failure("b")
    health.record_success("b")
    assert health.order(["a", "b"]) == ["a", "b"]


# --- request_json failover ---

def _request(llm, models, health=None, exclude=(), log=None):
    events = log if log is not None else []
    return request_json(
        llm, models, [{"role": "user", "content": "x"}],
        validate=lambda obj: obj, log=lambda e, p: events.append((e, p)), base={"symbol": "AAPL"},
        max_tokens=10, temperature=0, health=health, exclude=exclude,
    ), events


def test_dead_primary_fails_over_to_backup_with_a_short_retry_budget():
    llm = RoutedLlm({"p": DEAD, "b": '{"ok": 1}'})
    health = ModelHealth()
    reply, events = _request(llm, ["p", "b"], health=health)

    assert reply.ok and reply.model == "b" and reply.value == {"ok": 1}
    assert reply.fallbacks == [{"model": "p", "error": f"ConnectionError: {DEAD}", "call_failed": True}]
    assert llm.calls[0] == ("p", {"max_tokens": 10, "temperature": 0, "max_retries": BACKUP_CONNECT_RETRIES})
    assert "max_retries" not in llm.calls[1][1]  # the last candidate keeps the full budget
    assert ("fallback", {"symbol": "AAPL", "from_model": "p", "to_model": "b",
                         "reason": f"ConnectionError: {DEAD}"}) in events
    assert health.order(["p", "b"]) == ["b", "p"]


def test_unusable_reply_also_fails_over_but_does_not_cool_the_model():
    llm = RoutedLlm({"p": "not json", "b": '{"ok": 1}'})
    health = ModelHealth()
    reply, _ = _request(llm, ["p", "b"], health=health)
    assert reply.ok and reply.model == "b"
    assert [m for m, _ in llm.calls] == ["p", "p", "b"]  # p got its correction turn first
    assert reply.fallbacks[0]["call_failed"] is False
    assert health.order(["p", "b"]) == ["p", "b"]


def test_cooling_primary_is_tried_after_the_backup():
    llm = RoutedLlm({"p": '{"ok": "p"}', "b": '{"ok": "b"}'})
    health = ModelHealth()
    health.record_failure("p")
    reply, _ = _request(llm, ["p", "b"], health=health)
    assert reply.model == "b" and [m for m, _ in llm.calls] == ["b"]


def test_last_healthy_model_gets_full_budget_and_cooling_ones_only_a_probe():
    # live, Sept 25: the healthy model got 1 retry because a dead one
    # followed it, and the dead one then got the full ~7-minute budget
    llm = RoutedLlm({"dead": DEAD, "ok": DEAD})
    health = ModelHealth()
    health.record_failure("dead")
    _request(llm, ["dead", "ok"], health=health)
    assert llm.calls[0][0] == "ok" and "max_retries" not in llm.calls[0][1]
    assert llm.calls[1] == ("dead", {"max_tokens": 10, "temperature": 0, "max_retries": BACKUP_CONNECT_RETRIES})


def test_all_cooling_models_get_only_a_probe():
    llm = RoutedLlm({"a": DEAD, "b": DEAD})
    health = ModelHealth()
    health.record_failure("a")
    health.record_failure("b")
    _request(llm, ["a", "b"], health=health)
    assert all(kwargs.get("max_retries") == BACKUP_CONNECT_RETRIES for _, kwargs in llm.calls)


def test_excluded_models_are_never_called():
    llm = RoutedLlm({"b": '{"ok": 1}'})
    reply, events = _request(llm, ["p", "b"], exclude={"p"})
    assert reply.model == "b" and [m for m, _ in llm.calls] == ["b"]
    reply, _ = _request(RoutedLlm({}), ["p"], exclude={"p"})
    assert not reply.ok and reply.call_failed and "excluded" in reply.error


def test_every_candidate_failing_returns_the_last_error_and_all_fallbacks():
    llm = RoutedLlm({"p": DEAD, "b": DEAD})
    reply, _ = _request(llm, ["p", "b"])
    assert not reply.ok and reply.call_failed and reply.model == "b"
    assert [f["model"] for f in reply.fallbacks] == ["p"]


# --- agents ---

def test_analyst_records_the_backup_model_that_answered():
    llm = RoutedLlm({"p": DEAD, "b": VERDICT})
    verdict = AnalystAgent(llm, "analyst_1", models=["p", "b"], health=ModelHealth()).analyze(_ctx())
    assert verdict.ok and verdict.model == "b"
    assert verdict.fallbacks[0]["model"] == "p"


def test_analyst_excludes_the_model_the_other_analyst_used():
    llm = RoutedLlm({"b": VERDICT})
    verdict = AnalystAgent(llm, "analyst_2", models=["p", "b"], health=ModelHealth()).analyze(
        _ctx(), exclude={"p"})
    assert verdict.model == "b" and [m for m, _ in llm.calls] == ["b"]


def test_revise_prefers_the_model_that_wrote_the_verdict_and_never_the_other_analysts():
    llm = RoutedLlm({"b": VERDICT})
    agent = AnalystAgent(llm, "analyst_1", models=["p", "b", "x"], health=ModelHealth())
    own = AnalystVerdict(symbol="AAPL", role="analyst_1", model="b", generated_at="t", recommendation="buy",
                         confidence=0.7, thesis="t", drivers=["d"], risks=["r"])
    other = AnalystVerdict(symbol="AAPL", role="analyst_2", model="x", generated_at="t", recommendation="hold",
                           confidence=0.6, thesis="t", drivers=["d"], risks=["r"])
    revised = agent.revise(_ctx(), own, other, assessment="a", challenges_to_me=[],
                           challenges_to_other=[], review_round=1)
    assert revised.model == "b" and [m for m, _ in llm.calls] == ["b"]


def test_critic_never_runs_on_an_analysts_model():
    critique_json = json.dumps({"assessment": "a", "challenges": []})
    llm = RoutedLlm({"c2": critique_json})
    critic = CriticAgent(llm, models=["a1", "c2"], health=ModelHealth())
    verdicts = {
        role: AnalystVerdict(symbol="AAPL", role=role, model=model, generated_at="t", recommendation="buy",
                             confidence=0.7, thesis="t", drivers=["d"], risks=["r"])
        for role, model in (("analyst_1", "a1"), ("analyst_2", "a2"))
    }
    critique = critic.review(_ctx(), verdicts, 1)
    assert critique.ok and critique.model == "c2" and [m for m, _ in llm.calls] == ["c2"]


class _NoData:
    def get_series_observations(self, *a, **k):
        return []

    def __getattr__(self, name):
        def fail(*a, **k):
            raise RuntimeError("offline")
        return fail


def test_macro_briefing_fails_over_and_only_sends_thinking_flag_to_nemotron(tmp_path):
    good = ("MACRO: x\nCOMPANY FILINGS & FUNDAMENTALS: x\nPRICE & VOLUME: x\nNEWS: x\nDATA GAPS: x")
    llm = RoutedLlm({"nvidia/nemotron-x": DEAD, "mistralai/other": good})
    trace = TraceLogger(tmp_path / "t.jsonl")
    health = ModelHealth()
    agent = MacroContextAgent(_NoData(), _NoData(), _NoData(), _NoData(), llm, trace=trace,
                              models=["nvidia/nemotron-x", "mistralai/other"], health=health)
    ctx = agent.run(["AAPL"])["AAPL"]

    assert ctx.briefing == good and ctx.briefing_model == "mistralai/other"
    (m1, k1), (m2, k2) = llm.calls
    assert k1["chat_template_kwargs"] == {"enable_thinking": False} and k1["max_retries"] == BACKUP_CONNECT_RETRIES
    assert "chat_template_kwargs" not in k2 and "max_retries" not in k2
    events = [json.loads(l)["event"] for l in trace.path.read_text(encoding="utf-8").splitlines()]
    assert "fallback" in events
    assert health.order(["nvidia/nemotron-x", "mistralai/other"])[0] == "mistralai/other"


def test_primary_skipped_as_cooling_still_shows_as_backup_use():
    llm = RoutedLlm({"b": VERDICT})
    health = ModelHealth()
    health.record_failure("p")
    llm.behaviour["p"] = DEAD
    verdict = AnalystAgent(llm, "analyst_1", models=["p", "b"], health=health).analyze(_ctx())
    assert verdict.model == "b" and verdict.primary_model == "p"
    assert verdict.fallbacks == [] and verdict.used_backup
    ok = AnalystAgent(RoutedLlm({"p": VERDICT}), "analyst_1", models=["p", "b"], health=ModelHealth()).analyze(_ctx())
    assert not ok.used_backup


def test_critic_records_its_primary_when_excluded():
    critique_json = json.dumps({"assessment": "a", "challenges": []})
    critic = CriticAgent(RoutedLlm({"c2": critique_json}), models=["a1", "c2"], health=ModelHealth())
    verdicts = {
        role: AnalystVerdict(symbol="AAPL", role=role, model=model, generated_at="t", recommendation="buy",
                             confidence=0.7, thesis="t", drivers=["d"], risks=["r"])
        for role, model in (("analyst_1", "a1"), ("analyst_2", "a2"))
    }
    critique = critic.review(_ctx(), verdicts, 1)
    assert critique.model == "c2" and critique.primary_model == "a1"
