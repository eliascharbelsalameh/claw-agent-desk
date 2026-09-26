"""DeskPipeline directly: stage order, final outcome, data-only runs."""
from agents.critic_loop import AGREED_BUY
from agents.pipeline import DeskPipeline, SymbolRun
from tests.fakes import FakeAnalyst, FakeCritic, FakeMacro, patch_live_components


def _pipeline(critic=True, analysts=("analyst_1", "analyst_2")):
    return DeskPipeline(
        FakeMacro(),
        analysts={r: FakeAnalyst(None, r) for r in analysts},
        critic=FakeCritic(None) if critic else None,
    )


def test_events_arrive_in_stage_order_per_symbol():
    events = []
    _pipeline().run(["AAPL", "NVDA"], lambda stage, symbol, payload: events.append((symbol, stage)))
    assert events == [
        ("AAPL", "context"), ("AAPL", "verdict"), ("AAPL", "verdict"), ("AAPL", "cross_check"),
        ("AAPL", "critic_loop"), ("AAPL", "done"),
        ("NVDA", "context"), ("NVDA", "verdict"), ("NVDA", "verdict"), ("NVDA", "cross_check"), ("NVDA", "done"),
    ]


def test_final_outcome_prefers_the_critic_loop():
    runs = {r.symbol: r for r in _pipeline().run(["AAPL", "MSFT", "NVDA"])}
    assert (runs["AAPL"].cross_check.outcome, runs["AAPL"].outcome, runs["AAPL"].recommendation) == (
        "critic", "agree", "hold")
    assert runs["MSFT"].critic_loop.trigger == AGREED_BUY and runs["MSFT"].recommendation == "buy"
    assert runs["NVDA"].critic_loop is None and runs["NVDA"].recommendation == "hold"


def test_without_critic_the_cross_check_is_final():
    run = _pipeline(critic=False).run(["AAPL"])[0]
    assert run.critic_loop is None and run.outcome == "critic" and run.recommendation is None


def test_data_only_run_has_context_and_no_decision():
    run = _pipeline(critic=False, analysts=()).run(["AAPL"])[0]
    assert isinstance(run, SymbolRun) and run.context.symbol == "AAPL"
    assert run.verdicts == {} and run.outcome is None and run.recommendation is None


def test_from_settings_data_only_builds_no_llm_agents(monkeypatch, tmp_path):
    patch_live_components(monkeypatch, tmp_path)
    p = DeskPipeline.from_settings(use_llm=False)
    assert p.analysts == {} and p.critic is None
    p = DeskPipeline.from_settings(use_llm=True, run_critic=False)
    assert set(p.analysts) == {"analyst_1", "analyst_2"} and p.critic is None


# --- deferral and resume (decided Sept 26, 2026) ---

import json  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402

from agents.pipeline import ANALYST_BACKUP_AFTER, ModelOutages  # noqa: E402
from tests.fakes import _verdict  # noqa: E402

T0 = datetime(2026, 9, 28, 14, 0, tzinfo=timezone.utc)


class DownOnceAnalyst(FakeAnalyst):
    """Its model is unreachable on the first call, fine afterwards."""

    def __init__(self, llm, role, trace=None):
        super().__init__(llm, role, trace)
        self.calls = []

    def analyze(self, ctx, *, exclude=(), allow_backup=False):
        self.calls.append(allow_backup)
        if len(self.calls) == 1:
            return _verdict(ctx.symbol, self.role, None, error="ConnectionError: dropped", call_failed=True)
        return super().analyze(ctx, exclude=exclude, allow_backup=allow_backup)


def test_unreachable_analyst_defers_and_resume_asks_only_that_analyst():
    a1, a2 = FakeAnalyst(None, "analyst_1"), DownOnceAnalyst(None, "analyst_2")
    desk = DeskPipeline(FakeMacro(), analysts={"analyst_1": a1, "analyst_2": a2}, critic=FakeCritic(None))
    run = desk.run(["NVDA"])[0]
    assert run.deferred and run.outcome == "deferred" and run.recommendation is None
    assert "could not reach analyst_2" in run.cross_check.reason
    first_analyst_1 = run.verdicts["analyst_1"]

    # the scheduler stores the run as JSON and resumes it next cycle
    stored = SymbolRun.from_dict(json.loads(json.dumps(run.to_dict(), default=str)))
    events = []
    resumed = desk.resume_symbol(stored, lambda stage, symbol, payload: events.append(stage))
    assert resumed.outcome == "agree" and resumed.recommendation == "hold"
    assert len(a2.calls) == 2
    assert resumed.verdicts["analyst_1"].thesis == first_analyst_1.thesis  # kept, not asked again
    assert list(resumed.verdicts) == ["analyst_1", "analyst_2"]
    assert events == ["verdict", "cross_check"]


def test_resume_leaves_a_finished_run_alone():
    desk = _pipeline()
    run = desk.run(["NVDA"])[0]
    assert desk.resume_symbol(run) is run and run.outcome == "agree"


def test_verdicts_keep_role_order_when_analysts_finish_out_of_order():
    run = _pipeline().run(["AAPL"])[0]
    assert list(run.verdicts) == ["analyst_1", "analyst_2"]


def test_model_outages_track_the_first_failure_since_the_last_success():
    outages = ModelOutages()
    down = _verdict("AAPL", "analyst_1", None, error="ConnectionError", call_failed=True)
    outages.note(down, T0)
    outages.note(down, T0 + timedelta(hours=1))
    assert outages.down_for("lab-analyst_1/m", T0 + timedelta(hours=2)) == timedelta(hours=2)
    outages.note(_verdict("AAPL", "analyst_1", "buy"), T0 + timedelta(hours=3))
    assert outages.down_for("lab-analyst_1/m", T0 + timedelta(hours=3)) is None
    # a backup that answered after the primary failed marks the primary down
    backup = _verdict("AAPL", "analyst_1", "buy", fallbacks=[{"model": "p", "error": "x", "call_failed": True}])
    outages.note(backup, T0)
    assert "p" in outages.down_since and "lab-analyst_1/m" not in outages.down_since


def test_backups_only_after_the_primary_has_been_down_long_enough():
    analyst = FakeAnalyst(None, "analyst_1")
    analyst.models = ["lab-analyst_1/m", "backup/m"]
    clock = {"now": T0}
    desk = DeskPipeline(FakeMacro(), analysts={"analyst_1": analyst}, now=lambda: clock["now"])
    assert not desk.allow_backup("analyst_1")
    desk.outages.record("lab-analyst_1/m", ok=False, now=T0)
    clock["now"] = T0 + ANALYST_BACKUP_AFTER - timedelta(minutes=1)
    assert not desk.allow_backup("analyst_1")
    clock["now"] = T0 + ANALYST_BACKUP_AFTER
    assert desk.allow_backup("analyst_1")
    analyst.models = ["lab-analyst_1/m"]  # no backup configured: always defer
    assert not desk.allow_backup("analyst_1")


def test_parallel_run_matches_the_sequential_one_in_input_order():
    sequential = _pipeline().run(["AAPL", "MSFT", "NVDA"])
    parallel = _pipeline().run(["AAPL", "MSFT", "NVDA"], workers=3)
    assert [r.symbol for r in parallel] == ["AAPL", "MSFT", "NVDA"]
    assert [(r.outcome, r.recommendation) for r in parallel] == [(r.outcome, r.recommendation) for r in sequential]
