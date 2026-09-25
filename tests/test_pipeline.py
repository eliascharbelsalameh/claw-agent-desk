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
