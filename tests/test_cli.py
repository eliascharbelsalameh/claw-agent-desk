"""End-to-end wiring of `python -m agents` with every agent and client faked:
context -> analysts -> cross-check -> critic loop -> printed summary."""
import sys

import pytest

import agents.__main__ as cli
from agents.analyst_agent import AnalystVerdict
from agents.critic_agent import Critique
from agents.macro_agent import StockContext

FIRST = {"AAPL": ("buy", "hold"), "MSFT": ("buy", "buy"), "NVDA": ("hold", "hold")}
REVOTE = {"AAPL": ("hold", "hold"), "MSFT": ("buy", "buy")}


def _verdict(symbol, role, rec, **extra):
    return AnalystVerdict(symbol=symbol, role=role, model=f"lab-{role}/m", generated_at="t",
                          recommendation=rec, confidence=0.7, thesis=f"{role} says {rec}",
                          drivers=["d"], risks=["r"], **extra)


class FakeMacro:
    def __init__(self, **kwargs):
        pass

    def run(self, symbols):
        return {s: StockContext(symbol=s, generated_at="t", macro={}) for s in symbols}


class FakeAnalyst:
    def __init__(self, llm, role, trace=None):
        self.role, self.model = role, f"lab-{role}/m"
        self.index = 0 if role == "analyst_1" else 1

    def analyze(self, ctx):
        return _verdict(ctx.symbol, self.role, FIRST[ctx.symbol][self.index])

    def revise(self, ctx, own, other, *, assessment, challenges_to_me, challenges_to_other, review_round):
        return _verdict(ctx.symbol, self.role, REVOTE[ctx.symbol][self.index], review_round=review_round,
                        previous_recommendation=own.recommendation,
                        response_to_critique=[{"point": "p", "accept": True, "reason": "fair"}])


class FakeCritic:
    def __init__(self, llm, trace=None):
        pass

    def review(self, ctx, verdicts, review_round):
        return Critique(symbol=ctx.symbol, model="critic/m", review_round=review_round, generated_at="t",
                        assessment=f"{ctx.symbol} critique",
                        challenges=[{"to": "analyst_1", "point": "p", "why": "w"}])


@pytest.fixture
def run_cli(monkeypatch, tmp_path, capsys):
    for name in ("AlpacaClient", "FredClient", "EdgarClient", "FinnhubClient", "LlmClient", "DiskCache"):
        monkeypatch.setattr(cli, name, lambda *a, **k: None)
    monkeypatch.setattr(cli, "get_settings", lambda: type("S", (), {"cache_dir": tmp_path})())
    monkeypatch.setattr(cli, "MacroContextAgent", FakeMacro)
    monkeypatch.setattr(cli, "AnalystAgent", FakeAnalyst)
    monkeypatch.setattr(cli, "CriticAgent", FakeCritic)

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["agents", *args, "--log-dir", str(tmp_path)])
        cli.main()
        return capsys.readouterr().out

    return run


def test_full_pipeline_prints_cross_check_and_critic_loop(run_cli):
    out = run_cli("AAPL", "MSFT", "NVDA", "--analysts", "analyst_1", "analyst_2")

    assert "AAPL cross-check: CRITIC" in out
    assert "AAPL critic loop (split): AGREE (hold) - after round 1: both analysts recommend hold" in out
    assert "analyst_1 buy->hold (accepted 1, rejected 0)" in out
    assert "critic: AAPL critique" in out
    assert "MSFT cross-check: AGREE (buy)" in out
    assert "MSFT critic loop (agreed_buy): AGREE (buy)" in out
    assert "NVDA cross-check: AGREE (hold)" in out
    assert "NVDA critic loop (" not in out  # agreed hold skips the critic


def test_no_critic_stops_at_the_cross_check(run_cli):
    out = run_cli("AAPL", "--analysts", "analyst_1", "analyst_2", "--no-critic")
    assert "AAPL cross-check: CRITIC" in out
    assert "AAPL critic loop (" not in out


def test_single_analyst_skips_cross_check(run_cli):
    out = run_cli("AAPL", "--analysts", "analyst_1")
    assert "cross-check" not in out and "AAPL analyst_1" in out
