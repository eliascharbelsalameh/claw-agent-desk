"""End-to-end wiring of `python -m agents` with every agent and client faked:
context -> analysts -> cross-check -> critic loop -> printed summary."""
import sys

import pytest

import agents.__main__ as cli
from tests.fakes import FakeAnalyst, patch_live_components


@pytest.fixture
def run_cli(monkeypatch, tmp_path, capsys):
    patch_live_components(monkeypatch, tmp_path)

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
    assert "critic [critic/m]: AAPL critique" in out
    assert "MSFT cross-check: AGREE (buy)" in out
    assert "MSFT critic loop (agreed_buy): AGREE (buy)" in out
    assert "NVDA cross-check: AGREE (hold)" in out
    assert "NVDA critic loop (" not in out  # agreed hold skips the critic
    # analyst_2 may not use the model analyst_1 actually ran on
    assert FakeAnalyst.excludes[("AAPL", "analyst_1")] == set()
    assert FakeAnalyst.excludes[("AAPL", "analyst_2")] == {"lab-analyst_1/m"}


def test_no_critic_stops_at_the_cross_check(run_cli):
    out = run_cli("AAPL", "--analysts", "analyst_1", "analyst_2", "--no-critic")
    assert "AAPL cross-check: CRITIC" in out
    assert "AAPL critic loop (" not in out


def test_single_analyst_skips_cross_check(run_cli):
    out = run_cli("AAPL", "--analysts", "analyst_1")
    assert "cross-check" not in out and "AAPL analyst_1" in out
