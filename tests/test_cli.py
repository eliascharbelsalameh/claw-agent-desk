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
    assert "MSFT bias gate: PASSED - passed: neither bias agent flags the buy" in out
    assert "MSFT technical [tech/m]: ENTER - MSFT clean entry (4h trend up, support 90.0, resistance 110.0)" in out
    assert "NVDA bias gate" not in out and "AAPL bias gate" not in out  # agreed buys only
    assert "NVDA cross-check: AGREE (hold)" in out
    assert "NVDA critic loop (" not in out  # agreed hold skips the critic
    # the analysts run in parallel, each on its own model: independence comes
    # from disjoint model lists (test_backups), not from runtime exclusion
    assert FakeAnalyst.excludes[("AAPL", "analyst_1")] == set()
    assert FakeAnalyst.excludes[("AAPL", "analyst_2")] == set()


def test_no_critic_stops_at_the_cross_check(run_cli):
    out = run_cli("AAPL", "--analysts", "analyst_1", "analyst_2", "--no-critic")
    assert "AAPL cross-check: CRITIC" in out
    assert "AAPL critic loop (" not in out


def test_single_analyst_skips_cross_check(run_cli):
    out = run_cli("AAPL", "--analysts", "analyst_1")
    assert "cross-check" not in out and "AAPL analyst_1" in out
