"""The Streamlit app, driven headlessly with every live client and agent faked."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tests.fakes import patch_live_components
from ui.support import CREDENTIAL_NAMES

APP = str(Path(__file__).resolve().parent.parent / "ui" / "streamlit_app.py")


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAW_DESK_NO_REGISTRY", "1")
    monkeypatch.setenv("CLAW_DESK_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("CLAW_DESK_STATE", str(tmp_path / "no-state.json"))  # never the real state file
    for name in CREDENTIAL_NAMES:
        monkeypatch.setenv(name, "present-but-fake")
    patch_live_components(monkeypatch, tmp_path)
    return AppTest.from_file(APP, default_timeout=60)


def _all_markdown(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def _run(at, symbols="AAPL MSFT NVDA", mode=None):
    at.run()
    at.sidebar.text_input[0].set_value(symbols)
    if mode:
        at.sidebar.radio[0].set_value(mode)
    at.sidebar.button[0].click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_app_loads_with_credential_status_and_line_up(app):
    app.run()
    assert not app.exception
    assert app.title[0].value == "Claw Agent Desk"
    sidebar = "\n".join(m.value for m in app.sidebar.markdown)
    for name in CREDENTIAL_NAMES:
        assert f":green[set] `{name}`" in sidebar
    assert ":red[missing]" not in sidebar
    assert "present-but-fake" not in sidebar  # presence only, never values


def test_full_run_shows_every_stage(app):
    at = _run(app)
    summary = at.dataframe[0].value
    assert list(summary["symbol"]) == ["AAPL", "MSFT", "NVDA"]
    assert list(summary["final"]) == ["agree hold", "agree buy", "agree hold"]
    text = _all_markdown(at)
    assert "Critic loop** (split)" in text and "Critic loop** (agreed_buy)" in text
    assert "accepted 1, rejected 0" in text
    assert "Cross-check:" in text
    assert "**Bias gate:** :green[**PASSED**]" in text  # MSFT, the agreed buy
    assert list(summary["bias gate"]) == ["", "passed", ""]
    assert list(summary["entry"]) == ["", "enter", ""]
    assert "**Technical** `tech/m`: :green[**ENTER**] - MSFT clean entry" in text


def test_data_only_run_needs_no_nvidia_key_and_makes_no_decision(app, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY")
    at = _run(app, symbols="AAPL", mode="Data only (no Build credits)")
    assert list(at.dataframe[0].value["final"]) == ["data only"]


def test_full_run_refuses_without_the_nvidia_key(app, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY")
    at = _run(app, symbols="AAPL")
    assert any("Missing credentials: NVIDIA_API_KEY" in e.value for e in at.error)


def test_bad_tickers_are_ignored_with_a_warning(app):
    at = _run(app, symbols="AAPL, not-a-ticker!, MSFT")
    assert any("Ignored: NOT-A-TICKER!" in w.value for w in at.warning)
    assert list(at.dataframe[0].value["symbol"]) == ["AAPL", "MSFT"]


def test_trace_viewer_reads_the_run_trace(app):
    at = _run(app, symbols="AAPL")
    # the run wrote cross_check / critic_loop events into the tmp log dir
    assert at.selectbox[0].value.name.startswith("trace-")
    events_table = next(d.value for d in at.dataframe if "event" in d.value.columns)
    assert "decision" in set(events_table["event"])
    assert "cross_check" in set(events_table["agent"])


def test_deferred_stock_can_be_retried_from_the_app(app, monkeypatch):
    import agents.pipeline as pipeline
    from tests.fakes import FakeAnalyst, _verdict

    class FlakyAnalyst(FakeAnalyst):
        """analyst_2's model is unreachable on its first call only."""
        failed = False

        def analyze(self, ctx, *, exclude=(), allow_backup=False):
            if self.role == "analyst_2" and not FlakyAnalyst.failed:
                FlakyAnalyst.failed = True
                return _verdict(ctx.symbol, self.role, None, error="ConnectionError: dropped", call_failed=True)
            return super().analyze(ctx, exclude=exclude, allow_backup=allow_backup)

    monkeypatch.setattr(pipeline, "AnalystAgent", FlakyAnalyst)
    at = _run(app, symbols="NVDA")
    assert list(at.dataframe[0].value["final"]) == ["deferred"]
    assert any("Model unreachable (Build failure)" in w.value for w in at.warning)
    retry = next(b for b in at.button if b.label.startswith("Retry the 1 deferred"))
    retry.click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert list(at.dataframe[0].value["final"]) == ["agree hold"]


def test_desk_state_tab_shows_the_schedulers_memory(app, monkeypatch, tmp_path):
    from agents.state import DeskState

    path = tmp_path / "state" / "desk_state.json"
    DeskState(positions={"MSFT": {"qty": 98, "entry_session": "2026-09-28", "horizon_end": "2026-10-05",
                                  "reason": "both analysts recommend buy"}},
              pending={"AAPL": {"session": "2026-09-28", "attempts": 1, "reason": "could not reach analyst_2"}},
              decisions=[{"session": "2026-09-28", "symbol": "MSFT", "outcome": "agree", "recommendation": "buy"}],
              last_decision_session="2026-09-28").save(path)
    monkeypatch.setenv("CLAW_DESK_STATE", str(path))
    app.run()
    assert not app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Positions (desk)"] == "1" and metrics["Deferred"] == "1"
    assert metrics["Last decision session"] == "2026-09-28"
