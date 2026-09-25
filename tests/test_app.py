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
