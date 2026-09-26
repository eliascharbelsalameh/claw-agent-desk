"""Technical agent (entry timing, spec section 3 step 5) and how its answer
reaches the portfolio."""
import json

import pytest

from agents.bias_agent import VETOED
from agents.macro_agent import StockContext
from agents.pipeline import DeskPipeline, SymbolRun
from agents.portfolio import Decision
from agents.technical_agent import (
    DEFERRED,
    FAILED,
    PASSED,
    SYSTEM_PROMPT,
    WAITING,
    TechnicalAgent,
    chart,
    validate_timing,
)
from data_layer.llm_client import ModelHealth
from tests.fakes import FakeAnalyst, FakeBias, FakeCritic, FakeMacro, FakeTechnical
from tests.test_backups import DEAD, RoutedLlm


def _bars(n, start=100.0):
    return [{"t": f"2026-09-{i + 1:02d}T16:00:00Z", "o": start + i, "h": start + i + 1, "l": start + i - 1,
             "c": start + i + 0.5, "v": 1000, "n": 5, "vw": 1.0} for i in range(n)]


def _ctx(**kw):
    return StockContext(symbol="MSFT", generated_at="t", macro={}, price={"last_close": 124.5},
                        technicals={"rsi_14": 71.2}, bars_4h=_bars(40), bars_1h=_bars(20, 120.0), **kw)


def _reply(timing="enter", **extra):
    return json.dumps({"timing": timing, "trend_4h": "up", "support": 118.0, "resistance": 125.5,
                       "reason": f"{timing} because", "evidence": [
                           {"fact": "chart.bars_1h[19].c", "value": 139.5, "why": "last close"},
                           {"fact": "technicals.rsi_14", "value": 71.2, "why": "momentum"}], **extra})


def test_chart_keeps_recent_bars_in_ohlcv_only():
    bars = chart(_ctx())
    assert len(bars["bars_4h"]) == 30 and len(bars["bars_1h"]) == 20
    assert set(bars["bars_4h"][0]) == {"t", "o", "h", "l", "c", "v"}
    assert bars["bars_4h"][-1]["c"] == 139.5  # the most recent bar is kept


def test_prompt_times_the_entry_only():
    assert "You do not revisit that decision" in SYSTEM_PROMPT
    assert "not for general uncertainty" in SYSTEM_PROMPT
    assert "IEX" in SYSTEM_PROMPT


def test_validate_timing():
    assert validate_timing(json.loads(_reply()))["support"] == 118.0
    assert validate_timing({**json.loads(_reply()), "support": None})["support"] is None
    with pytest.raises(ValueError, match="timing"):
        validate_timing({**json.loads(_reply()), "timing": "later"})
    with pytest.raises(ValueError, match="trend_4h"):
        validate_timing({**json.loads(_reply()), "trend_4h": "bullish"})


@pytest.mark.parametrize("timing, outcome", [("enter", PASSED), ("wait", WAITING)])
def test_time_entry_checks_evidence_against_facts_and_chart(timing, outcome):
    result = TechnicalAgent(RoutedLlm({"t/m": _reply(timing)}), model="t/m", health=ModelHealth()).time_entry(_ctx())
    assert result.outcome == outcome and result.timing == timing
    assert result.evidence_check["matches_source"] == 2
    assert result.gate["passed"] is (outcome == PASSED) and result.gate["reason"] == f"{timing}: {timing} because"


def test_time_entry_failures():
    down = TechnicalAgent(RoutedLlm({"t/m": DEAD}), model="t/m", health=ModelHealth()).time_entry(_ctx())
    assert down.outcome == DEFERRED and down.call_failed
    garbled = TechnicalAgent(RoutedLlm({"t/m": "no json"}), model="t/m", health=ModelHealth()).time_entry(_ctx())
    assert garbled.outcome == FAILED and not garbled.gate["passed"]
    no_bars = TechnicalAgent(RoutedLlm({}), model="t/m", health=ModelHealth()).time_entry(
        StockContext(symbol="MSFT", generated_at="t", macro={}))
    assert no_bars.outcome == FAILED and "no 4-hour or 1-hour bars" in no_bars.error


def _desk(technical=None):
    analysts = {r: FakeAnalyst(None, r) for r in ("analyst_1", "analyst_2")}
    return DeskPipeline(FakeMacro(), analysts=analysts, critic=FakeCritic(None),
                        bias_agents={r: FakeBias(None, r) for r in ("bias_1", "bias_2")},
                        technical=technical or FakeTechnical(None))


def test_wait_stops_the_buy_for_today_and_the_reason_reaches_the_portfolio():
    FakeTechnical.wait = {"MSFT"}
    try:
        run = _desk().run(["MSFT"])[0]
    finally:
        FakeTechnical.wait = set()
    assert run.technical.outcome == WAITING and not run.deferred
    decision = Decision.from_run(run, "2026-09-28", "t")
    assert [g["gate"] for g in decision.gates] == ["bias", "technical"]
    assert decision.blocked and decision.gates[1]["reason"] == "wait: MSFT into resistance"


def test_no_timing_after_a_bias_veto():
    FakeBias.flag = {"MSFT"}
    try:
        run = _desk().run(["MSFT"])[0]
    finally:
        FakeBias.flag = set()
    assert run.bias.outcome == VETOED and run.technical is None


def test_deferred_timing_is_resumed():
    class DownOnce(FakeTechnical):
        calls = 0

        def time_entry(self, ctx):
            DownOnce.calls += 1
            result = super().time_entry(ctx)
            if DownOnce.calls == 1:
                result.outcome, result.timing, result.error, result.call_failed = DEFERRED, None, "dropped", True
            return result

    desk = _desk(technical=DownOnce(None))
    run = desk.run(["MSFT"])[0]
    assert run.deferred and run.outcome == "agree"
    stored = SymbolRun.from_dict(json.loads(json.dumps(run.to_dict(), default=str)))
    resumed = desk.resume_symbol(stored)
    assert not resumed.deferred and resumed.technical.outcome == PASSED and DownOnce.calls == 2
