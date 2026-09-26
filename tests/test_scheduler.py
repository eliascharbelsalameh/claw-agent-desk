"""The scheduler: when it runs what, deferral across passes, and what it keeps."""
import json
from datetime import datetime, timedelta, timezone

from agents.macro_agent import SharedContext, StockContext
from agents.pipeline import DeskPipeline
from agents.scheduler import DECISION, IDLE, RETRY, RETRY_EVERY, DeskScheduler, session_for
from agents.state import DeskState
from tests.fakes import FakeAnalyst, FakeCritic, _verdict
from tests.test_portfolio import FakeBroker


def _clock(ts, is_open=False, next_open="2026-09-28T09:30:00-04:00"):
    return {"timestamp": ts, "is_open": is_open, "next_open": next_open, "next_close": "2026-09-28T16:00:00-04:00"}


PRE_MARKET = _clock("2026-09-28T09:05:00-04:00")
EARLY = _clock("2026-09-28T08:30:00-04:00")
MIDDAY = _clock("2026-09-28T11:00:00-04:00", is_open=True, next_open="2026-09-29T09:30:00-04:00")
AFTER_CLOSE = _clock("2026-09-28T17:00:00-04:00", next_open="2026-09-29T09:30:00-04:00")
WEEKEND = _clock("2026-09-26T10:00:00-04:00", next_open="2026-09-28T09:30:00-04:00")


def test_session_for():
    assert session_for(PRE_MARKET) == "2026-09-28"
    assert session_for(MIDDAY) == "2026-09-28"
    assert session_for(AFTER_CLOSE) == "2026-09-29"
    assert session_for(WEEKEND) == "2026-09-28"


class PricedMacro:
    def prepare(self):
        return SharedContext(macro={}, gaps=[], clock=None)

    def build_context(self, symbol, shared):
        return StockContext(symbol=symbol, generated_at="t", macro={}, price={"last_close": 100.0})


class ScriptedAnalyst(FakeAnalyst):
    """buy/buy for MSFT, hold/hold for NVDA; analyst_2 can't be reached for
    AAPL on its first call (so AAPL is deferred once, then agreed hold)."""
    FIRST = {"MSFT": "buy", "NVDA": "hold", "AAPL": "hold", "AMD": "hold"}
    down_once = {"AAPL"}

    def analyze(self, ctx, *, exclude=(), allow_backup=False):
        if self.role == "analyst_2" and ctx.symbol in ScriptedAnalyst.down_once:
            ScriptedAnalyst.down_once.discard(ctx.symbol)
            return _verdict(ctx.symbol, self.role, None, error="ConnectionError: dropped", call_failed=True)
        return _verdict(ctx.symbol, self.role, self.FIRST[ctx.symbol])

    def revise(self, ctx, own, other, **kwargs):
        return _verdict(ctx.symbol, self.role, own.recommendation, review_round=kwargs["review_round"],
                        previous_recommendation=own.recommendation)


def _scheduler(tmp_path, broker, clock_box, watchlist=("AAPL", "MSFT", "NVDA"), dry_run=False):
    ScriptedAnalyst.down_once = {"AAPL"}

    def factory(trace, outages):
        analysts = {r: ScriptedAnalyst(None, r) for r in ("analyst_1", "analyst_2")}
        return DeskPipeline(PricedMacro(), analysts=analysts, critic=FakeCritic(None), trace=trace, outages=outages)

    broker.get_clock = lambda: clock_box["clock"]
    return DeskScheduler(broker=broker, pipeline_factory=factory, state_path=tmp_path / "state.json",
                         log_dir=tmp_path / "logs", watchlist=watchlist, dry_run=dry_run,
                         now=lambda: clock_box["now"])


T0 = datetime(2026, 9, 28, 13, 5, tzinfo=timezone.utc)


def test_a_trading_day(tmp_path):
    broker = FakeBroker()
    box = {"clock": EARLY, "now": T0}
    sched = _scheduler(tmp_path, broker, box)
    assert sched.tick() == IDLE  # before the decision time

    box["clock"] = PRE_MARKET
    assert sched.tick() == DECISION
    state = DeskState.load(tmp_path / "state.json")
    assert state.last_decision_session == "2026-09-28"
    assert {d["symbol"]: (d["outcome"], d["recommendation"]) for d in state.decisions} == {
        "MSFT": ("agree", "buy"), "NVDA": ("agree", "hold")}
    assert set(state.pending) == {"AAPL"} and state.pending["AAPL"]["attempts"] == 1
    assert "could not reach analyst_2" in state.pending["AAPL"]["reason"]
    assert set(state.positions) == {"MSFT"} and broker.submitted[0][:3] == ("MSFT", 98, "buy")
    assert state.cycles[-1]["kind"] == DECISION and state.cycles[-1]["pending"] == ["AAPL"]

    # the retry waits RETRY_EVERY after the decision cycle, to give Build time to recover
    box["clock"], box["now"] = MIDDAY, T0 + timedelta(minutes=10)
    assert sched.tick() == IDLE
    box["now"] = T0 + RETRY_EVERY
    assert sched.tick() == RETRY
    state = DeskState.load(tmp_path / "state.json")
    assert state.pending == {}
    assert state.decisions[-1]["symbol"] == "AAPL" and state.decisions[-1]["recommendation"] == "hold"
    assert state.cycles[-1]["kind"] == RETRY and state.cycles[-1]["decisions"] == {"AAPL": "agree hold"}
    assert sched.tick() == IDLE  # decided, nothing pending

    box["clock"] = AFTER_CLOSE
    assert sched.tick() == IDLE


def test_restart_keeps_the_days_work(tmp_path):
    broker = FakeBroker()
    box = {"clock": PRE_MARKET, "now": T0}
    assert _scheduler(tmp_path, broker, box).tick() == DECISION
    restarted = _scheduler(tmp_path, broker, box)
    assert restarted.state.last_decision_session == "2026-09-28" and set(restarted.state.pending) == {"AAPL"}
    box["now"] = T0 + timedelta(minutes=1)
    assert restarted.tick() == RETRY  # a fresh process retries pending work at once
    assert len(broker.submitted) == 1  # MSFT was not bought twice


def test_next_day_expires_stale_pending_and_revisits_held_stocks(tmp_path):
    broker = FakeBroker(positions=[{"symbol": "AMD", "qty": "10"}])
    box = {"clock": PRE_MARKET, "now": T0}
    sched = _scheduler(tmp_path, broker, box, watchlist=("NVDA",))
    sched.state.pending["AAPL"] = {"session": "2026-09-25", "run": {}, "attempts": 3}
    sched.state.positions["AMD"] = {"qty": 10, "entry_session": "2026-09-21", "horizon_end": "2026-09-28"}
    sched.decision_cycle("2026-09-28")
    assert sched.state.pending == {}
    assert [d["symbol"] for d in sched.state.decisions] == ["NVDA", "AMD"]
    # AMD's horizon ended today and the desk only agreed on hold: sold at the open
    sells = [s for s in broker.submitted if s[2] == "sell"]
    assert sells == [("AMD", 10, "sell", "desk-2026-09-28-AMD-sell")] and "AMD" not in sched.state.positions
    trace = [json.loads(l) for l in next((tmp_path / "logs").glob("*.jsonl")).read_text(encoding="utf-8").splitlines()]
    start = next(e for e in trace if e["event"] == "cycle_start")
    assert start["expired_pending"] == ["AAPL"] and start["symbols"] == ["NVDA", "AMD"]


def test_weekend_is_idle(tmp_path):
    box = {"clock": WEEKEND, "now": T0}
    assert _scheduler(tmp_path, FakeBroker(), box).tick() == IDLE


def test_a_failing_tick_is_logged_and_the_loop_goes_on(tmp_path):
    box = {"clock": PRE_MARKET, "now": T0}
    sched = _scheduler(tmp_path, FakeBroker(), box)

    def broken_clock():
        raise ConnectionError("alpaca down")

    sched._broker.get_clock = broken_clock

    class Stop(Exception):
        pass

    naps = []

    def sleep(seconds):
        naps.append(seconds)
        if len(naps) == 2:
            raise Stop

    try:
        sched.run_forever(sleep=sleep)
    except Stop:
        pass
    trace = next((tmp_path / "logs").glob("*.jsonl")).read_text(encoding="utf-8")
    assert trace.count("tick_failed") == 2 and "alpaca down" in trace
