"""Runs the desk unattended - the long-running part of the demo (spec
section 7: about two days of continuous running on the Oracle A1 box).

    python -m agents.scheduler                 # loop forever, log-only orders
    python -m agents.scheduler --paper-orders  # ... and place paper orders
    python -m agents.scheduler --once decision # one decision cycle now, then exit

Every trading day (Alpaca's market clock decides what one is):
- one decision cycle at DECISION_TIME_ET (08:00), before the open: every watchlist
  stock, plus every stock the desk holds, gets a fresh context and the full
  desk; final decisions go to the portfolio step, whose orders queue for
  the open. Before the open the latest daily bar is a finished session, so
  no stock is judged on half a day's data, and Build is quieter at that
  hour (Sept 25, 2026: 14 of 17 failures came between 21:00 and 24:00 UTC).
- retry passes every RETRY_EVERY until RETRY_UNTIL_ET: stocks deferred by a
  Build failure resume from the failed step with the morning's context.
  A stock still deferred after that waits for the next day's fresh cycle -
  its context would be stale by then.
Every pass is saved to the state file (agents/state.py) and summarized in
the trace and the state, including LLM successes and failures per UTC hour.
"""
from __future__ import annotations

import argparse
import sys
import time as time_module
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from data_layer import AlpacaClient

from .pipeline import DeskPipeline, ModelOutages, SymbolRun, default_trace_path
from .portfolio import Decision, Portfolio
from .state import DeskState
from .trace import TraceLogger, failures_by_hour, read_trace, summarize_trace

AGENT_NAME = "scheduler"

# Liquid large caps across sectors: the stocks the desk was tested on.
DEFAULT_WATCHLIST = ("AAPL", "MSFT", "NVDA", "AMD", "META", "INTC", "CVX", "NFLX", "NKE", "MRK", "ADBE")
# 90 minutes before the open: a full watchlist cycle took up to an hour on a
# slow Build day (Sept 2026), and orders sent before 09:30 queue for the open.
DECISION_TIME_ET = time(8, 0)
RETRY_EVERY = timedelta(minutes=30)
RETRY_UNTIL_ET = time(15, 0)
POLL_SECONDS = 60
# Stocks going through the desk at once in a decision cycle. Build's
# ceiling is 40 requests/minute per model; three stocks keep well under it.
WORKERS = 3

DECISION, RETRY, IDLE = "decision", "retry", "idle"


def session_for(clock: dict[str, Any]) -> str:
    """The trading session decisions made now are for: today's while the
    market is open, otherwise the next one to open (which is today before
    the open, and the next trading day after the close or on a weekend)."""
    if clock.get("is_open"):
        return datetime.fromisoformat(clock["timestamp"]).date().isoformat()
    return datetime.fromisoformat(clock["next_open"]).date().isoformat()


class DeskScheduler:
    def __init__(
        self,
        *,
        broker: Any,
        pipeline_factory: Callable[[TraceLogger, ModelOutages], DeskPipeline],
        state_path: str | Path,
        log_dir: str | Path = "logs",
        watchlist: tuple[str, ...] = DEFAULT_WATCHLIST,
        dry_run: bool = True,
        workers: int = WORKERS,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self._broker = broker
        self.workers = workers
        self._make_pipeline = pipeline_factory
        self.state_path = Path(state_path)
        self.log_dir = Path(log_dir)
        self.watchlist = tuple(s.upper() for s in watchlist)
        self.dry_run = dry_run
        self._now = now
        self.state = DeskState.load(self.state_path)
        # When the last pass (decision or retry) ended: a retry waits
        # RETRY_EVERY after it, so a Build outage has time to clear.
        self._last_pass: datetime | None = None

    # --- plumbing ---

    def _trace(self) -> TraceLogger:
        return TraceLogger(default_trace_path(self.log_dir))

    def _pipeline(self, trace: TraceLogger) -> DeskPipeline:
        return self._make_pipeline(trace, ModelOutages(dict(self.state.outages)))

    def _save(self, pipeline: DeskPipeline | None = None) -> None:
        if pipeline is not None:
            self.state.outages = dict(pipeline.outages.down_since)
        self.state.save(self.state_path)

    # --- one scheduling step ---

    def tick(self) -> str:
        """Do whatever is due now: the day's decision cycle, a retry pass,
        or nothing. Returns which."""
        clock = self._broker.get_clock()
        now_et = datetime.fromisoformat(clock["timestamp"])
        session = session_for(clock)
        if session != now_et.date().isoformat():
            return IDLE  # after the close, or no session today
        if self.state.last_decision_session != session and now_et.time() >= DECISION_TIME_ET:
            self.decision_cycle(session)
            return DECISION
        if self.state.pending and now_et.time() < RETRY_UNTIL_ET and (
            self._last_pass is None or self._now() - self._last_pass >= RETRY_EVERY
        ):
            self.retry_pass(session)
            return RETRY
        return IDLE

    def run_forever(self, sleep: Callable[[float], None] = time_module.sleep) -> None:
        while True:
            try:
                action = self.tick()
            except Exception as exc:  # noqa: BLE001 - a bad tick must not end a multi-day run
                self._trace().log(AGENT_NAME, "tick_failed", {"error": f"{type(exc).__name__}: {exc}"})
                action = IDLE
            if action == IDLE:
                sleep(POLL_SECONDS)

    # --- cycles ---

    def decision_cycle(self, session: str) -> dict[str, Any]:
        """Fresh contexts and the full desk for the watchlist and every stock
        the desk holds; deferred stocks go to `pending` for the retry passes."""
        started = self._now()
        trace = self._trace()
        pipeline = self._pipeline(trace)
        expired = self._expire_pending(session)
        symbols = list(dict.fromkeys([*self.watchlist, *sorted(self.state.positions)]))
        trace.log(AGENT_NAME, "cycle_start", {"kind": DECISION, "session": session, "symbols": symbols,
                                              "expired_pending": expired})
        decisions = []
        for run in pipeline.run(symbols, workers=self.workers):
            decision = self._settle(run, session)
            if decision is not None:
                decisions.append(decision)
        self._save(pipeline)
        orders = self._act(decisions, session, trace, check_horizons=True)
        self.state.last_decision_session = session
        return self._finish(DECISION, session, started, trace, pipeline, decisions, orders)

    def retry_pass(self, session: str) -> dict[str, Any]:
        """Resume every deferred stock of this session from its failed step."""
        started = self._now()
        trace = self._trace()
        pipeline = self._pipeline(trace)
        expired = self._expire_pending(session)
        trace.log(AGENT_NAME, "cycle_start", {"kind": RETRY, "session": session,
                                              "symbols": sorted(self.state.pending), "expired_pending": expired})
        decisions = []
        for symbol in sorted(self.state.pending):
            run = pipeline.resume_symbol(SymbolRun.from_dict(self.state.pending[symbol]["run"]))
            decision = self._settle(run, session)
            if decision is not None:
                decisions.append(decision)
            self._save(pipeline)
        orders = self._act(decisions, session, trace, check_horizons=False)
        return self._finish(RETRY, session, started, trace, pipeline, decisions, orders)

    def _expire_pending(self, session: str) -> list[str]:
        stale = [s for s, item in self.state.pending.items() if item["session"] != session]
        for symbol in stale:
            del self.state.pending[symbol]
        return stale

    def _settle(self, run: SymbolRun, session: str) -> Decision | None:
        """A deferred run goes (back) to pending; anything else is final."""
        now = self._now().isoformat()
        if run.deferred:
            item = self.state.pending.get(run.symbol) or {"session": session, "first_deferred": now, "attempts": 0}
            item.update(run=run.to_dict(), attempts=item["attempts"] + 1, last_attempt=now,
                        reason=(run.critic_loop or run.cross_check).reason)
            self.state.pending[run.symbol] = item
            return None
        self.state.pending.pop(run.symbol, None)
        decision = Decision.from_run(run, session, now)
        self.state.decisions.append(decision.to_dict())
        return decision

    def _act(self, decisions: list[Decision], session: str, trace: TraceLogger, *,
             check_horizons: bool) -> list[dict[str, Any]]:
        portfolio = Portfolio(self._broker, trace=trace, dry_run=self.dry_run)
        plans, notes = portfolio.plan(decisions, self.state.positions, session=session,
                                      check_horizons=check_horizons)
        results = portfolio.execute(plans)
        portfolio.record(self.state.positions, decisions, results, notes, session=session)
        self._save()  # right away: orders were just sent
        return results + [{"note": n} for n in notes]

    def _finish(self, kind: str, session: str, started: datetime, trace: TraceLogger,
                pipeline: DeskPipeline, decisions: list[Decision], orders: list[dict[str, Any]]) -> dict[str, Any]:
        events = read_trace(trace.path) if trace.path.exists() else []
        since = started.isoformat()
        recent = [e for e in events if e.get("ts", "") >= since]
        summary = {
            "kind": kind,
            "session": session,
            "started": since,
            "finished": self._now().isoformat(),
            "decisions": {d.symbol: f"{d.outcome} {d.recommendation or ''}".strip() for d in decisions},
            "pending": sorted(self.state.pending),
            "orders": orders,
            "llm": summarize_trace(recent),
            "llm_by_utc_hour": failures_by_hour(recent),
            "dry_run": self.dry_run,
        }
        self.state.cycles.append(summary)
        trace.log(AGENT_NAME, "cycle_end", summary)
        self._save(pipeline)
        self._last_pass = self._now()
        return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agents.scheduler", description=__doc__.split("\n\n")[0])
    parser.add_argument("--watchlist", nargs="+", default=list(DEFAULT_WATCHLIST))
    parser.add_argument("--state", default="state/desk_state.json")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--paper-orders", action="store_true",
                        help="send the orders to the Alpaca paper account (default: log only)")
    parser.add_argument("--workers", type=int, default=WORKERS, help="stocks processed at once")
    parser.add_argument("--once", choices=(DECISION, RETRY),
                        help="run one decision cycle or retry pass now for the current session, then exit")
    args = parser.parse_args(argv)

    broker = AlpacaClient()

    def factory(trace: TraceLogger, outages: ModelOutages) -> DeskPipeline:
        return DeskPipeline.from_settings(trace=trace, outages=outages)

    scheduler = DeskScheduler(broker=broker, pipeline_factory=factory, state_path=args.state,
                              log_dir=args.log_dir, watchlist=tuple(args.watchlist),
                              dry_run=not args.paper_orders, workers=args.workers)
    if args.once:
        session = session_for(broker.get_clock())
        run = scheduler.decision_cycle if args.once == DECISION else scheduler.retry_pass
        summary = run(session)
        print(f"{args.once} pass for session {session}:")
        for symbol, outcome in summary["decisions"].items():
            print(f"  {symbol}: {outcome}")
        if summary["pending"]:
            print(f"  deferred (retried later): {', '.join(summary['pending'])}")
        for order in summary["orders"]:
            print(f"  {order}")
        return 0
    print(f"Scheduler running: watchlist {', '.join(scheduler.watchlist)}; "
          f"{'paper orders' if args.paper_orders else 'log-only orders'}; state {args.state}", flush=True)
    scheduler.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
