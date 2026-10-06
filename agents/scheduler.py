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
A final decision is saved as it settles, and its orders go out at the end
of the pass. If they can't (a restart, a broker failure), the next tick
sends them - the desk is never re-run for orders it already decided.
Every pass is saved to the state file (agents/state.py) and summarized in
the trace and the state, including LLM successes and failures per UTC hour.
"""
from __future__ import annotations

import argparse
import sys
import time as time_module
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from data_layer import AlpacaClient

from .pipeline import DeskPipeline, ModelOutages, SymbolRun, default_trace_path
from .portfolio import Decision, Portfolio
from .state import DeskState
from .technical_agent import PASSED, WAITING
from .trace import TraceLogger, failures_by_hour, read_trace, summarize_trace

AGENT_NAME = "scheduler"

# Liquid US large caps, all US-GAAP filers with complete data (checked live
# Sept 26, 2026: every source loads and the valuation is computable). The
# first 11 are the stocks the desk was tested on; the rest widen it beyond
# tech (decided Sept 26). XOM was left out: SEC's data has no fiscal-2025
# annual figures for it, so no trailing valuation; CVX covers energy. Oct 6:
# 18 more (AVGO and others) once longer cycles were acceptable; each loaded
# with no data gap in a data-only run. V was left out: its filings give no
# trailing EPS or share count.
DEFAULT_WATCHLIST = (
    "AAPL", "MSFT", "NVDA", "AMD", "META", "INTC", "ADBE", "NFLX",  # tech and communication
    "CVX",                                                          # energy
    "JPM", "GS",                                                    # financials
    "MRK", "LLY", "UNH",                                            # health care
    "PG", "KO", "WMT",                                              # consumer staples
    "NKE", "HD",                                                    # consumer discretionary
    "CAT", "GE",                                                    # industrials
    "NEE",                                                          # utilities
    "AVGO", "GOOGL", "AMZN", "TSLA", "ORCL", "CRM", "QCOM", "TXN", "AMAT", "MU",  # more tech
    "MA", "BAC",                                                    # more financials
    "ABBV", "JNJ",                                                  # more health care
    "COST", "PEP", "MCD",                                           # more consumer
    "UNP",                                                          # more industrials
)
# 90 minutes before the open: a full watchlist cycle took up to an hour on a
# slow Build day (Sept 2026), and orders sent before 09:30 queue for the open.
DECISION_TIME_ET = time(8, 0)
RETRY_EVERY = timedelta(minutes=30)
RETRY_UNTIL_ET = time(15, 0)
# Once a session (decided Oct 6, 2026), the technical agent is asked again
# about every agreed buy it held back that morning with a wait: the pick
# stands, only the entry is re-timed on fresh bars. 11:30 ET gives the open's
# swings two hours to settle and leaves the afternoon to fill.
RECHECK_TIME_ET = time(11, 30)
POLL_SECONDS = 60
# A state file belongs to one mode. A log-only run's file must never steer
# a scheduler that places paper orders: it would skip the session the log-only
# run already decided, and later sell shares that were never bought (Sept 28,
# 2026: the README's dry run and the service shared one file).
PAPER_STATE = "state/desk_state.json"
LOG_ONLY_STATE = "state/desk_state-log-only.json"
# Stocks going through the desk at once in a decision cycle, so the whole
# watchlist is decided before the 09:30 open (about 9 minutes per stock on
# a slow Build morning). Build's ceiling is 40 requests/minute per model;
# four stocks (at most two calls each at a time) keep well under it.
WORKERS = 4

DECISION, RETRY, ORDERS, RECHECK, IDLE = "decision", "retry", "orders", "recheck", "idle"


def default_state_path(paper_orders: bool) -> str:
    return PAPER_STATE if paper_orders else LOG_ONLY_STATE


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
        self._claim_state()
        # When the last pass (decision or retry) ended: a retry waits
        # RETRY_EVERY after it, so a Build outage has time to clear.
        self._last_pass: datetime | None = None

    # --- plumbing ---

    def _claim_state(self) -> None:
        """Refuse a state file written in the other mode (see PAPER_STATE)."""
        recorded = self.state.dry_run
        if recorded is None and self.state.cycles:  # written before the mode was recorded
            recorded = bool(self.state.cycles[-1].get("dry_run"))
        if recorded is not None and recorded != self.dry_run:
            raise ValueError(
                f"{self.state_path} belongs to a {'log-only' if recorded else 'paper-order'} scheduler; "
                f"give this run its own --state (defaults: {PAPER_STATE} with --paper-orders, "
                f"{LOG_ONLY_STATE} without)")
        self.state.dry_run = self.dry_run

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
        """Do whatever is due now: the day's decision cycle, the orders of a
        pass that was cut short, a retry pass, or nothing. Returns which."""
        clock = self._broker.get_clock()
        now_et = datetime.fromisoformat(clock["timestamp"])
        session = session_for(clock)
        if session != now_et.date().isoformat():
            return IDLE  # after the close, or no session today
        if self.state.last_decision_session != session and now_et.time() >= DECISION_TIME_ET:
            self.decision_cycle(session)
            return DECISION
        if self.state.unsent and self.state.unsent["session"] == session:
            self.send_unsent(session)
            return ORDERS
        if (self.state.last_decision_session == session and self.state.last_recheck_session != session
                and RECHECK_TIME_ET <= now_et.time() < RETRY_UNTIL_ET):
            self.recheck_entries(session)
            return RECHECK
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
                                              "expired_pending": expired, "expired_unsent": self._expire_unsent(session)})
        self._unsent(session)["check_horizons"] = True  # horizon exits are due even if every stock is deferred
        decisions = []
        for run in pipeline.run(symbols, workers=self.workers):
            decision = self._settle(run, session)
            if decision is not None:
                decisions.append(decision)
        # Decided from here on: if the orders fail, the next tick sends them
        # rather than re-running the whole desk (an hour and ~1M tokens).
        self.state.last_decision_session = session
        self._save(pipeline)
        orders = self._act(session, trace)
        return self._finish(DECISION, session, started, trace, pipeline, decisions, orders)

    def retry_pass(self, session: str) -> dict[str, Any]:
        """Resume every deferred stock of this session from its failed step."""
        started = self._now()
        trace = self._trace()
        pipeline = self._pipeline(trace)
        expired = self._expire_pending(session)
        trace.log(AGENT_NAME, "cycle_start", {"kind": RETRY, "session": session,
                                              "symbols": sorted(self.state.pending), "expired_pending": expired,
                                              "expired_unsent": self._expire_unsent(session)})
        decisions = []
        for symbol in sorted(self.state.pending):
            run = pipeline.resume_symbol(SymbolRun.from_dict(self.state.pending[symbol]["run"]))
            decision = self._settle(run, session)
            if decision is not None:
                decisions.append(decision)
            self._save(pipeline)  # the decision and its place in `unsent` survive a restart
        orders = self._act(session, trace)
        return self._finish(RETRY, session, started, trace, pipeline, decisions, orders)

    def recheck_entries(self, session: str) -> list[dict[str, Any]]:
        """The midday pass: the technical agent re-times every agreed buy it
        waited on this morning (a stock not already held). An *enter* turns
        the decision into a buy, sent the usual way; a wait, a failure or an
        unreachable model leaves things as they were - once per session."""
        started = self._now()
        trace = self._trace()
        self.state.last_recheck_session = session  # once: a failed pass isn't retried in a loop
        latest = {}
        for d in self.state.decisions:
            if d["session"] == session:
                latest[d["symbol"]] = Decision(**d)  # the day's last word on each stock
        candidates = {
            symbol: decision for symbol, decision in latest.items()
            if any(g.get("gate") == "technical" and g.get("outcome") == WAITING for g in decision.gates)
            and decision.reaffirms and symbol not in self.state.positions}
        trace.log(AGENT_NAME, "recheck_start", {"session": session, "symbols": sorted(candidates)})
        if not candidates:
            self._save()
            return []
        pipeline = self._pipeline(trace)
        results = pipeline.recheck_entries(sorted(candidates), workers=self.workers)
        entering = []
        for symbol, result in sorted(results.items()):
            if result.outcome != PASSED:
                continue
            old = candidates[symbol]
            gates = [g for g in old.gates if g.get("gate") != "technical"] + [result.gate]
            decision = Decision(symbol=symbol, session=session, outcome=old.outcome, recommendation=old.recommendation,
                                reason=f"{old.reason} (entry timed at midday: {result.reason})",
                                decided_at=self._now().isoformat(), price=result.price or old.price, gates=gates)
            self.state.decisions.append(decision.to_dict())
            self._unsent(session)["decisions"].append(decision.to_dict())
            entering.append(symbol)
        self._save(pipeline)
        orders = self._act(session, trace)
        summary = {"session": session, "started": started.isoformat(), "finished": self._now().isoformat(),
                   "rechecked": {s: r.outcome for s, r in sorted(results.items())}, "entering": entering,
                   "orders": orders, "dry_run": self.dry_run}
        trace.log(AGENT_NAME, "recheck_end", summary)
        print(f"midday recheck for session {session}: {len(results)} stocks re-timed, {len(entering)} entering, "
              f"{sum(1 for o in orders if 'side' in o)} orders", flush=True)
        return orders

    def send_unsent(self, session: str) -> list[dict[str, Any]]:
        """Send the orders of decisions a pass settled but didn't act on."""
        trace = self._trace()
        symbols = [d["symbol"] for d in self.state.unsent["decisions"]]
        orders = self._act(session, trace)
        trace.log(AGENT_NAME, "unsent_orders", {"session": session, "symbols": symbols, "orders": orders})
        print(f"orders for session {session} sent after an interrupted pass: {len(symbols)} decisions, "
              f"{sum(1 for o in orders if 'side' in o)} orders", flush=True)
        return orders

    def _expire_pending(self, session: str) -> list[str]:
        stale = [s for s, item in self.state.pending.items() if item["session"] != session]
        for symbol in stale:
            del self.state.pending[symbol]
        return stale

    def _expire_unsent(self, session: str) -> list[str]:
        """Decisions of an earlier session never acted on are dropped: the
        market has moved on, and today's cycle decides afresh."""
        unsent = self.state.unsent
        if unsent is None or unsent["session"] == session:
            return []
        self.state.unsent = None
        return [d["symbol"] for d in unsent["decisions"]]

    def _unsent(self, session: str) -> dict[str, Any]:
        if self.state.unsent is None or self.state.unsent["session"] != session:
            self.state.unsent = {"session": session, "check_horizons": False, "decisions": []}
        return self.state.unsent

    def _settle(self, run: SymbolRun, session: str) -> Decision | None:
        """A deferred run goes (back) to pending; anything else is final."""
        now = self._now().isoformat()
        if run.deferred:
            item = self.state.pending.get(run.symbol) or {"session": session, "first_deferred": now, "attempts": 0}
            item.update(run=run.to_dict(), attempts=item["attempts"] + 1, last_attempt=now,
                        reason=run.deferred_reason)
            self.state.pending[run.symbol] = item
            return None
        self.state.pending.pop(run.symbol, None)
        decision = Decision.from_run(run, session, now)
        self.state.decisions.append(decision.to_dict())
        self._unsent(session)["decisions"].append(decision.to_dict())
        return decision

    def _act(self, session: str, trace: TraceLogger) -> list[dict[str, Any]]:
        """Orders for the session's unsent decisions. Everything that can
        fail is read before the first order goes out; a failure then leaves
        the decisions in `unsent` for the next tick."""
        batch = self.state.unsent
        if batch is None or batch["session"] != session:
            return []
        decisions = [Decision(**d) for d in batch["decisions"]]
        portfolio = Portfolio(self._broker, trace=trace, dry_run=self.dry_run)
        horizon_end = portfolio.horizon_end(session)
        plans, notes = portfolio.plan(decisions, self.state.positions, session=session,
                                      check_horizons=batch["check_horizons"], buys_allowed=self._buys_allowed(session))
        results = portfolio.execute(plans)
        portfolio.record(self.state.positions, decisions, results, notes, session=session, horizon_end=horizon_end)
        self.state.unsent = None
        self._save()  # right away: orders were just sent
        return results + [{"note": n} for n in notes]

    def _buys_allowed(self, session: str) -> bool:
        """False once the session's market has closed (a retry pass can run
        past 16:00 ET): a day order sent then waits for the next open, on a
        decision made for a session that is over."""
        return session_for(self._broker.get_clock()) == session

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
            "next_pass": self._next_pass(session),
        }
        self.state.cycles.append(summary)
        trace.log(AGENT_NAME, "cycle_end", summary)
        print(f"{kind} pass for session {session} done: {len(decisions)} decided, {len(self.state.pending)} deferred, "
              f"{sum(1 for o in orders if 'side' in o)} orders; {describe_next_pass(summary['next_pass'])}", flush=True)
        self._save(pipeline)
        self._last_pass = self._now()
        return summary


    def _next_pass(self, session: str) -> dict[str, Any] | None:
        """When the scheduler acts next, for the log: a retry pass RETRY_EVERY
        from now while a stock is deferred and there is still time today,
        otherwise the next session's decision cycle (from Alpaca's calendar,
        with the ET offset of its clock). None when either can't be read - a
        log line must never fail a pass."""
        try:
            now_et = datetime.fromisoformat(self._broker.get_clock()["timestamp"])
            retry_et = now_et + RETRY_EVERY
            if self.state.pending and retry_et.date() == now_et.date() and retry_et.time() < RETRY_UNTIL_ET:
                return {"kind": RETRY, "due_utc": (self._now() + RETRY_EVERY).isoformat(),
                        "note": f"{len(self.state.pending)} deferred"}
            day = date.fromisoformat(session)
            if self.state.last_decision_session == session:
                calendar = self._broker.get_calendar(day + timedelta(days=1), day + timedelta(days=10))
                day = min(date.fromisoformat(str(d["date"])) for d in calendar)
            due = datetime.combine(day, DECISION_TIME_ET, tzinfo=now_et.tzinfo)
            note = f"session {day.isoformat()}"
            if self.state.pending:
                note += f"; the {len(self.state.pending)} deferred are dropped then"
            return {"kind": DECISION, "due_utc": due.astimezone(timezone.utc).isoformat(), "note": note}
        except Exception:  # noqa: BLE001
            return None


def describe_next_pass(next_pass: dict[str, Any] | None) -> str:
    if next_pass is None:
        return "next pass: unknown (market clock or calendar unreadable)"
    what = "retry pass" if next_pass["kind"] == RETRY else "decision cycle"
    due = datetime.fromisoformat(next_pass["due_utc"])
    return f"next: {what} at {due:%Y-%m-%d %H:%M} UTC ({next_pass['note']})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agents.scheduler", description=__doc__.split("\n\n")[0])
    parser.add_argument("--watchlist", nargs="+", default=list(DEFAULT_WATCHLIST))
    parser.add_argument("--state", help=f"state file (default: {PAPER_STATE} with --paper-orders, "
                                        f"{LOG_ONLY_STATE} without, so the two never share one)")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--paper-orders", action="store_true",
                        help="send the orders to the Alpaca paper account (default: log only)")
    parser.add_argument("--workers", type=int, default=WORKERS, help="stocks processed at once")
    parser.add_argument("--once", choices=(DECISION, RETRY),
                        help="run one decision cycle or retry pass now for the current session, then exit")
    args = parser.parse_args(argv)
    state_path = args.state or default_state_path(args.paper_orders)

    broker = AlpacaClient()

    def factory(trace: TraceLogger, outages: ModelOutages) -> DeskPipeline:
        return DeskPipeline.from_settings(trace=trace, outages=outages)

    try:
        scheduler = DeskScheduler(broker=broker, pipeline_factory=factory, state_path=state_path,
                                  log_dir=args.log_dir, watchlist=tuple(args.watchlist),
                                  dry_run=not args.paper_orders, workers=args.workers)
    except ValueError as exc:  # a state file of the other mode, or of another version
        print(f"error: {exc}", file=sys.stderr)
        return 2
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
          f"{'paper orders' if args.paper_orders else 'log-only orders'}; state {state_path}", flush=True)
    scheduler.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
