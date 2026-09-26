"""The desk's output - step 6 of the pipeline (spec section 3): turn final
decisions into paper-trading orders under rules stated up front, so every
order can be traced to the decision that caused it.

Rules:
- Only an agreed "buy" opens a position (long-only, real shares, no
  leverage - decided Sept 25, 2026).
- A position the desk opened is sold when the desk agrees on "avoid" for
  it, or when the horizon it was bought for has run out: HOLDING_SESSIONS
  trading sessions after entry, matching the analysts' 2-5 trading-day
  horizon. An agreed "buy" for a stock already held starts a fresh horizon
  instead of adding shares.
- Everything else - hold, abort, deferred, a split the critic loop didn't
  resolve - changes nothing.
- Sizing: each new position gets POSITION_FRACTION of account equity in
  whole shares, at most MAX_POSITIONS open, never more than the cash left
  (no margin). Sells go first so they free cash for the buys.
- Positions the desk did not open itself are never touched.
- Every order carries a client_order_id built from session, symbol and
  side, so retrying a submission can never place a second order.

With dry_run (the default everywhere except the deployed scheduler) the
plan is logged and recorded but nothing is sent to the broker.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any

from .cross_check import AGREE
from .trace import TraceLogger

AGENT_NAME = "portfolio"

POSITION_FRACTION = 0.10
MAX_POSITIONS = 8
HOLDING_SESSIONS = 5
# Orders are sized on the last close but fill at the next open: keep a
# margin so a gap up doesn't push the order past the cash available.
PRICE_BUFFER = 1.02


@dataclass
class Decision:
    """The desk's final word on one stock for one trading session."""

    symbol: str
    session: str
    outcome: str | None
    recommendation: str | None
    reason: str
    decided_at: str
    price: float | None = None
    # Later gates on an agreed buy (bias and technical agents) - each
    # {"gate", "passed", "reason"}; a failed gate stops the buy.
    gates: list[dict[str, Any]] = field(default_factory=list)

    @property
    def agreed(self) -> str | None:
        """The agreed recommendation, or None when there is no agreement."""
        return self.recommendation if self.outcome == AGREE else None

    @property
    def blocked(self) -> bool:
        return any(not g.get("passed", False) for g in self.gates)

    @classmethod
    def from_run(cls, run: Any, session: str, decided_at: str) -> Decision:
        reason = ""
        if run.critic_loop is not None:
            reason = run.critic_loop.reason
        elif run.cross_check is not None:
            reason = run.cross_check.reason
        price = (run.context.price or {}).get("last_close")
        gates = [stage.gate for stage in (getattr(run, "bias", None), getattr(run, "technical", None))
                 if stage is not None]
        return cls(symbol=run.symbol, session=session, outcome=run.outcome, recommendation=run.recommendation,
                   reason=reason, decided_at=decided_at, price=float(price) if price else None, gates=gates)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OrderPlan:
    symbol: str
    side: str
    qty: int
    reason: str
    client_order_id: str


def client_order_id(session: str, symbol: str, side: str) -> str:
    return f"desk-{session}-{symbol}-{side}"


def session_after(sessions: list[str], start: str, n: int) -> str | None:
    """The trading session `n` sessions after `start`, from a sorted list of
    session dates; None when the list doesn't reach that far."""
    later = [s for s in sessions if s > start]
    return later[n - 1] if len(later) >= n else None


class Portfolio:
    def __init__(
        self,
        broker: Any,
        *,
        trace: TraceLogger | None = None,
        dry_run: bool = True,
        position_fraction: float = POSITION_FRACTION,
        max_positions: int = MAX_POSITIONS,
        holding_sessions: int = HOLDING_SESSIONS,
    ):
        self._broker = broker
        self._trace = trace
        self.dry_run = dry_run
        self.position_fraction = position_fraction
        self.max_positions = max_positions
        self.holding_sessions = holding_sessions

    def _log(self, event: str, payload: dict[str, Any]) -> None:
        if self._trace is not None:
            self._trace.log(AGENT_NAME, event, payload)

    def trading_sessions(self, start: str, days: int = 20) -> list[str]:
        """Session dates from `start` on (enough to reach any horizon end)."""
        first = date.fromisoformat(start)
        calendar = self._broker.get_calendar(first, first + timedelta(days=days))
        return sorted(str(day["date"]) for day in calendar)

    def plan(
        self,
        decisions: list[Decision],
        positions: dict[str, dict[str, Any]],
        *,
        session: str,
        check_horizons: bool,
    ) -> tuple[list[OrderPlan], list[dict[str, Any]]]:
        """Orders for these decisions (sells first), plus notes on anything
        that was decided but not acted on (no slot, no cash, a failed gate).
        `positions` is the desk's own record (DeskState.positions); horizon
        exits are only checked when `check_horizons` (the day's decision
        cycle)."""
        broker_qty = {p["symbol"]: int(float(p["qty"])) for p in self._broker.list_positions()}
        by_symbol = {d.symbol: d for d in decisions}
        sells: list[OrderPlan] = []
        buys: list[OrderPlan] = []
        notes: list[dict[str, Any]] = []

        for symbol, pos in positions.items():
            # a dry-run position exists only in the desk's record
            held = int(pos.get("qty") or 0) if pos.get("dry_run") else broker_qty.get(symbol, 0)
            decision = by_symbol.get(symbol)
            if decision is not None and decision.agreed == "avoid":
                why = f"the desk agreed on avoid: {decision.reason}"
            elif check_horizons and pos.get("horizon_end") and session >= pos["horizon_end"] \
                    and not (decision is not None and decision.agreed == "buy" and not decision.blocked):
                why = f"its {self.holding_sessions}-session horizon ended ({pos['horizon_end']})"
            else:
                continue
            if held <= 0:
                notes.append({"symbol": symbol, "note": "the desk's position is not at the broker "
                                                         "(order never filled?) - dropped from the record"})
                continue
            sells.append(OrderPlan(symbol, "sell", min(held, int(pos.get("qty") or held)), f"sell: {why}",
                                   client_order_id(session, symbol, "sell")))

        account = self._broker.get_account()
        equity = float(account.get("equity") or 0)
        cash = float(account.get("cash") or 0) + sum(
            (by_symbol[s.symbol].price or 0) * s.qty for s in sells if s.symbol in by_symbol)
        open_slots = self.max_positions - (len(positions) - len(sells))
        for decision in sorted(decisions, key=lambda d: d.symbol):
            if decision.agreed != "buy":
                continue
            if decision.symbol in positions:
                continue  # re-affirmed: a fresh horizon, handled by record()
            if decision.blocked:
                failed = [g for g in decision.gates if not g.get("passed", False)]
                notes.append({"symbol": decision.symbol, "note": "agreed buy stopped by "
                              + "; ".join(f"{g.get('gate')}: {g.get('reason')}" for g in failed)})
                continue
            if broker_qty.get(decision.symbol, 0) > 0:
                notes.append({"symbol": decision.symbol, "note": "already held outside the desk - not touched"})
                continue
            if open_slots <= 0:
                notes.append({"symbol": decision.symbol, "note": f"agreed buy, but {self.max_positions} "
                                                                 "positions are already open"})
                continue
            if not decision.price:
                notes.append({"symbol": decision.symbol, "note": "agreed buy, but no price to size it"})
                continue
            budget = min(equity * self.position_fraction, cash)
            qty = math.floor(budget / (decision.price * PRICE_BUFFER))
            if qty < 1:
                notes.append({"symbol": decision.symbol, "note": f"agreed buy, but the budget ({budget:.0f}) "
                                                                 "buys less than one share"})
                continue
            buys.append(OrderPlan(decision.symbol, "buy", qty, f"buy: the desk agreed on buy: {decision.reason}",
                                  client_order_id(session, decision.symbol, "buy")))
            cash -= qty * decision.price * PRICE_BUFFER
            open_slots -= 1
        plans = sells + buys
        self._log("plan", {"session": session, "orders": [asdict(p) for p in plans], "notes": notes,
                           "dry_run": self.dry_run})
        return plans, notes

    def execute(self, plans: list[OrderPlan]) -> list[dict[str, Any]]:
        """Send the orders (unless dry_run). A failed submission is recorded,
        never raised: one broker hiccup must not stop the other orders."""
        results = []
        for plan in plans:
            result: dict[str, Any] = {**asdict(plan)}
            if self.dry_run:
                result["status"] = "dry_run"
            else:
                try:
                    order = self._broker.submit_market_order(
                        plan.symbol, plan.qty, plan.side, client_order_id=plan.client_order_id)
                    result.update(status=order.get("status"), order_id=order.get("id"))
                except Exception as exc:  # noqa: BLE001 - recorded in the result and the trace
                    existing = self._existing_order(plan.client_order_id)
                    if existing is not None:  # sent before (a retry or a replayed cycle): same order
                        result.update(status=existing.get("status"), order_id=existing.get("id"), duplicate=True)
                    else:
                        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            self._log("order", result)
            results.append(result)
        return results

    def _existing_order(self, client_id: str) -> dict[str, Any] | None:
        try:
            return self._broker.get_order_by_client_id(client_id)
        except Exception:  # noqa: BLE001 - no such order (or the broker is down): the submit failed
            return None

    def record(
        self,
        positions: dict[str, dict[str, Any]],
        decisions: list[Decision],
        results: list[dict[str, Any]],
        notes: list[dict[str, Any]],
        *,
        session: str,
    ) -> None:
        """Update the desk's own position record from what was sent: new
        buys get their horizon, sells and positions missing at the broker
        are dropped, and a re-affirmed buy restarts the horizon."""
        horizon_end = session_after(self.trading_sessions(session), session, self.holding_sessions)
        for result in results:
            if result.get("status") == "failed":
                continue
            if result["side"] == "buy":
                decision = next(d for d in decisions if d.symbol == result["symbol"])
                positions[result["symbol"]] = {
                    "qty": result["qty"], "entry_session": session, "horizon_end": horizon_end,
                    "order_id": result.get("order_id"), "client_order_id": result["client_order_id"],
                    "decided_at": decision.decided_at, "reason": decision.reason,
                    "dry_run": result.get("status") == "dry_run",
                }
            else:
                positions.pop(result["symbol"], None)
        for note in notes:
            if "dropped from the record" in note["note"]:
                positions.pop(note["symbol"], None)
        for decision in decisions:
            pos = positions.get(decision.symbol)
            if pos is not None and decision.agreed == "buy" and not decision.blocked \
                    and pos.get("entry_session") != session:
                pos["horizon_end"] = horizon_end
                pos["reaffirmed"] = session
