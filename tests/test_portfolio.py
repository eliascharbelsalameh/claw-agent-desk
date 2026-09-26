"""Portfolio rules: what each final decision does to the paper account."""
from datetime import date, timedelta

import pytest

from agents.portfolio import Decision, Portfolio, client_order_id, session_after


class FakeBroker:
    def __init__(self, equity=100_000.0, cash=100_000.0, positions=None, fail_symbols=()):
        self.account = {"equity": str(equity), "cash": str(cash)}
        self.positions = positions or []
        self.fail_symbols = set(fail_symbols)
        self.orders = {}
        self.submitted = []

    def get_account(self):
        return self.account

    def list_positions(self):
        return self.positions

    def get_calendar(self, start, end):
        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        return [{"date": d.isoformat()} for d in days if d.weekday() < 5]

    def submit_market_order(self, symbol, qty, side, time_in_force="day", client_order_id=None):
        self.submitted.append((symbol, qty, side, client_order_id))
        if symbol in self.fail_symbols:
            raise ConnectionError("broker down")
        if client_order_id in self.orders:
            raise RuntimeError("422 client_order_id must be unique")
        order = {"id": f"order-{len(self.orders) + 1}", "status": "accepted", "client_order_id": client_order_id}
        self.orders[client_order_id] = order
        return order

    def get_order_by_client_id(self, client_id):
        if client_id not in self.orders:
            raise KeyError(client_id)
        return self.orders[client_id]


SESSION = "2026-09-28"  # a Monday


def _decision(symbol, outcome="agree", rec="buy", price=100.0, gates=None):
    return Decision(symbol=symbol, session=SESSION, outcome=outcome, recommendation=rec, reason=f"{symbol} reason",
                    decided_at="2026-09-28T13:00:00+00:00", price=price, gates=gates or [])


def _run_step(portfolio, decisions, positions, check_horizons=True, session=SESSION):
    plans, notes = portfolio.plan(decisions, positions, session=session, check_horizons=check_horizons)
    results = portfolio.execute(plans)
    portfolio.record(positions, decisions, results, notes, session=session)
    return plans, notes, results


def test_session_after_counts_trading_sessions():
    sessions = ["2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01", "2026-10-02", "2026-10-05"]
    assert session_after(sessions, "2026-09-28", 5) == "2026-10-05"
    assert session_after(sessions, "2026-09-28", 6) is None


def test_agreed_buy_opens_a_sized_position_with_its_horizon():
    broker = FakeBroker()
    positions = {}
    plans, notes, results = _run_step(Portfolio(broker, dry_run=False), [_decision("AAPL", price=341.0)], positions)
    # 10% of 100k at 341 * 1.02 -> 28 whole shares
    assert [(p.symbol, p.side, p.qty) for p in plans] == [("AAPL", "buy", 28)]
    assert broker.submitted == [("AAPL", 28, "buy", "desk-2026-09-28-AAPL-buy")]
    pos = positions["AAPL"]
    assert pos["entry_session"] == SESSION and pos["horizon_end"] == "2026-10-05"
    assert pos["order_id"] == "order-1" and pos["dry_run"] is False
    assert notes == []


@pytest.mark.parametrize("outcome, rec", [("abort", None), ("deferred", None), ("critic", None),
                                          ("agree", "hold"), ("agree", "avoid")])
def test_anything_but_an_agreed_buy_opens_nothing(outcome, rec):
    broker = FakeBroker()
    plans, _, _ = _run_step(Portfolio(broker, dry_run=False), [_decision("AAPL", outcome, rec)], {})
    assert plans == [] and broker.submitted == []


def test_agreed_avoid_sells_a_desk_position_with_the_brokers_quantity():
    broker = FakeBroker(positions=[{"symbol": "NFLX", "qty": "12"}])
    positions = {"NFLX": {"qty": 12, "entry_session": "2026-09-25", "horizon_end": "2026-10-02"}}
    plans, _, _ = _run_step(Portfolio(broker, dry_run=False), [_decision("NFLX", "agree", "avoid")], positions)
    assert [(p.symbol, p.side, p.qty) for p in plans] == [("NFLX", "sell", 12)]
    assert "desk agreed on avoid" in plans[0].reason
    assert "NFLX" not in positions


def test_horizon_exit_only_in_the_decision_cycle_and_not_when_reaffirmed():
    broker = FakeBroker(positions=[{"symbol": "AMD", "qty": "10"}])
    ended = {"AMD": {"qty": 10, "entry_session": "2026-09-21", "horizon_end": SESSION}}
    plans, _, _ = _run_step(Portfolio(broker, dry_run=False), [], dict(ended), check_horizons=False)
    assert plans == []
    plans, _, _ = _run_step(Portfolio(broker, dry_run=False), [], dict(ended))
    assert [(p.symbol, p.side) for p in plans] == [("AMD", "sell")] and "horizon ended" in plans[0].reason
    # an agreed buy the same morning restarts the horizon instead
    positions = {"AMD": dict(ended["AMD"])}
    plans, _, _ = _run_step(Portfolio(FakeBroker(positions=[{"symbol": "AMD", "qty": "10"}]), dry_run=False),
                            [_decision("AMD")], positions)
    assert plans == []
    assert positions["AMD"]["horizon_end"] == "2026-10-05" and positions["AMD"]["reaffirmed"] == SESSION


def test_limits_gates_and_outside_positions_become_notes():
    broker = FakeBroker(positions=[{"symbol": "MSFT", "qty": "3"}])
    held = {f"H{i}": {"qty": 1, "entry_session": SESSION, "horizon_end": "2026-10-05", "dry_run": True}
            for i in range(7)}
    decisions = [
        _decision("AAPL"),
        _decision("MSFT"),
        _decision("NVDA", gates=[{"gate": "bias_1", "passed": False, "reason": "news-driven"}]),
        _decision("ZZZ"),
    ]
    plans, notes, _ = _run_step(Portfolio(broker, dry_run=False), decisions, held)
    assert [(p.symbol, p.side) for p in plans] == [("AAPL", "buy")]  # the 8th and last slot
    by_symbol = {n["symbol"]: n["note"] for n in notes}
    assert "already held outside the desk" in by_symbol["MSFT"]
    assert by_symbol["NVDA"].startswith("agreed buy stopped by bias_1: news-driven")
    assert "8 positions are already open" in by_symbol["ZZZ"]


def test_dry_run_records_without_sending_and_can_sell_later():
    broker = FakeBroker()
    portfolio = Portfolio(broker)
    positions = {}
    _, _, results = _run_step(portfolio, [_decision("AAPL")], positions)
    assert broker.submitted == [] and results[0]["status"] == "dry_run"
    assert positions["AAPL"]["dry_run"] is True and positions["AAPL"]["qty"] == 98  # 10k / (100 * 1.02)
    plans, _, _ = _run_step(portfolio, [_decision("AAPL", "agree", "avoid")], positions, session="2026-09-29")
    assert [(p.side, p.qty) for p in plans] == [("sell", 98)]  # from the record: the broker has nothing
    assert "AAPL" not in positions


def test_a_replayed_order_is_recognized_not_duplicated():
    broker = FakeBroker()
    portfolio = Portfolio(broker, dry_run=False)
    first, _, _ = _run_step(portfolio, [_decision("AAPL")], {})
    positions = {}
    _, _, results = _run_step(portfolio, [_decision("AAPL")], positions)  # e.g. a crash before saving
    assert results[0]["duplicate"] is True and results[0]["order_id"] == "order-1"
    assert len(broker.orders) == 1 and positions["AAPL"]["order_id"] == "order-1"


def test_a_failed_submission_is_recorded_and_the_rest_still_go_out():
    broker = FakeBroker(fail_symbols={"AAPL"})
    positions = {}
    _, _, results = _run_step(Portfolio(broker, dry_run=False), [_decision("AAPL"), _decision("MSFT")], positions)
    status = {r["symbol"]: r["status"] for r in results}
    assert status == {"AAPL": "failed", "MSFT": "accepted"}
    assert set(positions) == {"MSFT"}


def test_position_missing_at_the_broker_is_dropped_when_it_should_be_sold():
    positions = {"AAPL": {"qty": 5, "entry_session": "2026-09-21", "horizon_end": SESSION}}
    _, notes, _ = _run_step(Portfolio(FakeBroker(), dry_run=False), [], positions)
    assert "not at the broker" in notes[0]["note"] and positions == {}


def test_client_order_id_is_deterministic():
    assert client_order_id(SESSION, "AAPL", "buy") == "desk-2026-09-28-AAPL-buy"
    assert date.fromisoformat(SESSION).weekday() == 0
