"""The Positions tab's data: trades rebuilt from the traces, the decision
behind each order, the snapshot file, the charts, and the tab itself."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tests.fakes import patch_live_components
from ui.charts import return_chart, trade_chart
from ui.snapshot import build_snapshot, fill_of, read_events
from ui.support import CREDENTIAL_NAMES
from ui.trades import collect_orders, decision_story, mark_open, pair_trades, trade_label

APP = str(Path(__file__).resolve().parent.parent / "ui" / "streamlit_app.py")


def _order(ts, symbol, side, qty, session, status="pending_new"):
    return {"ts": ts, "agent": "portfolio", "event": "order", "symbol": symbol, "side": side, "qty": qty,
            "reason": f"{side}: reason", "client_order_id": f"desk-{session}-{symbol}-{side}", "status": status}


def _events():
    return [
        {"ts": "2026-09-29T12:10:00+00:00", "agent": "analyst_1", "event": "verdict", "symbol": "META",
         "role": "analyst_1", "model": "openai/gpt-oss-20b", "recommendation": "buy", "confidence": 0.6,
         "thesis": "Strong revenue. " * 40, "review_round": 0, "response_to_critique": []},
        {"ts": "2026-09-29T12:20:00+00:00", "agent": "analyst_1", "event": "verdict", "symbol": "META",
         "role": "analyst_1", "model": "openai/gpt-oss-20b", "recommendation": "buy", "confidence": 0.65,
         "thesis": "Still buy.", "review_round": 1,
         "response_to_critique": [{"accept": True}, {"accept": False}, {"accept": None}]},
        {"ts": "2026-09-29T12:21:00+00:00", "agent": "critic", "event": "critique", "symbol": "META",
         "model": "m/critic", "review_round": 1, "assessment": "They agree on growth.",
         "challenges": [{"to": "analyst_1", "point": "Valuation is rich."}]},
        {"ts": "2026-09-29T12:30:00+00:00", "agent": "bias_1", "event": "bias_check", "symbol": "META",
         "role": "bias_1", "model": "m/b1", "verdict": "flag", "reason": "News-driven."},
        {"ts": "2026-09-29T12:31:00+00:00", "agent": "bias_gate", "event": "decision", "symbol": "META",
         "outcome": "passed", "reason": "passed: only bias_1 flags it"},
        {"ts": "2026-09-29T12:32:00+00:00", "agent": "technical", "event": "timing", "symbol": "META",
         "timing": "enter", "trend_4h": "down", "support": 713.3, "resistance": 779.8, "reason": "Testing support."},
        {"ts": "2026-09-29T12:33:00+00:00", "agent": "critic_loop", "event": "final", "symbol": "META",
         "outcome": "agree", "recommendation": "buy", "reason": "after round 1: both analysts recommend buy"},
        {"ts": "2026-09-30T12:10:00+00:00", "agent": "analyst_1", "event": "verdict", "symbol": "META",
         "role": "analyst_1", "model": "x", "recommendation": "hold", "confidence": 0.5, "thesis": "Next day.",
         "review_round": 0, "response_to_critique": []},
        _order("2026-09-29T12:34:00+00:00", "META", "buy", 13, "2026-09-29"),
        _order("2026-09-29T12:34:00.5+00:00", "META", "buy", 13, "2026-09-29"),  # a replayed order: counted once
        _order("2026-10-02T12:32:00+00:00", "META", "sell", 13, "2026-10-02"),
        _order("2026-10-01T12:47:00+00:00", "AAPL", "buy", 29, "2026-10-01"),
        _order("2026-10-01T12:48:00+00:00", "NVDA", "buy", 42, "2026-10-01", status="failed"),
        _order("2026-10-01T12:49:00+00:00", "MSFT", "buy", 5, "2026-10-01", status="dry_run"),
    ]


FILLS = {
    "desk-2026-09-29-META-buy": {"price": 722.0, "filled_at": "2026-09-29T13:31:00Z", "qty": 13, "status": "filled"},
    "desk-2026-10-02-META-sell": {"price": 738.38, "filled_at": "2026-10-02T13:33:00Z", "qty": 13, "status": "filled"},
    "desk-2026-10-01-AAPL-buy": {"price": 330.0, "filled_at": "2026-10-01T13:31:00Z", "qty": 29, "status": "filled"},
}


def test_collect_orders_skips_failed_dry_run_and_replays():
    orders = collect_orders(_events())
    assert [(o["symbol"], o["side"], o["session"]) for o in orders] == [
        ("META", "buy", "2026-09-29"), ("AAPL", "buy", "2026-10-01"), ("META", "sell", "2026-10-02")]


def test_decision_story_keeps_only_that_stock_and_session():
    story = decision_story(_events(), "META", "2026-09-29")
    assert [(a["round"], a["recommendation"]) for a in story["analysts"]] == [(0, "buy"), (1, "buy")]
    assert len(story["analysts"][0]["thesis"]) <= 240 and story["analysts"][0]["thesis"].endswith("…")
    assert (story["analysts"][1]["accepted"], story["analysts"][1]["rejected"]) == (1, 1)
    assert story["critic"][0]["challenges"] == [{"to": "analyst_1", "point": "Valuation is rich."}]
    assert story["bias"][0]["verdict"] == "flag" and "only bias_1" in story["bias_gate"]
    assert story["technical"]["timing"] == "enter"
    assert story["outcome"] == "agree buy: after round 1: both analysts recommend buy"


def test_pair_trades_matches_a_sell_to_its_buy_and_computes_the_result():
    trades = pair_trades(collect_orders(_events()), FILLS)
    assert [(t["symbol"], t["status"]) for t in trades] == [("AAPL", "open"), ("META", "closed")]
    meta = trades[1]
    assert meta["pnl"] == pytest.approx((738.38 - 722.0) * 13, abs=0.01) and meta["pnl_pct"] == 2.27
    mark_open(trades, {"AAPL": 333.0})
    assert trades[0]["pnl"] == 87.0 and trades[0]["now"] == 333.0
    assert trade_label(meta) == "META · closed · 2026-09-29 · +2.27%"
    unfilled = pair_trades(collect_orders(_events()), {})
    assert unfilled[0]["pnl"] is None and trade_label(unfilled[0]).endswith("pending fill")


class FakeBroker:
    def get_order_by_client_id(self, cid):
        fill = FILLS.get(cid)
        if fill is None:
            raise KeyError(cid)
        return {"status": fill["status"], "filled_avg_price": str(fill["price"]), "filled_at": fill["filled_at"],
                "filled_qty": str(fill["qty"])}

    def list_positions(self):
        return [{"symbol": "AAPL", "qty": "29", "avg_entry_price": "330", "current_price": "333",
                 "unrealized_pl": "87"}]

    def get_account(self):
        return {"equity": "100000", "cash": "50000", "last_equity": "99900"}

    def get_bars(self, symbol, timeframe, start=None, end=None, **kwargs):
        if symbol == "AAPL":
            raise ConnectionError("no bars")
        return [{"t": f"2026-09-29T{h}:00:00Z", "o": 720 + h, "h": 725 + h, "l": 718 + h, "c": 722 + h, "v": 1}
                for h in (14, 15, 16, 17)] + [{"t": "2026-10-02T14:00:00Z", "o": 735, "h": 740, "l": 733, "c": 738, "v": 1}]


def test_fill_of_reads_alpaca_order_fields():
    assert fill_of({"status": "new", "filled_qty": "0"}) == {"price": None, "filled_at": None, "qty": None, "status": "new"}
    assert fill_of({"status": "filled", "filled_avg_price": "1.5", "filled_qty": "3", "filled_at": "t"})["qty"] == 3


@pytest.fixture
def snapshot():
    return build_snapshot(_events(), FakeBroker(), now=datetime(2026, 10, 5, 9, tzinfo=timezone.utc))


def test_snapshot_has_trades_stories_bars_and_the_account(snapshot):
    assert snapshot["account"] == {"equity": 100000.0, "cash": 50000.0, "last_equity": 99900.0}
    by_symbol = {t["symbol"]: t for t in snapshot["trades"]}
    meta, aapl = by_symbol["META"], by_symbol["AAPL"]
    assert meta["entry"]["story"]["technical"]["timing"] == "enter"
    assert meta["exit"]["story"]["analysts"] == [] and len(meta["bars"]) == 5  # no agent events logged that day
    assert set(meta["bars"][0]) == {"t", "o", "h", "l", "c"}
    assert aapl["bars"] == [] and aapl["pnl"] == 87.0 and aapl["status"] == "open"  # bars failed: still shown
    json.dumps(snapshot)  # plain data, ready to write


def test_read_events_merges_files_and_folders_in_time_order(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "t1.jsonl").write_text(json.dumps({"ts": "2026-01-02T00:00:00+00:00"}) + "\n", encoding="utf-8")
    single = tmp_path / "t2.jsonl"
    single.write_text(json.dumps({"ts": "2026-01-01T00:00:00+00:00"}) + "\n", encoding="utf-8")
    assert [e["ts"][:10] for e in read_events([tmp_path / "a", single])] == ["2026-01-01", "2026-01-02"]


def test_charts_build_for_closed_open_and_barless_trades(snapshot):
    for trade in snapshot["trades"]:
        spec = trade_chart(trade).to_dict()
        assert ("layer" in spec) == bool(trade["bars"])  # no bars: an empty chart, not an error
    assert return_chart(snapshot["trades"]).to_dict()
    assert return_chart([]).to_dict()


@pytest.fixture
def positions_app(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAW_DESK_NO_REGISTRY", "1")
    monkeypatch.setenv("CLAW_DESK_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("CLAW_DESK_STATE", str(tmp_path / "no-state.json"))
    for name in CREDENTIAL_NAMES:
        monkeypatch.setenv(name, "present-but-fake")
    patch_live_components(monkeypatch, tmp_path)
    return monkeypatch, tmp_path


def test_positions_tab_without_a_snapshot_says_how_to_build_one(positions_app):
    monkeypatch, tmp_path = positions_app
    monkeypatch.setenv("CLAW_DESK_SNAPSHOT", str(tmp_path / "missing.json"))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("python -m ui.snapshot" in i.value for i in at.info)


def test_positions_tab_shows_trades_chart_and_both_stories(positions_app, snapshot):
    monkeypatch, tmp_path = positions_app
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    monkeypatch.setenv("CLAW_DESK_SNAPSHOT", str(path))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Equity"] == "$100,000" and metrics["Open positions"] == "1" and metrics["Closed trades"] == "1"
    assert metrics["Realized"] == "+212.94" and metrics["Unrealized"] == "+87.00"
    table = next(d for d in at.dataframe if "buy price" in d.value.columns).value
    assert list(table["stock"]) == ["AAPL", "META"]
    at.selectbox[0].select(1)  # META, the closed trade
    at.run()
    assert not at.exception
    text = "\n".join(m.value for m in at.markdown)
    assert "Why it was bought (2026-09-29)" in text and "Why it was sold (2026-10-02)" in text
    assert "only bias_1 flags it" in text and "Valuation is rich." in text
    assert len(at.get("vega_lite_chart")) >= 1
