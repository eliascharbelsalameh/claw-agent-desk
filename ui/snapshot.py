"""Freeze the desk's trades into one JSON file the app can show offline:

    .venv/Scripts/python -m ui.snapshot --logs logs/server logs --out demo/snapshot.json

Reads the traces (the orders and the decisions behind them), asks Alpaca for
each order's fill, the account, the open positions and the price bars around
every trade, and writes it all down. The Positions tab reads this file and
makes no API call of its own, so it works with no credentials and no network
(a booth with bad wifi). Re-run it after new trades to refresh. Only trace
text the desk already redacts is copied; no credentials or account numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from agents.trace import read_trace

from .trades import collect_orders, decision_story, mark_open, pair_trades

BAR_PADDING_DAYS = 4  # calendar days of price before the entry and after the exit
DEFAULT_OUT = "demo/snapshot.json"


def read_events(paths: list[str | Path]) -> list[dict[str, Any]]:
    """Every event of every trace in the given files or folders, oldest first."""
    files: list[Path] = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    events = [e for f in files for e in read_trace(f)]
    return sorted(events, key=lambda e: e.get("ts", ""))


def fill_of(order: dict[str, Any]) -> dict[str, Any]:
    price = order.get("filled_avg_price")
    return {"price": float(price) if price else None, "filled_at": order.get("filled_at"),
            "qty": int(float(order["filled_qty"])) if order.get("filled_qty") not in (None, "0") else None,
            "status": order.get("status")}


def build_snapshot(events: list[dict[str, Any]], broker: Any, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    orders = collect_orders(events)
    fills: dict[str, dict[str, Any]] = {}
    for o in orders:
        try:
            fills[o["client_order_id"]] = fill_of(broker.get_order_by_client_id(o["client_order_id"]))
        except Exception:  # noqa: BLE001 - an order Alpaca no longer knows is shown without a fill
            fills[o["client_order_id"]] = {}
    trades = pair_trades(orders, fills)

    positions = broker.list_positions()
    prices = {p["symbol"]: float(p["current_price"]) for p in positions}
    mark_open(trades, prices)

    for t in trades:
        entry, exit_ = t["entry"], t["exit"]
        entry["story"] = decision_story(events, t["symbol"], entry["session"])
        if exit_:
            exit_["story"] = decision_story(events, t["symbol"], exit_["session"])
        start = date.fromisoformat(entry["session"]) - timedelta(days=BAR_PADDING_DAYS)
        end = (date.fromisoformat(exit_["session"]) if exit_ else now.date()) + timedelta(days=BAR_PADDING_DAYS)
        end = min(end, now.date() + timedelta(days=1))
        try:
            bars = broker.get_bars(t["symbol"], "1Hour", start=datetime.combine(start, datetime.min.time(), timezone.utc),
                                   end=datetime.combine(end, datetime.min.time(), timezone.utc), cache_ttl=None)
        except Exception:  # noqa: BLE001 - a trade with no bars still shows its story
            bars = []
        t["bars"] = [{k: b[k] for k in ("t", "o", "h", "l", "c")} for b in bars]

    account = broker.get_account()
    return {
        "generated_at": now.isoformat(),
        "account": {"equity": float(account["equity"]), "cash": float(account["cash"]),
                    "last_equity": float(account.get("last_equity") or 0)},
        "positions": [{"symbol": p["symbol"], "qty": int(float(p["qty"])), "avg_entry_price": float(p["avg_entry_price"]),
                       "current_price": float(p["current_price"]), "unrealized_pl": float(p["unrealized_pl"])}
                      for p in positions],
        "trades": trades,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ui.snapshot", description=__doc__.split("\n\n")[0])
    parser.add_argument("--logs", nargs="+", default=["logs/server", "logs"], help="trace files or folders")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    from data_layer import AlpacaClient
    from ui.support import load_user_env

    load_user_env()
    snapshot = build_snapshot(read_events(args.logs), AlpacaClient())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(snapshot['trades'])} trades, {len(snapshot['positions'])} open positions -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
