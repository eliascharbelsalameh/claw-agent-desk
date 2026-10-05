"""The desk's trades as a story, for the Positions tab (Streamlit-free, so it
can be unit-tested): the orders in the traces, each paired buy -> sell, with
the decision behind each order pulled from the same traces.

The state file forgets a position once it is sold, so the history is rebuilt
from the traces' `portfolio` `order` events (they are never rewritten) and the
fills Alpaca reports for their client order ids. `ui/snapshot.py` freezes the
result, with price bars, into one JSON file the app can show offline.
"""
from __future__ import annotations

from typing import Any, Iterable

SHORT = 240  # characters of a thesis shown on a card line


def collect_orders(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every order the desk sent, once: failed submissions are skipped and a
    replayed order (the same client id) counts once. Oldest first."""
    orders: dict[str, dict[str, Any]] = {}
    for e in events:
        if e.get("agent") != "portfolio" or e.get("event") != "order":
            continue
        cid = e.get("client_order_id")
        if not cid or e.get("status") in (None, "failed", "dry_run") or cid in orders:
            continue
        orders[cid] = {"symbol": e["symbol"], "side": e["side"], "qty": int(e["qty"]), "reason": e.get("reason", ""),
                       "client_order_id": cid, "sent_at": e["ts"], "session": _session_of(cid)}
    return sorted(orders.values(), key=lambda o: o["sent_at"])


def _session_of(client_order_id: str) -> str:
    """desk-2026-10-02-META-sell -> 2026-10-02"""
    return client_order_id[5:15]


def _clip(text: Any, n: int = SHORT) -> str:
    text = str(text or "").strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def decision_story(events: Iterable[dict[str, Any]], symbol: str, session: str) -> dict[str, Any]:
    """What the desk said about one stock on one session's cycle: both
    analysts (first and last round), the critic's challenges, the cross-check,
    the bias checks and the entry timing. A cycle and its retries all fall on
    the session's date (the decision cycle starts at 08:00 ET)."""
    mine = [e for e in events if e.get("symbol") == symbol and str(e.get("ts", ""))[:10] == session]
    story: dict[str, Any] = {"symbol": symbol, "session": session, "analysts": [], "critic": [], "bias": [],
                             "technical": None, "outcome": None}
    for e in mine:
        kind = e.get("event")
        if kind == "verdict" and e.get("agent") in ("analyst_1", "analyst_2") and not e.get("error"):
            responses = e.get("response_to_critique") or []
            story["analysts"].append({
                "role": e["agent"], "model": e.get("model"), "round": e.get("review_round") or 0,
                "recommendation": e.get("recommendation"), "confidence": e.get("confidence"),
                "thesis": _clip(e.get("thesis")),
                "accepted": sum(1 for r in responses if r.get("accept") is True),
                "rejected": sum(1 for r in responses if r.get("accept") is False),
            })
        elif kind == "critique" and not e.get("error"):
            story["critic"].append({
                "round": e.get("review_round"), "model": e.get("model"), "assessment": _clip(e.get("assessment"), 400),
                "challenges": [{"to": c.get("to"), "point": _clip(c.get("point"), 200)}
                               for c in e.get("challenges") or []]})
        elif kind == "bias_check" and not e.get("error"):
            story["bias"].append({"role": e.get("role"), "model": e.get("model"), "verdict": e.get("verdict"),
                                  "reason": _clip(e.get("reason"), 300)})
        elif kind == "timing" and e.get("agent") == "technical":
            story["technical"] = {"timing": e.get("timing"), "trend_4h": e.get("trend_4h"),
                                  "support": e.get("support"), "resistance": e.get("resistance"),
                                  "reason": _clip(e.get("reason"), 400)}
        elif kind in ("decision", "final") and e.get("agent") in ("cross_check", "critic_loop"):
            story["outcome"] = f"{e.get('outcome')} {e.get('recommendation') or ''}".strip() + \
                               (f": {e.get('reason')}" if e.get("reason") else "")
        elif kind == "decision" and e.get("agent") == "bias_gate":
            story["bias_gate"] = e.get("reason")
    return story


def pair_trades(orders: list[dict[str, Any]], fills: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Each buy with the next sell of the same stock. `fills` maps a client
    order id to {price, filled_at, qty, status} as Alpaca reports it; an
    order with no fill yet has price None. Newest entry first."""
    open_buys: dict[str, dict[str, Any]] = {}
    trades: list[dict[str, Any]] = []
    for o in orders:
        fill = fills.get(o["client_order_id"]) or {}
        leg = {"session": o["session"], "sent_at": o["sent_at"], "reason": o["reason"],
               "price": fill.get("price"), "filled_at": fill.get("filled_at"), "status": fill.get("status"),
               "qty": int(fill.get("qty") or o["qty"]), "client_order_id": o["client_order_id"]}
        if o["side"] == "buy":
            trade = {"symbol": o["symbol"], "qty": leg["qty"], "entry": leg, "exit": None}
            open_buys[o["symbol"]] = trade
            trades.append(trade)
        elif o["symbol"] in open_buys:
            open_buys.pop(o["symbol"])["exit"] = leg
    for t in trades:
        t["status"] = "closed" if t["exit"] else "open"
        e, x = t["entry"]["price"], (t["exit"] or {}).get("price")
        t["pnl"] = round((x - e) * t["qty"], 2) if e and x else None
        t["pnl_pct"] = round((x / e - 1) * 100, 2) if e and x else None
    return sorted(trades, key=lambda t: t["entry"]["sent_at"], reverse=True)


def mark_open(trades: list[dict[str, Any]], prices: dict[str, float]) -> None:
    """Fill the unrealized result of open trades from the current prices."""
    for t in trades:
        price, entry = prices.get(t["symbol"]), t["entry"]["price"]
        if t["status"] == "open" and price and entry:
            t["now"] = price
            t["pnl"] = round((price - entry) * t["qty"], 2)
            t["pnl_pct"] = round((price / entry - 1) * 100, 2)


def trade_label(t: dict[str, Any]) -> str:
    pct = t.get("pnl_pct")
    result = f"{pct:+.2f}%" if pct is not None else "pending fill"
    return f"{t['symbol']} · {t['status']} · {t['entry']['session']} · {result}"
