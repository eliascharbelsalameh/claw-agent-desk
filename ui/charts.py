"""Altair chart of one trade for the Positions tab: the price around the
trade, the entry and exit markers with hover text, the holding period shaded
and the entry price as a dashed line. Altair ships with Streamlit, so there
is nothing new to install.

The x axis is the bar number, not the clock: overnight and weekend gaps would
otherwise draw as long steep lines between two hours that are 17 hours apart.
Labels mark the start of each trading day (US Eastern dates)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import altair as alt
import pandas as pd

GREEN, RED, AMBER, GREY = "#15803d", "#b91c1c", "#c2410c", "#64748b"
MARKET_TZ = "America/New_York"


def _when(ts: str | None) -> datetime | None:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None


def _money(value: float | None) -> str:
    return f"${value:,.2f}" if value is not None else "waiting for the fill"


def _bar_at(times: pd.Series, moment: datetime) -> int:
    """Index of the bar a fill belongs to: the last bar that started at or before it."""
    before = times[times <= pd.Timestamp(moment)]
    return int(before.index[-1]) if len(before) else 0


def _day_axis(bars: pd.DataFrame) -> alt.Axis:
    """Ticks at the first bar of each trading day, labelled with its date."""
    days = bars["time"].dt.tz_convert(MARKET_TZ).dt.strftime("%b %d")
    firsts = days[days != days.shift()]
    labels = {str(int(i)): d for i, d in firsts.items()}
    expr = "(" + repr(labels).replace("'", '"') + ")[datum.value] || ''"
    return alt.Axis(values=[int(i) for i in firsts.index], labelExpr=expr, title=None, labelAngle=0, grid=True)


def trade_chart(trade: dict[str, Any], height: int = 360) -> alt.Chart:
    bars = pd.DataFrame(trade["bars"])
    entry, exit_ = trade["entry"], trade["exit"]
    symbol, qty = trade["symbol"], trade["qty"]
    if bars.empty:
        return alt.Chart(pd.DataFrame({"x": [], "c": []})).mark_line().encode(x="x:Q", y="c:Q").properties(height=height)
    bars["time"] = pd.to_datetime(bars["t"], utc=True)
    bars["n"] = range(len(bars))
    bars["hour"] = bars["time"].dt.tz_convert(MARKET_TZ).dt.strftime("%b %d %H:%M ET")
    line = alt.Chart(bars).mark_line(color=GREY).encode(
        x=alt.X("n:Q", axis=_day_axis(bars), scale=alt.Scale(nice=False)),
        y=alt.Y("c:Q", title=f"{symbol} price (USD, 1-hour closes)", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("hour:N", title="Hour"), alt.Tooltip("c:Q", title="Close", format="$,.2f")])
    layers: list[alt.Chart] = [line]

    marks = []
    first = last = None
    if entry.get("price") and entry.get("filled_at"):
        first = _bar_at(bars["time"], _when(entry["filled_at"]))
        marks.append({"n": first, "price": entry["price"], "kind": "Bought", "color": GREEN, "shape": "triangle-up",
                      "info": f"Bought {qty} {symbol} at {_money(entry['price'])} on {entry['session']}"})
        layers.append(alt.Chart(pd.DataFrame({"y": [entry["price"]]})).mark_rule(
            strokeDash=[5, 4], color=GREEN, opacity=0.6).encode(y="y:Q"))
    if exit_ and exit_.get("price") and exit_.get("filled_at"):
        last = _bar_at(bars["time"], _when(exit_["filled_at"]))
        result = f", {trade['pnl_pct']:+.2f}% ({trade['pnl']:+,.2f} USD)" if trade.get("pnl") is not None else ""
        marks.append({"n": last, "price": exit_["price"], "kind": "Sold",
                      "color": RED if (trade.get("pnl") or 0) < 0 else AMBER, "shape": "triangle-down",
                      "info": f"Sold {qty} {symbol} at {_money(exit_['price'])} on {exit_['session']}{result}"})
    elif trade["status"] == "open" and trade.get("now"):
        last = len(bars) - 1
        marks.append({"n": last, "price": trade["now"], "kind": "Now", "color": GREY, "shape": "circle",
                      "info": f"Still held: {_money(trade['now'])}, {trade['pnl_pct']:+.2f}% ({trade['pnl']:+,.2f} USD)"})
    if first is not None and last is not None:
        layers.append(alt.Chart(pd.DataFrame({"a": [first], "b": [last]})).mark_rect(color=GREEN, opacity=0.07).encode(
            x="a:Q", x2="b:Q"))
    if marks:
        layers.append(alt.Chart(pd.DataFrame(marks)).mark_point(size=260, filled=True, opacity=1).encode(
            x="n:Q", y="price:Q", color=alt.Color("color:N", scale=None), shape=alt.Shape("shape:N", scale=None),
            tooltip=[alt.Tooltip("kind:N", title=""), alt.Tooltip("info:N", title="")]))
    return alt.layer(*layers).properties(height=height)


def return_chart(trades: list[dict[str, Any]], height: int = 260) -> alt.Chart:
    """Every trade's return since its entry, on one shared axis (days held)."""
    rows = []
    for t in trades:
        entry = t["entry"]
        if not entry.get("price") or not entry.get("filled_at"):
            continue
        start = _when(entry["filled_at"])
        end = _when(t["exit"]["filled_at"]) if t["exit"] and t["exit"].get("filled_at") else datetime.now(timezone.utc)
        for b in t["bars"]:
            ts = _when(b["t"])
            if start <= ts <= end:
                rows.append({"trade": f"{t['symbol']} ({entry['session'][5:]})", "days": (ts - start).total_seconds() / 86400,
                             "return": (b["c"] / entry["price"] - 1) * 100})
    if not rows:
        return alt.Chart(pd.DataFrame({"days": [], "return": [], "trade": []})).mark_line()
    return alt.Chart(pd.DataFrame(rows)).mark_line().encode(
        x=alt.X("days:Q", title="Days since entry"), y=alt.Y("return:Q", title="Return since entry (%)"),
        color=alt.Color("trade:N", title=None),
        tooltip=[alt.Tooltip("trade:N", title=""), alt.Tooltip("days:Q", format=".1f", title="Days"),
                 alt.Tooltip("return:Q", format="+.2f", title="Return %")]).properties(height=height)
