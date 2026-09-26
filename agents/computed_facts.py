"""Facts the desk computes itself from source data (no LLM).

Added Sept 26, 2026 because the analysts kept pointing at what the packet
lacked: 38 of 43 hold verdicts in the Sept 25-26 tests cited missing
valuation or forward data, and the only price facts were 1/5/20-day
changes. Everything here is deterministic and traceable to its inputs:

- valuation: trailing-twelve-month (TTM) revenue, net income and diluted
  EPS from SEC filings, combined with the latest price into market cap,
  P/E and price-to-sales;
- technicals: standard indicators from the daily bars already fetched
  (RSI, moving averages, average true range, 52-week range) and the
  stock's move relative to SPY;
- earnings: the next and the latest earnings report from Finnhub's
  calendar, with the consensus estimates it carries.

No interpretation labels ("overbought", "cheap") are attached: what the
numbers mean over the analysts' horizon is left to the analysts.
"""
from __future__ import annotations

from datetime import date
from typing import Any

# The same candidate tags the fundamentals use (macro_agent.FUNDAMENTAL_CONCEPTS).
REVENUE_CONCEPTS = ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet")
NET_INCOME_CONCEPTS = ("NetIncomeLoss",)
EPS_CONCEPTS = ("EarningsPerShareDiluted",)
DILUTED_SHARES_CONCEPT = "WeightedAverageNumberOfDilutedSharesOutstanding"

VALUATION_NOTE = (
    "Computed by the desk from SEC filings and the latest price. Trailing twelve months "
    "(TTM) = the last fiscal year + this fiscal year's year-to-date - last year's same "
    "year-to-date (or the fiscal year itself when it just ended). market_cap = price x "
    "diluted weighted-average shares of the latest quarter (share classes combined), an "
    "approximation of the market value. Trailing figures only: no forward estimates."
)

TECHNICALS_NOTE = (
    "Computed by the desk from daily bars of the IEX feed. rsi_14: Wilder's relative "
    "strength index over 14 sessions (0-100). sma_N: simple average of the last N closes; "
    "pct_from_sma_N: last close vs that average, in %. atr_14_pct: Wilder's average true "
    "range over 14 sessions as % of the last close (the typical daily range). high_52w/"
    "low_52w: highest high and lowest low over the last 252 sessions. vs_spy: the stock's "
    "% change minus SPY's % change over the same sessions, in percentage points."
)

EARNINGS_NOTE = (
    "From Finnhub's earnings calendar (free plan), which lists reports roughly one month "
    "back and a few months ahead; a missing report does not prove none is scheduled. "
    "Estimates are third-party consensus figures, not company guidance."
)
EARNINGS_HOURS = {"bmo": "before market open", "amc": "after market close", "dmh": "during market hours"}

RSI_PERIOD = 14
ATR_PERIOD = 14
SMA_PERIODS = (20, 50, 200)
SESSIONS_52W = 252


# --- trailing twelve months (SEC company facts) ---


def _days(entry: dict[str, Any]) -> int:
    return (date.fromisoformat(entry["end"]) - date.fromisoformat(entry["start"])).days


def _duration_entries(company_facts: dict[str, Any], concept: str) -> list[dict[str, Any]]:
    """10-K/10-Q entries with a start date (flows, not balance-sheet points)
    for one concept, one per (start, end): the latest filing wins, since
    later filings restate earlier periods."""
    us_gaap = company_facts.get("facts", {}).get("us-gaap", {})
    periods: dict[tuple[str, str], dict[str, Any]] = {}
    for entries in us_gaap.get(concept, {}).get("units", {}).values():
        for e in entries:
            if e.get("form") not in ("10-K", "10-Q") or "start" not in e or "end" not in e or "val" not in e:
                continue
            key = (e["start"], e["end"])
            if key not in periods or (e.get("filed") or "") > (periods[key].get("filed") or ""):
                periods[key] = e
    return list(periods.values())


def trailing_twelve_months(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """TTM value from one concept's duration entries, or None when the
    pieces aren't there. When the latest period is a fiscal year, that is
    the TTM. Otherwise TTM = prior fiscal year + current year-to-date -
    prior year's same year-to-date (a 10-Q reports both the quarter and the
    year-to-date; for a first quarter they are the same period). Windows
    allow for 52/53-week fiscal years."""
    if not entries:
        return None
    latest_end = max(e["end"] for e in entries)
    at_end = [e for e in entries if e["end"] == latest_end]
    annual = [e for e in at_end if 330 <= _days(e) <= 380]
    if annual:
        fy = annual[0]
        return {"value": fy["val"], "period_start": fy["start"], "period_end": fy["end"],
                "method": "fiscal year"}
    ytd = max(at_end, key=_days)
    ytd_start = date.fromisoformat(ytd["start"])
    ytd_end = date.fromisoformat(ytd["end"])
    prior_fy = [
        e for e in entries
        if 330 <= _days(e) <= 380 and 0 <= (ytd_start - date.fromisoformat(e["end"])).days <= 8
    ]
    prior_ytd = [
        e for e in entries
        if abs(_days(e) - _days(ytd)) <= 10 and 350 <= (ytd_end - date.fromisoformat(e["end"])).days <= 380
    ]
    if not prior_fy or not prior_ytd:
        return None
    fy = max(prior_fy, key=lambda e: e["end"])
    py = max(prior_ytd, key=lambda e: e["end"])
    return {
        "value": fy["val"] + ytd["val"] - py["val"],
        "period_end": ytd["end"],
        "method": f"fiscal year to {fy['end']} + year-to-date to {ytd['end']} - year-to-date to {py['end']}",
    }


def best_ttm(company_facts: dict[str, Any], concepts: tuple[str, ...]) -> dict[str, Any] | None:
    """TTM from the candidate tag whose TTM ends latest (companies switch
    revenue tags; see macro_agent.extract_fundamentals). Pieces are never
    mixed across tags: FY from one tag and year-to-date from another would
    silently combine two definitions."""
    results = []
    for concept in concepts:
        ttm = trailing_twelve_months(_duration_entries(company_facts, concept))
        if ttm is not None:
            results.append({**ttm, "concept": concept})
    return max(results, key=lambda r: r["period_end"]) if results else None


def latest_diluted_shares(company_facts: dict[str, Any]) -> dict[str, Any] | None:
    """Diluted weighted-average shares of the latest quarter. Used instead
    of the cover-page share count (dei:EntityCommonStockSharesOutstanding),
    which EDGAR's company facts omit for multi-class companies (META and
    GOOGL have none, NKE's stops in 2015 - checked live Sept 26, 2026); the
    weighted-average count combines the classes on the same basis as EPS."""
    entries = _duration_entries(company_facts, DILUTED_SHARES_CONCEPT)
    quarters = [e for e in entries if 80 <= _days(e) <= 100]
    pool = quarters or entries
    if not pool:
        return None
    best = max(pool, key=lambda e: (e["end"], -_days(e)))
    return {"value": best["val"], "period_end": best["end"]}


def compute_valuation(
    company_facts: dict[str, Any], price: dict[str, Any] | None
) -> tuple[dict[str, Any] | None, list[str]]:
    """Valuation block plus data gaps for whatever couldn't be computed."""
    if not price or not price.get("last_close"):
        return None, ["valuation: no price to value the shares at"]
    last = float(price["last_close"])
    revenue = best_ttm(company_facts, REVENUE_CONCEPTS)
    net_income = best_ttm(company_facts, NET_INCOME_CONCEPTS)
    eps = best_ttm(company_facts, EPS_CONCEPTS)
    shares = latest_diluted_shares(company_facts)
    gaps = [
        f"valuation: TTM {name} not computable from the filings (missing year-to-date or fiscal-year pieces)"
        for name, value in (("revenue", revenue), ("net income", net_income), ("diluted EPS", eps))
        if value is None
    ]
    if shares is None:
        gaps.append("valuation: no diluted share count in the filings, so no market cap or price-to-sales")
    if revenue is None and net_income is None and eps is None and shares is None:
        return None, gaps

    out: dict[str, Any] = {"price": last, "price_date": str(price.get("last_bar_date", ""))[:10]}
    market_cap = round(last * shares["value"]) if shares else None
    out["market_cap"] = market_cap
    if eps is not None:
        eps_value = round(float(eps["value"]), 4)
        out["pe_ttm"] = round(last / eps_value, 2) if eps_value > 0 else None
        if eps_value <= 0:
            out["pe_ttm_note"] = "not meaningful: trailing twelve-month EPS is zero or negative"
    else:
        out["pe_ttm"] = None
    out["ps_ttm"] = round(market_cap / revenue["value"], 2) if market_cap and revenue and revenue["value"] > 0 else None

    ttm: dict[str, Any] = {}
    period_ends = set()
    for key, value in (("revenue", revenue), ("net_income", net_income), ("eps_diluted", eps)):
        if value is not None:
            ttm[key] = round(value["value"], 4) if key == "eps_diluted" else value["value"]
            period_ends.add(value["period_end"])
    if len(period_ends) == 1:
        ttm["period_end"] = period_ends.pop()
    else:  # the three concepts end at different dates: say which is which
        for key, value in (("revenue", revenue), ("net_income", net_income), ("eps_diluted", eps)):
            if value is not None:
                ttm[f"{key}_period_end"] = value["period_end"]
    out["trailing_12m"] = ttm
    if shares:
        out["diluted_shares"] = shares["value"]
        out["diluted_shares_period_end"] = shares["period_end"]
    out["note"] = VALUATION_NOTE
    return out, gaps


# --- technicals (daily bars) ---


def _sma(values: list[float], n: int) -> float | None:
    return sum(values[-n:]) / n if len(values) >= n else None


def rsi(closes: list[float], period: int = RSI_PERIOD) -> float | None:
    """Wilder's RSI: seeded with the simple average gain/loss of the first
    `period` changes, then smoothed with weight 1/period."""
    if len(closes) <= period:
        return None
    changes = [b - a for a, b in zip(closes, closes[1:])]
    avg_gain = sum(max(c, 0.0) for c in changes[:period]) / period
    avg_loss = sum(max(-c, 0.0) for c in changes[:period]) / period
    for c in changes[period:]:
        avg_gain = (avg_gain * (period - 1) + max(c, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-c, 0.0)) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return round(100 - 100 / (1 + avg_gain / avg_loss), 2)


def atr(bars: list[dict[str, Any]], period: int = ATR_PERIOD) -> float | None:
    """Wilder's average true range (true range includes gaps from the
    previous close)."""
    if len(bars) <= period:
        return None
    ranges = [
        max(b["h"] - b["l"], abs(b["h"] - prev["c"]), abs(b["l"] - prev["c"]))
        for prev, b in zip(bars, bars[1:])
    ]
    value = sum(ranges[:period]) / period
    for r in ranges[period:]:
        value = (value * (period - 1) + r) / period
    return value


def _bar_day(bar: dict[str, Any]) -> str:
    return str(bar["t"])[:10]


def _change_pct(closes: list[float], sessions: int) -> float | None:
    if len(closes) <= sessions or not closes[-1 - sessions]:
        return None
    return (closes[-1] / closes[-1 - sessions] - 1) * 100


def _change_vs_benchmark(
    bars: list[dict[str, Any]], benchmark: list[dict[str, Any]], sessions: int
) -> tuple[float | None, float | None]:
    """(benchmark % change, stock minus benchmark in pp) over the same two
    dates. Aligned by date, not by position: the two symbols' bar lists
    can differ by a missing session."""
    if len(bars) <= sessions or not benchmark:
        return None, None
    start_day, end_day = _bar_day(bars[-1 - sessions]), _bar_day(bars[-1])
    by_day = {_bar_day(b): b["c"] for b in benchmark}
    if start_day not in by_day or end_day not in by_day or not by_day[start_day]:
        return None, None
    bench = (by_day[end_day] / by_day[start_day] - 1) * 100
    stock = _change_pct([b["c"] for b in bars], sessions)
    if stock is None:
        return None, None
    return round(bench, 2), round(stock - bench, 2)


def compute_technicals(
    bars: list[dict[str, Any]],
    benchmark: list[dict[str, Any]] | None = None,
    benchmark_symbol: str = "SPY",
    latest_in_progress: bool = False,
) -> dict[str, Any] | None:
    """Indicators from ascending daily bars. Anything needing more sessions
    than are available is None and listed under `unavailable`."""
    if not bars:
        return None
    closes = [float(b["c"]) for b in bars]
    last = closes[-1]
    out: dict[str, Any] = {"rsi_14": rsi(closes)}
    unavailable = []
    for n in SMA_PERIODS:
        sma = _sma(closes, n)
        out[f"sma_{n}"] = round(sma, 2) if sma is not None else None
        out[f"pct_from_sma_{n}"] = round((last / sma - 1) * 100, 2) if sma else None
        if sma is None:
            unavailable.append(f"sma_{n}: needs {n} sessions, {len(closes)} available")
    atr_value = atr(bars)
    out["atr_14_pct"] = round(atr_value / last * 100, 2) if atr_value is not None and last else None
    window = bars[-SESSIONS_52W:]
    out["high_52w"] = max(b["h"] for b in window)
    out["low_52w"] = min(b["l"] for b in window)
    out["pct_from_high_52w"] = round((last / out["high_52w"] - 1) * 100, 2) if out["high_52w"] else None
    if len(bars) < SESSIONS_52W:
        unavailable.append(f"high_52w/low_52w: cover only the {len(bars)} sessions available")
    for sessions in (5, 20):
        bench, relative = _change_vs_benchmark(bars, benchmark or [], sessions)
        out[f"{benchmark_symbol.lower()}_change_{sessions}d_pct"] = bench
        out[f"change_{sessions}d_vs_{benchmark_symbol.lower()}_pp"] = relative
    if not benchmark:
        unavailable.append(f"vs_{benchmark_symbol.lower()}: no {benchmark_symbol} bars")
    if out["rsi_14"] is None:
        unavailable.append(f"rsi_14: needs {RSI_PERIOD + 1} sessions")
    out["sessions_used"] = len(bars)
    if latest_in_progress:
        out["includes_running_session"] = True
    if unavailable:
        out["unavailable"] = unavailable
    out["note"] = TECHNICALS_NOTE
    return out


# --- earnings calendar (Finnhub) ---


def _surprise_pct(actual: Any, estimate: Any) -> float | None:
    if actual is None or estimate in (None, 0):
        return None
    return round((actual - estimate) / abs(estimate) * 100, 2)


def summarize_earnings(items: list[dict[str, Any]], today: date) -> dict[str, Any]:
    """The next scheduled report (on or after today) and the latest one
    already reported, from Finnhub calendar rows."""
    rows = sorted((r for r in items if r.get("date")), key=lambda r: r["date"])
    upcoming = [r for r in rows if r["date"] >= today.isoformat() and r.get("epsActual") is None]
    reported = [r for r in rows if r.get("epsActual") is not None or r.get("revenueActual") is not None]
    out: dict[str, Any] = {"next_report": None, "latest_report": None}
    if upcoming:
        r = upcoming[0]
        out["next_report"] = {
            "date": r["date"],
            "calendar_days_until": (date.fromisoformat(r["date"]) - today).days,
            "time": EARNINGS_HOURS.get(r.get("hour") or "", "not specified"),
            "fiscal_period": f"Q{r.get('quarter')} {r.get('year')}" if r.get("quarter") else None,
            "eps_estimate": r.get("epsEstimate"),
            "revenue_estimate": r.get("revenueEstimate"),
        }
    if reported:
        r = reported[-1]
        out["latest_report"] = {
            "date": r["date"],
            "calendar_days_ago": (today - date.fromisoformat(r["date"])).days,
            "fiscal_period": f"Q{r.get('quarter')} {r.get('year')}" if r.get("quarter") else None,
            "eps_actual": r.get("epsActual"),
            "eps_estimate": r.get("epsEstimate"),
            "eps_surprise_pct": _surprise_pct(r.get("epsActual"), r.get("epsEstimate")),
            "revenue_actual": r.get("revenueActual"),
            "revenue_estimate": r.get("revenueEstimate"),
            "revenue_surprise_pct": _surprise_pct(r.get("revenueActual"), r.get("revenueEstimate")),
        }
    out["note"] = EARNINGS_NOTE
    return out
