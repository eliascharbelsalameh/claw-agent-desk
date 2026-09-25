"""Macro/context agent - step 1 of the pipeline (spec section 3).

Only fetches and passes context to the analysts: macro data (FRED),
filings and fundamentals (EDGAR), company news (Finnhub), and price/volume
(Alpaca). It does no analysis itself. Its model (AGENT_MODELS["macro"]) is
used strictly to condense the raw facts into a neutral briefing; the prompt
forbids opinions, forecasts, and recommendations, and the structured facts
travel alongside the briefing so analysts can check every number against
its source instead of trusting the summary.

Every source is fetched independently and a failure is recorded in
StockContext.data_gaps instead of aborting the cycle - a 2-day unattended
run (spec section 7) should degrade to "analysts are told what's missing",
not crash on one flaky endpoint.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

from data_layer.alpaca_client import session_in_progress
from data_layer.llm_client import DEFAULT_MODEL_HEALTH, ModelHealth, role_models

from .llm_json import BACKUP_CONNECT_RETRIES
from .trace import TraceLogger

AGENT_NAME = "macro"

# series_id -> human label. Rates/spreads/VIX are daily, CPI/unemployment/
# fed funds are monthly; every one is reported as latest vs prior vs ~1y ago.
MACRO_SERIES = {
    "FEDFUNDS": "Effective federal funds rate (%)",
    "DGS2": "2-year Treasury yield (%)",
    "DGS10": "10-year Treasury yield (%)",
    "T10Y2Y": "10y minus 2y Treasury spread (pp)",
    "CPIAUCSL": "CPI, all urban consumers (index)",
    "UNRATE": "Unemployment rate (%)",
    "VIXCLS": "CBOE VIX volatility index",
}
# Index levels where only the % change is meaningful (CPI -> inflation).
# Rates and spreads get an absolute change instead: a fed funds move from
# 4.33 to 3.50 is "-0.83pp", not "-19%".
PERCENT_CHANGE_SERIES = {"CPIAUCSL"}

FILING_FORMS = ("10-K", "10-Q", "8-K")

# A 10-Q lands within ~45 days of quarter end, so anything older than a
# fiscal year plus slack means the concept stopped being reported (or the
# tag changed to one not in FUNDAMENTAL_CONCEPTS) and must not pass silently.
FUNDAMENTALS_STALE_DAYS = 450
MAX_FILINGS = 8

# us-gaap concept candidates, first one present wins. Companies differ in
# which revenue tag they use (ASC 606 introduced the long one).
FUNDAMENTAL_CONCEPTS = {
    "revenue": (
        ("RevenueFromContractWithCustomerExcludingAssessedTax", "USD"),
        ("Revenues", "USD"),
        ("SalesRevenueNet", "USD"),
    ),
    "net_income": (("NetIncomeLoss", "USD"),),
    "eps_diluted": (("EarningsPerShareDiluted", "USD/shares"),),
    "total_assets": (("Assets", "USD"),),
    "total_liabilities": (("Liabilities", "USD"),),
    "cash": (("CashAndCashEquivalentsAtCarryingValue", "USD"),),
}

NEWS_LOOKBACK_DAYS = 7
MAX_NEWS_ITEMS = 15
NEWS_SUMMARY_CHARS = 300

DAILY_LOOKBACK_DAYS = 90
BARS_4H_LOOKBACK_DAYS = 15

# The briefing is summarization, not reasoning, so thinking is switched off:
# with it on, the macro model spent its entire 4096-token budget thinking
# out loud (inside `content`, not reasoning_content) and never produced a
# briefing; with it off, the same prompt answered in ~20s with ~1.2k tokens
# (both live-tested Sept 2026). The flag is Nemotron's; if the macro role is
# ever reassigned to another family, re-check what it accepts.
BRIEFING_MAX_TOKENS = 2048
BRIEFING_EXTRA_PARAMS = {"chat_template_kwargs": {"enable_thinking": False}}


def briefing_params(model: str) -> dict[str, Any]:
    """The thinking-off flag, only for NVIDIA Nemotron models: it's their
    chat-template switch, and a backup from another family may reject or
    misread an unknown chat_template_kwargs."""
    return dict(BRIEFING_EXTRA_PARAMS) if model.startswith("nvidia/nemotron") else {}

# Stripped from EDGAR's registered name so "Apple Inc." matches headlines
# that say "Apple".
_NAME_SUFFIXES = re.compile(
    r"[,.]?\s+(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|"
    r"holdings?|group|n\.?v|s\.?a|ag|se)\.?$",
    re.IGNORECASE,
)

IEX_VOLUME_NOTE = (
    "Volume figures come from the IEX exchange only (free Alpaca feed), roughly 4% "
    "of consolidated US equity volume. Absolute volume is NOT comparable to "
    "market-wide volume; only relative volume against the stock's own recent IEX "
    "baseline is meaningful."
)

# What this desk never provides, identical for every stock and every run -
# a disclosure, not a judgment. Distinct from data_gaps, which lists what
# failed to load *this* run. Without it, gpt-oss-20b equated "data
# concerns" with the (empty) data_gaps and reported none, while treating
# valuation it had never been given as "implied" (live, Sept 2026).
DESK_LIMITATIONS = (
    "No valuation data: no share count, market capitalization, P/E or other multiples.",
    "No forward-looking data: no company guidance, analyst estimates or earnings calendar.",
    "Fundamentals cover only the latest reported period and the same period a year earlier.",
    "News headlines and summaries are unverified third-party reporting, keyword-matched to "
    "the company; they are not confirmed facts.",
    IEX_VOLUME_NOTE,
)

BRIEFING_SECTIONS = ("MACRO", "COMPANY FILINGS & FUNDAMENTALS", "PRICE & VOLUME", "NEWS", "DATA GAPS")

# A fresh call gets one more chance when a briefing fails validation; after
# that the analysts get facts only. Live (Sept 2026), 2 of 3 briefings
# degenerated mid-output into "The The The..." / ",,,,,,," - both on calls
# whose first stream had been dropped, so a sick backend is the likely
# cause, but the check doesn't depend on that being true.
BRIEFING_ATTEMPTS = 2

# Any word or punctuation mark repeated 6+ times in a row.
_DEGENERATE_RE = re.compile(r"(\b\w+\b|[^\w\s])(?:\s*\1){5,}", re.IGNORECASE)

SYSTEM_PROMPT = f"""You are the context officer on an equity research desk. You prepare a factual briefing for two analysts who will make the decision; you make no decision yourself.

Rules:
- Report only facts present in the data you are given. Never invent or estimate numbers.
- Do NOT give opinions, forecasts, price targets, ratings, or buy/sell/hold language. Do not say whether anything is good or bad for the stock.
- Keep every number you mention exactly as given, with its date and its unit. Do not derive new figures (no month-on-month changes, no unit conversions such as index points into basis points) that are not already in the data.
- Describe filings by form, date and item number; do not guess what an 8-K item number means.
- Flag explicitly any data listed under data_gaps, and any news older than 72 hours as potentially already priced in.
- The news list has already been filtered to items that mention the company (news_unrelated_dropped says how many were removed). Some remaining items still mention it only in passing: say so when that is the case rather than presenting them as company news.
- {IEX_VOLUME_NOTE}

Write plain text with these sections: {", ".join(BRIEFING_SECTIONS)}. Be concise: at most about 350 words."""


@dataclass
class StockContext:
    symbol: str
    generated_at: str
    macro: dict[str, Any]
    price: dict[str, Any] | None = None
    relative_volume: dict[str, Any] | None = None
    bars_4h: list[dict[str, Any]] = field(default_factory=list)
    filings: list[dict[str, Any]] = field(default_factory=list)
    fundamentals: dict[str, Any] = field(default_factory=dict)
    company_name: str | None = None
    news: list[dict[str, Any]] = field(default_factory=list)
    news_unrelated_dropped: int = 0
    data_gaps: list[str] = field(default_factory=list)
    briefing: str | None = None
    briefing_model: str | None = None

    def facts(self) -> dict[str, Any]:
        """Everything the analysts get as ground truth, minus the raw 4H
        bars (those are for the technical agent, too long for a prompt)."""
        data = asdict(self)
        for key in ("bars_4h", "briefing", "briefing_model"):
            data.pop(key)
        data["limitations"] = list(DESK_LIMITATIONS)
        return data

    def to_prompt(self) -> str:
        """Context block for downstream agents: briefing plus source facts."""
        briefing = self.briefing or "(no briefing available - rely on the facts below)"
        return (
            f"=== CONTEXT FOR {self.symbol} (generated {self.generated_at}) ===\n"
            f"--- Briefing (from the context agent, summary only) ---\n{briefing}\n\n"
            f"--- Source facts (authoritative; prefer these over the briefing) ---\n"
            f"{json.dumps(self.facts(), indent=1, default=str)}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_reasoning(text: str) -> str:
    """Some reasoning models inline their chain-of-thought in <think> tags
    instead of keeping it hidden; the analysts only need the answer."""
    return _THINK_RE.sub("", text).strip()


def briefing_problems(text: str) -> list[str]:
    """Why a briefing can't be passed on to the analysts (empty = fine).

    Deliberately checks shape, not truth: the analysts are told the source
    facts are authoritative, so the job here is only to stop visibly broken
    output (degenerate repetition, missing sections) from reaching them.
    """
    problems = []
    match = _DEGENERATE_RE.search(text)
    if match:
        problems.append(f"degenerate repetition: {match.group(0)[:40]!r}")
    upper = text.upper()
    missing = [s for s in BRIEFING_SECTIONS if s not in upper]
    if missing:
        problems.append(f"missing sections: {', '.join(missing)}")
    return problems


def summarize_series(
    observations: list[dict[str, Any]], pct_change: bool = False
) -> dict[str, Any] | None:
    """Latest vs prior vs ~1y-ago value. FRED marks missing days with ".",
    which are dropped rather than treated as zero."""
    points = []
    for obs in observations:
        try:
            points.append((obs["date"], float(obs["value"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not points:
        return None
    points.sort()
    latest_date, latest = points[-1]
    summary: dict[str, Any] = {"latest": latest, "date": latest_date}
    if len(points) > 1:
        summary["prior"] = points[-2][1]
        summary["prior_date"] = points[-2][0]
    year_ago_cutoff = (date.fromisoformat(latest_date) - timedelta(days=365)).isoformat()
    year_ago = [p for p in points if p[0] <= year_ago_cutoff]
    if year_ago:
        summary["year_ago"] = year_ago[-1][1]
        summary["year_ago_date"] = year_ago[-1][0]
        if pct_change:
            if year_ago[-1][1]:
                summary["yoy_pct_change"] = round((latest / year_ago[-1][1] - 1) * 100, 2)
        else:
            summary["yoy_abs_change"] = round(latest - year_ago[-1][1], 4)
    return summary


def extract_recent_filings(
    submissions: dict[str, Any], cik: str, forms: tuple[str, ...] = FILING_FORMS, limit: int = MAX_FILINGS
) -> list[dict[str, Any]]:
    """EDGAR's `recent` block is columnar (parallel arrays per field)."""
    recent = submissions.get("filings", {}).get("recent", {})
    form_list = recent.get("form", [])

    def col(name: str, i: int) -> Any:
        values = recent.get(name, [])
        return values[i] if i < len(values) else None

    filings = []
    for i, form in enumerate(form_list):
        if form not in forms:
            continue
        accession = col("accessionNumber", i) or ""
        document = col("primaryDocument", i) or ""
        filings.append(
            {
                "form": form,
                "filing_date": col("filingDate", i),
                "report_date": col("reportDate", i),
                "description": col("primaryDocDescription", i),
                "items": col("items", i),
                "url": (
                    f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                    f"{accession.replace('-', '')}/{document}"
                ),
            }
        )
    filings.sort(key=lambda f: f["filing_date"] or "", reverse=True)
    return filings[:limit]


def _duration_days(entry: dict[str, Any]) -> int:
    if "start" not in entry:
        return 0
    return (date.fromisoformat(entry["end"]) - date.fromisoformat(entry["start"])).days


def extract_fundamentals(company_facts: dict[str, Any]) -> dict[str, Any]:
    """Most recent 10-K/10-Q value per concept.

    The candidate tag with the most recent period wins, not the first one
    present: companies switch revenue tags over the years, in both
    directions (live, Sept 2026: NVDA's RevenueFromContractWith... stops in
    2022 and its current revenue is under Revenues, while AAPL and MSFT
    went the other way). Among entries sharing the latest period end, the
    shortest duration wins (a 10-Q reports both the quarter and
    year-to-date for the same end date), then the latest filing. start/end
    are always reported so readers can tell a quarter from a fiscal year.
    """
    us_gaap = company_facts.get("facts", {}).get("us-gaap", {})
    out: dict[str, Any] = {}
    for name, candidates in FUNDAMENTAL_CONCEPTS.items():
        entries = [
            (concept, unit, e)
            for concept, unit in candidates
            for e in us_gaap.get(concept, {}).get("units", {}).get(unit, [])
            if e.get("form") in ("10-K", "10-Q") and "end" in e and "val" in e
        ]
        if not entries:
            continue
        latest_end = max(e["end"] for _, _, e in entries)
        concept, unit, best = min(
            (item for item in entries if item[2]["end"] == latest_end),
            key=lambda item: (_duration_days(item[2]), _neg_date(item[2].get("filed"))),
        )
        out[name] = {
            "value": best["val"],
            "unit": unit,
            "concept": concept,
            "period_start": best.get("start"),
            "period_end": best["end"],
            "form": best["form"],
            "filed": best.get("filed"),
        }
        prior = _year_ago_entry(best, [e for _, _, e in entries])
        if prior is not None:
            out[name]["year_ago_value"] = prior["val"]
            out[name]["year_ago_period_end"] = prior["end"]
            if prior["val"]:
                out[name]["yoy_pct_change"] = round(
                    (best["val"] / prior["val"] - 1) * 100 * (1 if prior["val"] > 0 else -1), 2
                )
    return out


def _year_ago_entry(best: dict[str, Any], entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The same kind of period one year earlier: a quarter for a quarter,
    a fiscal year for a fiscal year, a balance-sheet date for a balance-
    sheet date. Without this, analysts asserted "earnings growth" from a
    single quarter (live, Sept 2026). Searched across all candidate tags,
    since the year-ago value may sit under a tag the company has since
    dropped. Fiscal calendars drift (52/53-week years), hence the windows.
    """
    end = date.fromisoformat(best["end"])
    duration = _duration_days(best)
    matches = [
        e
        for e in entries
        if ("start" in e) == ("start" in best)
        and abs(_duration_days(e) - duration) <= 10
        and 350 <= (end - date.fromisoformat(e["end"])).days <= 380
    ]
    if not matches:
        return None
    return min(matches, key=lambda e: _neg_date(e.get("filed")))


def _neg_date(iso: str | None) -> int:
    """Sort key making later dates sort first under min()."""
    return -date.fromisoformat(iso).toordinal() if iso else 0


def summarize_daily_bars(
    bars: list[dict[str, Any]], latest_in_progress: bool = False
) -> dict[str, Any] | None:
    """Price summary from ascending daily bars. When the latest bar is a
    session still running, its "close" is just the latest trade so far;
    that is kept (it is the current price) but flagged, so no reader takes
    it for a settled close."""
    if not bars:
        return None
    closes = [b["c"] for b in bars]
    last = bars[-1]

    def pct_change(days: int) -> float | None:
        if len(closes) <= days or not closes[-1 - days]:
            return None
        return round((closes[-1] / closes[-1 - days] - 1) * 100, 2)

    window = bars[-20:]
    summary = {
        "last_close": last["c"],
        "last_bar_date": last["t"],
        "change_1d_pct": pct_change(1),
        "change_5d_pct": pct_change(5),
        "change_20d_pct": pct_change(20),
        "high_20d": max(b["h"] for b in window),
        "low_20d": min(b["l"] for b in window),
        "daily_bars_available": len(bars),
        "latest_bar_in_progress": latest_in_progress,
    }
    if latest_in_progress:
        summary["note"] = (
            "The latest session is still running: last_close is the latest intraday price "
            "(IEX feed, may lag), and change_1d_pct compares it with the previous session's close."
        )
    return summary


def company_short_name(name: str) -> str:
    short = name.strip()
    while True:
        stripped = _NAME_SUFFIXES.sub("", short).strip(" ,.&")
        if stripped == short or not stripped:
            return short
        short = stripped


def filter_news(
    items: list[dict[str, Any]], symbol: str, company_name: str | None
) -> tuple[list[dict[str, Any]], int]:
    """Keep items whose headline or summary mentions the company.

    Finnhub's company-news feed is keyword-matched and, live, returned
    Garmin/Dell/Cloudflare stories for AAPL. Filtering here is deterministic
    and cheap; the no-thinking briefing model miscounted when asked to do
    it. Ticker matches are case-sensitive (tickers like "ON" or "A" are
    ordinary words); name matches are not. First-word matching ("Meta" for
    "Meta Platforms") can let a few false positives through - acceptable,
    since dropping a real story is worse than keeping a stray one. Without
    a company name, nothing is dropped.
    """
    if not company_name:
        return items, 0
    short = company_short_name(company_name)
    names = {short}
    first_word = short.split()[0]
    if len(first_word) >= 4:
        names.add(first_word)
    alternatives = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    name_re = re.compile(rf"\b({alternatives})\b", re.IGNORECASE)
    ticker_re = re.compile(rf"\b{re.escape(symbol.upper())}\b")
    kept = []
    for item in items:
        text = f"{item.get('headline') or ''} {item.get('summary') or ''}"
        if ticker_re.search(text) or name_re.search(text):
            kept.append(item)
    return kept, len(items) - len(kept)


def trim_news(items: list[dict[str, Any]], limit: int = MAX_NEWS_ITEMS) -> list[dict[str, Any]]:
    newest_first = sorted(items, key=lambda n: n.get("datetime", 0), reverse=True)
    trimmed = []
    for i, item in enumerate(newest_first[:limit]):
        summary = (item.get("summary") or "").strip()
        if len(summary) > NEWS_SUMMARY_CHARS:
            summary = summary[:NEWS_SUMMARY_CHARS].rstrip() + "..."
        trimmed.append(
            {
                # Explicit position, equal to i in news[i]: analysts citing
                # news paths miscounted by 1-3 positions when they had to count
                # (live, Sept 2026).
                "index": i,
                "headline": item.get("headline"),
                "source": item.get("source"),
                "published_at": item.get("published_at"),
                "age_hours": round(item["age_hours"], 1) if "age_hours" in item else None,
                "summary": summary,
                "url": item.get("url"),
            }
        )
    return trimmed


class MacroContextAgent:
    def __init__(
        self,
        alpaca: Any,
        fred: Any,
        edgar: Any,
        finnhub: Any,
        llm: Any | None,
        *,
        trace: TraceLogger | None = None,
        model: str | None = None,
        models: Sequence[str] | None = None,
        health: ModelHealth | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        """`models` is the ordered candidate list for the briefing (primary,
        then backups); by default the macro role's list. Passing `model`
        pins a single model with no backups."""
        self._alpaca = alpaca
        self._fred = fred
        self._edgar = edgar
        self._finnhub = finnhub
        self._llm = llm
        self._trace = trace
        self._models = list(models) if models else [model] if model else role_models(AGENT_NAME)
        self._health = health if health is not None else DEFAULT_MODEL_HEALTH
        self._now = now

    def _log(self, event: str, payload: dict[str, Any]) -> None:
        if self._trace is not None:
            self._trace.log(AGENT_NAME, event, payload)

    # --- gathering (no LLM) ---

    def gather_macro(self) -> tuple[dict[str, Any], list[str]]:
        start = (self._now().date() - timedelta(days=400)).isoformat()
        macro: dict[str, Any] = {}
        gaps: list[str] = []
        for series_id, label in MACRO_SERIES.items():
            try:
                summary = summarize_series(
                    self._fred.get_series_observations(series_id, start_date=start),
                    pct_change=series_id in PERCENT_CHANGE_SERIES,
                )
            except Exception as exc:  # noqa: BLE001 - any source failure becomes a data gap
                gaps.append(f"macro {series_id}: {type(exc).__name__}: {exc}")
                continue
            if summary is None:
                gaps.append(f"macro {series_id}: no observations")
                continue
            macro[series_id] = {"label": label, **summary}
        return macro, gaps

    def gather_stock(
        self,
        symbol: str,
        macro: dict[str, Any],
        macro_gaps: list[str],
        clock: dict[str, Any] | None = None,
    ) -> StockContext:
        now = self._now()
        ctx = StockContext(
            symbol=symbol.upper(),
            generated_at=now.isoformat(),
            macro=macro,
            data_gaps=list(macro_gaps),
        )

        def attempt(label: str, fn: Callable[[], Any]) -> Any:
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - any source failure becomes a data gap
                ctx.data_gaps.append(f"{label}: {type(exc).__name__}: {exc}")
                return None

        daily = attempt(
            "daily bars",
            lambda: self._alpaca.get_bars(
                ctx.symbol,
                timeframe="1Day",
                start=now - timedelta(days=DAILY_LOOKBACK_DAYS),
                end=now,
            ),
        )
        if daily is not None:
            in_progress = bool(daily) and clock is not None and session_in_progress(daily[-1], clock)
            ctx.price = summarize_daily_bars(daily, latest_in_progress=in_progress)
            if ctx.price is None:
                ctx.data_gaps.append("daily bars: none returned")

        ctx.relative_volume = attempt(
            "relative volume", lambda: self._alpaca.get_relative_volume(ctx.symbol, clock=clock)
        )
        ctx.bars_4h = (
            attempt(
                "4h bars",
                lambda: self._alpaca.get_bars_4h(
                    ctx.symbol, start=now - timedelta(days=BARS_4H_LOOKBACK_DAYS), end=now
                ),
            )
            or []
        )
        if not ctx.bars_4h and not any(g.startswith("4h bars") for g in ctx.data_gaps):
            ctx.data_gaps.append("4h bars: none returned")

        cik = attempt("EDGAR CIK lookup", lambda: self._edgar.get_cik(ctx.symbol))
        if cik is not None:
            submissions = attempt(
                "EDGAR filings", lambda: self._edgar.get_company_submissions(ctx.symbol)
            )
            if submissions is not None:
                ctx.filings = extract_recent_filings(submissions, cik)
                ctx.company_name = submissions.get("name")
            facts = attempt(
                "EDGAR fundamentals", lambda: self._edgar.get_company_facts(ctx.symbol)
            )
            if facts is not None:
                ctx.fundamentals = extract_fundamentals(facts)
                if not ctx.fundamentals:
                    ctx.data_gaps.append("EDGAR fundamentals: no recognized us-gaap concepts")
                stale_cutoff = (now.date() - timedelta(days=FUNDAMENTALS_STALE_DAYS)).isoformat()
                for name, fact in ctx.fundamentals.items():
                    if fact["period_end"] < stale_cutoff:
                        ctx.data_gaps.append(
                            f"EDGAR fundamentals {name}: latest value is for the period "
                            f"ending {fact['period_end']} (stale)"
                        )

        news = attempt(
            "news",
            lambda: self._finnhub.get_company_news(
                ctx.symbol, now.date() - timedelta(days=NEWS_LOOKBACK_DAYS), now.date()
            ),
        )
        if news is not None:
            relevant, ctx.news_unrelated_dropped = filter_news(
                news, ctx.symbol, ctx.company_name
            )
            ctx.news = trim_news(relevant)
            if not news:
                ctx.data_gaps.append(f"news: no items in the last {NEWS_LOOKBACK_DAYS} days")
            elif not ctx.news:
                ctx.data_gaps.append(
                    f"news: {len(news)} items in the last {NEWS_LOOKBACK_DAYS} days, "
                    "none mentioning the company"
                )

        return ctx

    # --- briefing (LLM, summary only) ---

    def write_briefing(self, ctx: StockContext) -> None:
        if self._llm is None:
            return
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Prepare the briefing for {ctx.symbol}. Data:\n"
                    f"{json.dumps(ctx.facts(), indent=1, default=str)}"
                ),
            },
        ]
        candidates = self._health.order(self._models)
        self._log("llm_request", {"symbol": ctx.symbol, "models": candidates, "messages": messages})
        failure = ""
        tries = 0
        for i, model in enumerate(candidates):
            has_backup = i < len(candidates) - 1
            retry_budget = {"max_retries": BACKUP_CONNECT_RETRIES} if has_backup else {}
            for attempt in range(1, BRIEFING_ATTEMPTS + 1):
                tries += 1
                base = {"symbol": ctx.symbol, "model": model, "attempt": attempt}
                try:
                    response = self._llm.chat_completion(
                        model, messages, max_tokens=BRIEFING_MAX_TOKENS,
                        **retry_budget, **briefing_params(model),
                    )
                    choice = response["choices"][0]
                    content = _strip_reasoning(choice["message"].get("content") or "")
                    finish_reason = choice.get("finish_reason")
                except Exception as exc:  # noqa: BLE001 - analysts still get the raw facts
                    failure = f"{type(exc).__name__}: {exc}"
                    self._log("llm_error", {**base, "error": repr(exc)})
                    self._health.record_failure(model)
                    break  # the call itself failed: move on to the next model

                self._log(
                    "llm_response",
                    {**base, "finish_reason": finish_reason, "content": content,
                     "usage": response.get("usage")},
                )
                if not content:
                    failure = f"empty content (finish_reason={finish_reason})"
                    continue
                problems = briefing_problems(content)
                if problems:
                    failure = "rejected: " + "; ".join(problems)
                    self._log("briefing_rejected", {**base, "problems": problems})
                    continue
                self._health.record_success(model)
                ctx.briefing = content
                ctx.briefing_model = model
                return
            if has_backup:
                self._log("fallback", {"symbol": ctx.symbol, "from_model": model,
                                       "to_model": candidates[i + 1], "reason": failure})

        ctx.data_gaps.append(
            f"briefing: {failure} ({tries} attempt{'s' if tries != 1 else ''} across "
            f"{len(candidates)} model{'s' if len(candidates) != 1 else ''})"
        )

    # --- entry point ---

    def market_clock(self) -> tuple[dict[str, Any] | None, list[str]]:
        """Alpaca's market clock, read once per run so every stock is judged
        against the same moment. Without it, a session still running can't
        be told apart from a finished one."""
        try:
            return self._alpaca.get_clock(), []
        except Exception as exc:  # noqa: BLE001 - becomes a data gap
            return None, [
                f"market clock: {type(exc).__name__}: {exc} - can't tell whether today's "
                "session is still running, so the latest price and volume may be partial"
            ]

    def run(self, symbols: list[str]) -> dict[str, StockContext]:
        """Context packet per symbol. Macro and the market clock are fetched
        once and shared."""
        macro, macro_gaps = self.gather_macro()
        clock, clock_gaps = self.market_clock()
        contexts: dict[str, StockContext] = {}
        for symbol in symbols:
            ctx = self.gather_stock(symbol, macro, macro_gaps + clock_gaps, clock=clock)
            self._log("context_gathered", {"symbol": ctx.symbol, "facts": ctx.facts()})
            self.write_briefing(ctx)
            contexts[ctx.symbol] = ctx
        return contexts
