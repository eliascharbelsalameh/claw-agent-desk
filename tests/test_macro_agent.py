import json
from datetime import datetime, timezone

from agents.macro_agent import (
    BRIEFING_ATTEMPTS,
    MacroContextAgent,
    briefing_problems,
    company_short_name,
    extract_fundamentals,
    filter_news,
    extract_recent_filings,
    summarize_daily_bars,
    summarize_series,
    trim_news,
)
from agents.trace import TraceLogger

NOW = datetime(2026, 9, 25, 15, 0, tzinfo=timezone.utc)


def _daily_bars(n=25, start_close=100.0):
    return [
        {"t": f"2026-08-{i + 1:02d}T04:00:00Z", "o": 0, "h": start_close + i + 1,
         "l": start_close + i - 1, "c": start_close + i, "v": 1000}
        for i in range(n)
    ]


class FakeAlpaca:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def get_bars(self, symbol, timeframe="1Hour", start=None, end=None, **kw):
        self.calls.append(("get_bars", symbol, timeframe, start, end))
        if self.fail:
            raise ConnectionError("alpaca down")
        return _daily_bars()

    def get_relative_volume(self, symbol, lookback_days=20):
        if self.fail:
            raise ConnectionError("alpaca down")
        return {"symbol": symbol, "relative_volume": 1.4, "feed": "iex"}

    def get_bars_4h(self, symbol, start=None, end=None):
        if self.fail:
            raise ConnectionError("alpaca down")
        return [{"t": "2026-09-24T12:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10}]


class FakeFred:
    def get_series_observations(self, series_id, start_date=None, end_date=None):
        return [
            {"date": "2025-09-01", "value": "4.00"},
            {"date": "2026-08-01", "value": "."},
            {"date": "2026-09-01", "value": "3.50"},
        ]


class FakeEdgar:
    def get_cik(self, ticker):
        if ticker == "ZZZZ":
            raise KeyError("no CIK found for ticker 'ZZZZ'")
        return "0000320193"

    def get_company_submissions(self, ticker):
        return {
            "name": "Apple Inc.",
            "filings": {
                "recent": {
                    "form": ["4", "10-Q", "8-K"],
                    "filingDate": ["2026-09-01", "2026-08-01", "2026-09-10"],
                    "reportDate": ["", "2026-06-28", "2026-09-09"],
                    "accessionNumber": ["x", "0000320193-26-000010", "0000320193-26-000020"],
                    "primaryDocument": ["x.xml", "q3.htm", "8k.htm"],
                    "primaryDocDescription": ["", "10-Q", "8-K"],
                    "items": ["", "", "2.02"],
                }
            }
        }

    def get_company_facts(self, ticker):
        return {"facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [
            {"start": "2026-03-30", "end": "2026-06-28", "val": 100, "form": "10-Q", "filed": "2026-08-01"},
        ]}}}}}


class FakeFinnhub:
    def get_company_news(self, symbol, from_date, to_date):
        return [
            {"headline": "old", "datetime": 1, "age_hours": 100.0, "summary": "Apple ships"},
            {"headline": "new", "datetime": 2, "age_hours": 2.04, "summary": "AAPL moves"},
            {"headline": "Garmin rallies", "datetime": 3, "age_hours": 1.0, "summary": "watches"},
        ]


GOOD_BRIEFING = (
    "MACRO: rates at 3.50%.\nCOMPANY FILINGS & FUNDAMENTALS: 10-Q filed 2026-08-01.\n"
    "PRICE & VOLUME: last close 124.0.\nNEWS: 2 items.\nDATA GAPS: none."
)
DEGENERATE_BRIEFING = GOOD_BRIEFING.replace("last close", "The The The The The The The last close")


class FakeLlm:
    """`content` may be a list: one entry per successive call."""

    def __init__(self, content=GOOD_BRIEFING, fail=False, finish_reason="stop"):
        self.contents = list(content) if isinstance(content, list) else None
        self.content = content
        self.fail = fail
        self.finish_reason = finish_reason
        self.calls = []

    def chat_completion(self, model, messages, **kwargs):
        self.calls.append((model, messages, kwargs))
        if self.fail:
            raise TimeoutError("build timed out")
        content = self.contents.pop(0) if self.contents is not None else self.content
        return {
            "choices": [{"message": {"content": content}, "finish_reason": self.finish_reason}],
            "usage": {"total_tokens": 42},
        }


def _agent(alpaca=None, llm=None, trace=None):
    return MacroContextAgent(
        alpaca=alpaca or FakeAlpaca(),
        fred=FakeFred(),
        edgar=FakeEdgar(),
        finnhub=FakeFinnhub(),
        llm=llm,
        trace=trace,
        model="test/model",
        now=lambda: NOW,
    )


def test_summarize_series_skips_missing_and_uses_absolute_change_for_rates():
    obs = FakeFred().get_series_observations("FEDFUNDS")
    summary = summarize_series(obs)
    assert summary["latest"] == 3.5
    assert summary["prior"] == 4.0  # the "." row is dropped, not treated as 0
    assert summary["yoy_abs_change"] == -0.5
    assert "yoy_pct_change" not in summary


def test_summarize_series_percent_change_for_index():
    obs = [{"date": "2025-09-01", "value": "300"}, {"date": "2026-09-01", "value": "309"}]
    assert summarize_series(obs, pct_change=True)["yoy_pct_change"] == 3.0


def test_summarize_series_empty():
    assert summarize_series([{"date": "2026-01-01", "value": "."}]) is None


def test_extract_recent_filings_filters_forms_and_builds_url():
    filings = extract_recent_filings(FakeEdgar().get_company_submissions("AAPL"), "0000320193")
    assert [f["form"] for f in filings] == ["8-K", "10-Q"]  # newest first, Form 4 dropped
    assert filings[1]["url"] == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019326000010/q3.htm"
    )


def test_extract_fundamentals_prefers_quarter_over_ytd_for_same_end():
    facts = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [
            {"start": "2025-09-28", "end": "2026-06-28", "val": 900, "form": "10-Q", "filed": "2026-08-01"},
            {"start": "2026-03-30", "end": "2026-06-28", "val": 300, "form": "10-Q", "filed": "2026-08-01"},
            {"start": "2025-03-30", "end": "2025-06-28", "val": 250, "form": "10-Q", "filed": "2025-08-01"},
            {"start": "2026-03-30", "end": "2026-06-28", "val": 1, "form": "8-K", "filed": "2026-08-01"},
        ]}},
        "Assets": {"units": {"USD": [
            {"end": "2026-06-28", "val": 5000, "form": "10-Q", "filed": "2026-08-01"},
        ]}},
    }}}
    out = extract_fundamentals(facts)
    assert out["revenue"]["value"] == 300
    assert out["revenue"]["concept"] == "Revenues"
    assert out["total_assets"]["value"] == 5000
    assert "cash" not in out


def test_summarize_daily_bars():
    price = summarize_daily_bars(_daily_bars(25))
    assert price["last_close"] == 124.0
    assert price["change_1d_pct"] == round((124 / 123 - 1) * 100, 2)
    assert price["change_20d_pct"] == round((124 / 104 - 1) * 100, 2)
    assert price["high_20d"] == 125.0
    assert summarize_daily_bars([]) is None


def test_trim_news_newest_first_and_truncates():
    items = [{"headline": "h", "datetime": 0, "age_hours": 1.0, "summary": "x" * 1000}]
    trimmed = trim_news(items + FakeFinnhub().get_company_news("A", None, None))
    assert trimmed[0]["headline"] == "Garmin rallies"
    assert trimmed[1]["headline"] == "new"
    assert [t["index"] for t in trimmed] == list(range(len(trimmed)))
    assert trimmed[1]["age_hours"] == 2.0
    long = next(t for t in trimmed if t["headline"] == "h")
    assert long["summary"].endswith("...") and len(long["summary"]) == 303


def test_run_builds_full_context_without_llm():
    contexts = _agent().run(["aapl"])
    ctx = contexts["AAPL"]
    assert ctx.macro["FEDFUNDS"]["latest"] == 3.5
    assert ctx.macro["CPIAUCSL"]["yoy_pct_change"] == -12.5
    assert ctx.price["last_close"] == 124.0
    assert ctx.relative_volume["relative_volume"] == 1.4
    assert len(ctx.bars_4h) == 1
    assert ctx.filings[0]["form"] == "8-K"
    assert ctx.fundamentals["net_income"]["value"] == 100
    assert [n["headline"] for n in ctx.news] == ["new", "old"]
    assert ctx.news_unrelated_dropped == 1
    assert ctx.company_name == "Apple Inc."
    assert ctx.data_gaps == []
    assert ctx.briefing is None


def test_run_passes_explicit_date_range_to_alpaca():
    # get_bars returns 0 rows without explicit start/end (live-verified).
    alpaca = FakeAlpaca()
    _agent(alpaca=alpaca).run(["AAPL"])
    _, _, timeframe, start, end = alpaca.calls[0]
    assert timeframe == "1Day" and start is not None and end == NOW


def test_source_failures_become_data_gaps_not_exceptions():
    ctx = _agent(alpaca=FakeAlpaca(fail=True)).run(["ZZZZ"])["ZZZZ"]
    assert ctx.price is None and ctx.relative_volume is None and ctx.bars_4h == []
    gaps = " | ".join(ctx.data_gaps)
    assert "daily bars: ConnectionError" in gaps
    assert "relative volume" in gaps and "4h bars" in gaps
    assert "EDGAR CIK lookup: KeyError" in gaps
    assert ctx.news  # unaffected sources still come through


def test_briefing_prompt_forbids_recommendations_and_carries_facts():
    llm = FakeLlm(content="<think>hmm</think>\n" + GOOD_BRIEFING)
    ctx = _agent(llm=llm).run(["AAPL"])["AAPL"]

    assert ctx.briefing == GOOD_BRIEFING  # inline reasoning stripped
    assert ctx.briefing_model == "test/model"
    model, messages, kwargs = llm.calls[0]
    assert model == "test/model"
    assert kwargs["max_tokens"] >= 1024
    assert kwargs["chat_template_kwargs"] == {"enable_thinking": False}
    system = messages[0]["content"]
    assert "Do NOT give opinions" in system and "IEX" in system
    assert "news_unrelated_dropped" in system
    assert '"last_close": 124.0' in messages[1]["content"]
    assert "bars_4h" not in messages[1]["content"]


def test_briefing_failure_is_recorded_and_context_still_usable():
    ctx = _agent(llm=FakeLlm(fail=True)).run(["AAPL"])["AAPL"]
    assert ctx.briefing is None
    assert any(g.startswith("briefing: TimeoutError") for g in ctx.data_gaps)
    assert f"after {BRIEFING_ATTEMPTS} attempts" in ctx.data_gaps[-1]
    assert "no briefing available" in ctx.to_prompt()


def test_empty_briefing_from_length_cutoff_is_a_gap():
    ctx = _agent(llm=FakeLlm(content=None, finish_reason="length")).run(["AAPL"])["AAPL"]
    assert ctx.briefing is None
    assert ctx.data_gaps[-1].startswith("briefing: empty content (finish_reason=length)")


def test_to_prompt_contains_briefing_facts_and_limitations():
    ctx = _agent(llm=FakeLlm()).run(["AAPL"])["AAPL"]
    prompt = ctx.to_prompt()
    assert "MACRO: rates at 3.50%." in prompt
    assert "IEX exchange only" in prompt
    assert "Source facts (authoritative" in prompt


def test_trace_logs_context_request_and_response(tmp_path):
    trace = TraceLogger(tmp_path / "trace.jsonl")
    _agent(llm=FakeLlm(), trace=trace).run(["AAPL"])
    records = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]
    assert [r["event"] for r in records] == ["context_gathered", "llm_request", "llm_response"]
    assert all(r["agent"] == "macro" and r["symbol"] == "AAPL" and r["ts"] for r in records)
    assert records[2]["usage"] == {"total_tokens": 42}


def test_empty_4h_bars_is_a_gap():
    class NoIntradayAlpaca(FakeAlpaca):
        def get_bars_4h(self, symbol, start=None, end=None):
            return []

    ctx = _agent(alpaca=NoIntradayAlpaca()).run(["AAPL"])["AAPL"]
    assert "4h bars: none returned" in ctx.data_gaps


def test_company_short_name():
    assert company_short_name("Apple Inc.") == "Apple"
    assert company_short_name("Meta Platforms, Inc.") == "Meta Platforms"
    assert company_short_name("JPMORGAN CHASE & CO") == "JPMORGAN CHASE"
    assert company_short_name("ASML Holding N.V.") == "ASML"


def test_filter_news_matches_ticker_name_and_first_word():
    items = [
        {"headline": "Meta's Muse tops the charts", "summary": ""},
        {"headline": "Garmin rallies", "summary": "before Apple took the category"},
        {"headline": "ON Semiconductor", "summary": "turned on the lights"},
        {"headline": "Dell's story", "summary": "servers"},
    ]
    kept, dropped = filter_news(items, "META", "Meta Platforms, Inc.")
    assert [k["headline"] for k in kept] == ["Meta's Muse tops the charts"]
    assert dropped == 3
    # tickers are case-sensitive, so the word "on" doesn't match ticker ON
    kept, _ = filter_news(items, "ON", "ON Semiconductor Corp")
    assert [k["headline"] for k in kept] == ["ON Semiconductor"]


def test_filter_news_keeps_everything_without_company_name():
    items = [{"headline": "anything", "summary": ""}]
    assert filter_news(items, "AAPL", None) == (items, 0)


def test_all_news_filtered_out_is_a_distinct_gap():
    class OffTopicFinnhub(FakeFinnhub):
        def get_company_news(self, symbol, from_date, to_date):
            return [{"headline": "Dell", "datetime": 1, "age_hours": 1.0, "summary": "servers"}]

    agent = _agent()
    agent._finnhub = OffTopicFinnhub()
    ctx = agent.run(["AAPL"])["AAPL"]
    assert ctx.news == []
    assert "news: 1 items in the last 7 days, none mentioning the company" in ctx.data_gaps


def test_briefing_problems_flags_repetition_and_missing_sections():
    assert briefing_problems(GOOD_BRIEFING) == []
    assert briefing_problems(DEGENERATE_BRIEFING)[0].startswith("degenerate repetition")
    assert briefing_problems(GOOD_BRIEFING + " ,,,,,,,,")[0].startswith("degenerate repetition")
    missing = briefing_problems("MACRO: only this")
    assert missing == [
        "missing sections: COMPANY FILINGS & FUNDAMENTALS, PRICE & VOLUME, NEWS, DATA GAPS"
    ]
    # ordinary formatting must not trip it
    assert briefing_problems(GOOD_BRIEFING + " ... ** 1,000,000,000 --") == []


def test_degenerate_briefing_is_retried_then_accepted(tmp_path):
    trace = TraceLogger(tmp_path / "trace.jsonl")
    llm = FakeLlm(content=[DEGENERATE_BRIEFING, GOOD_BRIEFING])
    ctx = _agent(llm=llm, trace=trace).run(["AAPL"])["AAPL"]

    assert ctx.briefing == GOOD_BRIEFING
    assert len(llm.calls) == 2
    assert ctx.data_gaps == []
    events = [json.loads(l)["event"] for l in trace.path.read_text(encoding="utf-8").splitlines()]
    assert events == [
        "context_gathered", "llm_request", "llm_response", "briefing_rejected", "llm_response",
    ]


def test_persistently_degenerate_briefing_falls_back_to_facts():
    llm = FakeLlm(content=[DEGENERATE_BRIEFING] * BRIEFING_ATTEMPTS)
    ctx = _agent(llm=llm).run(["AAPL"])["AAPL"]

    assert ctx.briefing is None
    assert ctx.data_gaps[-1].startswith("briefing: rejected: degenerate repetition")
    assert "no briefing available" in ctx.to_prompt()


def test_extract_fundamentals_takes_most_recent_tag_not_first_listed():
    # NVDA, live: the ASC 606 tag stops in 2022, current revenue is under Revenues.
    facts = {"facts": {"us-gaap": {
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
            {"start": "2021-02-01", "end": "2022-01-30", "val": 26914, "form": "10-K", "filed": "2022-03-18"},
        ]}},
        "Revenues": {"units": {"USD": [
            {"start": "2026-04-27", "end": "2026-07-26", "val": 70000, "form": "10-Q", "filed": "2026-08-26"},
        ]}},
    }}}
    revenue = extract_fundamentals(facts)["revenue"]
    assert revenue["value"] == 70000
    assert revenue["concept"] == "Revenues"


def test_stale_fundamentals_become_a_gap():
    class StaleEdgar(FakeEdgar):
        def get_company_facts(self, ticker):
            return {"facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [
                {"start": "2021-11-01", "end": "2022-01-30", "val": 1, "form": "10-K", "filed": "2022-03-18"},
            ]}}}}}

    agent = _agent()
    agent._edgar = StaleEdgar()
    ctx = agent.run(["AAPL"])["AAPL"]
    assert (
        "EDGAR fundamentals net_income: latest value is for the period ending 2022-01-30 (stale)"
        in ctx.data_gaps
    )


def test_extract_fundamentals_adds_same_period_year_ago():
    facts = {"facts": {"us-gaap": {
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
            # latest quarter + its YTD twin (same end)
            {"start": "2026-03-29", "end": "2026-06-27", "val": 110, "form": "10-Q", "filed": "2026-07-31"},
            {"start": "2025-09-28", "end": "2026-06-27", "val": 330, "form": "10-Q", "filed": "2026-07-31"},
            # year-ago YTD must not be matched against the quarter
            {"start": "2024-09-29", "end": "2025-06-28", "val": 300, "form": "10-Q", "filed": "2025-08-01"},
        ]}},
        # year-ago quarter sits under an older tag
        "Revenues": {"units": {"USD": [
            {"start": "2025-03-30", "end": "2025-06-28", "val": 100, "form": "10-Q", "filed": "2025-08-01"},
        ]}},
        "Assets": {"units": {"USD": [
            {"end": "2026-06-27", "val": 500, "form": "10-Q", "filed": "2026-07-31"},
            {"end": "2025-06-28", "val": 400, "form": "10-Q", "filed": "2025-08-01"},
        ]}},
        "NetIncomeLoss": {"units": {"USD": [
            {"start": "2026-03-29", "end": "2026-06-27", "val": -50, "form": "10-Q", "filed": "2026-07-31"},
            {"start": "2025-03-30", "end": "2025-06-28", "val": -100, "form": "10-Q", "filed": "2025-08-01"},
        ]}},
    }}}
    out = extract_fundamentals(facts)
    assert out["revenue"]["year_ago_value"] == 100
    assert out["revenue"]["year_ago_period_end"] == "2025-06-28"
    assert out["revenue"]["yoy_pct_change"] == 10.0
    assert out["total_assets"]["yoy_pct_change"] == 25.0
    # a smaller loss is an improvement, so the change reads positive
    assert out["net_income"]["yoy_pct_change"] == 50.0
    assert "year_ago_value" not in out.get("cash", {})


def test_limitations_are_identical_for_every_stock_and_separate_from_gaps():
    ok = _agent().run(["AAPL"])["AAPL"]
    broken = _agent(alpaca=FakeAlpaca(fail=True)).run(["ZZZZ"])["ZZZZ"]
    assert ok.facts()["limitations"] == broken.facts()["limitations"]
    assert any("valuation" in item for item in ok.facts()["limitations"])
    assert any("IEX" in item for item in ok.facts()["limitations"])
    assert ok.data_gaps == []  # limitations never leak into this run's gaps
