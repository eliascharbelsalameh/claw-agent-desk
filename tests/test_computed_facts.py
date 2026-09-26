from datetime import date

import pytest

from agents.computed_facts import (
    atr,
    best_ttm,
    compute_technicals,
    compute_valuation,
    latest_diluted_shares,
    rsi,
    summarize_earnings,
    trailing_twelve_months,
)


def _e(start, end, val, form="10-Q", filed="2026-08-01"):
    return {"start": start, "end": end, "val": val, "form": form, "filed": filed}


def _facts(**concepts):
    units = {"EarningsPerShareDiluted": "USD/shares", "WeightedAverageNumberOfDilutedSharesOutstanding": "shares"}
    return {"facts": {"us-gaap": {
        name: {"units": {units.get(name, "USD"): entries}} for name, entries in concepts.items()
    }}}


# Apple's fiscal calendar, EPS as reported (live, Sept 26, 2026). Finnhub's
# epsTTM for AAPL that day was 8.7233.
AAPL_EPS = [
    _e("2024-09-29", "2025-06-28", 5.62, filed="2025-08-01"),
    _e("2025-03-30", "2025-06-28", 1.57, filed="2025-08-01"),
    _e("2024-09-29", "2025-09-27", 7.46, form="10-K", filed="2025-10-31"),
    _e("2025-09-28", "2025-12-27", 2.84, filed="2026-01-30"),
    _e("2025-09-28", "2026-06-27", 6.88),
    _e("2026-03-29", "2026-06-27", 2.02),
]


def test_ttm_is_fiscal_year_plus_ytd_minus_prior_ytd():
    ttm = trailing_twelve_months(AAPL_EPS)
    assert ttm["value"] == pytest.approx(7.46 + 6.88 - 5.62)
    assert ttm["value"] == pytest.approx(8.72, abs=0.01)  # Finnhub: 8.7233
    assert ttm["period_end"] == "2026-06-27"
    assert ttm["method"] == (
        "fiscal year to 2025-09-27 + year-to-date to 2026-06-27 - year-to-date to 2025-06-28"
    )


def test_ttm_when_the_latest_period_is_a_fiscal_year():
    entries = AAPL_EPS[:3]
    ttm = trailing_twelve_months(entries)
    assert ttm == {"value": 7.46, "period_start": "2024-09-29", "period_end": "2025-09-27",
                   "method": "fiscal year"}


def test_ttm_after_a_first_quarter_uses_the_quarter_as_ytd():
    entries = [
        _e("2024-12-30", "2025-03-29", 10, filed="2025-05-01"),
        _e("2024-12-30", "2025-12-27", 50, form="10-K", filed="2026-02-01"),
        _e("2025-12-28", "2026-03-28", 14, filed="2026-05-01"),
    ]
    assert trailing_twelve_months(entries)["value"] == 50 + 14 - 10


def test_ttm_missing_the_prior_ytd_is_none():
    assert trailing_twelve_months([e for e in AAPL_EPS if e["val"] != 5.62]) is None
    assert trailing_twelve_months([]) is None


def test_later_filing_of_the_same_period_wins():
    restated = [*AAPL_EPS, _e("2025-09-28", "2026-06-27", 7.00, filed="2026-09-01")]
    ttm = best_ttm(_facts(EarningsPerShareDiluted=restated), ("EarningsPerShareDiluted",))
    assert ttm["value"] == pytest.approx(7.46 + 7.00 - 5.62)


def test_best_ttm_takes_the_latest_tag_and_never_mixes_tags():
    facts = _facts(
        RevenueFromContractWithCustomerExcludingAssessedTax=[
            _e("2020-01-27", "2021-01-31", 1, form="10-K", filed="2021-03-01"),
        ],
        Revenues=[
            _e("2025-01-27", "2026-01-25", 100, form="10-K", filed="2026-03-01"),
            _e("2025-01-27", "2025-07-27", 40, filed="2025-08-27"),
            _e("2026-01-26", "2026-07-26", 60, filed="2026-08-26"),
        ],
    )
    ttm = best_ttm(facts, ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"))
    assert ttm["concept"] == "Revenues" and ttm["value"] == 120


def test_latest_diluted_shares_prefers_the_latest_quarter():
    facts = _facts(WeightedAverageNumberOfDilutedSharesOutstanding=[
        _e("2025-09-28", "2026-06-27", 14_750_302_000),
        _e("2026-03-29", "2026-06-27", 14_714_676_000),
        _e("2025-12-28", "2026-03-28", 14_725_873_000),
    ])
    assert latest_diluted_shares(facts) == {"value": 14_714_676_000, "period_end": "2026-06-27"}


def _full_facts(eps_entries=AAPL_EPS):
    scale = 1_000_000_000
    flow = [dict(e, val=e["val"] * scale) for e in AAPL_EPS]
    return _facts(
        Revenues=[dict(e, val=e["val"] * 10) for e in flow],
        NetIncomeLoss=flow,
        EarningsPerShareDiluted=eps_entries,
        WeightedAverageNumberOfDilutedSharesOutstanding=[_e("2026-03-29", "2026-06-27", 1_000_000_000)],
    )


PRICE = {"last_close": 200.0, "last_bar_date": "2026-09-25T04:00:00Z"}


def test_valuation_combines_price_with_trailing_results():
    valuation, gaps = compute_valuation(_full_facts(), PRICE)
    assert gaps == []
    assert valuation["price"] == 200.0 and valuation["price_date"] == "2026-09-25"
    assert valuation["market_cap"] == 200_000_000_000
    assert valuation["pe_ttm"] == round(200 / 8.72, 2)
    revenue_ttm = valuation["trailing_12m"]["revenue"]
    assert revenue_ttm == pytest.approx(87.2e9)
    assert valuation["ps_ttm"] == round(200e9 / revenue_ttm, 2)
    assert valuation["trailing_12m"]["period_end"] == "2026-06-27"
    assert "Trailing figures only" in valuation["note"]


def test_negative_trailing_eps_has_no_pe():
    losses = [dict(e, val=-e["val"]) for e in AAPL_EPS]
    valuation, _ = compute_valuation(_full_facts(eps_entries=losses), PRICE)
    assert valuation["pe_ttm"] is None
    assert "not meaningful" in valuation["pe_ttm_note"]


def test_valuation_without_price_or_share_count():
    assert compute_valuation(_full_facts(), None) == (None, ["valuation: no price to value the shares at"])
    facts = _full_facts()
    del facts["facts"]["us-gaap"]["WeightedAverageNumberOfDilutedSharesOutstanding"]
    valuation, gaps = compute_valuation(facts, PRICE)
    assert valuation["market_cap"] is None and valuation["ps_ttm"] is None
    assert valuation["pe_ttm"] == round(200 / 8.72, 2)
    assert gaps == ["valuation: no diluted share count in the filings, so no market cap or price-to-sales"]


def test_valuation_with_nothing_computable():
    valuation, gaps = compute_valuation({"facts": {}}, PRICE)
    assert valuation is None and len(gaps) == 4


# The closes of StockCharts' RSI worked example. By hand: the first 14
# changes gain 3.34 and lose 1.40 in total, so avg gain 0.238571, avg loss
# 0.1, RSI 70.46; the next change (-0.28) smooths them to 0.221531 and
# 0.112857, RSI 66.25. (StockCharts prints 70.53 and 66.32 because its
# table rounds the averages to 0.24 and 0.10.)
WILDER_CLOSES = [
    44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61,
    46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64, 46.21, 46.25, 45.71, 46.45, 45.78, 45.35,
    44.03, 44.18, 44.22, 44.57, 43.42, 42.66, 43.13,
]


def test_rsi_matches_the_worked_example_by_hand():
    assert rsi(WILDER_CLOSES[:15]) == 70.46
    assert rsi(WILDER_CLOSES[:16]) == 66.25
    assert 0 < rsi(WILDER_CLOSES) < 50  # the series ends in a slide
    assert rsi(WILDER_CLOSES[:14]) is None
    assert rsi([1.0] * 20) == 50.0
    assert rsi([float(i) for i in range(20)]) == 100.0


def test_atr_counts_gaps_from_the_previous_close():
    flat = [{"h": 11.0, "l": 9.0, "c": 10.0}] * 20
    assert atr(flat) == pytest.approx(2.0)
    gapped = flat[:-1] + [{"h": 14.0, "l": 13.0, "c": 13.5}]  # true range 4 (14 - previous close 10)
    assert atr(gapped) == pytest.approx((2.0 * 13 + 4.0) / 14)
    assert atr(flat[:14]) is None


def _bars(closes, start_day=1, month=6):
    bars = []
    for i, c in enumerate(closes):
        day = date.fromordinal(date(2025, month, start_day).toordinal() + i)
        bars.append({"t": f"{day.isoformat()}T04:00:00Z", "o": c, "h": c + 1, "l": c - 1, "c": c, "v": 1})
    return bars


def test_technicals_with_short_history_lists_what_is_unavailable():
    bars = _bars([100.0 + i for i in range(30)])
    tech = compute_technicals(bars, bars)
    assert tech["sma_20"] == round(sum(100.0 + i for i in range(10, 30)) / 20, 2)
    assert tech["sma_50"] is None and tech["pct_from_sma_200"] is None
    assert "sma_50: needs 50 sessions, 30 available" in tech["unavailable"]
    assert "high_52w/low_52w: cover only the 30 sessions available" in tech["unavailable"]
    assert tech["high_52w"] == 130.0 and tech["low_52w"] == 99.0
    assert tech["change_20d_vs_spy_pp"] == 0.0
    assert tech["sessions_used"] == 30
    assert "includes_running_session" not in tech


def test_technicals_relative_to_benchmark_aligns_by_date():
    stock = _bars([100.0] * 25 + [110.0])  # +10% over the last 5 sessions
    spy = _bars([400.0] * 25 + [404.0])  # +1%
    tech = compute_technicals(stock, spy)
    assert tech["spy_change_5d_pct"] == 1.0
    assert tech["change_5d_vs_spy_pp"] == 9.0
    # a benchmark missing the start date gives no comparison rather than a wrong one
    tech = compute_technicals(stock, spy[:-6] + spy[-5:])
    assert tech["change_5d_vs_spy_pp"] is None


def test_technicals_52_week_window_and_running_session():
    closes = [50.0] + [100.0] * 299
    tech = compute_technicals(_bars(closes), None, latest_in_progress=True)
    assert tech["low_52w"] == 99.0  # the 50 is older than 252 sessions
    assert tech["pct_from_sma_200"] == 0.0
    assert tech["includes_running_session"] is True
    assert "vs_spy: no SPY bars" in tech["unavailable"]
    assert compute_technicals([], None) is None


def test_summarize_earnings_next_and_latest():
    rows = [
        {"date": "2026-10-28", "hour": "amc", "quarter": 4, "year": 2026, "epsEstimate": 2.02,
         "epsActual": None, "revenueEstimate": 1.15e11, "revenueActual": None},
        {"date": "2026-08-30", "hour": "bmo", "quarter": 3, "year": 2026, "epsEstimate": 2.0,
         "epsActual": 1.9, "revenueEstimate": 1.0e11, "revenueActual": 1.1e11},
    ]
    out = summarize_earnings(rows, date(2026, 9, 26))
    assert out["next_report"] == {
        "date": "2026-10-28", "calendar_days_until": 32, "time": "after market close",
        "fiscal_period": "Q4 2026", "eps_estimate": 2.02, "revenue_estimate": 1.15e11,
    }
    latest = out["latest_report"]
    assert latest["calendar_days_ago"] == 27
    assert latest["eps_surprise_pct"] == -5.0 and latest["revenue_surprise_pct"] == 10.0
    assert "not company guidance" in out["note"]


def test_summarize_earnings_empty_and_unknown_hour():
    out = summarize_earnings([], date(2026, 9, 26))
    assert out["next_report"] is None and out["latest_report"] is None
    row = {"date": "2026-09-26", "hour": "", "epsEstimate": None, "epsActual": None}
    assert summarize_earnings([row], date(2026, 9, 26))["next_report"]["time"] == "not specified"
