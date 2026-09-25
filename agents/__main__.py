"""Run the macro/context agent, optionally followed by analysts, against live APIs.

    .venv/Scripts/python -m agents AAPL MSFT NVDA
    .venv/Scripts/python -m agents AAPL --no-llm                 # data only, no Build credits
    .venv/Scripts/python -m agents AAPL --analysts analyst_1     # context + one analyst
    .venv/Scripts/python -m agents AAPL --analysts analyst_1 analyst_2

Writes the trace to logs/trace-<UTC date>.jsonl. Without --analysts it
prints each symbol's analyst-facing context block; with them, a one-line
verdict per analyst per symbol.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from data_layer import AlpacaClient, DiskCache, EdgarClient, FinnhubClient, FredClient, get_settings
from data_layer.llm_client import LlmClient

from .analyst_agent import AnalystAgent, AnalystVerdict
from .macro_agent import MacroContextAgent
from .trace import TraceLogger


def _summary(v: AnalystVerdict) -> str:
    head = f"{v.symbol} {v.role} ({v.model})"
    if not v.ok:
        return f"{head}: FAILED - {v.error}"
    check = v.evidence_check
    return (
        f"{head}: {v.recommendation.upper()} conf={v.confidence} | evidence "
        f"{check.get('verified', 0)} verified, {check.get('wrong_index', 0)} wrong index, "
        f"{check.get('mismatch', 0)} mismatch, {check.get('unknown_path', 0)} unknown path\n"
        f"  {v.thesis}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("symbols", nargs="+")
    parser.add_argument("--no-llm", action="store_true", help="skip every LLM call")
    parser.add_argument(
        "--analysts", nargs="*", default=[], metavar="ROLE",
        help="analyst roles to run after the context agent, e.g. analyst_1 analyst_2",
    )
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()
    if args.no_llm and args.analysts:
        parser.error("--analysts needs the LLM; drop --no-llm")

    settings = get_settings()
    cache = DiskCache(settings.cache_dir)
    trace = TraceLogger(
        Path(args.log_dir) / f"trace-{datetime.now(timezone.utc):%Y%m%d}.jsonl"
    )
    llm = None if args.no_llm else LlmClient(settings=settings)
    agent = MacroContextAgent(
        alpaca=AlpacaClient(settings=settings, cache=cache),
        fred=FredClient(settings=settings, cache=cache),
        edgar=EdgarClient(settings=settings, cache=cache),
        finnhub=FinnhubClient(settings=settings, cache=cache),
        llm=llm,
        trace=trace,
    )
    analysts = [AnalystAgent(llm, role, trace=trace) for role in args.analysts]

    for ctx in agent.run(args.symbols).values():
        if not analysts:
            print(ctx.to_prompt())
            print()
            continue
        for analyst in analysts:
            print(_summary(analyst.analyze(ctx)), flush=True)
    print(f"trace: {trace.path}")


if __name__ == "__main__":
    main()
